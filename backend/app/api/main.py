import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.errors import log, map_error
from app.api.limits import RateLimiter
from app.config import settings
from app.optimizer.levers import DIMENSIONS, is_valid
from app.optimizer.test_plan import EVAL_CONFIGS
from app.rag.pipeline import RAGConfig

DEFAULT_CONFIG = EVAL_CONFIGS["optimized"]  # the optimizer's accepted result: dense, 256/32, top_k 6


class AskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_size: int | None = None
    overlap: int | None = None
    top_k: int | None = None
    prompt: str | None = None
    temperature: float | None = None
    retriever: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    config: AskConfig = Field(default_factory=AskConfig)

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


def resolve_config(overrides: AskConfig) -> RAGConfig:
    values = {**asdict(DEFAULT_CONFIG), **overrides.model_dump(exclude_none=True)}
    try:
        cfg = RAGConfig(**values)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not is_valid(cfg):
        allowed = {k: list(v) for k, v in DIMENSIONS.items()}
        raise HTTPException(422, f"that combination of settings is not allowed. Allowed values: {allowed}")
    return cfg


class ServiceHolder:
    """Builds the RAG service on first use, so importing the app loads no models."""

    def __init__(self, service=None):
        self._service = service
        self._build_lock = threading.Lock()
        self.ask_lock = threading.Lock()

    def get(self):
        if self._service is None:
            with self._build_lock:
                if self._service is None:
                    from app.graphs.rag_graph import RAGService
                    self._service = RAGService()
        return self._service


def create_app(service=None, results_path: str = "experiments/export/results.json", docs_dir: str = "docs",
               limiter: RateLimiter | None = None) -> FastAPI:
    app = FastAPI(title="Self-Tuning RAG", version="0.1.0")
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])
    holder = ServiceHolder(service)
    limiter = limiter or RateLimiter(settings.ask_per_ip_per_hour, 3600, settings.ask_daily_cap, 86400)
    results_file, docs = Path(results_path), Path(docs_dir)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "results_available": results_file.exists(), "generator": settings.generator_model}

    @app.get("/api/defaults")
    def defaults():
        return {"config": asdict(DEFAULT_CONFIG), "allowed": {k: list(v) for k, v in DIMENSIONS.items()}}

    @app.get("/api/results")
    def results():
        if not results_file.exists():
            raise HTTPException(404, "No exported results yet. Run backend/scripts/export_results.py.")
        return Response(content=results_file.read_text(encoding="utf-8"), media_type="application/json")

    @app.get("/api/graphs")
    def graphs():
        out = {}
        for name, filename in (("rag", "rag_graph.mmd"), ("optimizer", "optimizer_graph.mmd")):
            path = docs / filename
            if path.exists():
                out[name] = path.read_text(encoding="utf-8-sig")  # the files were written with a BOM
        return out

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request):
        cfg = resolve_config(body.config)  # validate first, so a bad request never uses up the limit
        client = request.client.host if request.client else "unknown"
        allowed, reason, retry_after = limiter.check(client)
        if not allowed:
            raise HTTPException(429, f"Rate limit: {reason}. Try again later.",
                                headers={"Retry-After": str(retry_after)})
        try:
            svc = holder.get()
            with holder.ask_lock:
                return svc.ask(body.question, asdict(cfg))
        except Exception as exc:  # noqa: BLE001 - mapped to a safe response below
            status, message = map_error(exc)
            log.exception("ask failed with status %s", status)
            raise HTTPException(status, message)

    return app


app = create_app()