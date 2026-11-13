import hmac
import json
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.errors import log, map_error
from app.api.jobs import ACTIVE, JobBusy, JobManager, NotAwaiting, build_runtime, validate_start
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


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_run: int = Field(ge=1)
    budget: int = Field(default=150, ge=1)
    max_experiments: int = Field(default=2, ge=1)
    dry_run: bool = True  # safe by default: diagnose and estimate, call no API
    approve: bool = True  # pause before each experiment


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool


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
               limiter: RateLimiter | None = None, jobs: JobManager | None = None,
               admin_token: str | None = None, validate_start_fn=None) -> FastAPI:
    app = FastAPI(title="Self-Tuning RAG", version="0.2.0")
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Admin-Token"])
    holder = ServiceHolder(service)
    limiter = limiter or RateLimiter(settings.ask_per_ip_per_hour, 3600, settings.ask_daily_cap, 86400)
    results_file, docs = Path(results_path), Path(docs_dir)
    admin = settings.admin_token if admin_token is None else admin_token
    job_manager = jobs if jobs is not None else JobManager(build_runtime, settings.job_approval_timeout_s)
    validate = validate_start_fn or validate_start

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if not admin:
            raise HTTPException(403, "Optimizer jobs are disabled on this server (no ADMIN_TOKEN is set).")
        if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), admin.encode()):
            raise HTTPException(401, "Missing or wrong admin token.")

    def job_view(job) -> dict:
        with job.cond:
            return {"id": job.id, "status": job.status, "params": job.params, "created": job.created,
                    "proposal": job.proposal, "result": job.result, "error": job.error,
                    "n_events": len(job.events)}

    def find_job(job_id: str):
        try:
            return job_manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "No such job.")

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

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(require_admin)])
    def start_job(body: JobRequest):
        if body.budget > settings.job_max_budget:
            raise HTTPException(422, f"budget is capped at {settings.job_max_budget} API calls on this server")
        if body.max_experiments > settings.job_max_experiments:
            raise HTTPException(422, f"max_experiments is capped at {settings.job_max_experiments}")
        try:
            validate(body.start_run)
        except ValueError as e:
            raise HTTPException(422, str(e))
        try:
            job = job_manager.start(body.model_dump())
        except JobBusy as e:
            raise HTTPException(409, str(e))
        return job_view(job)

    @app.get("/api/jobs", dependencies=[Depends(require_admin)])
    def list_jobs():
        return [job_view(j) for j in job_manager.all()]

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_admin)])
    def get_job(job_id: str):
        return job_view(find_job(job_id))

    @app.post("/api/jobs/{job_id}/decision", dependencies=[Depends(require_admin)])
    def decide(job_id: str, body: Decision):
        find_job(job_id)
        try:
            job_manager.decide(job_id, body.approve)
        except NotAwaiting as e:
            raise HTTPException(409, str(e))
        return {"id": job_id, "decision": "approve" if body.approve else "reject"}

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(require_admin)])
    def job_events(job_id: str, after: int = -1):
        job = find_job(job_id)

        def stream():
            i = after + 1
            while True:
                with job.cond:
                    if i >= len(job.events) and job.status in ACTIVE:
                        job.cond.wait(timeout=15)
                    batch = job.events[i:]
                    finished = job.status not in ACTIVE
                if batch:
                    for ev in batch:
                        yield f"id: {ev['i']}\ndata: {json.dumps(ev)}\n\n"
                    i += len(batch)
                elif finished:
                    return
                else:
                    yield ": keepalive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()