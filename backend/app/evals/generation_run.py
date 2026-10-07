import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings
from app.evals.golden import GoldenItem
from app.evals.retrieval_eval import RetrievalEvaluator
from app.llm import get_generator
from app.rag.chunker import count_tokens
from app.rag.generator import build_prompt
from app.rag.pipeline import RAGConfig

CACHE_PATH = "data/cache/generations.sqlite"
MAX_TOKENS = 256
CONTEXT_LIMIT = 3800  # Ollama's default window is 4096; longer prompts get silently truncated
STALL_S = 600  # a call slower than this almost certainly hit a sleeping laptop, not a slow model

_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "don't know based on",
    "cannot find",
    "can't find",
    "not mentioned in the",
    "does not contain",
    "do not contain",
)


def is_refusal(answer: str) -> bool:
    """Rough, rule-based check. The judge in 6.3 gives the real verdict."""
    a = answer.lower().replace("\u2019", "'")
    return any(m in a for m in _REFUSAL_MARKERS)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def make_key(model: str, base_url: str, temperature: float, max_tokens: int, prompt: str) -> str:
    payload = json.dumps(
        {"model": model, "base_url": base_url, "temperature": temperature,
         "max_tokens": max_tokens, "prompt": prompt},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GenerationCache:
    def __init__(self, path: str = CACHE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS generations ("
            "key TEXT PRIMARY KEY, answer TEXT NOT NULL, "
            "latency_s REAL NOT NULL, created_at REAL NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> tuple[str, float] | None:
        row = self.conn.execute(
            "SELECT answer, latency_s FROM generations WHERE key = ?", (key,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, key: str, answer: str, latency_s: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?)",
            (key, answer, latency_s, time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class GenerationRecord:
    id: str
    type: str
    question: str
    answer: str
    prompt_tokens: int
    latency_s: float  # for cache hits this is the latency of the original call
    cache_hit: bool
    over_context: bool
    error: str | None
    used_chunks: list[dict]


def _call_llm(llm, prompt: str, retries: int = 2) -> tuple[str, float]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            start = time.perf_counter()
            reply = llm.invoke(prompt)
            return str(reply.content).strip(), time.perf_counter() - start
        except Exception as e:  # noqa: BLE001 - any provider error should trigger a retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise last  # type: ignore[misc]


def summarize_generation(records: list[GenerationRecord]) -> dict:
    ok = [r for r in records if r.error is None]
    ans = [r for r in ok if r.type != "unanswerable"]
    un = [r for r in ok if r.type == "unanswerable"]
    lat = [r.latency_s for r in ok if r.latency_s <= STALL_S]
    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "cache_hits": sum(1 for r in ok if r.cache_hit),
        "over_context": sum(1 for r in ok if r.over_context),
        "mean_latency_s": sum(lat) / len(lat) if lat else 0.0,
        "p95_latency_s": percentile(lat, 95),
        "mean_prompt_tokens": sum(r.prompt_tokens for r in ok) / len(ok) if ok else 0.0,
        "refusal_rate_unanswerable": sum(is_refusal(r.answer) for r in un) / len(un) if un else 0.0,
        "false_refusal_rate_answerable": sum(is_refusal(r.answer) for r in ans) / len(ans) if ans else 0.0,
    }


def run_generation(
    evaluator: RetrievalEvaluator,
    config: RAGConfig,
    items: list[GoldenItem],
    cache: GenerationCache | None = None,
    verbose: bool = True,
) -> list[GenerationRecord]:
    cache = cache or GenerationCache()
    index = evaluator.index_for(config.chunk_size, config.overlap)
    llm = get_generator(temperature=config.temperature, max_tokens=MAX_TOKENS)

    records: list[GenerationRecord] = []
    for n, it in enumerate(items, start=1):
        retrieved = index.search(it.question, config.top_k)
        prompt = build_prompt(config.prompt, it.question, retrieved)
        prompt_tokens = count_tokens(prompt)
        key = make_key(settings.generator_model, settings.generator_base_url,
                       config.temperature, MAX_TOKENS, prompt)

        error = None
        hit = cache.get(key)
        if hit is not None:
            answer, latency = hit
            cache_hit = True
        else:
            cache_hit = False
            try:
                answer, latency = _call_llm(llm, prompt)
                if latency <= STALL_S:  # a stalled call has a meaningless latency, so don't cache it
                    cache.put(key, answer, latency)
            except Exception as e:  # noqa: BLE001
                answer, latency, error = "", 0.0, f"{type(e).__name__}: {e}"

        records.append(GenerationRecord(
            id=it.id, type=it.type, question=it.question, answer=answer,
            prompt_tokens=prompt_tokens, latency_s=latency, cache_hit=cache_hit,
            over_context=prompt_tokens > CONTEXT_LIMIT, error=error,
            used_chunks=[{"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                          "start": r.chunk.start, "end": r.chunk.end,
                          "score": round(r.score, 4), "rank": r.rank} for r in retrieved],
        ))
        if verbose:
            tag = "cached" if cache_hit else f"{latency:.1f}s"
            flag = f"  ERROR {error}" if error else ""
            print(f"[{n}/{len(items)}] {it.id} ({tag}){flag}", flush=True)
    return records


def save_generation(records: list[GenerationRecord], config: RAGConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": asdict(config),
        "model": settings.generator_model,
        "summary": summarize_generation(records),
        "records": [asdict(r) for r in records],
    }
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")