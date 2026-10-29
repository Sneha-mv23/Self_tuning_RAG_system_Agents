import hashlib
import json

from app.rag.pipeline import RAGConfig

# The configs compared on the held-out test split. They are fixed in code on purpose: changing them
# shows up in git history, which is the audit trail showing the test split was not used for tuning.
EVAL_CONFIGS = {
    "baseline": RAGConfig(),                           # dense, 256/32, top_k 4: where the optimizer started
    "optimized": RAGConfig(top_k=6),                   # the optimizer's accepted result
    "cheap": RAGConfig(top_k=3, retriever="hybrid"),   # a cost-saving point from the Pareto frontier
}
MAX_REPLICATES = 3


def fingerprint(items) -> str:
    """Hash of the test questions, so the lock can detect that they changed after results existed."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i.id):
        h.update(json.dumps(it.model_dump(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def run_order(names: list[str], replicates: int) -> list[tuple[str, int]]:
    """Interleaved, so that if the quota runs out every config has one replicate before any has two."""
    return [(n, r) for r in range(replicates) for n in names]


def pending(order: list[tuple[str, int]], completed: list[dict]) -> list[tuple[str, int]]:
    done = {(c["config"], c["replicate"]) for c in completed}
    return [x for x in order if x not in done]