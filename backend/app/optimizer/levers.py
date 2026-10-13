from collections import Counter
from dataclasses import dataclass, replace

from app.evals.failure_classifier import QuestionOutcome, classify
from app.rag.pipeline import RAGConfig

CHUNK_SIZES = (128, 192, 256)  # above 256 the embedder cannot see the tail of each chunk
OVERLAPS = (0, 16, 32)
TOP_KS = (3, 4, 5, 6, 8, 10)
PROMPTS = ("strict", "basic")  # a middle prompt is added in step 8
TEMPERATURES = (0.0, 0.3)
MAX_PROMPT_TOKENS = 2400  # keeps CPU generation affordable; Ollama's hard window is 4096

DIMENSIONS = {
    "chunk_size": CHUNK_SIZES,
    "overlap": OVERLAPS,
    "top_k": TOP_KS,
    "prompt": PROMPTS,
    "temperature": TEMPERATURES,
}

# Which levers can address which failure category. Levers not in AVAILABLE are built in step 8.
DIAGNOSIS = {
    "ranking_failure": ["top_k", "reranker", "hybrid_search"],
    "retrieval_miss": ["hybrid_search", "query_rewrite", "chunk_size"],
    "partial_retrieval": ["query_decomposition", "hybrid_search"],
    "over_refusal": ["prompt"],
    "mixed_refusal": ["prompt"],
    "failed_to_decline": ["prompt"],
    "incorrect_answer": ["prompt", "temperature"],
    "hallucination": ["prompt", "temperature"],
    "unsupported_claims": ["prompt"],
}
AVAILABLE = {"top_k", "chunk_size", "overlap", "prompt", "temperature"}


def estimated_prompt_tokens(c: RAGConfig) -> int:
    """Upper bound: each chunk is at most chunk_size tokens, plus ~15 for its header, plus instructions."""
    return c.top_k * (c.chunk_size + 15) + 120


def is_valid(c: RAGConfig) -> bool:
    return (
        c.chunk_size in CHUNK_SIZES and c.overlap in OVERLAPS and c.top_k in TOP_KS
        and c.prompt in PROMPTS and c.temperature in TEMPERATURES
        and c.overlap <= c.chunk_size // 4
        and estimated_prompt_tokens(c) <= MAX_PROMPT_TOKENS
    )


def neighbors(c: RAGConfig, dimension: str) -> list[RAGConfig]:
    """Valid configs that differ from c in exactly one dimension, one step away."""
    values = DIMENSIONS[dimension]
    current = getattr(c, dimension)
    if current not in values:
        return []
    i = values.index(current)
    out = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(values):
            cand = replace(c, **{dimension: values[j]})
            if is_valid(cand):
                out.append(cand)
    return out


def recoverable_by_top_k(outcomes: list[QuestionOutcome], top_k: int) -> dict[int, list[str]]:
    """For each larger top_k: ids of ranking failures whose gold evidence would then be in the prompt.
    An upper bound on questions recovered, because the model must still answer them correctly."""
    ranking = [o for o in outcomes if classify(o, top_k) == "ranking_failure"]
    return {
        k: sorted(o.question_id for o in ranking if o.rank_needed is not None and o.rank_needed <= k)
        for k in TOP_KS if k > top_k
    }


@dataclass(frozen=True)
class Proposal:
    lever: str
    n_addressed: int  # top_k: ranking failures recoverable. Others: failures in the categories it targets.
    available: bool
    candidates: list[RAGConfig]
    note: str


def plan(outcomes: list[QuestionOutcome], config: RAGConfig) -> list[Proposal]:
    counts = Counter(classify(o, config.top_k) for o in outcomes)
    targets: dict[str, dict[str, int]] = {}
    for category, levers in DIAGNOSIS.items():
        if counts.get(category):
            for lever in levers:
                targets.setdefault(lever, {})[category] = counts[category]

    recoverable = recoverable_by_top_k(outcomes, config.top_k)
    valid_ks = [k for k in recoverable if is_valid(replace(config, top_k=k))]

    proposals = []
    for lever, addr in targets.items():
        available = lever in AVAILABLE
        if lever == "top_k":
            best = max((len(recoverable[k]) for k in valid_ks), default=0)
            ks = [k for k in valid_ks if best > 0 and len(recoverable[k]) == best][:1]
            candidates = [replace(config, top_k=k) for k in ks]
            n = best
            note = "ranking failures recoverable at top_k " + ", ".join(
                f"{k}: {len(recoverable[k])}" + ("" if k in valid_ks else " (prompt too long)")
                for k in recoverable
            )
        else:
            candidates = neighbors(config, lever) if available else []
            n = sum(addr.values())
            note = "targets " + ", ".join(f"{c} {v}" for c, v in addr.items())
        proposals.append(Proposal(lever, n, available, candidates, note))
    proposals.sort(key=lambda p: (not p.available, -p.n_addressed, p.lever))
    return proposals