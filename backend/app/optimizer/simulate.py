from app.evals.failure_classifier import QuestionOutcome
from app.optimizer.candidates import GENERATION_ONLY, Candidate, flatten, rank_candidates
from app.optimizer.levers import DIMENSIONS, neighbors, plan
from app.rag.pipeline import RAGConfig


class Landscape:
    """Free retrieval results for every (chunk_size, overlap, retriever), so any config is scored instantly.
    ranks[(chunk_size, overlap, retriever)] = [(question id, rank_needed, evidence_ranks), ...] (answerable only)."""

    def __init__(self, ranks: dict[tuple, list]):
        self.ranks = ranks

    def coverage(self, c: RAGConfig) -> int:
        rows = self.ranks[(c.chunk_size, c.overlap, c.retriever)]
        return sum(1 for _, r, _ in rows if r is not None and r <= c.top_k)

    def outcomes(self, c: RAGConfig) -> list[QuestionOutcome]:
        """Assumes the model answers correctly whenever all gold evidence is in the prompt, and refuses otherwise."""
        out = []
        for qid, rank, ev in self.ranks[(c.chunk_size, c.overlap, c.retriever)]:
            ok = rank is not None and rank <= c.top_k
            out.append(QuestionOutcome(
                question_id=qid, type="factual", rank_needed=rank, evidence_ranks=list(ev),
                answer_correct=ok, faithful=True if ok else None,
                pure_refusal=not ok, mixed_refusal=False, error=None))
        return out


def random_search(land: Landscape, start: RAGConfig, space: list[RAGConfig], rng, max_exp: int) -> list[int]:
    """Best coverage after each experiment. Every sampled config costs one experiment."""
    best = land.coverage(start)
    curve = []
    for c in rng.sample(space, min(max_exp, len(space))):
        best = max(best, land.coverage(c))
        curve.append(best)
    return curve + [best] * (max_exp - len(curve))


def climb(land: Landscape, start: RAGConfig, rng, max_exp: int, guided: bool) -> dict:
    """Hill climbing with a free retrieval screen. guided=True uses the diagnosis to propose candidates,
    guided=False tries every one-step neighbor in random order, including prompt and temperature."""
    cfg, cur = start, land.coverage(start)
    best, tried = cur, {start}
    curve: list[int] = []
    wasted, path = 0, []
    while len(curve) < max_exp:
        if guided:
            found = flatten(plan(land.outcomes(cfg), cfg), tried)
        else:
            found = [(dim, c) for dim in DIMENSIONS for c in neighbors(cfg, dim) if c not in tried]
            rng.shuffle(found)
        screened = [Candidate(lever, c, i, None if lever in GENERATION_ONLY else land.coverage(c) - cur)
                    for i, (lever, c) in enumerate(found)]
        to_try, rejected = rank_candidates(screened)
        tried.update(c.config for c in rejected)
        if not to_try:
            break
        chosen = to_try[0]
        tried.add(chosen.config)
        new = land.coverage(chosen.config)
        if new > cur:
            path.append((chosen.lever, chosen.config, new - cur))
            cfg, cur = chosen.config, new
            best = max(best, new)
        else:
            wasted += 1
        curve.append(best)
    return {"curve": curve + [best] * (max_exp - len(curve)), "wasted": wasted, "path": path}