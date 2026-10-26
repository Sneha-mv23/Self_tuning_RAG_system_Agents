from dataclasses import dataclass

from app.evals.golden import GoldenItem
from app.evals.retrieval_eval import RetrievalEvaluator
from app.optimizer.levers import Proposal
from app.rag.pipeline import RAGConfig

GENERATION_ONLY = {"prompt", "temperature"}  # cannot be screened without calling the model
CONFIG_FIELDS = ("chunk_size", "overlap", "top_k", "prompt", "temperature", "retriever")


@dataclass(frozen=True)
class Candidate:
    lever: str
    config: RAGConfig
    order: int  # position in the diagnosis ranking, used to break ties
    screen_gain: int | None  # extra questions with ALL gold evidence in the prompt; None = cannot be screened


def config_diff(a: RAGConfig, b: RAGConfig) -> dict:
    return {f: [getattr(a, f), getattr(b, f)] for f in CONFIG_FIELDS if getattr(a, f) != getattr(b, f)}


def covered_count(evaluator: RetrievalEvaluator, config: RAGConfig, items: list[GoldenItem]) -> int:
    report = evaluator.evaluate(config, items)
    return sum(1 for q in report.questions
               if q.type != "unanswerable" and q.rank_needed is not None and q.rank_needed <= config.top_k)


def flatten(proposals: list[Proposal], tried: set) -> list[tuple[str, RAGConfig]]:
    seen: set = set()
    out: list[tuple[str, RAGConfig]] = []
    for p in proposals:
        if not p.available:
            continue
        for c in p.candidates:
            if c in tried or c in seen:
                continue
            seen.add(c)
            out.append((p.lever, c))
    return out


def screen(evaluator: RetrievalEvaluator, items: list[GoldenItem], current: RAGConfig,
           candidates: list[tuple[str, RAGConfig]]) -> list[Candidate]:
    base = covered_count(evaluator, current, items)
    return [
        Candidate(lever, cfg, order,
                  None if lever in GENERATION_ONLY else covered_count(evaluator, cfg, items) - base)
        for order, (lever, cfg) in enumerate(candidates)
    ]


def rank_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """(candidates worth an API run, best first; candidates the free screen rejects)."""
    rejected = [c for c in candidates if c.screen_gain is not None and c.screen_gain <= 0]
    keep = [c for c in candidates if c not in rejected]
    keep.sort(key=lambda c: (0 if (c.screen_gain or 0) > 0 else 1, -(c.screen_gain or 0), c.order))
    return keep, rejected