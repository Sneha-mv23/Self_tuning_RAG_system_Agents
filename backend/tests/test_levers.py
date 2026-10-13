from dataclasses import replace

from app.evals.failure_classifier import QuestionOutcome
from app.optimizer.levers import estimated_prompt_tokens, is_valid, neighbors, plan, recoverable_by_top_k
from app.rag.pipeline import RAGConfig


def o(qid, **kw):
    base = dict(question_id=qid, type="factual", rank_needed=1, evidence_ranks=[1], answer_correct=True,
                faithful=True, pure_refusal=False, mixed_refusal=False, error=None)
    base.update(kw)
    return QuestionOutcome(**base)


def ranking_failure(qid, rank):
    return o(qid, rank_needed=rank, evidence_ranks=[rank], answer_correct=False, faithful=None, pure_refusal=True)


def over_refusal(qid):
    return o(qid, answer_correct=False, faithful=None, pure_refusal=True)


def test_default_config_is_valid():
    assert is_valid(RAGConfig())


def test_invalid_configs():
    assert not is_valid(RAGConfig(chunk_size=512, overlap=32))
    assert not is_valid(RAGConfig(chunk_size=256, overlap=32, top_k=10))  # prompt too long
    assert not is_valid(RAGConfig(chunk_size=128, overlap=48))
    assert estimated_prompt_tokens(RAGConfig(chunk_size=256, top_k=10)) > 2400


def test_neighbors_change_exactly_one_field():
    base = RAGConfig()
    for dim in ("chunk_size", "overlap", "top_k", "prompt", "temperature"):
        for cand in neighbors(base, dim):
            changed = [f for f in ("chunk_size", "overlap", "top_k", "prompt", "temperature")
                       if getattr(cand, f) != getattr(base, f)]
            assert changed == [dim]
            assert is_valid(cand)


def test_recoverable_by_top_k():
    outcomes = [ranking_failure("a", 5), ranking_failure("b", 6), over_refusal("c")]
    rec = recoverable_by_top_k(outcomes, 4)
    assert rec[5] == ["a"]
    assert rec[6] == ["a", "b"]


def test_plan_orders_available_levers_first_and_picks_smallest_best_top_k():
    outcomes = [ranking_failure("a", 5), ranking_failure("b", 6),
                over_refusal("c"), over_refusal("d"), over_refusal("e")]
    proposals = plan(outcomes, RAGConfig())
    assert [p.lever for p in proposals[:2]] == ["prompt", "top_k"]
    assert proposals[0].n_addressed == 3
    top_k = proposals[1]
    assert top_k.n_addressed == 2 and [c.top_k for c in top_k.candidates] == [6]
    assert all(not p.available and p.candidates == [] for p in proposals[2:])
    assert {p.lever for p in proposals[2:]} == {"reranker", "hybrid_search"}