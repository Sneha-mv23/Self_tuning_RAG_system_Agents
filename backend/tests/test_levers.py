from app.evals.failure_classifier import QuestionOutcome
from app.optimizer.levers import estimated_prompt_tokens, is_valid, neighbors, plan, recoverable_by_top_k
from app.rag.pipeline import RAGConfig

FIELDS = ("chunk_size", "overlap", "top_k", "prompt", "temperature", "retriever")


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
    for dim in FIELDS:
        for cand in neighbors(base, dim):
            changed = [f for f in FIELDS if getattr(cand, f) != getattr(base, f)]
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
    assert [p.lever for p in proposals] == ["prompt", "hybrid_search", "top_k", "reranker"]
    assert proposals[0].n_addressed == 3
    assert [c.retriever for c in proposals[1].candidates] == ["hybrid"]
    top_k = proposals[2]
    assert top_k.n_addressed == 2 and [c.top_k for c in top_k.candidates] == [6]
    assert not proposals[3].available and proposals[3].candidates == []

def failed_to_decline(qid):
    return o(qid, type="unanswerable", rank_needed=None, evidence_ranks=[], answer_correct=False, faithful=None)


def _prompt_candidates(outcomes, prompt):
    proposal = [p for p in plan(outcomes, RAGConfig(prompt=prompt)) if p.lever == "prompt"][0]
    return [c.prompt for c in proposal.candidates]


def test_prompt_direction_depends_on_the_failure():
    assert _prompt_candidates([failed_to_decline("a")], "strict") == []  # already the strictest
    assert _prompt_candidates([failed_to_decline("a")], "basic") == ["strict"]
    assert _prompt_candidates([over_refusal("a")], "strict") == ["basic"]