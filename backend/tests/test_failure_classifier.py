from app.db.experiments import connect
from app.evals.failure_classifier import (
    CATEGORIES, STAGE, QuestionOutcome, classify, load_outcomes,
)


def o(**kw):
    base = dict(question_id="q", type="factual", rank_needed=1, evidence_ranks=[1],
                answer_correct=True, faithful=True, pure_refusal=False, mixed_refusal=False, error=None)
    base.update(kw)
    return QuestionOutcome(**base)


def test_stage_covers_every_category():
    assert set(STAGE) == set(CATEGORIES)


def test_pass_and_unanswerable():
    assert classify(o(), 4) == "pass"
    assert classify(o(type="unanswerable", rank_needed=None, evidence_ranks=[],
                      answer_correct=True, faithful=None, pure_refusal=True), 4) == "pass"
    assert classify(o(type="unanswerable", rank_needed=None, evidence_ranks=[],
                      answer_correct=False, faithful=None), 4) == "failed_to_decline"


def test_retrieval_categories_take_priority_over_refusal():
    refused = dict(answer_correct=False, faithful=None, pure_refusal=True)
    assert classify(o(rank_needed=None, evidence_ranks=[None], **refused), 4) == "retrieval_miss"
    assert classify(o(type="multi_hop", rank_needed=None, evidence_ranks=[2, None], **refused), 4) == "partial_retrieval"
    assert classify(o(rank_needed=6, evidence_ranks=[6], **refused), 4) == "ranking_failure"


def test_generation_categories_when_gold_is_in_prompt():
    bad = dict(rank_needed=2, evidence_ranks=[2])
    assert classify(o(answer_correct=False, faithful=None, pure_refusal=True, **bad), 4) == "over_refusal"
    assert classify(o(mixed_refusal=True, **bad), 4) == "mixed_refusal"
    assert classify(o(answer_correct=False, faithful=False, **bad), 4) == "hallucination"
    assert classify(o(answer_correct=False, faithful=True, **bad), 4) == "incorrect_answer"
    assert classify(o(answer_correct=True, faithful=False, **bad), 4) == "unsupported_claims"


def test_correct_answer_passes_even_if_retrieval_missed_the_label():
    assert classify(o(rank_needed=None, evidence_ranks=[None]), 4) == "pass"


def test_errors():
    assert classify(o(error="boom"), 4) == "error"
    assert classify(o(answer_correct=None), 4) == "error"


def test_load_outcomes_from_database():
    conn = connect(":memory:")
    conn.execute("INSERT INTO runs (created_at, split, top_k, config_key, generator_model, judge_model, metrics_json) "
                 "VALUES (0, 'dev', 4, 'k', 'g', 'j', '{}')")
    conn.execute("INSERT INTO question_results (run_id, question_id, type, rank_needed, evidence_ranks_json, "
                 "answer_correct, faithful, pure_refusal, mixed_refusal) "
                 "VALUES (1, 'a', 'factual', 6, '[6]', 0, NULL, 1, 0)")
    top_k, outcomes = load_outcomes(conn, 1)
    assert top_k == 4
    assert outcomes[0].pure_refusal is True and outcomes[0].faithful is None
    assert classify(outcomes[0], top_k) == "ranking_failure"