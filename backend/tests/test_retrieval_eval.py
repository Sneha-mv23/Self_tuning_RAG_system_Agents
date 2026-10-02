from app.evals.retrieval_eval import QuestionRetrieval, summarize


def q(id, type, rank, ev_ranks):
    return QuestionRetrieval(id=id, type=type, rank_needed=rank, evidence_ranks=ev_ranks,
                             top1_score=0.5, retrieved=[])


RESULTS = [
    q("a", "factual", 1, [1]),
    q("b", "factual", 5, [5]),
    q("c", "multi_hop", None, [2, None]),
    q("d", "unanswerable", None, []),
]


def test_recall_counts_only_answerable():
    s = summarize(RESULTS, top_k=4)
    assert abs(s["recall"][4] - 1 / 3) < 1e-9
    assert abs(s["recall"][5] - 2 / 3) < 1e-9


def test_evidence_recall_gives_partial_credit():
    s = summarize(RESULTS, top_k=4)
    assert abs(s["evidence_recall"][4] - 0.5) < 1e-9  # (1 + 0 + 0.5) / 3


def test_mrr():
    s = summarize(RESULTS, top_k=4)
    assert abs(s["mrr"] - 0.4) < 1e-9  # (1 + 0.2 + 0) / 3


def test_recall_by_type_uses_top_k():
    s = summarize(RESULTS, top_k=4)
    assert s["recall_by_type"] == {"factual": 0.5, "multi_hop": 0.0}