from types import SimpleNamespace

from app.db.experiments import connect, list_runs, log_run, recall_at

CONFIG = {"chunk_size": 256, "overlap": 32, "top_k": 4, "prompt": "strict", "temperature": 0.0}


def q(id, type_, rank):
    return SimpleNamespace(id=id, type=type_, rank_needed=rank, evidence_ranks=[rank], top1_score=0.5)


REPORT = SimpleNamespace(
    questions=[q("a", "factual", 1), q("b", "factual", 6), q("c", "unanswerable", None)],
    mrr=0.58, n_answerable=2, n_unanswerable=1,
    recall={4: 0.5}, evidence_recall={4: 0.5}, recall_by_type={"factual": 0.5},
)
GENERATION = {
    "model": "gen-model",
    "summary": {"mean_prompt_tokens": 1000.0},
    "records": [
        {"id": i, "answer": "x", "prompt_tokens": 1000, "latency_s": lat, "error": None}
        for i, lat in (("a", 90.0), ("b", 110.0), ("c", 99999.0))  # c is a stalled call
    ],
}
JUDGED = {
    "judge_model": "judge-model",
    "summary": {"answer_correctness": 0.5, "faithfulness": 1.0, "proper_refusal_rate_unanswerable": 1.0,
                "pure_refusal_rate_answerable": 0.0, "mixed_refusal_rate": 0.0},
    "judgments": [
        {"id": "a", "correct": True, "faithful": True, "pure_refusal": False, "mixed_refusal": False, "error": None},
        {"id": "b", "correct": False, "faithful": True, "pure_refusal": False, "mixed_refusal": False, "error": None},
        {"id": "c", "correct": True, "faithful": None, "pure_refusal": True, "mixed_refusal": False, "error": None},
    ],
}


def test_recall_at_ignores_unanswerable():
    assert recall_at(REPORT, 4) == 0.5
    assert recall_at(REPORT, 6) == 1.0


def test_log_run_stores_metrics_and_excludes_stalls():
    conn = connect(":memory:")
    log_run(conn, "dev", CONFIG, REPORT, GENERATION, JUDGED)
    row = list_runs(conn)[0]
    assert row["recall_at_k"] == 0.5 and row["answer_correctness"] == 0.5
    assert row["median_latency_s"] == 100.0  # the stalled call is excluded
    assert conn.execute("SELECT COUNT(*) FROM question_results").fetchone()[0] == 3


def test_logging_twice_replaces_instead_of_duplicating():
    conn = connect(":memory:")
    log_run(conn, "dev", CONFIG, REPORT, GENERATION, JUDGED)
    log_run(conn, "dev", CONFIG, REPORT, GENERATION, JUDGED)
    assert len(list_runs(conn)) == 1
    assert conn.execute("SELECT COUNT(*) FROM question_results").fetchone()[0] == 3


def test_different_judge_is_a_separate_run():
    conn = connect(":memory:")
    log_run(conn, "dev", CONFIG, REPORT, GENERATION, JUDGED)
    log_run(conn, "dev", CONFIG, REPORT, GENERATION, {**JUDGED, "judge_model": "other"})
    assert len(list_runs(conn)) == 2