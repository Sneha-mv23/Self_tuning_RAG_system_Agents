import json
import sqlite3
import statistics
import time
from pathlib import Path

from app.evals.generation_run import STALL_S, percentile

DB_PATH = "experiments/experiments.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    split TEXT NOT NULL,
    chunk_size INTEGER, overlap INTEGER, top_k INTEGER, prompt TEXT, temperature REAL,
    config_key TEXT NOT NULL,
    generator_model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    n_answerable INTEGER, n_unanswerable INTEGER,
    recall_at_k REAL, mrr REAL,
    answer_correctness REAL, faithfulness REAL,
    proper_refusal REAL, pure_refusal_answerable REAL, mixed_refusal REAL,
    median_latency_s REAL, p95_latency_s REAL, mean_prompt_tokens REAL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS question_results (
    run_id INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    type TEXT NOT NULL,
    rank_needed INTEGER,
    evidence_ranks_json TEXT,
    top1_score REAL,
    answer TEXT,
    answer_correct INTEGER,
    faithful INTEGER,
    pure_refusal INTEGER,
    mixed_refusal INTEGER,
    prompt_tokens INTEGER,
    latency_s REAL,
    error TEXT,
    PRIMARY KEY (run_id, question_id)
);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def recall_at(report, k: int) -> float:
    answerable = [q for q in report.questions if q.type != "unanswerable"]
    if not answerable:
        return 0.0
    return sum(1 for q in answerable if q.rank_needed is not None and q.rank_needed <= k) / len(answerable)


def _b(x):
    return None if x is None else int(bool(x))


def log_run(conn: sqlite3.Connection, split: str, config: dict, report, generation: dict, judged: dict) -> int:
    """Insert one run. Logging the same (split, config, models) again replaces the old row."""
    config_key = json.dumps(config, sort_keys=True)
    gen_model, judge_model = generation["model"], judged["judge_model"]

    old = conn.execute(
        "SELECT id FROM runs WHERE split=? AND config_key=? AND generator_model=? AND judge_model=?",
        (split, config_key, gen_model, judge_model),
    ).fetchall()
    for row in old:
        conn.execute("DELETE FROM question_results WHERE run_id=?", (row["id"],))
        conn.execute("DELETE FROM runs WHERE id=?", (row["id"],))

    records = {r["id"]: r for r in generation["records"]}
    judgments = {j["id"]: j for j in judged["judgments"]}
    lat = [r["latency_s"] for r in generation["records"] if r["error"] is None and r["latency_s"] <= STALL_S]
    js, gs = judged["summary"], generation["summary"]

    values = {
        "created_at": time.time(), "split": split,
        "chunk_size": config["chunk_size"], "overlap": config["overlap"], "top_k": config["top_k"],
        "prompt": config["prompt"], "temperature": config["temperature"],
        "config_key": config_key, "generator_model": gen_model, "judge_model": judge_model,
        "n_answerable": report.n_answerable, "n_unanswerable": report.n_unanswerable,
        "recall_at_k": recall_at(report, config["top_k"]), "mrr": report.mrr,
        "answer_correctness": js["answer_correctness"], "faithfulness": js["faithfulness"],
        "proper_refusal": js["proper_refusal_rate_unanswerable"],
        "pure_refusal_answerable": js["pure_refusal_rate_answerable"],
        "mixed_refusal": js["mixed_refusal_rate"],
        "median_latency_s": statistics.median(lat) if lat else 0.0,
        "p95_latency_s": percentile(lat, 95),
        "mean_prompt_tokens": gs["mean_prompt_tokens"],
        "metrics_json": json.dumps({
            "recall": report.recall, "evidence_recall": report.evidence_recall,
            "recall_by_type": report.recall_by_type,
            "generation_summary": gs, "judge_summary": js,
        }),
    }
    cols = ", ".join(values)
    marks = ", ".join(f":{k}" for k in values)
    run_id = conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", values).lastrowid

    for q in report.questions:
        r, j = records.get(q.id), judgments.get(q.id)
        conn.execute(
            "INSERT INTO question_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, q.id, q.type, q.rank_needed, json.dumps(q.evidence_ranks), q.top1_score,
             r["answer"] if r else None,
             _b(j["correct"]) if j else None, _b(j["faithful"]) if j else None,
             _b(j["pure_refusal"]) if j else None, _b(j["mixed_refusal"]) if j else None,
             r["prompt_tokens"] if r else None, r["latency_s"] if r else None,
             (j["error"] if j and j["error"] else (r["error"] if r else None))),
        )
    conn.commit()
    return run_id


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()