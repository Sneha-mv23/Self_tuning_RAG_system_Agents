import json
import sqlite3
import time

STEPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS optimizer_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    step INTEGER NOT NULL,
    created_at REAL NOT NULL,
    parent_run_id INTEGER NOT NULL,
    candidate_run_id INTEGER,
    lever TEXT NOT NULL,
    change_json TEXT NOT NULL,
    screen_gain INTEGER,
    parent_passes INTEGER,
    candidate_passes INTEGER,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    api_calls INTEGER NOT NULL DEFAULT 0
);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(STEPS_SCHEMA)
    conn.commit()


def log_step(conn: sqlite3.Connection, session: str, step: int, parent_run_id: int, lever: str,
             change: dict, decision: str, reason: str, candidate_run_id: int | None = None,
             screen_gain: int | None = None, parent_passes: int | None = None,
             candidate_passes: int | None = None, api_calls: int = 0) -> int:
    cur = conn.execute(
        "INSERT INTO optimizer_steps (session, step, created_at, parent_run_id, candidate_run_id, lever, "
        "change_json, screen_gain, parent_passes, candidate_passes, decision, reason, api_calls) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (session, step, time.time(), parent_run_id, candidate_run_id, lever, json.dumps(change),
         screen_gain, parent_passes, candidate_passes, decision, reason, api_calls),
    )
    conn.commit()
    return cur.lastrowid