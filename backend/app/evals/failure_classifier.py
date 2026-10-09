import json
import sqlite3
from dataclasses import dataclass

CATEGORIES = (
    "pass",
    "retrieval_miss", "partial_retrieval", "ranking_failure",
    "over_refusal", "mixed_refusal", "hallucination", "incorrect_answer", "unsupported_claims",
    "failed_to_decline",
    "error",
)

STAGE = {
    "pass": "none",
    "retrieval_miss": "retrieval", "partial_retrieval": "retrieval", "ranking_failure": "retrieval",
    "over_refusal": "generation", "mixed_refusal": "generation", "hallucination": "generation",
    "incorrect_answer": "generation", "unsupported_claims": "generation",
    "failed_to_decline": "generation",
    "error": "error",
}


@dataclass(frozen=True)
class QuestionOutcome:
    question_id: str
    type: str
    rank_needed: int | None  # smallest top_k covering ALL evidence; None if some is missing from the top 20
    evidence_ranks: list[int | None]
    answer_correct: bool | None  # answerable: matches the reference. unanswerable: properly declined.
    faithful: bool | None
    pure_refusal: bool
    mixed_refusal: bool
    error: str | None


def classify(o: QuestionOutcome, top_k: int) -> str:
    if o.error or o.answer_correct is None:
        return "error"

    if o.type == "unanswerable":
        return "pass" if o.answer_correct else "failed_to_decline"

    if o.answer_correct and o.faithful is not False and not o.mixed_refusal:
        return "pass"

    # The question failed. Retrieval first: did the gold evidence reach the prompt?
    if o.rank_needed is None:
        found = [r for r in o.evidence_ranks if r is not None]
        return "partial_retrieval" if found else "retrieval_miss"
    if o.rank_needed > top_k:
        return "ranking_failure"

    # The gold evidence was in the prompt, so this is a generation failure.
    if o.pure_refusal:
        return "over_refusal"
    if o.mixed_refusal:
        return "mixed_refusal"
    if not o.answer_correct:
        return "hallucination" if o.faithful is False else "incorrect_answer"
    return "unsupported_claims"


def _opt_bool(x) -> bool | None:
    return None if x is None else bool(x)


def load_outcomes(conn: sqlite3.Connection, run_id: int) -> tuple[int, list[QuestionOutcome]]:
    run = conn.execute("SELECT top_k FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise ValueError(f"no run with id {run_id}")
    rows = conn.execute(
        "SELECT * FROM question_results WHERE run_id = ? ORDER BY question_id", (run_id,)
    ).fetchall()
    outcomes = [
        QuestionOutcome(
            question_id=r["question_id"], type=r["type"], rank_needed=r["rank_needed"],
            evidence_ranks=json.loads(r["evidence_ranks_json"] or "[]"),
            answer_correct=_opt_bool(r["answer_correct"]), faithful=_opt_bool(r["faithful"]),
            pure_refusal=bool(r["pure_refusal"]), mixed_refusal=bool(r["mixed_refusal"]),
            error=r["error"],
        )
        for r in rows
    ]
    return run["top_k"], outcomes