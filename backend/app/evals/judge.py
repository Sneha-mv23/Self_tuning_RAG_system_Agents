import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace  # noqa: F401  (handy for tests)

from app.config import settings
from app.evals.generation_run import is_refusal
from app.evals.golden import GoldenItem
from app.llm import get_judge
from app.rag.loader import Document

JUDGE_CACHE_PATH = "data/cache/judgments.sqlite"
PURE_REFUSAL_MAX_CHARS = 120

_JSON_FORMAT = _JSON_FORMAT = 'Reply with JSON only, with the verdict FIRST: {{"verdict": "yes" or "no", "reason": "<one short sentence>"}}'

CORRECT_PROMPT = (
    "You are grading an answer to a question about software documentation.\n\n"
    "Question: {question}\n\n"
    "Reference answer: {reference}\n\n"
    "Candidate answer: {answer}\n\n"
    "Does the candidate answer give the key information of the reference answer? "
    "Different wording and extra detail are fine. Answer \"no\" if it contradicts the reference, "
    "misses the key point, or only says it does not know. Ignore a trailing sentence such as "
    "\"I don't know based on the provided documents\" if the rest of the answer gives the information.\n\n"
    + _JSON_FORMAT
)

FAITHFUL_PROMPT = (
    "You are checking whether an answer is supported by the context it was given.\n\n"
    "Context:\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Is every factual claim in the answer supported by the context? Ignore citation markers like [1] "
    "and any sentence saying \"I don't know\". Answer \"no\" if the answer states something the context "
    "does not say or clearly imply.\n\n"
    + _JSON_FORMAT
)

DECLINE_PROMPT = (
    "A user asked a question that the documentation does NOT answer. The correct behavior is to "
    "decline instead of inventing an answer.\n\n"
    "Question: {question}\n\n"
    "Answer: {answer}\n\n"
    "Does the answer decline to answer (for example by saying the documents do not contain the "
    "information) without presenting an invented answer? Answer \"no\" if it gives a substantive "
    "answer, even if it also says it does not know.\n\n"
    + _JSON_FORMAT
)


def is_pure_refusal(answer: str) -> bool:
    return is_refusal(answer) and len(answer.strip()) < PURE_REFUSAL_MAX_CHARS


def is_mixed_refusal(answer: str) -> bool:
    return is_refusal(answer) and not is_pure_refusal(answer)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_VERDICT_RE = re.compile(r"verdict\W{0,4}(yes|no)\b", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """Extract (verdict, reason) from judge output. Returns (None, text) if unparseable."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    for m in _JSON_RE.finditer(cleaned):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        v = str(obj.get("verdict", "")).strip().lower()
        if v in ("yes", "no"):
            return v == "yes", str(obj.get("reason", "")).strip()
    m = _VERDICT_RE.search(cleaned)
    if m:
        return m.group(1).lower() == "yes", cleaned[:200]
    return None, cleaned[:200]


class JudgeCache:
    def __init__(self, path: str = JUDGE_CACHE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS judgments ("
            "key TEXT PRIMARY KEY, verdict INTEGER NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> tuple[bool, str] | None:
        row = self.conn.execute("SELECT verdict, reason FROM judgments WHERE key = ?", (key,)).fetchone()
        return (bool(row[0]), row[1]) if row else None

    def put(self, key: str, verdict: bool, reason: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?)",
            (key, int(verdict), reason, time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@dataclass
class Verdict:
    value: bool | None
    reason: str
    cached: bool
    error: str | None


class Judge:
    def __init__(self, cache: JudgeCache | None = None, llm=None, min_interval_s: float | None = None):
        self.cache = cache or JudgeCache()
        self.llm = llm or get_judge(max_tokens=1024)
        self.min_interval_s = settings.judge_min_interval_s if min_interval_s is None else min_interval_s
        self._last_call = 0.0

    def _key(self, kind: str, prompt: str) -> str:
        payload = json.dumps(
            {"model": settings.judge_model, "base_url": settings.judge_base_url, "kind": kind, "prompt": prompt},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_cached(self, kind: str, prompt: str) -> bool:
        return self.cache.get(self._key(kind, prompt)) is not None

    def ask(self, kind: str, prompt: str) -> Verdict:
            key = self._key(kind, prompt)
            hit = self.cache.get(key)
            if hit is not None:
                return Verdict(hit[0], hit[1], True, None)

            last_error = None
            for attempt in range(4):
                wait = self.min_interval_s - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                try:
                    self._last_call = time.monotonic()
                    reply = self.llm.invoke(prompt)
                except Exception as e:  # noqa: BLE001 - rate limits and network errors both land here
                    last_error = f"{type(e).__name__}: {e}"
                    if _is_rate_limit(last_error):
                        # A daily quota will not recover by waiting. A per-minute one usually does within ~60s.
                        if _is_daily_quota(last_error) or attempt >= 2:
                            raise QuotaExhausted(last_error) from e
                        time.sleep(20 * (attempt + 1))
                    else:
                        time.sleep(5 * 3**attempt)  # connection errors: 5s, 15s, 45s, 135s
                    continue
                value, reason = parse_verdict(str(reply.content))
                if value is None:
                    # Same prompt at temperature 0 would give the same output, so don't retry or cache.
                    return Verdict(None, reason, False, f"unparseable judge output: {reason[:80]!r}")
                self.cache.put(key, value, reason)
                return Verdict(value, reason, False, None)
            return Verdict(None, "", False, last_error)


def context_from_spans(used_chunks: list[dict], docs_by_id: dict[str, Document]) -> str:
    parts = []
    for c in used_chunks:
        text = docs_by_id[c["doc_id"]].text[c["start"] : c["end"]].strip()
        parts.append(f"[{c['rank']}] ({c['doc_id']})\n{text}")
    return "\n\n".join(parts)


def build_jobs(record: dict, item: GoldenItem, docs_by_id: dict[str, Document]) -> list[tuple[str, str]]:
    """The (kind, prompt) judge calls this answer needs. Empty for refusals handled by rule."""
    answer = record["answer"]
    if record.get("error") or is_pure_refusal(answer):
        return []
    if item.type == "unanswerable":
        return [("declined", DECLINE_PROMPT.format(question=item.question, answer=answer))]
    return [
        ("correct", CORRECT_PROMPT.format(
            question=item.question, reference=item.reference_answer, answer=answer)),
        ("faithful", FAITHFUL_PROMPT.format(
            context=context_from_spans(record["used_chunks"], docs_by_id), answer=answer)),
    ]


@dataclass
class Judgment:
    id: str
    type: str
    correct: bool | None  # answerable: matches the reference. unanswerable: properly declined.
    faithful: bool | None  # None when not applicable (refusals, unanswerable questions)
    pure_refusal: bool
    mixed_refusal: bool
    correct_reason: str
    faithful_reason: str
    error: str | None


def judge_record(judge: Judge, record: dict, item: GoldenItem, docs_by_id: dict[str, Document]) -> Judgment:
    answer = record["answer"]
    base = {"id": item.id, "type": item.type}
    if record.get("error"):
        return Judgment(**base, correct=None, faithful=None, pure_refusal=False, mixed_refusal=False,
                        correct_reason="", faithful_reason="", error=f"generation failed: {record['error']}")
    if is_pure_refusal(answer):
        return Judgment(**base, correct=(item.type == "unanswerable"), faithful=None, pure_refusal=True,
                        mixed_refusal=False, correct_reason="pure refusal (rule-based, no LLM call)",
                        faithful_reason="", error=None)

    verdicts = {kind: judge.ask(kind, prompt) for kind, prompt in build_jobs(record, item, docs_by_id)}
    primary = verdicts.get("declined") or verdicts["correct"]
    faith = verdicts.get("faithful")
    errors = [v.error for v in verdicts.values() if v.error]
    return Judgment(
        **base,
        correct=primary.value,
        faithful=faith.value if faith else None,
        pure_refusal=False,
        mixed_refusal=is_mixed_refusal(answer),
        correct_reason=primary.reason,
        faithful_reason=faith.reason if faith else "",
        error="; ".join(errors) or None,
    )


def _rate(xs: list[bool]) -> float:
    return sum(1 for x in xs if x) / len(xs) if xs else 0.0


def summarize_judgments(js: list[Judgment]) -> dict:
    ok = [j for j in js if j.error is None]
    ans = [j for j in ok if j.type != "unanswerable"]
    un = [j for j in ok if j.type == "unanswerable"]
    faith = [j for j in ans if j.faithful is not None]
    return {
        "n": len(js),
        "errors": len(js) - len(ok),
        "answer_correctness": _rate([j.correct for j in ans]),
        "faithfulness": _rate([j.faithful for j in faith]),
        "n_faithfulness_scored": len(faith),
        "proper_refusal_rate_unanswerable": _rate([j.correct for j in un]),
        "pure_refusal_rate_answerable": _rate([j.pure_refusal for j in ans]),
        "mixed_refusal_rate": _rate([j.mixed_refusal for j in ok]),
    }

class QuotaExhausted(RuntimeError):
    """Raised when the judge API refuses further calls (rate limit or daily quota)."""


def _is_rate_limit(err: str) -> bool:
    e = err.lower()
    return any(s in e for s in ("429", "ratelimit", "resource_exhausted", "quota"))


def _is_daily_quota(err: str) -> bool:
    return "perday" in err.lower().replace("_", "").replace(" ", "")