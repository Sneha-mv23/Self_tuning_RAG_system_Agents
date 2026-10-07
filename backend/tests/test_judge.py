from types import SimpleNamespace
import pytest

from app.evals.golden import Evidence, GoldenItem
from app.evals.judge import (
    Judge, JudgeCache, build_jobs, context_from_spans, is_mixed_refusal,
    is_pure_refusal, judge_record, parse_verdict,QuotaExhausted,strip_citations,
)
from app.rag.loader import Document

DOCS = {"a.md": Document(doc_id="a.md", text="Hello world, this is the context text.")}
CHUNKS = [{"doc_id": "a.md", "start": 0, "end": 11, "rank": 1}]


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.content)


def make_judge(tmp_path, content='{"reason": "ok", "verdict": "yes"}'):
    llm = FakeLLM(content)
    return Judge(cache=JudgeCache(str(tmp_path / "j.sqlite")), llm=llm, min_interval_s=0), llm


def item(type_="factual"):
    ev = [Evidence(doc_id="a.md", quote="Hello", start=0, end=5)] if type_ != "unanswerable" else []
    return GoldenItem(id="x", question="Q?", type=type_,
                      reference_answer=None if type_ == "unanswerable" else "ref", evidence=ev)


def test_parse_clean_json():
    assert parse_verdict('{"reason": "matches", "verdict": "yes"}') == (True, "matches")


def test_parse_fenced_json():
    assert parse_verdict('```json\n{"reason": "no", "verdict": "no"}\n```') == (False, "no")


def test_parse_fallback_on_broken_json():
    value, _ = parse_verdict('verdict: no, because {oops')
    assert value is False


def test_parse_garbage_is_none():
    assert parse_verdict("I think maybe?")[0] is None


def test_refusal_kinds():
    assert is_pure_refusal("I don't know based on the provided documents.")
    long_mixed = "You can send the email in the background. " * 4 + "I don't know based on the provided documents."
    assert is_mixed_refusal(long_mixed) and not is_pure_refusal(long_mixed)
    assert not is_pure_refusal("Use @app.middleware.")


def test_context_from_spans():
    assert context_from_spans(CHUNKS, DOCS) == "[1] (a.md)\nHello world"


def test_jobs_for_each_case():
    rec = {"answer": "Use the decorator.", "error": None, "used_chunks": CHUNKS}
    assert [k for k, _ in build_jobs(rec, item(), DOCS)] == ["correct", "faithful"]
    assert [k for k, _ in build_jobs(rec, item("unanswerable"), DOCS)] == ["declined"]
    refusal = {"answer": "I don't know based on the provided documents.", "error": None, "used_chunks": CHUNKS}
    assert build_jobs(refusal, item(), DOCS) == []


def test_cache_prevents_second_call(tmp_path):
    judge, llm = make_judge(tmp_path)
    assert judge.ask("correct", "p").value is True
    again = judge.ask("correct", "p")
    assert again.value is True and again.cached and llm.calls == 1


def test_pure_refusal_makes_no_llm_call(tmp_path):
    judge, llm = make_judge(tmp_path)
    rec = {"answer": "I don't know based on the provided documents.", "error": None, "used_chunks": CHUNKS}
    assert judge_record(judge, rec, item(), DOCS).correct is False
    assert judge_record(judge, rec, item("unanswerable"), DOCS).correct is True
    assert llm.calls == 0


def test_answerable_gets_two_verdicts(tmp_path):
    judge, llm = make_judge(tmp_path)
    rec = {"answer": "Use the decorator.", "error": None, "used_chunks": CHUNKS}
    j = judge_record(judge, rec, item(), DOCS)
    assert j.correct is True and j.faithful is True and llm.calls == 2


def test_unparseable_output_is_an_error_and_not_cached(tmp_path):
    judge, llm = make_judge(tmp_path, content="no idea")
    v = judge.ask("correct", "p")
    assert v.value is None and v.error and llm.calls == 1
    judge.ask("correct", "p")
    assert llm.calls == 2


def test_parse_truncated_reply_still_gets_verdict():
    value, _ = parse_verdict('{"verdict": "yes", "reason": "The cand')
    assert value is True
    value, _ = parse_verdict('{"verdict": "no", "reas')
    assert value is False

class RaisingLLM:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise RuntimeError(self.message)


def test_daily_quota_stops_immediately(tmp_path):
    llm = RaisingLLM("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    judge = Judge(cache=JudgeCache(str(tmp_path / "j.sqlite")), llm=llm, min_interval_s=0)
    with pytest.raises(QuotaExhausted):
        judge.ask("correct", "p")
    assert llm.calls == 1


def test_strip_citations():
    assert strip_citations("Because it is evaluated in order [3].") == "Because it is evaluated in order."
    assert strip_citations("See [1] and [2].") == "See and."


def test_faithful_prompt_has_no_citation_markers():
    rec = {"answer": "Use the decorator [3].", "error": None, "used_chunks": CHUNKS}
    jobs = dict(build_jobs(rec, item(), DOCS))
    assert "[3]" not in jobs["faithful"]
    assert "[3]" in jobs["correct"]