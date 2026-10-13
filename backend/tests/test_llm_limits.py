import pytest

from app.evals.generation_run import _call_llm
from app.llm_limits import QuotaExhausted, is_daily_quota, is_rate_limit


class RaisingLLM:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise RuntimeError(self.message)


def test_classify_messages():
    assert is_rate_limit("RateLimitError: Error code: 429 - quota")
    assert not is_rate_limit("APIConnectionError: Connection error.")
    assert is_daily_quota("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert not is_daily_quota("429 GenerateRequestsPerMinutePerProjectPerModel")


def test_daily_quota_stops_generation_at_once():
    llm = RaisingLLM("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    with pytest.raises(QuotaExhausted):
        _call_llm(llm, "p")
    assert llm.calls == 1