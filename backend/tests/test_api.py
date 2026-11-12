import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from app.api.limits import RateLimiter
from app.api.main import create_app
from app.llm_limits import QuotaExhausted


class FakeService:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def ask(self, question, config):
        self.calls.append((question, config))
        if self.error:
            raise self.error
        return {"question": question, "config": config, "answer": "an answer", "sources": [],
                "prompt_tokens": 10, "timings": {"retrieve": 0.1, "generate": 0.2}}


def client_for(tmp_path, service=None, limiter=None):
    app = create_app(service or FakeService(), results_path=str(tmp_path / "results.json"),
                     docs_dir=str(tmp_path), limiter=limiter or RateLimiter(100, 3600, 1000, 86400))
    return TestClient(app)


GOOD = {"question": "How do I make a query parameter required?"}


def test_health(tmp_path):
    r = client_for(tmp_path).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["results_available"] is False


def test_results_missing_then_present(tmp_path):
    c = client_for(tmp_path)
    assert c.get("/api/results").status_code == 404
    (tmp_path / "results.json").write_text('{"runs": []}', encoding="utf-8")
    r = c.get("/api/results")
    assert r.status_code == 200 and r.json() == {"runs": []}


def test_graphs_strip_the_bom(tmp_path):
    (tmp_path / "rag_graph.mmd").write_text("graph TD;\n", encoding="utf-8-sig")
    out = client_for(tmp_path).get("/api/graphs").json()
    assert out == {"rag": "graph TD;\n"}


def test_ask_uses_the_default_config_and_applies_overrides(tmp_path):
    svc = FakeService()
    c = client_for(tmp_path, svc)
    assert c.post("/api/ask", json=GOOD).status_code == 200
    assert svc.calls[0][1]["top_k"] == 6 and svc.calls[0][1]["retriever"] == "dense"
    r = c.post("/api/ask", json={**GOOD, "config": {"top_k": 3, "retriever": "hybrid"}})
    assert r.status_code == 200 and r.json()["answer"] == "an answer"
    assert svc.calls[1][1]["top_k"] == 3 and svc.calls[1][1]["retriever"] == "hybrid"


@pytest.mark.parametrize("body", [
    {"question": "hi"},
    {"question": "   ab   "},
    {"question": "a valid question", "config": {"top_k": 50}},
    {"question": "a valid question", "config": {"bogus": 1}},
    {"question": "a valid question", "config": {"retriever": "nonsense"}},
])
def test_bad_input_is_rejected_before_the_service_is_called(tmp_path, body):
    svc = FakeService()
    assert client_for(tmp_path, svc).post("/api/ask", json=body).status_code == 422
    assert svc.calls == []


@pytest.mark.parametrize("error, status", [
    (QuotaExhausted("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel"), 429),
    (APIConnectionError(request=httpx.Request("POST", "http://example.test")), 503),
    (RuntimeError("secret-key-123 exploded"), 500),
])
def test_errors_become_clean_responses(tmp_path, error, status):
    r = client_for(tmp_path, FakeService(error)).post("/api/ask", json=GOOD)
    assert r.status_code == status
    assert "secret-key-123" not in r.text and "Traceback" not in r.text


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    c = client_for(tmp_path, limiter=RateLimiter(2, 3600, 1000, 86400))
    assert [c.post("/api/ask", json=GOOD).status_code for _ in range(3)] == [200, 200, 429]
    assert int(c.post("/api/ask", json=GOOD).headers["retry-after"]) >= 1


def test_invalid_requests_do_not_use_the_limit(tmp_path):
    c = client_for(tmp_path, limiter=RateLimiter(1, 3600, 1000, 86400))
    assert c.post("/api/ask", json={**GOOD, "config": {"top_k": 50}}).status_code == 422
    assert c.post("/api/ask", json=GOOD).status_code == 200


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_limiter_per_client_window():
    clock = Clock()
    lim = RateLimiter(2, 100, 1000, 1000, clock=clock)
    assert lim.check("a")[0] and lim.check("a")[0]
    ok, _, retry = lim.check("a")
    assert not ok and 1 <= retry <= 100
    assert lim.check("b")[0]
    clock.t = 101
    assert lim.check("a")[0]


def test_limiter_global_cap():
    lim = RateLimiter(10, 100, 2, 1000, clock=Clock())
    assert lim.check("a")[0] and lim.check("b")[0]
    ok, reason, _ = lim.check("c")
    assert not ok and "daily" in reason


def test_denied_requests_are_not_recorded():
    clock = Clock()
    lim = RateLimiter(1, 100, 1000, 1000, clock=clock)
    assert lim.check("a")[0]
    clock.t = 50
    assert not lim.check("a")[0]
    clock.t = 101  # the first request has expired; the denied one must not have been counted
    assert lim.check("a")[0]