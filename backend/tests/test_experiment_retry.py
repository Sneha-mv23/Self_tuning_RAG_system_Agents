from types import SimpleNamespace

import app.optimizer.experiment as ex
from app.rag.pipeline import RAGConfig


def test_retries_once_before_giving_up(monkeypatch):
    calls = {"n": 0}

    def fake_attempt(*args, **kwargs):
        calls["n"] += 1
        detail = {"q1": "boom"} if calls["n"] == 1 else {}
        return [], [], [], detail, 3

    monkeypatch.setattr(ex, "_attempt", fake_attempt)
    monkeypatch.setattr(ex, "log_run", lambda *a, **k: 7)
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    evaluator = SimpleNamespace(evaluate=lambda config, items: None)
    result = ex.run_experiment(None, evaluator, RAGConfig(), [], None, SimpleNamespace(calls_made=0))
    assert result.run_id == 7 and calls["n"] == 2 and result.new_generation_calls == 6


def test_gives_up_after_one_retry(monkeypatch):
    monkeypatch.setattr(ex, "_attempt", lambda *a, **k: ([], [], [], {"q1": "boom"}, 0))
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)
    evaluator = SimpleNamespace(evaluate=lambda config, items: None)
    result = ex.run_experiment(None, evaluator, RAGConfig(), [], None, SimpleNamespace(calls_made=0))
    assert result.run_id is None and result.errors == ["q1"]