from app.evals.golden import GoldenItem
from app.optimizer.levers import is_valid
from app.optimizer.test_plan import EVAL_CONFIGS, fingerprint, pending, run_order


def item(id, q="q"):
    return GoldenItem(id=id, question=q, type="unanswerable")


def test_configs_are_valid_and_distinct():
    assert all(is_valid(c) for c in EVAL_CONFIGS.values())
    assert len(set(EVAL_CONFIGS.values())) == len(EVAL_CONFIGS)


def test_run_order_is_interleaved():
    assert run_order(["a", "b"], 2) == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]


def test_pending_skips_completed():
    order = run_order(["a", "b"], 2)
    assert pending(order, [{"config": "a", "replicate": 0}]) == [("b", 0), ("a", 1), ("b", 1)]


def test_fingerprint_detects_changes_and_ignores_order():
    a, b = item("1"), item("2")
    assert fingerprint([a, b]) == fingerprint([b, a])
    assert fingerprint([a, b]) != fingerprint([a, item("2", q="changed")])