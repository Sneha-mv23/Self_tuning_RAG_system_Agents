import pytest

from app.optimizer.acceptance import decide, sign_test_p


def test_sign_test():
    assert sign_test_p(0, 0) == 1.0
    assert abs(sign_test_p(4, 0) - 0.125) < 1e-9
    assert sign_test_p(3, 3) == 1.0


def test_accepts_only_with_enough_net_gain():
    parent = {"a": False, "b": False, "c": True, "d": True}
    better = {"a": True, "b": True, "c": True, "d": True}
    d = decide(parent, better)
    assert d.accepted and d.net == 2 and d.fixed == ["a", "b"] and d.broken == []
    one = {"a": True, "b": False, "c": True, "d": True}
    assert not decide(parent, one).accepted


def test_breakage_counts_against_the_candidate():
    parent = {"a": False, "b": False, "c": True, "d": True}
    churn = {"a": True, "b": True, "c": False, "d": True}
    d = decide(parent, churn)
    assert d.net == 1 and d.broken == ["c"] and not d.accepted


def test_requires_the_same_questions():
    with pytest.raises(ValueError):
        decide({"a": True}, {"b": True})