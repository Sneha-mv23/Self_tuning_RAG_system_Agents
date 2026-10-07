from app.evals.agreement import agreement_stats, wilson


def test_perfect_agreement():
    pairs = [(True, True)] * 5 + [(False, False)] * 5
    s = agreement_stats(pairs, {True: 50, False: 5})
    assert s["weighted_agreement"] == 1.0
    assert abs(s["kappa"] - 1.0) < 1e-9


def test_weights_follow_pool():
    pairs = [(True, True), (True, False), (False, False), (False, False)]
    s = agreement_stats(pairs, {True: 20, False: 2})
    assert abs(s["weighted_agreement"] - 12 / 22) < 1e-9


def test_stratum_counts():
    pairs = [(True, True), (True, False), (False, False)]
    s = agreement_stats(pairs, {True: 10, False: 10})
    assert (s["judge_yes_n"], s["judge_yes_agree"]) == (2, 1)
    assert (s["judge_no_n"], s["judge_no_agree"]) == (1, 1)


def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert 0 < lo < 0.8 < hi < 1
    assert wilson(0, 0) == (0.0, 1.0)