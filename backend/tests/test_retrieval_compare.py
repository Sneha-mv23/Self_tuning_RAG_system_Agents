from app.evals.retrieval_compare import compare_ranks, covered


def test_covered():
    assert covered(3, 4) and covered(4, 4)
    assert not covered(5, 4) and not covered(None, 20)


def test_compare_ranks_groups():
    dense = {"a": 1, "b": 6, "c": None, "d": 2}
    hybrid = {"a": 2, "b": 3, "c": None, "d": 9}
    assert compare_ranks(dense, hybrid, 4) == {
        "fixed": ["b"], "broken": ["d"], "both_hit": ["a"], "both_miss": ["c"],
    }