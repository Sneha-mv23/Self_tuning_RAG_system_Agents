from app.optimizer.levers import is_valid
from app.optimizer.space import pareto, retrieval_space


def test_pareto_keeps_only_non_dominated_points():
    points = [(10, 100), (12, 200), (12, 300), (9, 100), (10, 100)]
    assert pareto(points) == [0, 1, 4]


def test_retrieval_space_contains_only_valid_configs():
    space = retrieval_space()
    assert all(is_valid(c) for c in space)
    assert len(space) == 102  # 3 chunk sizes x 3 overlaps x 6 top_k x 2 retrievers, minus 256 with top_k 10
    assert {c.retriever for c in space} == {"dense", "hybrid"}