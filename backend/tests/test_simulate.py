import random

from app.optimizer.simulate import Landscape, climb, random_search
from app.optimizer.space import retrieval_space
from app.rag.pipeline import RAGConfig

KEYS = {(c.chunk_size, c.overlap, c.retriever) for c in retrieval_space()}
# three answerable questions found at ranks 2, 5 and 9, whatever the chunking
ROWS = [("a", 2, [2]), ("b", 5, [5]), ("c", 9, [9])]
LAND = Landscape({k: ROWS for k in KEYS})


def test_coverage_and_outcomes():
    assert LAND.coverage(RAGConfig(top_k=4)) == 1
    assert LAND.coverage(RAGConfig(top_k=5)) == 2
    assert [o.answer_correct for o in LAND.outcomes(RAGConfig(top_k=4))] == [True, False, False]


def test_curves_never_decrease_and_have_fixed_length():
    start = RAGConfig()
    curves = [
        random_search(LAND, start, retrieval_space(), random.Random(1), 12),
        climb(LAND, start, random.Random(1), 12, guided=False)["curve"],
        climb(LAND, start, random.Random(1), 12, guided=True)["curve"],
    ]
    for curve in curves:
        assert len(curve) == 12
        assert all(a <= b for a, b in zip(curve, curve[1:]))
        assert curve[0] >= LAND.coverage(start)


def test_guided_is_deterministic_and_improves():
    a = climb(LAND, RAGConfig(), random.Random(1), 12, guided=True)
    b = climb(LAND, RAGConfig(), random.Random(99), 12, guided=True)
    assert a["curve"] == b["curve"] and a["path"] == b["path"]
    assert a["curve"][-1] > LAND.coverage(RAGConfig())
    assert a["wasted"] == 0


def test_blind_search_wastes_experiments_on_generation_only_levers():
    assert climb(LAND, RAGConfig(), random.Random(3), 25, guided=False)["wasted"] >= 1


def test_random_search_is_reproducible():
    a = random_search(LAND, RAGConfig(), retrieval_space(), random.Random(5), 10)
    b = random_search(LAND, RAGConfig(), retrieval_space(), random.Random(5), 10)
    assert a == b