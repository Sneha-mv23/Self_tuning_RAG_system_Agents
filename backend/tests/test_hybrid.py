import numpy as np

from app.rag.chunker import Chunk
from app.rag.hybrid import HybridIndex, rrf_fuse, tokenize
from app.rag.index import VectorIndex


def chunk(i, text):
    return Chunk(chunk_id=f"d::{i}", doc_id="d.md", text=text, start=i * 100, end=i * 100 + 50, n_tokens=3)


CHUNKS = [chunk(0, "alpha beta"), chunk(1, "gamma delta unique"), chunk(2, "epsilon zeta")]
VECTORS = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
QUERY_VEC = np.array([1, 0], dtype=np.float32)


def test_tokenize():
    assert tokenize("Use `Depends()` for path_params!") == ["use", "depends", "for", "path_params"]


def test_rrf_fuse_math():
    fused = rrf_fuse([[0, 1], [0]], [1.0, 1.0], k=60)
    assert abs(fused[0] - 2 / 61) < 1e-9
    assert abs(fused[1] - 1 / 62) < 1e-9


def test_keyword_match_can_outrank_dense_order():
    dense = VectorIndex(CHUNKS, VECTORS)
    assert dense.search_vec(QUERY_VEC, 1)[0].chunk.chunk_id == "d::0"
    hybrid = HybridIndex(dense)
    top = hybrid.search_vec("gamma delta", QUERY_VEC, 3)
    assert top[0].chunk.chunk_id == "d::1"
    assert [r.rank for r in top] == [1, 2, 3]


def test_k_larger_than_index_and_no_keyword_match():
    hybrid = HybridIndex(VectorIndex(CHUNKS, VECTORS))
    top = hybrid.search_vec("zzzz qqqq", QUERY_VEC, 10)
    assert len(top) == 3
    assert top[0].chunk.chunk_id == "d::0"  # falls back to dense order