import numpy as np
import pytest

from app.rag.chunker import Chunk
from app.rag.generator import build_prompt
from app.rag.index import VectorIndex


def make_chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"d::{i}", doc_id="d.md", text=f"text {i}",
                 start=i * 10, end=i * 10 + 9, n_tokens=2)


def test_search_returns_best_first():
    chunks = [make_chunk(i) for i in range(3)]
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    idx = VectorIndex(chunks, vectors)
    results = idx.search_vec(np.array([0, 1], dtype=np.float32), k=2)
    assert [r.chunk.chunk_id for r in results] == ["d::1", "d::2"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score >= results[1].score


def test_k_larger_than_index():
    chunks = [make_chunk(0)]
    idx = VectorIndex(chunks, np.array([[1, 0]], dtype=np.float32))
    assert len(idx.search_vec(np.array([1, 0], dtype=np.float32), k=10)) == 1


def test_prompt_contains_question_and_context():
    chunks = [make_chunk(0)]
    idx = VectorIndex(chunks, np.array([[1, 0]], dtype=np.float32))
    retrieved = idx.search_vec(np.array([1, 0], dtype=np.float32), k=1)
    prompt = build_prompt("strict", "What is X?", retrieved)
    assert "What is X?" in prompt
    assert "text 0" in prompt
    assert "[1]" in prompt


def test_unknown_prompt_raises():
    with pytest.raises(ValueError):
        build_prompt("nope", "q", [])