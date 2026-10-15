import pytest

from app.evals.naming import run_name
from app.rag.pipeline import RAGConfig


def test_dense_keeps_the_original_file_names():
    assert run_name("dev", RAGConfig()) == "dev_cs256_ov32_k4_strict_t0.0"


def test_hybrid_gets_a_suffix():
    assert run_name("dev", RAGConfig(retriever="hybrid")) == "dev_cs256_ov32_k4_strict_t0.0_hybrid"


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError):
        RAGConfig(retriever="bm25")