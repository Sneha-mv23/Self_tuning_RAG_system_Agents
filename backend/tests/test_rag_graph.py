from types import SimpleNamespace

import pytest

from app.graphs.rag_graph import build_rag_graph
from app.rag.chunker import Chunk
from app.rag.index import Retrieved


def chunk(i):
    return Chunk(chunk_id=f"d::{i}", doc_id="d.md", text=f"text {i}", start=i * 10, end=i * 10 + 9, n_tokens=2)


class FakeIndex:
    def __init__(self):
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return [Retrieved(chunk=chunk(i), score=1 - i * 0.1, rank=i + 1) for i in range(k)]


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="  an answer [1]  ")


def make_graph():
    index, requested = FakeIndex(), []

    def index_for(size, overlap, retriever):
        requested.append((size, overlap, retriever))
        return index

    return build_rag_graph(index_for, lambda temperature, max_tokens: FakeLLM()), index, requested


def test_graph_runs_end_to_end():
    graph, index, requested = make_graph()
    out = graph.invoke({"question": "What?", "config": {"top_k": 3}})
    assert out["answer"] == "an answer [1]"
    assert index.calls == [("What?", 3)] and requested == [(256, 32, "dense")]
    assert [s["rank"] for s in out["sources"]] == [1, 2, 3]
    assert "What?" in out["prompt"] and "text 0" in out["prompt"] and out["prompt_tokens"] > 0
    assert set(out["timings"]) == {"retrieve", "generate"}


def test_config_is_passed_through_to_the_index():
    graph, _, requested = make_graph()
    graph.invoke({"question": "q", "config": {"chunk_size": 128, "overlap": 16, "retriever": "hybrid"}})
    assert requested == [(128, 16, "hybrid")]


def test_invalid_config_is_rejected():
    graph, _, _ = make_graph()
    with pytest.raises(ValueError):
        graph.invoke({"question": "q", "config": {"retriever": "nonsense"}})