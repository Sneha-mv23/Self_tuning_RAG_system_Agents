from app.evals.golden import Evidence, GoldenItem
from app.evals.retrieval_metrics import first_hit_rank, is_hit, rank_needed
from app.rag.chunker import Chunk
from app.rag.index import Retrieved


def chunk(doc, start, end):
    return Chunk(chunk_id=f"{doc}::{start}", doc_id=doc, text="t", start=start, end=end, n_tokens=1)


def ev(doc, start, end):
    return Evidence(doc_id=doc, quote="q", start=start, end=end)


def test_overlap_is_a_hit():
    assert is_hit(chunk("a.md", 100, 200), ev("a.md", 150, 250))


def test_touching_edges_are_not_a_hit():
    assert not is_hit(chunk("a.md", 100, 200), ev("a.md", 200, 300))


def test_other_document_is_not_a_hit():
    assert not is_hit(chunk("b.md", 100, 200), ev("a.md", 100, 200))


def test_rank_needed_uses_worst_evidence():
    retrieved = [
        Retrieved(chunk("a.md", 0, 100), 0.9, 1),
        Retrieved(chunk("z.md", 0, 100), 0.8, 2),
        Retrieved(chunk("b.md", 0, 100), 0.7, 3),
    ]
    item = GoldenItem(id="x", question="q", type="multi_hop", reference_answer="a",
                      evidence=[ev("a.md", 10, 20), ev("b.md", 10, 20)])
    assert first_hit_rank(retrieved, item.evidence[0]) == 1
    assert rank_needed(retrieved, item) == 3


def test_rank_needed_none_when_any_evidence_missing():
    retrieved = [Retrieved(chunk("a.md", 0, 100), 0.9, 1)]
    item = GoldenItem(id="x", question="q", type="multi_hop", reference_answer="a",
                      evidence=[ev("a.md", 10, 20), ev("b.md", 10, 20)])
    assert rank_needed(retrieved, item) is None