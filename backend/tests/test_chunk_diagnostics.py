from app.evals.chunk_diagnostics import span_status, visible_end_char
from app.rag.chunker import Chunk


def chunk(i, start, end):
    return Chunk(chunk_id=f"d::{i}", doc_id="d.md", text="t", start=start, end=end, n_tokens=1)


def test_no_cut_when_short():
    offsets = [(i * 5, i * 5 + 4) for i in range(100)]
    assert visible_end_char(offsets) is None


def test_cut_position_after_limit():
    offsets = [(i * 5, i * 5 + 4) for i in range(300)]
    assert visible_end_char(offsets) == 253 * 5 + 4


def test_span_status_cases():
    chunks = [chunk(0, 0, 1000)]
    cuts = {"d::0": 600}
    assert span_status(chunks, cuts, "d.md", 100, 200) == "full"
    assert span_status(chunks, cuts, "d.md", 550, 650) == "partial"
    assert span_status(chunks, cuts, "d.md", 700, 800) == "none"
    assert span_status(chunks, cuts, "d.md", 900, 1100) == "split"
    assert span_status(chunks, {"d::0": None}, "d.md", 700, 800) == "full"


def test_best_chunk_wins_when_chunks_overlap():
    chunks = [chunk(0, 0, 1000), chunk(1, 500, 1500)]
    cuts = {"d::0": 600, "d::1": None}
    assert span_status(chunks, cuts, "d.md", 700, 800) == "full"