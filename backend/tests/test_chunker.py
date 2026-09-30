from app.rag.chunker import chunk_corpus, chunk_document
from app.rag.loader import Document

TEXT = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(30)) + "\n"
DOC = Document(doc_id="demo.md", text=TEXT)


def test_offsets_match_text():
    for c in chunk_document(DOC, chunk_size=128, overlap=16):
        assert DOC.text[c.start : c.end] == c.text


def test_chunks_respect_size():
    for c in chunk_document(DOC, chunk_size=128, overlap=16):
        assert c.n_tokens <= 128


def test_overlap_shares_text():
    chunks = chunk_document(DOC, chunk_size=128, overlap=64)
    assert len(chunks) > 1
    assert any(a.end > b.start for a, b in zip(chunks, chunks[1:]))


def test_no_overlap_when_zero():
    chunks = chunk_document(DOC, chunk_size=128, overlap=0)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))


def test_deterministic():
    a = chunk_corpus([DOC], 128, 16)
    b = chunk_corpus([DOC], 128, 16)
    assert a == b


def test_smaller_chunks_give_more_chunks():
    small = chunk_document(DOC, chunk_size=64, overlap=8)
    large = chunk_document(DOC, chunk_size=256, overlap=8)
    assert len(small) > len(large)