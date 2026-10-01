import pytest

from app.evals.golden import Evidence, GoldenItem, resolve_evidence
from app.rag.loader import Document

DOC = Document(doc_id="a.md", text="Alpha beta gamma.\n\nDelta epsilon. Alpha again.\n")


def test_unanswerable_must_have_no_evidence():
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="q", type="unanswerable",
                   evidence=[Evidence(doc_id="a.md", quote="beta")])


def test_answerable_requires_evidence_and_answer():
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="q", type="factual", reference_answer="a")
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="q", type="factual",
                   evidence=[Evidence(doc_id="a.md", quote="beta")])


def test_multi_hop_requires_two_quotes():
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="q", type="multi_hop", reference_answer="a",
                   evidence=[Evidence(doc_id="a.md", quote="beta")])


def test_resolve_finds_span():
    item = GoldenItem(id="x", question="q", type="factual", reference_answer="a",
                      evidence=[Evidence(doc_id="a.md", quote="epsilon")])
    ev = resolve_evidence([item], [DOC])[0].evidence[0]
    assert DOC.text[ev.start : ev.end] == "epsilon"


def test_resolve_missing_quote_raises():
    item = GoldenItem(id="x", question="q", type="factual", reference_answer="a",
                      evidence=[Evidence(doc_id="a.md", quote="zzz")])
    with pytest.raises(ValueError, match="not found"):
        resolve_evidence([item], [DOC])


def test_resolve_ambiguous_quote_raises():
    item = GoldenItem(id="x", question="q", type="factual", reference_answer="a",
                      evidence=[Evidence(doc_id="a.md", quote="Alpha")])
    with pytest.raises(ValueError, match="appears 2 times"):
        resolve_evidence([item], [DOC])