import re
from dataclasses import dataclass

import tiktoken

from app.rag.loader import Document

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>::<index>"
    doc_id: str
    text: str
    start: int  # char offset in the document text (inclusive)
    end: int  # char offset in the document text (exclusive)
    n_tokens: int


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split on blank lines, returning (start, end) character spans."""
    spans, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, pos = [], start
    for m in re.finditer(r"\n", text[start:end]):
        cut = start + m.start()
        if cut > pos:
            spans.append((pos, cut))
        pos = cut + 1
    if pos < end:
        spans.append((pos, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _units(text: str, chunk_size: int) -> list[tuple[int, int, int]]:
    """Smallest pieces we pack into chunks: paragraphs, or lines if a
    paragraph alone is bigger than chunk_size."""
    units = []
    for s, e in _paragraph_spans(text):
        n = count_tokens(text[s:e])
        if n <= chunk_size:
            units.append((s, e, n))
        else:
            for ls, le in _line_spans(text, s, e):
                units.append((ls, le, count_tokens(text[ls:le])))
    return units


def chunk_document(doc: Document, chunk_size: int = 256, overlap: int = 32) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = doc.text
    units = _units(text, chunk_size)
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        # Pack units greedily until the next one would exceed chunk_size.
        j, total = i, 0
        while j < len(units) and (j == i or total + units[j][2] <= chunk_size):
            total += units[j][2]
            j += 1

        start, end = units[i][0], units[j - 1][1]
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{len(chunks)}",
                doc_id=doc.doc_id,
                text=chunk_text,
                start=start,
                end=end,
                n_tokens=count_tokens(chunk_text),
            )
        )
        if j >= len(units):
            break

        # Step back over trailing units to create the overlap.
        k, ov = j, 0
        while k - 1 > i and ov + units[k - 1][2] <= overlap:
            k -= 1
            ov += units[k][2]
        i = k
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int = 256, overlap: int = 32) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size, overlap))
    return out