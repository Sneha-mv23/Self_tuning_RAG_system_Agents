MAX_SEQ = 256  # the embedding model's input limit, including 2 special tokens
SPECIAL_TOKENS = 2


def visible_end_char(offsets: list[tuple[int, int]], max_seq: int = MAX_SEQ) -> int | None:
    """Character index (relative to the chunk text) where the embedder stops reading.
    None means the whole chunk fits."""
    limit = max_seq - SPECIAL_TOKENS
    if len(offsets) <= limit:
        return None
    return offsets[limit - 1][1]


def chunk_cut_positions(chunks, offsets_fn, max_seq: int = MAX_SEQ) -> dict[str, int | None]:
    """Absolute document position after which each chunk's text is invisible to the embedder."""
    cuts: dict[str, int | None] = {}
    for c in chunks:
        v = visible_end_char(offsets_fn(c.text), max_seq)
        cuts[c.chunk_id] = None if v is None else c.start + v
    return cuts


def span_status(chunks, cuts: dict[str, int | None], doc_id: str, start: int, end: int) -> str:
    """'split'   no single chunk contains the whole span
       'full'    some containing chunk shows the whole span to the embedder
       'partial' the best containing chunk shows only part of it
       'none'    every containing chunk hides the span from the embedder"""
    containing = [c for c in chunks if c.doc_id == doc_id and c.start <= start and c.end >= end]
    if not containing:
        return "split"
    best = "none"
    for c in containing:
        cut = cuts[c.chunk_id]
        if cut is None or end <= cut:
            return "full"
        if start < cut:
            best = "partial"
    return best