from app.evals.golden import Evidence, GoldenItem
from app.rag.chunker import Chunk
from app.rag.index import Retrieved


def is_hit(chunk: Chunk, ev: Evidence) -> bool:
    """A chunk is a hit if it overlaps the gold span in the same document."""
    return chunk.doc_id == ev.doc_id and chunk.start < ev.end and chunk.end > ev.start


def first_hit_rank(retrieved: list[Retrieved], ev: Evidence) -> int | None:
    """Rank (1 = best) of the first chunk that covers this evidence, or None."""
    for r in retrieved:
        if is_hit(r.chunk, ev):
            return r.rank
    return None


def rank_needed(retrieved: list[Retrieved], item: GoldenItem) -> int | None:
    """Smallest top_k that covers ALL of the item's evidence, or None if some is missing.

    For single-evidence questions this is just the rank of the gold chunk.
    For multi-hop questions it is the rank of the worst-ranked evidence.
    """
    ranks = [first_hit_rank(retrieved, ev) for ev in item.evidence]
    if not ranks or any(r is None for r in ranks):
        return None
    return max(ranks)