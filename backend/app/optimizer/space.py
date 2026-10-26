from itertools import product

from app.optimizer.levers import CHUNK_SIZES, OVERLAPS, TOP_KS, is_valid
from app.rag.pipeline import RETRIEVERS, RAGConfig


def retrieval_space(prompt: str = "strict", temperature: float = 0.0) -> list[RAGConfig]:
    """Every valid combination of the retrieval-side settings."""
    out = []
    for cs, ov, k, r in product(CHUNK_SIZES, OVERLAPS, TOP_KS, RETRIEVERS):
        c = RAGConfig(chunk_size=cs, overlap=ov, top_k=k, prompt=prompt, temperature=temperature, retriever=r)
        if is_valid(c):
            out.append(c)
    return out


def pareto(points: list[tuple[int, int]]) -> list[int]:
    """Indices of (coverage, tokens) points that nothing beats: no other point has coverage >= and
    tokens <= with at least one strictly better."""
    keep = []
    for i, (cov_i, tok_i) in enumerate(points):
        dominated = any(
            cov_j >= cov_i and tok_j <= tok_i and (cov_j > cov_i or tok_j < tok_i)
            for j, (cov_j, tok_j) in enumerate(points) if j != i
        )
        if not dominated:
            keep.append(i)
    return keep