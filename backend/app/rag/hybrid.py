import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.embedder import embed_texts
from app.rag.index import Retrieved, VectorIndex

_TOKEN = re.compile(r"[a-z0-9_]+")
RRF_K = 60
DEPTH = 50


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[int]], weights: list[float], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: each ranking adds weight / (k + rank) to every item it contains."""
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for pos, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + pos)
    return fused


class HybridIndex:
    """Dense embeddings plus BM25 keyword search, combined with reciprocal rank fusion.

    Same search(query, k) interface as VectorIndex. Scores are fused rank scores,
    not cosine similarities, so they are not comparable with dense scores.
    """

    def __init__(self, dense: VectorIndex, dense_weight: float = 1.0, bm25_weight: float = 1.0):
        self.dense = dense
        self.chunks = dense.chunks
        self.weights = [dense_weight, bm25_weight]
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in self.chunks])

    def search_vec(self, query: str, query_vec: np.ndarray, k: int) -> list[Retrieved]:
        depth = min(max(DEPTH, k), len(self.chunks))
        dense_scores = self.dense.vectors @ query_vec
        dense_rank = [int(i) for i in np.argsort(-dense_scores)[:depth]]
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = [int(i) for i in np.argsort(-bm25_scores)[:depth] if bm25_scores[i] > 0]
        fused = rrf_fuse([dense_rank, bm25_rank], self.weights)
        top = sorted(fused, key=lambda i: (-fused[i], i))[: min(k, len(self.chunks))]
        return [Retrieved(chunk=self.chunks[i], score=float(fused[i]), rank=r + 1) for r, i in enumerate(top)]

    def search(self, query: str, k: int) -> list[Retrieved]:
        return self.search_vec(query, embed_texts([query])[0], k)