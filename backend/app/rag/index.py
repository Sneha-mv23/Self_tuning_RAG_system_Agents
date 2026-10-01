from dataclasses import dataclass

import numpy as np

from app.rag.chunker import Chunk
from app.rag.embedder import embed_texts, embed_texts_cached


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float
    rank: int  # 1 = best


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        assert len(chunks) == len(vectors)
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorIndex":
        vectors = embed_texts_cached([c.text for c in chunks])
        return cls(chunks, vectors)

    def search_vec(self, query_vec: np.ndarray, k: int) -> list[Retrieved]:
        scores = self.vectors @ query_vec
        k = min(k, len(self.chunks))
        top = np.argsort(-scores)[:k]
        return [
            Retrieved(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(top)
        ]

    def search(self, query: str, k: int) -> list[Retrieved]:
        return self.search_vec(embed_texts([query])[0], k)