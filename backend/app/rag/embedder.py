import hashlib
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_CACHE_PATH = Path("data/vectorstore/embeddings.pkl")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts into unit-length vectors (so dot product = cosine similarity)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
    )
    return np.asarray(vecs, dtype=np.float32)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_texts_cached(texts: list[str]) -> np.ndarray:
    """Like embed_texts, but never re-embeds text it has seen before."""
    if not texts:
        return embed_texts(texts)
    prefix = settings.embedding_model + "|"
    keys = [hashlib.sha1((prefix + t).encode("utf-8")).hexdigest() for t in texts]
    cache = _load_cache()
    missing = sorted({i for i, k in enumerate(keys) if k not in cache})
    if missing:
        new_vecs = embed_texts([texts[i] for i in missing])
        for i, v in zip(missing, new_vecs):
            cache[keys[i]] = v
        _save_cache(cache)
    return np.stack([cache[k] for k in keys])