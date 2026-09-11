"""Sentence embeddings with a disk cache keyed by text hash."""
from __future__ import annotations

import hashlib

import numpy as np

from . import config as C

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(C.EMBED_MODEL)
    return _model


def embed(texts: list[str], cache_name: str | None = None) -> np.ndarray:
    """Return L2-normalised embeddings. Caches full arrays under data/processed when named."""
    if cache_name:
        digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
        path = C.PROCESSED / f"emb_{cache_name}_{digest}.npy"
        if path.exists():
            return np.load(path)
    vecs = _get_model().encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    if cache_name:
        np.save(path, vecs)
    return vecs
