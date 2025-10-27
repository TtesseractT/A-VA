# core/embeddings.py
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import config

_model_cache = {}

def _get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = _get_model(model_name)
    vecs = model.encode(texts, normalize_embeddings=config.EMBEDDING_NORMALIZE, convert_to_numpy=True)
    return vecs.astype(np.float32)
