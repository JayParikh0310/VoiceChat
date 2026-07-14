# Embedding wrapper — loads all-MiniLM-L6-v2 once and exposes embed(text) -> vector.
# Singleton pattern: model loaded once at startup, reused across all memory ops.
# Implementation: Part 3
from __future__ import annotations

import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around SentenceTransformer. Model loaded once at init and reused."""

    def __init__(self, config: dict) -> None:
        model_name: str = config["memory"]["long_term"]["embedding_model"]

        logger.info("Loading embedding model '%s'...", model_name)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - t0)

        self.dimension: int = self._model.get_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed one string. Returns a normalized float32 vector (shape: (dimension,))."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32)
