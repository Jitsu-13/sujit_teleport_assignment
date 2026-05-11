from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Mirrors the dimensionality behaviour of textembedding-gecko (768-d).
# all-MiniLM-L6-v2 produces 384-d L2-normalised vectors and is fast enough
# to run locally without a GPU.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbeddingModel:
    """sentence-transformers wrapper that simulates Vertex AI textembedding-gecko.

    Embeddings are L2-normalised so that inner-product and cosine similarity
    are equivalent, matching the behaviour of the managed gecko model.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return normalised float32 embeddings of shape (N, D)."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.array(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single text string."""
        return self.embed([text])[0]
