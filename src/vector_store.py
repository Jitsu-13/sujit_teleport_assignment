from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    title: str
    text: str
    score: float


class VectorStore:
    """FAISS-backed vector store with document metadata.

    Embeddings must be L2-normalised before insertion so that inner-product
    search (IndexFlatIP) is equivalent to cosine similarity.  LocalEmbeddingModel
    already guarantees this invariant.

    Cosine similarity is preferred over Euclidean distance here because:
      - It is magnitude-invariant, so short and long passages are treated fairly.
      - For unit vectors, cos(a,b) = a·b, letting us use the fast IP kernel.
      - Vertex AI Vector Search (Matching Engine) uses dot-product by default,
        so this choice maps directly to the production metric with no changes.
    """

    def __init__(self, embedding_dim: int) -> None:
        self._dim = embedding_dim
        # IndexFlatIP — exact brute-force inner-product search.
        # Suitable for corpora of this size; swap to IndexIVFFlat or HNSW
        # when scaling to millions of vectors.
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._metadata: List[dict] = []  # parallel list keyed by FAISS row id

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(self, documents: List[dict], embeddings: np.ndarray) -> None:
        """Index a batch of documents.

        Args:
            documents: list of dicts with at minimum keys 'id', 'title', 'text'.
            embeddings: float32 array of shape (N, D), L2-normalised.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"documents length ({len(documents)}) must match "
                f"embeddings rows ({embeddings.shape[0]})"
            )

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._metadata.extend(documents)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """Return the top-K most similar documents.

        Args:
            query_vector: 1-D float32 array, L2-normalised.
            top_k: number of results to return.

        Returns:
            List of SearchResult ordered by descending cosine similarity.
        """
        if self._index.ntotal == 0:
            return []

        query = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            doc = self._metadata[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    doc_id=doc["id"],
                    title=doc["title"],
                    text=doc["text"],
                    score=float(score),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return f"VectorStore(dim={self._dim}, docs={len(self)})"
