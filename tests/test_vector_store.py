"""Tests for VectorStore."""
from __future__ import annotations

import numpy as np
import pytest

from src.vector_store import SearchResult, VectorStore

_DIM = 8  # small dimension so tests stay fast without a real embedding model


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture()
def sample_docs() -> list:
    return [
        {"id": f"doc_{i:02d}", "title": f"Title {i}", "text": f"Body text {i}"}
        for i in range(5)
    ]


@pytest.fixture()
def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((5, _DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture()
def populated_store(sample_docs, sample_embeddings) -> VectorStore:
    store = VectorStore(embedding_dim=_DIM)
    store.add(sample_docs, sample_embeddings)
    return store


class TestVectorStoreAdd:
    def test_len_increases_after_add(self, sample_docs, sample_embeddings) -> None:
        store = VectorStore(embedding_dim=_DIM)
        assert len(store) == 0
        store.add(sample_docs, sample_embeddings)
        assert len(store) == 5

    def test_add_raises_on_shape_mismatch(self, sample_docs) -> None:
        store = VectorStore(embedding_dim=_DIM)
        bad_embeddings = np.ones((3, _DIM), dtype=np.float32)  # 3 != 5 docs
        with pytest.raises(ValueError, match="must match"):
            store.add(sample_docs, bad_embeddings)

    def test_add_is_cumulative(self, sample_docs, sample_embeddings) -> None:
        store = VectorStore(embedding_dim=_DIM)
        store.add(sample_docs[:2], sample_embeddings[:2])
        store.add(sample_docs[2:], sample_embeddings[2:])
        assert len(store) == 5


class TestVectorStoreSearch:
    def test_search_returns_search_results(self, populated_store, sample_embeddings) -> None:
        results = populated_store.search(sample_embeddings[0], top_k=3)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_respects_top_k(self, populated_store, sample_embeddings) -> None:
        for k in (1, 3, 5):
            results = populated_store.search(sample_embeddings[0], top_k=k)
            assert len(results) == k

    def test_scores_are_descending(self, populated_store, sample_embeddings) -> None:
        results = populated_store.search(sample_embeddings[0], top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_is_top_result(self, populated_store, sample_embeddings) -> None:
        # Querying with doc_02's own vector should return doc_02 first.
        query = sample_embeddings[2]
        results = populated_store.search(query, top_k=3)
        assert results[0].doc_id == "doc_02"

    def test_top_result_score_near_one_for_self_query(
        self, populated_store, sample_embeddings
    ) -> None:
        results = populated_store.search(sample_embeddings[0], top_k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_ranks_are_sequential(self, populated_store, sample_embeddings) -> None:
        results = populated_store.search(sample_embeddings[0], top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_search_on_empty_store_returns_empty(self) -> None:
        store = VectorStore(embedding_dim=_DIM)
        query = _unit(np.ones(_DIM, dtype=np.float32))
        assert store.search(query, top_k=3) == []

    def test_top_k_clamped_to_store_size(self, sample_docs, sample_embeddings) -> None:
        store = VectorStore(embedding_dim=_DIM)
        store.add(sample_docs[:2], sample_embeddings[:2])
        results = store.search(sample_embeddings[0], top_k=10)
        assert len(results) == 2

    def test_result_metadata_matches_document(
        self, populated_store, sample_docs, sample_embeddings
    ) -> None:
        results = populated_store.search(sample_embeddings[0], top_k=1)
        top = results[0]
        matching_doc = next(d for d in sample_docs if d["id"] == top.doc_id)
        assert top.title == matching_doc["title"]
        assert top.text == matching_doc["text"]
