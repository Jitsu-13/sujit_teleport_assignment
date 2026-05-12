"""Tests for LocalEmbeddingModel and MockTextEmbeddingModel."""
from __future__ import annotations

import numpy as np
import pytest

from src.embeddings import LocalEmbeddingModel
from src.mocks import MockTextEmbeddingModel


@pytest.fixture(scope="module")
def embedder() -> LocalEmbeddingModel:
    return LocalEmbeddingModel()


class TestLocalEmbeddingModel:
    def test_embed_returns_correct_shape(self, embedder: LocalEmbeddingModel) -> None:
        texts = ["hello world", "semantic search", "vector database"]
        result = embedder.embed(texts)
        assert result.shape == (3, 384)

    def test_embed_returns_float32(self, embedder: LocalEmbeddingModel) -> None:
        result = embedder.embed(["test"])
        assert result.dtype == np.float32

    def test_embeddings_are_l2_normalised(self, embedder: LocalEmbeddingModel) -> None:
        result = embedder.embed(["normalisation check", "another sentence"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_one_is_1d(self, embedder: LocalEmbeddingModel) -> None:
        result = embedder.embed_one("single sentence")
        assert result.ndim == 1
        assert result.shape[0] == 384

    def test_embed_one_matches_embed_batch(self, embedder: LocalEmbeddingModel) -> None:
        text = "consistency check"
        single = embedder.embed_one(text)
        batch = embedder.embed([text])[0]
        np.testing.assert_allclose(single, batch, atol=1e-6)

    def test_similar_sentences_have_higher_cosine(
        self, embedder: LocalEmbeddingModel
    ) -> None:
        a = embedder.embed_one("auto-scaling handles traffic spikes")
        b = embedder.embed_one("horizontal scaling manages peak load")
        c = embedder.embed_one("chocolate cake recipe")
        # a and b should be more similar to each other than a and c
        assert float(np.dot(a, b)) > float(np.dot(a, c))


class TestMockTextEmbeddingModel:
    def test_from_pretrained_returns_instance(self) -> None:
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert isinstance(model, MockTextEmbeddingModel)

    def test_from_pretrained_is_singleton(self) -> None:
        m1 = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        m2 = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert m1 is m2

    def test_get_embeddings_returns_list(self) -> None:
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        results = model.get_embeddings(["hello", "world"])
        assert len(results) == 2

    def test_get_embeddings_values_are_floats(self) -> None:
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        results = model.get_embeddings(["test sentence"])
        assert isinstance(results[0].values, list)
        assert all(isinstance(v, float) for v in results[0].values)

    def test_get_embeddings_values_length(self) -> None:
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        results = model.get_embeddings(["check dimension"])
        assert len(results[0].values) == 384
