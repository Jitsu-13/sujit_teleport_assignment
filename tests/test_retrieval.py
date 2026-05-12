"""Tests for StrategyA, StrategyB, and RAGPipeline.

GCP SDK mocking: MockGenerativeModel.generate_content is patched via
unittest.mock.patch to verify that StrategyB calls the model with the
correct prompt and that the pipeline behaves correctly for any model response.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.mocks import MockGenerativeModel, _GenerativeResponse
from src.pipeline import RAGPipeline
from src.retrieval import StrategyA, StrategyB
from src.vector_store import SearchResult

# ---------------------------------------------------------------------------
# Minimal corpus — keeps test runtime low
# ---------------------------------------------------------------------------

_MINI_CORPUS = [
    {
        "id": "doc_a",
        "title": "Auto-scaling",
        "text": (
            "Auto-scaling adds instances during peak load to absorb traffic spikes "
            "and removes them during quiet periods to reduce cost."
        ),
    },
    {
        "id": "doc_b",
        "title": "Circuit Breakers",
        "text": (
            "Circuit breakers prevent cascading failures by fast-failing calls to "
            "an unhealthy dependency and reopening after a cooldown window."
        ),
    },
    {
        "id": "doc_c",
        "title": "Vector Search",
        "text": (
            "Semantic vector search uses dense embeddings and cosine similarity to "
            "retrieve the most relevant document chunks for a given query."
        ),
    },
]


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    return RAGPipeline(documents=_MINI_CORPUS)


# ---------------------------------------------------------------------------
# StrategyA
# ---------------------------------------------------------------------------

class TestStrategyA:
    def test_retrieve_returns_search_results(self, pipeline: RAGPipeline) -> None:
        results = pipeline.search_a("how does scaling work?", top_k=2)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_retrieve_respects_top_k(self, pipeline: RAGPipeline) -> None:
        results = pipeline.search_a("peak load handling", top_k=1)
        assert len(results) == 1

    def test_retrieve_peak_load_surfaces_autoscaling(
        self, pipeline: RAGPipeline
    ) -> None:
        results = pipeline.search_a("peak load traffic spike", top_k=1)
        assert results[0].doc_id == "doc_a"

    def test_retrieve_failure_surfaces_circuit_breaker(
        self, pipeline: RAGPipeline
    ) -> None:
        results = pipeline.search_a("cascading failure prevention", top_k=1)
        assert results[0].doc_id == "doc_b"

    def test_scores_are_descending(self, pipeline: RAGPipeline) -> None:
        results = pipeline.search_a("semantic search embeddings", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# StrategyB — mocking the GCP GenerativeModel
# ---------------------------------------------------------------------------

class TestStrategyB:
    def test_expand_query_calls_generate_content(
        self, pipeline: RAGPipeline
    ) -> None:
        with patch.object(
            pipeline._strategy_b._gen_model,  # type: ignore[union-attr]
            "generate_content",
            return_value=_GenerativeResponse("expanded query text"),
        ) as mock_gen:
            pipeline._strategy_b.expand_query("peak load")  # type: ignore[union-attr]
            mock_gen.assert_called_once()

    def test_expand_query_prompt_contains_original_query(
        self, pipeline: RAGPipeline
    ) -> None:
        original = "How does the system handle peak load?"
        with patch.object(
            pipeline._strategy_b._gen_model,  # type: ignore[union-attr]
            "generate_content",
            return_value=_GenerativeResponse("expanded"),
        ) as mock_gen:
            pipeline._strategy_b.expand_query(original)  # type: ignore[union-attr]
            call_prompt = mock_gen.call_args[0][0]
            assert original in call_prompt

    def test_retrieve_uses_expanded_query(self, pipeline: RAGPipeline) -> None:
        # Force the model to return an expansion that strongly targets doc_c
        forced_expansion = (
            "semantic vector search dense embeddings cosine similarity retrieval"
        )
        with patch.object(
            pipeline._strategy_b._gen_model,  # type: ignore[union-attr]
            "generate_content",
            return_value=_GenerativeResponse(forced_expansion),
        ):
            results = pipeline.search_b("tell me something", top_k=1)
            assert results[0].doc_id == "doc_c"

    def test_retrieve_returns_search_results(self, pipeline: RAGPipeline) -> None:
        results = pipeline.search_b("peak load scaling", top_k=2)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_mock_generative_model_expand_peak_load(self) -> None:
        model = MockGenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            "Rewrite the following query:\nHow does the system handle peak load?"
        )
        assert "peak load" in response.text.lower() or "scal" in response.text.lower()

    def test_mock_generative_model_generic_fallback(self) -> None:
        model = MockGenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            "Rewrite the following query:\nxyzzy frobnicate quux"
        )
        # Generic fallback should still return a non-empty string
        assert len(response.text) > 0

    def test_strategy_b_with_fully_mocked_gcp_sdk(self) -> None:
        """Patch MockGenerativeModel at the class level to simulate full SDK mock."""
        mock_model = MagicMock(spec=MockGenerativeModel)
        mock_model.generate_content.return_value = _GenerativeResponse(
            "auto-scaling load balancing traffic peak"
        )

        from src.embeddings import LocalEmbeddingModel
        from src.vector_store import VectorStore

        embedder = LocalEmbeddingModel()
        embeddings = embedder.embed([d["text"] for d in _MINI_CORPUS])
        store = VectorStore(embedding_dim=embeddings.shape[1])
        store.add(_MINI_CORPUS, embeddings)

        strategy = StrategyB(store, embedder, generative_model=mock_model)
        results = strategy.retrieve("vague query", top_k=2)

        mock_model.generate_content.assert_called_once()
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def test_ingest_populates_store(self, pipeline: RAGPipeline) -> None:
        assert pipeline._store is not None
        assert len(pipeline._store) == len(_MINI_CORPUS)  # type: ignore[arg-type]

    def test_run_benchmark_returns_one_entry_per_query(
        self, pipeline: RAGPipeline
    ) -> None:
        queries = ["peak load", "failure handling"]
        report = pipeline.run_benchmark(queries=queries, top_k=2)
        assert len(report) == 2

    def test_benchmark_entry_has_required_keys(self, pipeline: RAGPipeline) -> None:
        report = pipeline.run_benchmark(queries=["peak load"], top_k=2)
        entry = report[0]
        assert {"query", "expanded_query", "strategy_a", "strategy_b"} <= entry.keys()

    def test_benchmark_strategy_results_are_lists(
        self, pipeline: RAGPipeline
    ) -> None:
        report = pipeline.run_benchmark(queries=["peak load"], top_k=2)
        assert isinstance(report[0]["strategy_a"], list)
        assert isinstance(report[0]["strategy_b"], list)

    def test_benchmark_result_entries_have_required_fields(
        self, pipeline: RAGPipeline
    ) -> None:
        report = pipeline.run_benchmark(queries=["scaling"], top_k=1)
        result = report[0]["strategy_a"][0]
        assert {"rank", "doc_id", "title", "score", "snippet"} <= result.keys()

    def test_expanded_query_is_longer_than_original(
        self, pipeline: RAGPipeline
    ) -> None:
        query = "peak load"
        report = pipeline.run_benchmark(queries=[query], top_k=1)
        assert len(report[0]["expanded_query"]) > len(query)
