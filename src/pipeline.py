"""RAGPipeline — top-level orchestration class.

Usage:
    python -m src.pipeline
"""
from __future__ import annotations

import json
from typing import List

from data.corpus import DOCUMENTS
from src.embeddings import LocalEmbeddingModel
from src.retrieval import StrategyA, StrategyB
from src.vector_store import SearchResult, VectorStore

_BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What mechanisms prevent cascading failures?",
    "How is search relevance improved for ambiguous queries?",
]


class RAGPipeline:
    """Ingests a document corpus and exposes Strategy A / B retrieval."""

    def __init__(self, documents: list | None = None) -> None:
        self._embedder = LocalEmbeddingModel()
        self._store: VectorStore | None = None
        self._strategy_a: StrategyA | None = None
        self._strategy_b: StrategyB | None = None

        docs = documents if documents is not None else DOCUMENTS
        self.ingest(docs)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list) -> None:
        """Embed all documents and load them into the vector store."""
        texts = [doc["text"] for doc in documents]
        embeddings = self._embedder.embed(texts)

        dim = embeddings.shape[1]
        self._store = VectorStore(embedding_dim=dim)
        self._store.add(documents, embeddings)

        self._strategy_a = StrategyA(self._store, self._embedder)
        self._strategy_b = StrategyB(self._store, self._embedder)

        print(f"Ingested {len(self._store)} documents (dim={dim})")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search_a(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Strategy A: raw vector search."""
        if self._strategy_a is None:
            raise RuntimeError("Pipeline not initialised — call ingest() first.")
        return self._strategy_a.retrieve(query, top_k=top_k)

    def search_b(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Strategy B: query-expanded vector search."""
        if self._strategy_b is None:
            raise RuntimeError("Pipeline not initialised — call ingest() first.")
        return self._strategy_b.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def run_benchmark(
        self, queries: List[str] | None = None, top_k: int = 3
    ) -> List[dict]:
        """Run both strategies on each query and return a comparison report."""
        queries = queries or _BENCHMARK_QUERIES
        report = []

        for query in queries:
            expanded = self._strategy_b.expand_query(query)  # type: ignore[union-attr]
            results_a = self.search_a(query, top_k=top_k)
            results_b = self.search_b(query, top_k=top_k)

            report.append(
                {
                    "query": query,
                    "expanded_query": expanded,
                    "strategy_a": _serialise(results_a),
                    "strategy_b": _serialise(results_b),
                }
            )

        return report


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _serialise(results: List[SearchResult]) -> List[dict]:
    return [
        {
            "rank": r.rank,
            "doc_id": r.doc_id,
            "title": r.title,
            "score": round(r.score, 4),
            "snippet": r.text[:120] + "…",
        }
        for r in results
    ]


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = RAGPipeline()
    report = pipeline.run_benchmark()
    print(json.dumps(report, indent=2))
