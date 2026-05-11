"""Mock implementations of vertexai SDK classes.

These stand in for:
  - vertexai.language_models.TextEmbeddingModel
  - vertexai.generative_models.GenerativeModel

Both classes present the same public interface as their real counterparts so
they can be swapped in via unittest.mock.patch or used directly in Strategy B
without installing the heavyweight GCP SDK.
"""
from __future__ import annotations

from typing import List

from src.embeddings import LocalEmbeddingModel

# ---------------------------------------------------------------------------
# TextEmbeddingModel mock
# ---------------------------------------------------------------------------

class _TextEmbedding:
    """Mirrors the vertexai TextEmbedding response object."""

    def __init__(self, values: List[float]) -> None:
        self.values = values


class MockTextEmbeddingModel:
    """Drop-in for vertexai.language_models.TextEmbeddingModel.

    Usage mirrors the real SDK:
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embeddings = model.get_embeddings(["some text"])
        vector = embeddings[0].values
    """

    _shared: MockTextEmbeddingModel | None = None

    @classmethod
    def from_pretrained(cls, model_name: str) -> MockTextEmbeddingModel:
        # Singleton — avoid reloading sentence-transformers on every call.
        if cls._shared is None:
            cls._shared = cls(model_name)
        return cls._shared

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._embedder = LocalEmbeddingModel()

    def get_embeddings(self, texts: List[str]) -> List[_TextEmbedding]:
        vectors = self._embedder.embed(texts)
        return [_TextEmbedding(vec.tolist()) for vec in vectors]


# ---------------------------------------------------------------------------
# GenerativeModel mock — query expansion
# ---------------------------------------------------------------------------

# Maps query keywords to domain-specific expansion terms.
# Each entry broadens the query in the direction a real LLM would rewrite it,
# making the subsequent embedding search pick up more relevant chunks.
_EXPANSION_MAP = {
    "peak load": (
        "traffic spike auto-scaling horizontal scaling load balancing "
        "capacity throughput burst handling"
    ),
    "scale": (
        "horizontal scaling distributed replica sharding partitioning "
        "elasticity capacity"
    ),
    "performance": (
        "latency throughput response time optimisation bottleneck "
        "benchmark profiling"
    ),
    "failure": (
        "fault tolerance circuit breaker retry fallback resilience "
        "degraded mode availability"
    ),
    "cache": (
        "in-memory Redis Memcached eviction write-through hit ratio "
        "invalidation TTL"
    ),
    "search": (
        "semantic similarity vector embedding retrieval nearest neighbour "
        "cosine distance index"
    ),
    "database": (
        "sharding read replica connection pool consistency ACID "
        "transaction query optimisation"
    ),
    "queue": (
        "message broker Pub/Sub Kafka consumer group dead-letter "
        "backpressure asynchronous decoupling"
    ),
    "monitor": (
        "observability metrics traces logs SLO SLA alerting dashboard "
        "capacity planning"
    ),
    "retrieval": (
        "RAG context grounding embedding similarity top-k chunk "
        "document relevance"
    ),
}


class _GenerativeResponse:
    """Mirrors the vertexai GenerativeModel response object."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockGenerativeModel:
    """Drop-in for vertexai.generative_models.GenerativeModel.

    Implements deterministic query expansion: it scans the prompt for known
    technical keywords and appends domain-specific expansion terms, producing
    a richer embedding-friendly query string.

    Usage mirrors the real SDK:
        model = MockGenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        expanded_query = response.text
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def generate_content(self, prompt: str) -> _GenerativeResponse:
        original_query = self._extract_query(prompt)
        expanded = self._expand(original_query)
        return _GenerativeResponse(expanded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_query(self, prompt: str) -> str:
        # Convention used by retrieval.py:
        #   "Rewrite the following query … :\n<query>"
        # Fall back to using the whole prompt if the delimiter is absent.
        if ":\n" in prompt:
            return prompt.split(":\n", 1)[-1].strip()
        if ":" in prompt:
            return prompt.split(":", 1)[-1].strip()
        return prompt.strip()

    def _expand(self, query: str) -> str:
        query_lower = query.lower()
        additions: List[str] = []
        for keyword, terms in _EXPANSION_MAP.items():
            if keyword in query_lower:
                additions.append(terms)

        if additions:
            return f"{query} {' '.join(additions)}"

        # Generic expansion when no specific keyword matches.
        return (
            f"{query} system architecture scalability reliability "
            "performance distributed infrastructure"
        )
