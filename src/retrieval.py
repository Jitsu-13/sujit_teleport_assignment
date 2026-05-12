from __future__ import annotations

from typing import List

from src.embeddings import LocalEmbeddingModel
from src.mocks import MockGenerativeModel
from src.vector_store import SearchResult, VectorStore

_EXPANSION_PROMPT_TEMPLATE = (
    "Rewrite the following search query to be more specific and include "
    "related technical terminology that will improve semantic search results. "
    "Return only the rewritten query, nothing else:\n{query}"
)


class StrategyA:
    """Raw vector search — embed the query as-is and retrieve top-K chunks."""

    def __init__(self, store: VectorStore, embedder: LocalEmbeddingModel) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        vector = self._embedder.embed_one(query)
        return self._store.search(vector, top_k=top_k)


class StrategyB:
    """AI-enhanced retrieval — expand the query via a generative model first.

    The MockGenerativeModel rewrites the query into a richer, more specific
    form before embedding and searching, improving recall for vague or
    ambiguous queries.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: LocalEmbeddingModel,
        generative_model: MockGenerativeModel | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._gen_model = generative_model or MockGenerativeModel("gemini-1.5-pro")

    def expand_query(self, query: str) -> str:
        prompt = _EXPANSION_PROMPT_TEMPLATE.format(query=query)
        response = self._gen_model.generate_content(prompt)
        return response.text

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        expanded = self.expand_query(query)
        vector = self._embedder.embed_one(expanded)
        return self._store.search(vector, top_k=top_k)
