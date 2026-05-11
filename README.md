# Context-Aware Retrieval Engine

A local Retrieval-Augmented Generation (RAG) pipeline that compares two retrieval strategies using semantic embeddings and a vector database.

## Overview

This project implements and benchmarks two retrieval strategies:

- **Strategy A — Raw Vector Search**: Embeds the user query directly and retrieves the top-K most similar document chunks via cosine similarity.
- **Strategy B — AI-Enhanced Retrieval**: A mocked `GenerativeModel` (simulating Vertex AI) rewrites/expands the query into a richer, embedding-friendly form before the vector search.

## Project Structure

```
.
├── src/
│   ├── embeddings.py       # Local sentence-transformers wrapper (simulates textembedding-gecko)
│   ├── vector_store.py     # FAISS-backed vector store
│   ├── retrieval.py        # Strategy A and Strategy B retrieval logic
│   └── pipeline.py         # RAGPipeline orchestration class
├── data/
│   └── corpus.py           # Sample technical document corpus (5-10 paragraphs)
├── tests/
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_retrieval.py
├── retrieval_benchmark.md  # Strategy A vs Strategy B comparison report
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Running the Benchmark

```bash
python -m src.pipeline
```

## Running Tests

```bash
pytest tests/ -v
```

## Design Decisions

See `retrieval_benchmark.md` for the full comparison report, similarity metric rationale (Cosine vs. Euclidean), and notes on migrating to Vertex AI Vector Search (Matching Engine) in production.
