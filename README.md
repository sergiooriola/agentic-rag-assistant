# agentic-rag-assistant

Conversational RAG agent built with **LangGraph** and **LangChain**: it decides whether a
question requires retrieving documentary context, performs **hybrid retrieval** (embeddings +
BM25 with RRF fusion) over **ChromaDB**, generates answers **with source citations**, and
verifies their **groundedness** before responding. Exposed as a REST API with **FastAPI**.

> 🚧 Work in progress, built in phases. This README will be completed in the final phase
> with the graph diagram, architecture decisions, and deployment instructions.

## Structure

```
src/agentic_rag/
├── config.py      # typed settings (pydantic-settings)
├── ingestion/     # PDF/Markdown loading, chunking, Chroma indexing
├── retrieval/     # hybrid search (semantic + BM25 + RRF)
├── agent/         # LangGraph graph (intent → retrieval → generation → verification)
└── api/           # FastAPI: /chat, /ingest, /health
data/              # source documents (PDF and Markdown)
scripts/           # ingestion and evaluation
tests/             # pytest
```

## Getting started (development)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # then fill in OPENAI_API_KEY
pytest
```
