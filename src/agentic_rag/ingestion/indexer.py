"""Embed chunks and persist them into a local ChromaDB collection.

Rebuild strategy: full reset. On every ingest the collection is deleted and
recreated, which guarantees the index exactly mirrors ``data/`` (no stale
chunks from deleted or renamed files). For corpora that fit a portfolio
project, re-embedding everything costs cents; incremental sync (hashing each
chunk and upserting only changes) is the natural evolution for large corpora.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import Settings

# Keeps each embedding request and Chroma write comfortably under API limits.
_BATCH_SIZE = 100


def get_vector_store(settings: Settings) -> Chroma:
    """Open (or create) the project's persistent Chroma collection."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def index_chunks(chunks: list[Document], settings: Settings) -> int:
    """Rebuild the collection from ``chunks`` and return how many were indexed."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    # get_or_create + delete works whether or not the collection exists yet.
    client.get_or_create_collection(settings.collection_name)
    client.delete_collection(settings.collection_name)

    store = get_vector_store(settings)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    for start in range(0, len(chunks), _BATCH_SIZE):
        store.add_documents(
            chunks[start : start + _BATCH_SIZE],
            ids=ids[start : start + _BATCH_SIZE],
        )
    return len(chunks)
