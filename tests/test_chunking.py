"""Chunking tests: token-size limits, metadata propagation, and deterministic IDs."""

import pytest
import tiktoken
from langchain_core.documents import Document

from agentic_rag.ingestion.chunking import chunk_documents

ENCODING = tiktoken.get_encoding("cl100k_base")

LONG_TEXT = "\n\n".join(
    f"Paragraph {i}. " + "The quick brown fox jumps over the lazy dog. " * 8
    for i in range(12)
)


def _make_doc(page: int | None = None) -> Document:
    metadata = {"source": "sample.md"} if page is None else {"source": "sample.pdf", "page": page}
    return Document(page_content=LONG_TEXT, metadata=metadata)


def test_chunks_respect_token_budget() -> None:
    chunks = chunk_documents([_make_doc()], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(ENCODING.encode(c.page_content)) <= 100 for c in chunks)


def test_metadata_is_propagated() -> None:
    chunks = chunk_documents([_make_doc(page=2)], chunk_size=100, chunk_overlap=20)
    assert all(c.metadata["source"] == "sample.pdf" for c in chunks)
    assert all(c.metadata["page"] == 2 for c in chunks)


def test_chunk_ids_are_unique_and_deterministic() -> None:
    docs = [_make_doc(), _make_doc(page=1)]
    first = [c.metadata["chunk_id"] for c in chunk_documents(docs, 100, 20)]
    second = [c.metadata["chunk_id"] for c in chunk_documents(docs, 100, 20)]
    assert first == second  # same input -> same IDs (idempotent ingestion)
    assert len(first) == len(set(first))  # no collisions
    assert first[0] == "sample.md:p0:c0"


def test_smaller_chunk_size_yields_more_chunks() -> None:
    small = chunk_documents([_make_doc()], chunk_size=80, chunk_overlap=10)
    large = chunk_documents([_make_doc()], chunk_size=400, chunk_overlap=10)
    assert len(small) > len(large)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_documents([_make_doc()], chunk_size=100, chunk_overlap=100)
