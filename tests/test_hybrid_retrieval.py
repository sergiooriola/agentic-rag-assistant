"""BM25 index and hybrid retriever tests — no ChromaDB, no API calls."""

from langchain_core.documents import Document

from agentic_rag.retrieval.bm25 import BM25Index
from agentic_rag.retrieval.hybrid import HybridRetriever


def _doc(chunk_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "source": "test.md"})


CORPUS = [
    _doc("c1", "HTTP status code 429 signals rate limiting on the server"),
    _doc("c2", "Git rebase rewrites commit history to produce a linear log"),
    _doc("c3", "Cursor pagination returns an opaque token for the next page"),
]


class FakeVectorStore:
    """Returns a preset ranking regardless of the query."""

    def __init__(self, ranked_docs: list[Document]) -> None:
        self._ranked_docs = ranked_docs

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._ranked_docs[:k]


# --- BM25 ---


def test_bm25_finds_exact_terms() -> None:
    index = BM25Index(CORPUS)
    results = index.search("429 rate limiting", k=2)
    assert results and results[0].metadata["chunk_id"] == "c1"


def test_bm25_excludes_zero_score_documents() -> None:
    index = BM25Index(CORPUS)
    assert index.search("quantum entanglement", k=3) == []


def test_bm25_empty_corpus_returns_empty() -> None:
    assert BM25Index([]).search("anything", k=3) == []


# --- Hybrid fusion ---


def test_hybrid_surfaces_documents_from_both_systems() -> None:
    # Vector search only knows about c2; BM25 will only match c1.
    retriever = HybridRetriever(
        FakeVectorStore([CORPUS[1]]),
        BM25Index(CORPUS),
        top_k_final=4,
    )
    ids = {doc.metadata["chunk_id"] for doc in retriever.retrieve("429 rate limiting")}
    assert {"c1", "c2"} <= ids


def test_hybrid_ranks_consensus_first_and_attaches_scores() -> None:
    # c3 appears in both rankings; c2 only in the semantic one.
    retriever = HybridRetriever(
        FakeVectorStore([CORPUS[2], CORPUS[1]]),
        BM25Index(CORPUS),
        top_k_final=4,
    )
    results = retriever.retrieve("cursor pagination opaque token")
    assert results[0].metadata["chunk_id"] == "c3"
    assert all("rrf_score" in doc.metadata for doc in results)


def test_hybrid_respects_top_k_final() -> None:
    retriever = HybridRetriever(
        FakeVectorStore(CORPUS),
        BM25Index(CORPUS),
        top_k_final=2,
    )
    assert len(retriever.retrieve("pagination token git rebase http 429")) == 2
