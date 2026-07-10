"""Hybrid retriever: semantic and lexical candidates fused with RRF.

Why hybrid beats pure-vector: embeddings are robust to paraphrase but blur
exact identifiers (status codes, RFC numbers, command names, acronyms), which
is precisely where BM25 excels — and vice versa, BM25 is blind to synonyms.
Real query traffic mixes both kinds, so running both searches and fusing the
rankings covers each system's blind spot.
"""

from typing import Protocol

from langchain_core.documents import Document

from agentic_rag.config import Settings
from agentic_rag.retrieval.bm25 import BM25Index
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion


class VectorSearcher(Protocol):
    """The one method we need from a vector store (lets tests inject a fake)."""

    def similarity_search(self, query: str, k: int) -> list[Document]: ...


class HybridRetriever:
    """Runs semantic + BM25 search and returns the RRF-fused top results."""

    def __init__(
        self,
        vector_store: VectorSearcher,
        bm25: BM25Index,
        *,
        top_k_semantic: int = 8,
        top_k_lexical: int = 8,
        top_k_final: int = 4,
        rrf_k: int = 60,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25
        self._top_k_semantic = top_k_semantic
        self._top_k_lexical = top_k_lexical
        self._top_k_final = top_k_final
        self._rrf_k = rrf_k

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridRetriever":
        """Wire the retriever against the persistent Chroma collection."""
        # Imported here so unit tests can use HybridRetriever without chromadb I/O.
        from agentic_rag.ingestion.indexer import get_vector_store

        store = get_vector_store(settings)
        return cls(
            store,
            BM25Index.from_vector_store(store),
            top_k_semantic=settings.top_k_semantic,
            top_k_lexical=settings.top_k_lexical,
            top_k_final=settings.top_k_final,
            rrf_k=settings.rrf_k,
        )

    def retrieve(self, query: str) -> list[Document]:
        """Return the fused top ``top_k_final`` chunks for ``query``.

        Each returned Document carries an ``rrf_score`` in its metadata so
        downstream consumers (API responses, evaluation) can expose ranking
        strength without recomputing it.
        """
        semantic = self._vector_store.similarity_search(query, k=self._top_k_semantic)
        lexical = self._bm25.search(query, k=self._top_k_lexical)

        by_id: dict[str, Document] = {}
        for doc in [*semantic, *lexical]:
            by_id.setdefault(doc.metadata["chunk_id"], doc)

        fused = reciprocal_rank_fusion(
            [
                [doc.metadata["chunk_id"] for doc in semantic],
                [doc.metadata["chunk_id"] for doc in lexical],
            ],
            k=self._rrf_k,
        )

        results: list[Document] = []
        for chunk_id, score in fused[: self._top_k_final]:
            doc = by_id[chunk_id]
            doc.metadata["rrf_score"] = round(score, 6)
            results.append(doc)
        return results
