"""In-memory BM25 lexical index over the chunk corpus.

The index is *derived* from the Chroma collection rather than persisted
separately: Chroma remains the single source of truth for chunks and
metadata, and the BM25 index is rebuilt from it in memory at startup (and
after re-ingestion). For a portfolio-sized corpus this takes milliseconds and
eliminates a whole class of index-out-of-sync bugs.
"""

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer (alphanumerics only).

    Deliberately simple. A production lexical engine would add stemming,
    stop-word handling, and language-aware analysis (or delegate to something
    like SQLite FTS5 / Elasticsearch), but for exact-term matching over a
    small technical corpus this captures most of BM25's value.
    """
    return _WORD.findall(text.lower())


class BM25Index:
    """BM25 ranking over a list of chunk Documents."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._bm25 = (
            BM25Okapi([tokenize(doc.page_content) for doc in documents]) if documents else None
        )

    @classmethod
    def from_vector_store(cls, store: Chroma) -> "BM25Index":
        """Build the lexical index from every chunk stored in Chroma."""
        data = store.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(data["documents"], data["metadatas"], strict=True)
        ]
        return cls(documents)

    def search(self, query: str, k: int) -> list[Document]:
        """Return up to ``k`` documents by BM25 score, best first.

        Zero-score documents (no term overlap with the query) are excluded:
        BM25 has nothing meaningful to say about them, and letting them
        through would inject noise into the fusion step.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # Stable sort: equal scores keep corpus order, so results are deterministic.
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._documents[i] for i in ranked[:k] if scores[i] > 0]
