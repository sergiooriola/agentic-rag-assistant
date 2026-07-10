"""Split documents into retrieval-sized chunks.

Strategy: *structure-aware splitting with token-based sizing*. The recursive
splitter cuts on the strongest boundary available (paragraph, then line, then
word), which keeps semantically coherent units together, while chunk size and
overlap are measured in **tokens** (tiktoken ``cl100k_base``, the encoding
used by OpenAI embedding models). Measuring in tokens ties the chunk budget to
what the embedding model actually sees, instead of characters, whose token
density varies wildly between prose, code, and tables.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split ``documents`` into chunks of at most ``chunk_size`` tokens.

    Source metadata (``source``, ``page``) is propagated to every chunk, and a
    deterministic, human-readable ``chunk_id`` (``file:pN:cM``) is added so
    that re-ingesting the same corpus produces the same IDs — which makes
    indexing idempotent and citations debuggable.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    counters: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)  # 0 = no page concept (Markdown)
        index = counters.get((source, page), 0)
        counters[(source, page)] = index + 1
        chunk.metadata["chunk_id"] = f"{source}:p{page}:c{index}"
    return chunks
