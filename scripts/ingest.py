"""(Re)build the vector index from the documents in data/.

Usage:
    python scripts/ingest.py             # full rebuild (calls the OpenAI embeddings API)
    python scripts/ingest.py --dry-run   # load + chunk only; no API calls, no writes
"""

import argparse
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.indexer import index_chunks
from agentic_rag.ingestion.loaders import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="(Re)build the Chroma index from data/")
    parser.add_argument("--data-dir", type=Path, default=None, help="override the data directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and chunk documents without embedding or indexing",
    )
    args = parser.parse_args()

    settings = get_settings()
    data_dir: Path = args.data_dir or settings.data_dir

    documents = load_documents(data_dir)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    sources = sorted({chunk.metadata["source"] for chunk in chunks})
    print(f"Loaded {len(documents)} documents from {len(sources)} files: {', '.join(sources)}")
    print(
        f"Produced {len(chunks)} chunks "
        f"(chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap} tokens)"
    )

    if args.dry_run:
        print("Dry run: skipped embedding and indexing.")
        return

    total = index_chunks(chunks, settings)
    print(f"Indexed {total} chunks into '{settings.collection_name}' at {settings.chroma_dir}/")


if __name__ == "__main__":
    main()
