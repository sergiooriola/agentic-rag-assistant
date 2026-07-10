"""Load source documents from the data directory.

PDFs are loaded page by page (one ``Document`` per page) so that citations can
point to an exact page. Markdown files have no page concept, so each file
becomes a single ``Document``; page-level granularity is recovered later by
the chunking step.

We read PDFs with ``pypdf`` directly instead of pulling in
``langchain-community`` loaders: for two file types, a thin loader is less
code than the dependency it would replace.
"""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: Path) -> list[Document]:
    """Return one Document per non-empty page, with ``source`` and 1-based ``page`` metadata."""
    reader = PdfReader(path)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_number},
                )
            )
    return documents


def load_markdown(path: Path) -> list[Document]:
    """Return the whole file as a single Document (empty files are skipped)."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name})]


def load_documents(data_dir: Path) -> list[Document]:
    """Load every supported document (PDF, Markdown) under ``data_dir``, recursively.

    Files are visited in sorted order so ingestion is deterministic across runs
    and operating systems. Unsupported extensions are silently ignored.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(load_pdf(path))
        elif suffix in {".md", ".markdown"}:
            documents.extend(load_markdown(path))
    return documents
