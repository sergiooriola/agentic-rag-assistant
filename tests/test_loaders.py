"""Loader tests: Markdown via temp files, PDF against the repo's sample document."""

from pathlib import Path

import pytest

from agentic_rag.ingestion.loaders import load_documents, load_markdown, load_pdf

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_PDF = DATA_DIR / "rest_api_design.pdf"


def test_load_markdown(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nSome content.", encoding="utf-8")
    docs = load_markdown(md)
    assert len(docs) == 1
    assert docs[0].page_content.startswith("# Title")
    assert docs[0].metadata == {"source": "notes.md"}  # no page concept for Markdown


def test_load_markdown_skips_empty_file(tmp_path: Path) -> None:
    md = tmp_path / "empty.md"
    md.write_text("   \n\n", encoding="utf-8")
    assert load_markdown(md) == []


def test_load_pdf_one_document_per_page() -> None:
    docs = load_pdf(SAMPLE_PDF)
    assert len(docs) == 3
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]  # 1-based, in order
    assert all(d.metadata["source"] == "rest_api_design.pdf" for d in docs)
    assert "REST API Design Guidelines" in docs[0].page_content


def test_load_documents_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("hello", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.metadata["source"] for d in docs] == ["doc.md"]


def test_load_documents_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(Path("no-such-dir"))
