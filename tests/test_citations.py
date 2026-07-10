"""Citation parsing tests: ordering, deduplication, and phantom markers."""

from agentic_rag.agent.citations import extract_citations

CONTEXT = [
    {"chunk_id": "a.md:p0:c0", "source": "a.md", "page": None, "text": "alpha"},
    {"chunk_id": "b.pdf:p2:c1", "source": "b.pdf", "page": 2, "text": "beta"},
    {"chunk_id": "c.md:p0:c3", "source": "c.md", "page": None, "text": "gamma"},
]


def test_citations_ordered_by_first_mention() -> None:
    citations = extract_citations("Claim one [2]. Claim two [1]. More on one [2].", CONTEXT)
    assert [c["index"] for c in citations] == [2, 1]
    assert citations[0]["source"] == "b.pdf"
    assert citations[0]["page"] == 2


def test_duplicate_markers_are_collapsed() -> None:
    citations = extract_citations("A [1]. B [1]. C [1].", CONTEXT)
    assert len(citations) == 1


def test_out_of_range_markers_are_ignored() -> None:
    # The model can hallucinate [7] with three excerpts; that must not crash
    # or fabricate a source.
    citations = extract_citations("Real [3]. Phantom [7]. Zero [0].", CONTEXT)
    assert [c["index"] for c in citations] == [3]


def test_answer_without_markers_yields_no_citations() -> None:
    assert extract_citations("No citations here.", CONTEXT) == []
    assert extract_citations("", CONTEXT) == []
