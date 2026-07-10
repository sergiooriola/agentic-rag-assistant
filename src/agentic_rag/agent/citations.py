"""Map ``[n]`` citation markers in a generated answer back to their chunks."""

import re
from typing import Any

_MARKER = re.compile(r"\[(\d+)\]")


def extract_citations(
    answer: str,
    context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the context entries cited in ``answer``, ordered by first mention.

    Markers are 1-based indexes into ``context`` (the numbering produced by
    ``format_context``). Duplicates are collapsed and out-of-range markers are
    ignored — a model can always emit ``[7]`` with only four excerpts, and a
    phantom citation must not crash the request or fabricate a source.
    """
    citations: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in _MARKER.finditer(answer):
        index = int(match.group(1))
        if index in seen or not 1 <= index <= len(context):
            continue
        seen.add(index)
        chunk = context[index - 1]
        citations.append(
            {
                "index": index,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk.get("page"),
            }
        )
    return citations
