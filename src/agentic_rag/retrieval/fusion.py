"""Reciprocal rank fusion (RRF).

RRF combines multiple ranked result lists into one using only the *rank* of
each document, never its raw score. This matters because cosine similarities
(vector search) and BM25 scores live on incomparable scales — averaging them
directly would require per-corpus normalization and tuning. RRF sidesteps the
problem entirely and is a strong, parameter-light baseline (Cormack et al.,
SIGIR 2009).
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into a single ranking, best first.

    Each document scores ``sum(1 / (k + rank))`` over every list it appears
    in (rank is 1-based), so items ranked well by *several* systems beat items
    ranked well by only one. The constant ``k`` dampens the gap between top
    ranks: with a small ``k``, rank 1 dwarfs rank 2; with the conventional
    ``k=60``, differences are smoothed and consensus dominates.

    Ties are broken alphabetically by ID so results are deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
