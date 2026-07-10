"""RRF fusion tests: consensus wins, determinism, and edge cases."""

from agentic_rag.retrieval.fusion import reciprocal_rank_fusion


def test_document_in_both_lists_beats_single_list_winners() -> None:
    # "b" is mid-ranked in both systems; "a" and "c" are top-1 in only one.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=60)
    assert fused[0][0] == "b"


def test_rank_order_within_a_single_list_is_preserved() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    assert [doc_id for doc_id, _ in fused] == ["x", "y", "z"]


def test_scores_follow_the_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=60))
    assert fused["a"] == 1 / 61
    assert fused["b"] == 1 / 62 + 1 / 61


def test_ties_break_alphabetically_for_determinism() -> None:
    fused = reciprocal_rank_fusion([["z"], ["a"]], k=60)  # same score for both
    assert [doc_id for doc_id, _ in fused] == ["a", "z"]


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
