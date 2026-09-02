from unittest.mock import MagicMock, patch

import pytest

from src.retrieve.hybrid import real_categories, _sanitize_filters, build_filter_clauses, rrf_fuse, rrf_fuse_with_scores

pytestmark = pytest.mark.unit


def test_no_filters_yields_no_clauses():
    assert build_filter_clauses({}) == []


def test_author_filter_uses_analyzed_text_subfield():
    clauses = build_filter_clauses({"author": "LeCun"})
    assert clauses == [{"match": {"authors.text": "LeCun"}}]


def test_category_filter_uses_exact_keyword_term():
    clauses = build_filter_clauses({"category": "cs.CL"})
    assert clauses == [{"term": {"category": "cs.CL"}}]


def test_date_range_filter_both_bounds():
    clauses = build_filter_clauses({"date_from": "2026-01-01", "date_to": "2026-06-01"})
    assert clauses == [{"range": {"published_date": {"gte": "2026-01-01", "lte": "2026-06-01"}}}]


def test_date_range_filter_one_bound_only():
    clauses = build_filter_clauses({"date_from": "2026-01-01"})
    assert clauses == [{"range": {"published_date": {"gte": "2026-01-01"}}}]


def test_all_filters_combine():
    clauses = build_filter_clauses({"author": "X", "category": "cs.IR", "date_from": "2026-01-01"})
    assert len(clauses) == 3


def test_rrf_top_rank_in_both_lists_wins():
    fused = rrf_fuse([["a", "b", "c"], ["a", "d", "b"]], k=60)
    assert fused[0] == "a"


def test_rrf_matches_hand_computed_scores():
    fused = rrf_fuse([["a", "b"], ["a", "d", "b"]], k=60)
    score_a = 1 / 61 + 1 / 61
    score_b = 1 / 62 + 1 / 63
    score_d = 1 / 62
    expected_order = sorted(["a", "b", "d"], key=lambda x: {"a": score_a, "b": score_b, "d": score_d}[x], reverse=True)
    assert fused == expected_order


def test_rrf_doc_only_in_one_list_still_included():
    fused = rrf_fuse([["a"], ["b"]], k=60)
    assert set(fused) == {"a", "b"}


def test_rrf_empty_lists_yield_empty_result():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_fuse_with_scores_exposes_real_arithmetic():
    # rrf_fuse() is rrf_fuse_with_scores() reduced to just an ordering —
    # this checks the underlying scores it's built from, which is what
    # A5's fusion-arithmetic panel actually displays.
    scores = rrf_fuse_with_scores([["a", "b"], ["a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_rrf_fuse_and_rrf_fuse_with_scores_agree_on_order():
    lists = [["a", "b", "c"], ["c", "a"]]
    order = rrf_fuse(lists)
    scores = rrf_fuse_with_scores(lists)
    assert order == sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


# ---------------------------------------------------------------------
# _sanitize_filters / real_categories — added 2026-08-24 after a real
# bug found in A8's first genuinely complete run: the rewrite LLM
# extracted a plausible-sounding category ("cs.LG") that was never
# actually assigned to the real paper it was asking about, and the
# category filter's hard exact `term` match then zeroed out every
# retrieval arm — a real, silent 0-candidate result that then correctly
# abstained given genuinely empty context. Two real cases (q025, q073)
# traced to exactly this.
# ---------------------------------------------------------------------


def _fake_client_with_categories(categories: list[str]) -> MagicMock:
    client = MagicMock()
    client.search.return_value = {"aggregations": {"cats": {"buckets": [{"key": c} for c in categories]}}}
    return client


def testreal_categories_returns_the_real_distinct_set():
    client = _fake_client_with_categories(["cs.IR", "cs.CL", "cs.AI"])
    with patch("src.retrieve.hybrid.get_json", return_value=None), patch("src.retrieve.hybrid.set_json"):
        result = real_categories(client, "rag_chunks")
    assert result == {"cs.IR", "cs.CL", "cs.AI"}


def testreal_categories_uses_cache_and_skips_a_second_real_query():
    client = _fake_client_with_categories(["cs.IR"])
    with patch("src.retrieve.hybrid.get_json", return_value=["cs.IR", "cs.CL"]):
        result = real_categories(client, "rag_chunks")
    assert result == {"cs.IR", "cs.CL"}
    client.search.assert_not_called()


def test_sanitize_filters_drops_a_hallucinated_category():
    # Exactly q025/q073's real, live-traced failure mode: the extracted
    # category doesn't exist anywhere in the real corpus.
    client = _fake_client_with_categories(["cs.IR", "cs.CL", "cs.AI", "cs.CV", "cs.SI", "cs.DB"])
    with patch("src.retrieve.hybrid.get_json", return_value=None), patch("src.retrieve.hybrid.set_json"):
        result = _sanitize_filters(client, "rag_chunks", {"category": "cs.LG", "author": "LeCun"})
    assert "category" not in result
    assert result["author"] == "LeCun"  # untouched — only category is validated


def test_sanitize_filters_keeps_a_real_category():
    client = _fake_client_with_categories(["cs.IR", "cs.CL"])
    with patch("src.retrieve.hybrid.get_json", return_value=None), patch("src.retrieve.hybrid.set_json"):
        result = _sanitize_filters(client, "rag_chunks", {"category": "cs.IR"})
    assert result == {"category": "cs.IR"}


def test_sanitize_filters_no_category_is_a_no_op():
    client = MagicMock()
    result = _sanitize_filters(client, "rag_chunks", {"author": "LeCun"})
    assert result == {"author": "LeCun"}
    client.search.assert_not_called()  # never even checks categories when none was extracted
