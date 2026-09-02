import math

import pytest

from evals.metrics import dcg_at_k, mean, mrr_at_k, ndcg_at_k, percentile, recall_at_k

pytestmark = pytest.mark.unit


def test_recall_at_k_partial_match():
    ranked = ["a", "b", "c"]
    relevant = {"a", "b", "z"}
    assert recall_at_k(ranked, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_only_counts_within_k():
    ranked = ["z", "a", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(ranked, relevant, k=1) == pytest.approx(0.0)
    assert recall_at_k(ranked, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_no_relevant_docs_returns_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_mrr_at_k_first_hit_at_rank_one():
    assert mrr_at_k(["a", "b"], {"a"}, k=5) == pytest.approx(1.0)


def test_mrr_at_k_first_hit_at_rank_three():
    assert mrr_at_k(["x", "y", "a"], {"a"}, k=5) == pytest.approx(1 / 3)


def test_mrr_at_k_no_hit_within_k_returns_zero():
    assert mrr_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_dcg_matches_hand_computed_value():
    # relevance: a=1, b=1, c=0 (irrelevant), ranked as [c, a, b]
    # DCG = 0 (c, rel=0) + 1/log2(3) (a at position 2) + 1/log2(4) (b at position 3)
    relevance = {"a": 1, "b": 1, "c": 0}
    ranked = ["c", "a", "b"]
    expected = 0 + (1 / math.log2(3)) + (1 / math.log2(4))
    assert dcg_at_k(ranked, relevance, k=3) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b"], relevance, k=2) == pytest.approx(1.0)


def test_ndcg_single_relevant_doc_at_rank_two():
    # DCG = 1/log2(3) (only hit, at position 2); IDCG = 1/log2(2) = 1
    # (ideal places it at position 1) -> nDCG = 1/log2(3)
    relevance = {"a": 1}
    ndcg = ndcg_at_k(["z", "a"], relevance, k=2)
    assert ndcg == pytest.approx(1 / math.log2(3))


def test_ndcg_worst_ranking_against_hand_computed_value():
    # From the SciFact-shaped worked example: relevance a=1,b=1,c=0,
    # retrieved in the worst order [c, a, b].
    relevance = {"a": 1, "b": 1, "c": 0}
    dcg = 0 + (1 / math.log2(3)) + (1 / math.log2(4))
    idcg = (1 / math.log2(2)) + (1 / math.log2(3))
    assert ndcg_at_k(["c", "a", "b"], relevance, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_no_relevant_docs_returns_zero_not_nan():
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=2) == 0.0


def test_ndcg_graded_relevance_prefers_higher_grade_first():
    # rel 2 (highly relevant) should score higher when placed first than
    # when a rel 1 doc is placed first, at otherwise-equal positions.
    relevance = {"high": 2, "low": 1}
    ndcg_high_first = ndcg_at_k(["high", "low"], relevance, k=2)
    ndcg_low_first = ndcg_at_k(["low", "high"], relevance, k=2)
    assert ndcg_high_first > ndcg_low_first
    assert ndcg_high_first == pytest.approx(1.0)  # correct order = ideal = perfect score


def test_mean_of_empty_list_is_zero_not_error():
    assert mean([]) == 0.0


def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_percentile_empty_list_is_zero():
    assert percentile([], 0.95) == 0.0


def test_percentile_p50_matches_median_for_odd_length():
    assert percentile([10.0, 20.0, 30.0], 0.50) == 20.0


def test_percentile_p95_picks_a_high_value_not_the_max():
    values = list(range(1, 101))  # 1..100
    p95 = percentile([float(v) for v in values], 0.95)
    assert p95 == 96.0  # nearest-rank at index int(100*0.95)=95 -> values[95] == 96


def test_percentile_matches_run_retrieval_evals_old_inline_p50_p95_logic():
    # regression guard for the extraction out of run_retrieval_eval.py's
    # _aggregate — same sorted-index method, same real values.
    latencies = [100.0, 200.0, 150.0, 300.0, 5000.0]
    ordered = sorted(latencies)
    assert percentile(latencies, 0.50) == ordered[len(ordered) // 2]
    assert percentile(latencies, 0.95) == ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
