import pytest

from evals.run_chunking_eval import select_variants

pytestmark = pytest.mark.unit


def test_winner_is_highest_ndcg():
    results = {
        "a": {"ndcg@10": 0.5, "chunk_count": 100},
        "b": {"ndcg@10": 0.9, "chunk_count": 100},
        "c": {"ndcg@10": 0.3, "chunk_count": 100},
    }
    selection = select_variants(results)
    assert selection["winner"] == "b"


def test_median_is_the_middle_ranked_strategy():
    results = {
        "a": {"ndcg@10": 0.9, "chunk_count": 100},
        "b": {"ndcg@10": 0.5, "chunk_count": 100},
        "c": {"ndcg@10": 0.1, "chunk_count": 100},
    }
    selection = select_variants(results)
    assert selection["median"] == "b"


def test_efficient_favors_fewer_chunks_over_raw_score():
    results = {
        "expensive_best": {"ndcg@10": 0.9, "chunk_count": 900},  # 0.001 per chunk
        "cheap_good": {"ndcg@10": 0.6, "chunk_count": 100},  # 0.006 per chunk
    }
    selection = select_variants(results)
    assert selection["efficient"] == "cheap_good"
    assert selection["winner"] == "expensive_best"  # confirms winner and efficient can genuinely differ


def test_winner_and_efficient_can_be_the_same_strategy():
    results = {
        "best_and_cheapest": {"ndcg@10": 0.9, "chunk_count": 50},
        "worse_and_pricier": {"ndcg@10": 0.5, "chunk_count": 200},
    }
    selection = select_variants(results)
    assert selection["winner"] == selection["efficient"] == "best_and_cheapest"
