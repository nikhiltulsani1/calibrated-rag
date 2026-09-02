import pytest

from evals.run_generation_eval import load_gold

pytestmark = pytest.mark.unit


def test_load_gold_has_real_entries_with_references():
    gold = load_gold()
    assert len(gold) > 0
    for row in gold:
        assert row["query_id"]
        assert row["query"]
        assert row["reference"]  # ground-truth answer, not just a relevant chunk id
        assert row["relevant_chunk_id"]
        assert row["paper_id"]


def test_load_gold_query_ids_are_a_subset_of_the_retrieval_qrels():
    import json
    from pathlib import Path

    qrels_ids = {
        json.loads(line)["query_id"]
        for line in open(Path(__file__).resolve().parents[2] / "evals" / "datasets" / "qrels.jsonl", encoding="utf-8")
    }
    gold_ids = {row["query_id"] for row in load_gold()}
    assert gold_ids.issubset(qrels_ids)
