from unittest.mock import patch

import pytest

from evals.scifact_loader import BenchmarkDoc
from evals.validate_harness import BENCHMARK_INDEX, _bm25_search, build_benchmark_index
from src.index.client import get_client

pytestmark = pytest.mark.integration

# Deliberately NOT the full 5,183-doc / 300-query validation — that's a
# one-time/periodic gate (`python -m evals.validate_harness`), documented
# separately, not something that should run on every `pytest -m
# integration`. This proves the OpenSearch plumbing (index creation,
# bulk indexing, BM25 search) is wired correctly, fast.
_FAKE_DOCS = [
    BenchmarkDoc(doc_id="d1", title="Water boils at 100C", text="At sea level, water boils at 100 degrees Celsius."),
    BenchmarkDoc(doc_id="d2", title="Photosynthesis", text="Plants convert sunlight into chemical energy."),
    BenchmarkDoc(doc_id="d3", title="Unrelated", text="A completely unrelated sentence about cars."),
]


@pytest.fixture
def tiny_benchmark_index():
    client = get_client()
    with patch("evals.validate_harness.load_corpus", return_value=_FAKE_DOCS):
        count = build_benchmark_index(client)
    yield client, count
    client.indices.delete(index=BENCHMARK_INDEX, ignore=[404])


def test_build_benchmark_index_indexes_expected_count(tiny_benchmark_index):
    _, count = tiny_benchmark_index
    assert count == 3


def test_bm25_search_finds_relevant_doc_via_contents_field(tiny_benchmark_index):
    client, _ = tiny_benchmark_index
    results = _bm25_search(client, "boiling point of water", size=3)
    assert results[0] == "d1"


def test_bm25_search_respects_size_limit(tiny_benchmark_index):
    client, _ = tiny_benchmark_index
    results = _bm25_search(client, "energy sunlight water cars", size=1)
    assert len(results) == 1
