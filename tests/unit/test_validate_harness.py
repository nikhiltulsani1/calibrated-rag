import pytest

from evals.validate_harness import _benchmark_index_body

pytestmark = pytest.mark.unit


def test_index_body_uses_beir_tuned_bm25_parameters():
    # k1=0.9, b=0.4 — the Anserini/Lucene-tuned values the official BEIR
    # BM25 baseline was computed with, not OpenSearch's stock defaults
    # (k1=1.2, b=0.75). Verified by actually running the harness both
    # ways and finding this changed the measured number, not assumed.
    body = _benchmark_index_body()
    similarity = body["settings"]["index"]["similarity"]["beir_bm25"]
    assert similarity["k1"] == 0.9
    assert similarity["b"] == 0.4


def test_index_body_fields_use_the_custom_similarity():
    body = _benchmark_index_body()
    props = body["mappings"]["properties"]
    assert props["contents"]["similarity"] == "beir_bm25"
    assert props["title"]["similarity"] == "beir_bm25"
    assert props["text"]["similarity"] == "beir_bm25"


def test_index_body_has_contents_field_for_concatenated_search():
    # The fix that actually closed the gap to the published baseline —
    # searching title+text as two separately-weighted fields measured
    # 0.61 against a published 0.665; the concatenated single-field
    # representation (matching Anserini's standard BEIR document
    # representation) measured 0.679. See the validate_harness.py
    # comment on build_benchmark_index for the investigation.
    body = _benchmark_index_body()
    assert "contents" in body["mappings"]["properties"]
