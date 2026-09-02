import pytest

from src.index.mapping import build_index_body, locked_embed_dimension

pytestmark = pytest.mark.unit


def test_locked_dimension_matches_committed_lock_file():
    # models.lock.yaml is a committed file, not a live service — this is
    # a unit test, not an integration one.
    assert locked_embed_dimension() == 1024


def test_build_index_body_uses_locked_dimension_by_default():
    body = build_index_body()
    assert body["mappings"]["properties"]["embedding"]["dimension"] == 1024


def test_build_index_body_dimension_override():
    body = build_index_body(dimension=256)
    assert body["mappings"]["properties"]["embedding"]["dimension"] == 256


def test_build_index_body_has_knn_enabled():
    body = build_index_body()
    assert body["settings"]["index"]["knn"] is True


def test_authors_field_has_analyzed_text_subfield():
    # The specific fix that made author filtering work at all — see A2's
    # hybrid.py addition and the mapping.py comment on why.
    body = build_index_body()
    authors_field = body["mappings"]["properties"]["authors"]
    assert authors_field["type"] == "keyword"
    assert authors_field["fields"]["text"]["type"] == "text"


def test_category_and_date_fields_are_filterable_types():
    body = build_index_body()
    props = body["mappings"]["properties"]
    assert props["category"]["type"] == "keyword"
    assert props["published_date"]["type"] == "date"
