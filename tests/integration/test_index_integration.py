import pytest

from src.index.client import create_index, get_client
from src.index.mapping import INDEX_NAME

pytestmark = pytest.mark.integration


def test_create_index_is_idempotent():
    client = get_client()
    # Whatever the current state, calling twice must not error and the
    # second call must report "already existed".
    create_index(client)
    second_call_created = create_index(client)
    assert second_call_created is False


def test_live_mapping_matches_intent():
    client = get_client()
    create_index(client)
    mapping = client.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]["properties"]

    assert mapping["embedding"]["type"] == "knn_vector"
    assert mapping["embedding"]["dimension"] == 1024
    assert mapping["embedding"]["method"]["name"] == "hnsw"
    assert mapping["text"]["analyzer"] == "english"
    assert mapping["authors"]["type"] == "keyword"
    assert mapping["authors"]["fields"]["text"]["type"] == "text"
