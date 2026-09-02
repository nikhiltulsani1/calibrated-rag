import pytest

from src.index.client import get_client
from src.index.mapping import INDEX_NAME
from src.reason.pipeline import fetch_metadata

pytestmark = pytest.mark.integration

_TEST_ID = "test_pipeline_meta_doc"


@pytest.fixture
def synthetic_doc():
    client = get_client()
    body = {
        "chunk_id": _TEST_ID,
        "paper_id": "9999.9999",
        "title": "A Test Paper About Transformers",
        "text": "some chunk text",
        "section": "methodology",
        "authors": ["Test Author"],
        "category": ["cs.IR"],
    }
    client.index(index=INDEX_NAME, id=_TEST_ID, body=body, refresh=True)
    yield
    client.delete(index=INDEX_NAME, id=_TEST_ID, ignore=[404])


def test_fetch_metadata_against_real_opensearch(synthetic_doc):
    result = fetch_metadata([_TEST_ID])
    assert result[_TEST_ID]["title"] == "A Test Paper About Transformers"
    assert result[_TEST_ID]["paper_id"] == "9999.9999"
    assert result[_TEST_ID]["section"] == "methodology"


def test_fetch_metadata_skips_not_found_ids(synthetic_doc):
    result = fetch_metadata([_TEST_ID, "does_not_exist_at_all"])
    assert set(result.keys()) == {_TEST_ID}
