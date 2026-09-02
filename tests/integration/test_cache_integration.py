import pytest

from src.platform.cache import get_client, get_json, set_json

pytestmark = pytest.mark.integration

_TEST_KEY = "test:cache_integration:round_trip"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    get_client().delete(_TEST_KEY)


def test_round_trip_against_real_redis():
    set_json(_TEST_KEY, {"hello": "world"}, ttl_seconds=30)
    assert get_json(_TEST_KEY) == {"hello": "world"}


def test_missing_key_returns_none():
    assert get_json("test:cache_integration:definitely_missing") is None


def test_ttl_is_actually_set():
    set_json(_TEST_KEY, {"x": 1}, ttl_seconds=30)
    ttl = get_client().ttl(_TEST_KEY)
    assert 0 < ttl <= 30
