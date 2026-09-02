import pytest

from src.app.rate_limit import check_rate_limit
from src.platform.cache import get_client

pytestmark = pytest.mark.integration

_TEST_CALLER = "test:rate_limit_integration:caller"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for key in get_client().keys(f"ratelimit:{_TEST_CALLER}:*"):
        get_client().delete(key)


def test_real_redis_enforces_the_limit_within_a_window():
    for _ in range(3):
        assert check_rate_limit(_TEST_CALLER, limit=3, window_seconds=60) is True
    assert check_rate_limit(_TEST_CALLER, limit=3, window_seconds=60) is False


def test_real_redis_sets_a_real_ttl_on_the_bucket_key():
    check_rate_limit(_TEST_CALLER, limit=10, window_seconds=60)
    matching_keys = get_client().keys(f"ratelimit:{_TEST_CALLER}:*")
    assert len(matching_keys) == 1
    ttl = get_client().ttl(matching_keys[0])
    assert 0 < ttl <= 60
