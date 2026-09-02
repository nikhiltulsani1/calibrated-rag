import hashlib
from unittest.mock import patch

import pytest

import src.platform.cache as cache
from src.platform.cache import build_cache_key

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cached_client():
    # get_client() caches a module-global singleton — reset it around
    # every test in this file so one test's env vars can't leak into
    # another's assertions via the cached instance.
    cache._client = None
    yield
    cache._client = None


# get_client — Phase 2: Upstash (the free-tier Redis this project's live
# deployment uses) requires TLS + a password; local docker compose Redis
# has neither. Both must be env-gated and default OFF, so local dev stays
# exactly as before.


def test_get_client_defaults_to_no_password_and_no_tls(monkeypatch):
    # Real gap found live (Phase 2): this test only cleared
    # REDIS_PASSWORD/REDIS_SSL, not REDIS_HOST/REDIS_PORT — harmless
    # while .env had no real Redis host set, but broke the moment a
    # real Upstash host landed in .env for the live deployment. Tests
    # must not depend on whatever happens to be in the real .env.
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)
    with patch("src.platform.cache.redis.Redis") as mock_redis:
        cache.get_client()
    mock_redis.assert_called_once_with(host="localhost", port=6379, password=None, ssl=False, decode_responses=True)


def test_get_client_uses_password_and_tls_when_configured(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "usw1-example.upstash.io")
    monkeypatch.setenv("REDIS_PASSWORD", "real-token")
    monkeypatch.setenv("REDIS_SSL", "true")
    with patch("src.platform.cache.redis.Redis") as mock_redis:
        cache.get_client()
    mock_redis.assert_called_once_with(
        host="usw1-example.upstash.io", port=6379, password="real-token", ssl=True, decode_responses=True
    )


def test_get_client_treats_empty_password_as_none(monkeypatch):
    # An empty-string env var (e.g. a blank .env line) must not be passed
    # through as a literal empty-string password.
    monkeypatch.setenv("REDIS_PASSWORD", "")
    with patch("src.platform.cache.redis.Redis") as mock_redis:
        cache.get_client()
    assert mock_redis.call_args.kwargs["password"] is None

# build_cache_key — promoted 2026-08-25 (full-codebase review) out of two
# independent hand-rolled copies (query_planner.py's plan cache,
# reason/answer_cache.py's answer cache) doing the identical
# "join parts with |, sha256, prefix" pattern.


def test_single_part_matches_the_original_query_planner_scheme():
    # query_planner.py's old scheme: prefix + sha256(query).hexdigest()
    # — must produce byte-identical output so any already-cached plan
    # entries in Redis stay reachable after this refactor.
    expected = "query_plan:" + hashlib.sha256(b"what is rlhf").hexdigest()
    assert build_cache_key("query_plan:", "what is rlhf") == expected


def test_multi_part_matches_the_original_answer_cache_scheme():
    # answer_cache.py's old scheme: prefix + sha256(f"{q}|{strategy}|{provider}")
    # — same byte-identical requirement.
    raw = "what is rlhf|default|jina"
    expected = "answer_cache:" + hashlib.sha256(raw.encode()).hexdigest()
    assert build_cache_key("answer_cache:", "what is rlhf", "default", "jina") == expected


def test_different_parts_produce_different_keys():
    assert build_cache_key("p:", "a") != build_cache_key("p:", "b")
    assert build_cache_key("p:", "a", "x") != build_cache_key("p:", "a", "y")


def test_different_prefix_produces_a_different_key_for_the_same_parts():
    assert build_cache_key("p1:", "a") != build_cache_key("p2:", "a")
