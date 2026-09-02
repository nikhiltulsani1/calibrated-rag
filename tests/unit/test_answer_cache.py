from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.reason.answer_cache import _cache_key, get_cached_trace, set_cached_trace

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _FakeTrace:
    # A minimal stand-in for StageTrace — set_cached_trace only needs
    # something serialize_trace() can walk, and this exercises the same
    # dataclass-recursion path without pulling in the real StageTrace's
    # full dependency graph. `fusion` mirrors real StageTrace's own
    # `fusion: FusionTrace | None = None` field — set_cached_trace reads
    # it to detect a real embed-provider mismatch (see its own docstring),
    # so a fake with no such attribute at all would silently pass tests a
    # real StageTrace could never trigger.
    original_query: str
    answer_text: str
    fusion: object | None = None


@pytest.fixture(autouse=True)
def _toggles():
    with patch("src.reason.answer_cache.get_active_strategy", return_value="default"), patch(
        "src.reason.answer_cache.get_active_embed_provider", return_value="jina"
    ):
        yield


def test_cache_key_varies_by_query_text():
    key_a = _cache_key("what is RLHF")
    key_b = _cache_key("what is PPO")
    assert key_a != key_b


def test_cache_key_stable_for_same_query_modulo_case_and_whitespace():
    assert _cache_key("What is RLHF") == _cache_key("  what is rlhf  ")


def test_cache_key_varies_by_chunking_strategy():
    with patch("src.reason.answer_cache.get_active_strategy", return_value="default"):
        key_default = _cache_key("what is RLHF")
    with patch("src.reason.answer_cache.get_active_strategy", return_value="winner"):
        key_winner = _cache_key("what is RLHF")
    assert key_default != key_winner


def test_cache_key_varies_by_embed_provider():
    with patch("src.reason.answer_cache.get_active_embed_provider", return_value="jina"):
        key_jina = _cache_key("what is RLHF")
    with patch("src.reason.answer_cache.get_active_embed_provider", return_value="mistral"):
        key_mistral = _cache_key("what is RLHF")
    assert key_jina != key_mistral


def test_get_cached_trace_miss_returns_none():
    with patch("src.reason.answer_cache.get_json", return_value=None) as mock_get:
        result = get_cached_trace("what is RLHF")
    assert result is None
    mock_get.assert_called_once()


def test_get_cached_trace_hit_returns_the_cached_dict():
    fake_cached = {"original_query": "what is RLHF", "answer": {"text": "..."}}
    with patch("src.reason.answer_cache.get_json", return_value=fake_cached):
        result = get_cached_trace("what is RLHF")
    assert result == fake_cached


def test_set_cached_trace_serializes_and_writes_with_the_configured_ttl():
    trace = _FakeTrace(original_query="what is RLHF", answer_text="RLHF is...")
    with patch("src.reason.answer_cache.set_json") as mock_set, patch(
        "src.reason.answer_cache._ANSWER_CACHE_TTL_SECONDS", 3600
    ):
        set_cached_trace("what is RLHF", trace)
    mock_set.assert_called_once()
    key_arg, value_arg, ttl_arg = mock_set.call_args.args
    assert key_arg == _cache_key("what is RLHF")
    assert value_arg == {"original_query": "what is RLHF", "answer_text": "RLHF is...", "fusion": None}
    assert ttl_arg == 3600


def test_miss_then_hit_round_trip_through_a_fake_redis():
    store: dict[str, tuple] = {}

    def fake_get_json(key):
        return store.get(key)

    def fake_set_json(key, value, ttl_seconds):
        store[key] = value

    trace = _FakeTrace(original_query="what is RLHF", answer_text="RLHF is...")
    with patch("src.reason.answer_cache.get_json", side_effect=fake_get_json), patch(
        "src.reason.answer_cache.set_json", side_effect=fake_set_json
    ):
        assert get_cached_trace("what is RLHF") is None  # miss
        set_cached_trace("what is RLHF", trace)
        hit = get_cached_trace("what is RLHF")  # hit
    assert hit == {"original_query": "what is RLHF", "answer_text": "RLHF is...", "fusion": None}


# ---------------------------------------------------------------------
# set_cached_trace's real embed-provider-mismatch guard — real bug found
# live 2026-08-24: _cache_key reads the NOMINAL get_active_embed_provider()
# toggle, but embed_queries_with_fallback can silently serve a request
# off the OTHER provider (its own designed failover) without the toggle
# ever flipping. Caching that trace under the nominal key would poison
# it with cross-provider-vector-space content. trace.fusion.dense_index_name
# (real, set by retrieve_with_trace) is the ground truth of which index
# actually got queried — compared here against what the nominal provider
# should have produced.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeFusion:
    dense_index_name: str


def test_set_cached_trace_skips_caching_on_a_real_provider_mismatch():
    # Nominal toggle says "jina" (INDEX_NAME="rag_chunks" expected), but
    # the trace's real fusion data shows the dense arm actually queried
    # the mistral index — a real fallback engaged silently.
    trace = _FakeTrace(
        original_query="what is RLHF", answer_text="RLHF is...", fusion=_FakeFusion(dense_index_name="rag_chunks_mistral_embed")
    )
    with patch("src.reason.answer_cache.set_json") as mock_set:
        set_cached_trace("what is RLHF", trace)
    mock_set.assert_not_called()


def test_set_cached_trace_caches_normally_when_provider_matches():
    trace = _FakeTrace(
        original_query="what is RLHF", answer_text="RLHF is...", fusion=_FakeFusion(dense_index_name="rag_chunks")
    )
    with patch("src.reason.answer_cache.set_json") as mock_set:
        set_cached_trace("what is RLHF", trace)
    mock_set.assert_called_once()
