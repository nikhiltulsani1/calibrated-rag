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


# ---------------------------------------------------------------------
# Phase 2 (stage 6, uploads) — real cross-visitor leak found while wiring
# up private uploads: this cache is one shared Redis keyspace with no
# privacy dimension before session_id was added to the key. A
# session-scoped answer (one that may cite a visitor's own private
# upload) cached under the plain query text would be served verbatim to
# a different visitor asking the identical question — exactly the class
# of bug the Phase 2 plan's multi-visitor isolation test exists to catch.
# ---------------------------------------------------------------------


def test_cache_key_varies_by_session_id():
    key_no_session = _cache_key("what is RLHF")
    key_session_a = _cache_key("what is RLHF", session_id="visitor-a")
    key_session_b = _cache_key("what is RLHF", session_id="visitor-b")
    assert len({key_no_session, key_session_a, key_session_b}) == 3


def test_set_cached_trace_still_skips_on_a_real_mismatch_on_the_opensearch_path(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    trace = _FakeTrace(
        original_query="what is RLHF", answer_text="RLHF is...", fusion=_FakeFusion(dense_index_name="rag_chunks_mistral_embed")
    )
    with patch("src.reason.answer_cache.set_json") as mock_set:
        set_cached_trace("what is RLHF", trace)
    mock_set.assert_not_called()


def test_set_cached_trace_caches_on_the_postgres_backend_despite_a_non_opensearch_index_name(monkeypatch):
    # Real bug found in review (Phase 2, stage 6): this guard was written
    # for hybrid.py's OpenSearch-only embed-provider-fallback case (see
    # its own docstring) and compares against provider_to_index(), which
    # only ever returns an OpenSearch index name. hybrid_postgres.py
    # always sets dense_index_name="postgres:chunks", which can never
    # equal an OpenSearch index name — so, ungated, this `return` fired
    # on EVERY postgres-backend query, silently disabling the answer
    # cache entirely on the one backend BYOK/uploads runs on. This test
    # fails on the pre-fix code (mock_set is never called) and passes
    # once the guard is scoped to the OpenSearch path only.
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    trace = _FakeTrace(
        original_query="what is RLHF", answer_text="RLHF is...", fusion=_FakeFusion(dense_index_name="postgres:chunks")
    )
    with patch("src.reason.answer_cache.set_json") as mock_set:
        set_cached_trace("what is RLHF", trace)
    mock_set.assert_called_once()


def test_a_session_scoped_cache_write_is_not_readable_without_that_session_id():
    store: dict[str, tuple] = {}

    def fake_get_json(key):
        return store.get(key)

    def fake_set_json(key, value, ttl_seconds):
        store[key] = value

    trace = _FakeTrace(original_query="what is RLHF", answer_text="visitor A's private answer")
    with patch("src.reason.answer_cache.get_json", side_effect=fake_get_json), patch(
        "src.reason.answer_cache.set_json", side_effect=fake_set_json
    ):
        set_cached_trace("what is RLHF", trace, session_id="visitor-a")

        # Visitor B, asking the exact same question text, must miss —
        # never see visitor A's cached (possibly private) answer.
        assert get_cached_trace("what is RLHF", session_id="visitor-b") is None
        assert get_cached_trace("what is RLHF") is None

        # Visitor A themself still gets their own cache hit.
        hit = get_cached_trace("what is RLHF", session_id="visitor-a")
    assert hit == {"original_query": "what is RLHF", "answer_text": "visitor A's private answer", "fusion": None}
