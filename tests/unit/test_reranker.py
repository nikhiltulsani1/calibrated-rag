from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.platform.credentials import Credentials, reset_credentials, set_credentials
from src.retrieve.reranker import Candidate, RerankResult, rerank

pytestmark = pytest.mark.unit

# COHERE_API_KEY is deleted by default for every unit test (see
# tests/unit/conftest.py) — the fallback-specific tests below re-set it
# explicitly via monkeypatch.setenv.


def _candidates(n: int) -> list[Candidate]:
    return [Candidate(id=f"c{i}", text=f"text {i}") for i in range(n)]


def test_empty_candidates_short_circuits():
    result = rerank("q", [])
    assert result == RerankResult(items=[], degraded=False, reason=None, model_served=None)


def test_missing_api_key_degrades_not_raises(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    result = rerank("q", _candidates(5), top_n=3)
    assert result.degraded is True
    assert result.reason == "JINA_API_KEY is not set"
    assert [item.id for item in result.items] == ["c0", "c1", "c2"]
    assert all(item.score is None for item in result.items)


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    monkeypatch.setenv("RAG_MODEL_RERANK", "voyage:rerank-2")
    with pytest.raises(NotImplementedError, match="voyage"):
        rerank("q", _candidates(2))


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown RERANKER_BACKEND"):
        rerank("q", _candidates(2), backend="quantum")


def test_local_backend_without_sentence_transformers_raises_clear_error(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "local")
    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        rerank("q", _candidates(1), backend="local")


def test_hosted_timeout_degrades(monkeypatch):
    # Also implicitly verifies no automatic Cohere fallback fires when
    # COHERE_API_KEY isn't configured (the autouse fixture deletes it) —
    # today's exact behavior, unchanged.
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        result = rerank("q", _candidates(5), top_n=3, backend="hosted")
    assert result.degraded is True
    assert "TimeoutException" in result.reason
    assert [item.id for item in result.items] == ["c0", "c1", "c2"]


# ---------------------------------------------------------------------
# Phase 2 (BYOK) — a visitor's own key must win over the server's
# os.environ, for both the primary hosted (Jina) path and the Cohere
# fallback path.
# ---------------------------------------------------------------------


def test_hosted_uses_the_visitors_own_jina_key_over_server_env(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "server-key")
    token = set_credentials(Credentials(jina="visitor-key"))
    try:
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.9}]}
        with patch("httpx.post", return_value=fake_response) as mock_post:
            rerank("q", _candidates(1), backend="hosted")
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer visitor-key"
    finally:
        reset_credentials(token)


def test_cohere_uses_the_visitors_own_key_over_server_env(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    token = set_credentials(Credentials(cohere="visitor-key"))
    try:
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.9}]}
        with patch("httpx.post", return_value=fake_response) as mock_post:
            result = rerank("q", _candidates(1), backend="cohere")
        assert result.degraded is False
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer visitor-key"
    finally:
        reset_credentials(token)


def test_visitor_only_cohere_key_still_engages_the_hosted_fallback(monkeypatch):
    # The rerank() dispatcher's own fallback-gate check (not just
    # _rerank_cohere's key lookup) must also see a visitor-only Cohere
    # key — real gap this guards: only fixing _rerank_cohere would leave
    # the fallback silently never engaging for a visitor with no server-
    # side COHERE_API_KEY configured at all.
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    token = set_credentials(Credentials(cohere="visitor-key"))
    try:
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.9}]}
        with patch("httpx.post", return_value=fake_response) as mock_post:
            result = rerank("q", _candidates(1), backend="hosted")
        assert result.degraded is False
        assert result.model_served == "cohere:rerank-v3.5"
    finally:
        reset_credentials(token)


# ---------------------------------------------------------------------
# automatic jina->cohere fallback — added 2026-08-22 after Jina's
# account balance ran out repeatedly this session. Only engages on
# backend="hosted" (the default) when Jina degrades AND a Cohere key is
# configured; backend="local"/"cohere" stay direct, no-fallback choices.
# ---------------------------------------------------------------------


def test_falls_back_to_cohere_when_jina_fails_and_cohere_key_present(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    candidates = [
        Candidate(id="c0", text="irrelevant"),
        Candidate(id="c1", text="most relevant"),
    ]

    def fake_post(url, **kwargs):
        if "jina.ai" in url:
            raise httpx.TimeoutException("jina timed out")
        fake = MagicMock()
        fake.raise_for_status = lambda: None
        fake.json = lambda: {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
        return fake

    with patch("httpx.post", side_effect=fake_post):
        result = rerank("q", candidates, top_n=2, backend="hosted")

    assert result.degraded is False
    assert result.model_served == "cohere:rerank-v3.5"
    assert [item.id for item in result.items] == ["c1", "c0"]


def test_no_fallback_attempted_when_jina_succeeds(monkeypatch):
    # The fallback must only fire on a real Jina failure, never as a
    # "double-check" on a perfectly good Jina result.
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    candidates = [Candidate(id="c0", text="a")]
    fake_jina_response = MagicMock()
    fake_jina_response.raise_for_status = lambda: None
    fake_jina_response.json = lambda: {"model": "jina-reranker-v3.5", "results": [{"index": 0, "relevance_score": 0.5}]}
    with patch("httpx.post", return_value=fake_jina_response) as mock_post:
        result = rerank("q", candidates, top_n=1, backend="hosted")
    assert result.model_served == "jina-reranker-v3.5"
    assert mock_post.call_count == 1  # only the Jina call, no fallback attempt


def test_no_fallback_when_cohere_key_absent(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    # COHERE_API_KEY deliberately absent (autouse fixture default).
    with patch("httpx.post", side_effect=httpx.TimeoutException("jina timed out")) as mock_post:
        result = rerank("q", _candidates(3), top_n=2, backend="hosted")
    assert result.degraded is True
    assert mock_post.call_count == 1  # never attempted the Cohere call at all


def test_degrades_when_both_jina_and_cohere_fail(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    with patch("httpx.post", side_effect=httpx.TimeoutException("both down")):
        result = rerank("q", _candidates(3), top_n=2, backend="hosted")
    assert result.degraded is True
    assert [item.id for item in result.items] == ["c0", "c1"]


def test_hosted_success_maps_out_of_order_results_correctly(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    candidates = [
        Candidate(id="c0", text="irrelevant"),
        Candidate(id="c1", text="most relevant"),
        Candidate(id="c2", text="somewhat relevant"),
    ]
    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {
        "model": "jina-reranker-v3.5",
        "results": [
            {"index": 1, "relevance_score": 0.93},
            {"index": 2, "relevance_score": 0.55},
            {"index": 0, "relevance_score": 0.10},
        ],
    }
    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = rerank("a query", candidates, top_n=3, backend="hosted")

    assert mock_post.call_args.kwargs["json"]["model"] == "jina-reranker-v3.5"
    assert result.degraded is False
    assert [item.id for item in result.items] == ["c1", "c2", "c0"]
    assert result.items[0].score == 0.93


# ---------------------------------------------------------------------
# cohere backend — added 2026-08-22 as a real, verified hosted
# alternative to Jina (comparable latency, unlike the local backend
# which was measured and ruled out as too slow on modest hardware).
# ---------------------------------------------------------------------


def test_cohere_missing_api_key_degrades_not_raises(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    result = rerank("q", _candidates(5), top_n=3, backend="cohere")
    assert result.degraded is True
    assert result.reason == "COHERE_API_KEY is not set"
    assert [item.id for item in result.items] == ["c0", "c1", "c2"]


def test_cohere_timeout_degrades(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        result = rerank("q", _candidates(5), top_n=3, backend="cohere")
    assert result.degraded is True
    assert "TimeoutException" in result.reason
    assert [item.id for item in result.items] == ["c0", "c1", "c2"]


def test_cohere_success_maps_out_of_order_results_correctly(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    candidates = [
        Candidate(id="c0", text="irrelevant"),
        Candidate(id="c1", text="most relevant"),
        Candidate(id="c2", text="somewhat relevant"),
    ]
    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {
        "results": [
            {"index": 1, "relevance_score": 0.87},
            {"index": 2, "relevance_score": 0.39},
            {"index": 0, "relevance_score": 0.01},
        ]
    }
    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = rerank("a query", candidates, top_n=3, backend="cohere")

    assert mock_post.call_args.kwargs["json"]["model"] == "rerank-v3.5"
    assert mock_post.call_args.kwargs["json"]["documents"] == ["irrelevant", "most relevant", "somewhat relevant"]
    assert result.degraded is False
    assert [item.id for item in result.items] == ["c1", "c2", "c0"]
    assert result.items[0].score == 0.87
    assert result.model_served == "cohere:rerank-v3.5"


def test_cohere_model_override_via_env_var(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    monkeypatch.setenv("RERANKER_COHERE_MODEL", "rerank-english-v3.0")
    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {"results": [{"index": 0, "relevance_score": 0.5}]}
    with patch("httpx.post", return_value=fake_response) as mock_post:
        rerank("q", _candidates(1), top_n=1, backend="cohere")
    assert mock_post.call_args.kwargs["json"]["model"] == "rerank-english-v3.0"
