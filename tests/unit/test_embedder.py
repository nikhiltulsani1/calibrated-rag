from unittest.mock import MagicMock, patch

import pytest

from src.index.embedder import EmbeddingResult, embed_passages, embed_queries, embed_queries_with_fallback, embed_query

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _default_jina_provider():
    # Every test here is about the HTTP-call shape for a given provider,
    # not about embed_toggle's own resolution logic (see
    # test_embed_toggle.py for that) — patched here so these tests don't
    # depend on a real Redis connection succeeding or failing.
    with patch("src.index.embedder.get_active_embed_provider", return_value="jina"):
        yield


def test_embed_query_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JINA_API_KEY"):
        embed_query("hello")


def test_embed_raises_on_unsupported_provider():
    # Provider selection moved from RAG_MODEL_EMBED to the embed_toggle
    # switch (EMBED_PROVIDER / Redis) on 2026-08-22 — see
    # src/index/embed_toggle.py. get_active_embed_provider() itself
    # already validates against {"jina", "mistral"} and falls back to
    # "jina" on anything else, so the only way to reach this branch is
    # an explicit provider= override (bypassing the toggle), which must
    # still fail loud rather than silently fall through to a live call.
    with pytest.raises(NotImplementedError, match="voyage"):
        embed_query("hello", provider="voyage")


def _fake_jina_response(vectors: list[list[float]]) -> MagicMock:
    fake = MagicMock()
    fake.raise_for_status = lambda: None
    fake.json = lambda: {
        "model": "jina-embeddings-v3",
        "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)],
        # Jina's real response key is "total_tokens", not the OpenAI-style
        # "prompt_tokens" this mock used to use — that mismatch is exactly
        # what let embedder.py's wrong assumption ship unnoticed until a
        # real API key existed to test against. Fixed here so this mock
        # actually matches reality, not the code's prior belief about it.
        "usage": {"total_tokens": 7},
    }
    return fake


def test_embed_query_uses_retrieval_query_task(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    with patch("httpx.post", return_value=_fake_jina_response([[0.1, 0.2]])) as mock_post:
        result = embed_query("what is RLHF")
    assert mock_post.call_args.kwargs["json"]["task"] == "retrieval.query"
    assert mock_post.call_args.kwargs["json"]["input"] == ["what is RLHF"]
    assert result.vectors == [[0.1, 0.2]]
    assert result.dimension == 2
    assert result.prompt_tokens == 7


def test_embed_queries_batches_many_query_texts_in_one_call(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    with patch("httpx.post", return_value=_fake_jina_response([[0.1, 0.2], [0.3, 0.4]])) as mock_post:
        result = embed_queries(["first query", "second query"])
    assert mock_post.call_count == 1  # one HTTP call for both texts, not two
    assert mock_post.call_args.kwargs["json"]["task"] == "retrieval.query"
    assert mock_post.call_args.kwargs["json"]["input"] == ["first query", "second query"]
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_passages_uses_retrieval_passage_task(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    with patch("httpx.post", return_value=_fake_jina_response([[0.1], [0.2]])) as mock_post:
        embed_passages(["chunk one", "chunk two"])
    assert mock_post.call_args.kwargs["json"]["task"] == "retrieval.passage"
    assert mock_post.call_args.kwargs["json"]["input"] == ["chunk one", "chunk two"]


def test_embed_passages_dimensions_param_passed_through(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    with patch("httpx.post", return_value=_fake_jina_response([[0.1]])) as mock_post:
        embed_passages(["x"], dimensions=512)
    assert mock_post.call_args.kwargs["json"]["dimensions"] == 512


def test_embed_response_vectors_reordered_by_index(monkeypatch):
    # The API doesn't guarantee response order matches input order — this
    # is defensive, verified by feeding a deliberately out-of-order
    # response and confirming vectors land back in input order.
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    fake = MagicMock()
    fake.raise_for_status = lambda: None
    fake.json = lambda: {
        "model": "jina-embeddings-v3",
        "data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ],
        "usage": {"total_tokens": 2},
    }
    with patch("httpx.post", return_value=fake):
        result = embed_passages(["first", "second"])
    assert result.vectors == [[1.0], [2.0]]


def test_jina_raises_loud_on_a_partial_batch_response(monkeypatch):
    # Real gap found in a full-codebase review, fixed 2026-08-25: a real
    # 200 response with fewer vectors than texts sent (a plausible
    # partial-batch failure) used to sail through unchecked despite
    # embed's own "fails LOUD" design — src/ingest/pipeline.py's
    # `zip(new_chunks, result.vectors)` would silently misalign every
    # chunk_id<->vector pair after the missing one, not error.
    monkeypatch.setenv("JINA_API_KEY", "dummy")
    with patch("httpx.post", return_value=_fake_jina_response([[0.1], [0.2]])):
        with pytest.raises(RuntimeError, match="2 vectors for 3 input texts"):
            embed_passages(["a", "b", "c"])


# ---------------------------------------------------------------------
# mistral provider — the real embed-provider switch added 2026-08-22
# after repeatedly hitting Jina's real account-balance limit mid-eval.
# ---------------------------------------------------------------------


def _fake_mistral_response(vectors: list[list[float]]) -> MagicMock:
    fake = MagicMock()
    fake.raise_for_status = lambda: None
    fake.json = lambda: {
        "model": "mistral-embed",
        "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)],
        # Real response shape, confirmed live 2026-08-22: has BOTH
        # prompt_tokens and total_tokens (unlike Jina, total_tokens only).
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    return fake


def test_active_provider_mistral_dispatches_to_mistral_endpoint(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    with patch("src.index.embedder.get_active_embed_provider", return_value="mistral"), patch(
        "httpx.post", return_value=_fake_mistral_response([[0.1, 0.2, 0.3]])
    ) as mock_post:
        result = embed_query("what is RLHF")
    assert mock_post.call_args.args[0] == "https://api.mistral.ai/v1/embeddings"
    assert mock_post.call_args.kwargs["json"]["model"] == "mistral-embed"
    assert mock_post.call_args.kwargs["json"]["input"] == ["what is RLHF"]
    # No task-asymmetry param for mistral, unlike jina.
    assert "task" not in mock_post.call_args.kwargs["json"]
    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert result.dimension == 3
    assert result.model == "mistral-embed"


def test_explicit_provider_override_bypasses_the_toggle():
    # embed_toggle defaults to jina in this fixture, but an explicit
    # provider= argument (used by scripts/build_mistral_embed_index.py,
    # which must always target mistral regardless of what's live) wins.
    with patch("src.index.embedder.get_active_embed_provider", return_value="jina") as mock_toggle, patch(
        "httpx.post", return_value=_fake_mistral_response([[0.1]])
    ):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "dummy"}):
            embed_query("hello", provider="mistral")
    # The toggle was never even consulted — provider= short-circuits it.
    assert not mock_toggle.called


def test_mistral_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with patch("src.index.embedder.get_active_embed_provider", return_value="mistral"):
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            embed_query("hello")


def test_mistral_rejects_dimensions_param_as_unverified(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    with patch("src.index.embedder.get_active_embed_provider", return_value="mistral"):
        with pytest.raises(NotImplementedError, match="dimensions"):
            embed_passages(["x"], dimensions=512)


def test_mistral_batches_large_inputs_to_avoid_the_real_token_limit(monkeypatch):
    # Real, live-discovered limit (2026-08-22): a 571-text batch got a
    # real 400 "Too many tokens overall, split into more batches" from
    # mistral-embed. This reproduces it with a mocked response and
    # checks the fix: more than _MISTRAL_BATCH_SIZE texts must become
    # multiple requests, not one, and results must reassemble in order.
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    from src.index.embedder import _MISTRAL_BATCH_SIZE

    texts = [f"text {i}" for i in range(_MISTRAL_BATCH_SIZE + 5)]

    def fake_post(url, json, headers, timeout):
        batch = json["input"]
        fake = MagicMock()
        fake.raise_for_status = lambda: None
        fake.json = lambda: {
            "model": "mistral-embed",
            "data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(batch)],
            "usage": {"total_tokens": len(batch)},
        }
        return fake

    with patch("src.index.embedder.get_active_embed_provider", return_value="mistral"), patch(
        "httpx.post", side_effect=fake_post
    ) as mock_post:
        result = embed_passages(texts)

    assert mock_post.call_count == 2  # (batch_size + 5) texts -> two requests
    assert len(result.vectors) == len(texts)
    assert result.prompt_tokens == len(texts)  # summed across both batches


def test_mistral_raises_loud_on_a_partial_batch_response(monkeypatch):
    # Same real gap as Jina's — checked per-batch so the mismatch is
    # blamed on the exact batch that produced it.
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    with patch("src.index.embedder.get_active_embed_provider", return_value="mistral"), patch(
        "httpx.post", return_value=_fake_mistral_response([[0.1], [0.2]])
    ):
        with pytest.raises(RuntimeError, match="2 vectors for 3 input"):
            embed_passages(["a", "b", "c"])


# ---------------------------------------------------------------------
# embed_queries_with_fallback — automatic Jina<->Mistral embed failover,
# added 2026-08-23 after a real gap bit: a full 61-question Groq run
# scored 0/61 because Jina's embed capability ran dry mid-run with no
# fallback, unlike rerank (which already recovers automatically).
# ---------------------------------------------------------------------


def test_fallback_uses_primary_provider_when_it_succeeds():
    # Default fixture sets active provider to "jina".
    fake = EmbeddingResult(model="jina-embeddings-v4", dimension=3, vectors=[[0.1, 0.2, 0.3]], prompt_tokens=5)
    with patch("src.index.embedder.embed_queries", return_value=fake) as mock_embed:
        result, index_name = embed_queries_with_fallback(["a query"])
    mock_embed.assert_called_once_with(["a query"], provider="jina")
    assert result is fake
    assert index_name == "rag_chunks"


def test_fallback_switches_to_mistral_and_matching_index_when_jina_fails():
    fake = EmbeddingResult(model="mistral-embed", dimension=3, vectors=[[0.4, 0.5, 0.6]], prompt_tokens=5)

    def fake_embed_queries(texts, provider=None):
        if provider == "jina":
            raise RuntimeError("JINA_API_KEY exhausted")
        return fake

    with patch("src.index.embedder.embed_queries", side_effect=fake_embed_queries):
        result, index_name = embed_queries_with_fallback(["a query"])
    assert result is fake
    # The real point of this whole feature: the returned index MATCHES
    # the provider that actually served the embedding, not the one
    # originally active.
    assert index_name == "rag_chunks_mistral_embed"


def test_fallback_from_mistral_falls_back_to_jina():
    fake = EmbeddingResult(model="jina-embeddings-v4", dimension=3, vectors=[[0.7, 0.8, 0.9]], prompt_tokens=5)

    def fake_embed_queries(texts, provider=None):
        if provider == "mistral":
            raise RuntimeError("mistral down")
        return fake

    with patch("src.index.embedder.get_active_embed_provider", return_value="mistral"), patch(
        "src.index.embedder.embed_queries", side_effect=fake_embed_queries
    ):
        result, index_name = embed_queries_with_fallback(["a query"])
    assert result is fake
    assert index_name == "rag_chunks"


def test_fallback_raises_naming_both_failures_when_both_providers_fail():
    def fake_embed_queries(texts, provider=None):
        raise RuntimeError(f"{provider} is down")

    with patch("src.index.embedder.embed_queries", side_effect=fake_embed_queries):
        with pytest.raises(RuntimeError) as exc_info:
            embed_queries_with_fallback(["a query"])
    # Fails LOUD (per A0), and names both real failures for debuggability.
    assert "jina is down" in str(exc_info.value)
    assert "mistral is down" in str(exc_info.value)
