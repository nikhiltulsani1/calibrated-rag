from unittest.mock import MagicMock, patch

import pytest

from src.platform.credentials import Credentials, reset_credentials, set_credentials
from src.platform.models import (
    CompletionResult,
    ModelBinding,
    complete,
    get_model,
    get_model_ladder,
    provider_has_key,
)

pytestmark = pytest.mark.unit


def test_get_model_resolves_default_from_yaml():
    binding = get_model("embed")
    assert binding == ModelBinding(provider="jina", model_id="jina-embeddings-v3")


def test_get_model_env_override_wins(monkeypatch):
    monkeypatch.setenv("RAG_MODEL_EMBED", "voyage:voyage-3")
    binding = get_model("embed")
    assert binding == ModelBinding(provider="voyage", model_id="voyage-3")


def test_get_model_ladder_resolves_top_rung():
    binding = get_model("rewrite")
    assert binding.provider == "groq"


def test_get_model_unknown_role_raises_keyerror():
    with pytest.raises(KeyError):
        get_model("does_not_exist")


def test_get_model_rejects_malformed_spec(monkeypatch):
    monkeypatch.setenv("RAG_MODEL_EMBED", "not-a-valid-spec")
    with pytest.raises(ValueError):
        get_model("embed")


def test_complete_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        complete("rewrite", [{"role": "user", "content": "hi"}])


def test_complete_raises_on_unsupported_provider(monkeypatch):
    # "mistral" used to be this test's example of an unsupported provider
    # — no longer true as of 2026-08-20 (real provider, added for a 3-way
    # model comparison run). Switched to a genuinely fictional provider
    # name so this test still tests what it says it tests.
    monkeypatch.setenv("RAG_MODEL_REWRITE", "notaprovider:some-model")
    with pytest.raises(NotImplementedError, match="notaprovider"):
        complete("rewrite", [{"role": "user", "content": "hi"}])


def test_complete_sends_expected_request_and_parses_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")

    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {
        "model": "openai/gpt-oss-20b",
        "choices": [{"message": {"content": "hello back"}}],
    }

    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = complete("rewrite", [{"role": "user", "content": "hi"}], json_mode=True)

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "openai/gpt-oss-20b"
    assert call_kwargs["json"]["response_format"] == {"type": "json_object"}
    assert call_kwargs["headers"]["Authorization"] == "Bearer dummy-key"

    assert result == CompletionResult(
        provider="groq", model_served="openai/gpt-oss-20b", content="hello back"
    )


def test_get_model_ladder_returns_every_rung():
    ladder = get_model_ladder("rewrite")
    assert len(ladder) == 2
    assert ladder[0].provider == "groq"
    assert ladder[1] == ModelBinding(provider="openrouter", model_id="google/gemma-4-26b-a4b-it:free")


def test_get_model_ladder_single_spec_role_returns_one_item():
    ladder = get_model_ladder("embed")
    assert ladder == [ModelBinding(provider="jina", model_id="jina-embeddings-v3")]


def test_get_model_ladder_env_override_replaces_whole_ladder(monkeypatch):
    monkeypatch.setenv("RAG_MODEL_REWRITE", "nvidia:some-model")
    assert get_model_ladder("rewrite") == [ModelBinding(provider="nvidia", model_id="some-model")]


def test_provider_has_key_true_when_env_var_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    assert provider_has_key("groq") is True


def test_provider_has_key_false_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert provider_has_key("openrouter") is False


def test_provider_has_key_false_for_unknown_provider():
    assert provider_has_key("not_a_real_provider") is False


# ---------------------------------------------------------------------
# Phase 2 (BYOK) — a visitor's own key must win over the server's
# os.environ, and provider_has_key must see it too (the real gap the
# plan flagged: without this, usable_ladder() silently filters out a
# rung the visitor DOES have a key for).
# ---------------------------------------------------------------------


def test_complete_uses_the_visitors_own_key_over_server_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "server-key")
    token = set_credentials(Credentials(groq="visitor-key"))
    try:
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"model": "m", "choices": [{"message": {"content": "hi"}}]}
        with patch("httpx.post", return_value=fake_response) as mock_post:
            complete("rewrite", [{"role": "user", "content": "hi"}])
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer visitor-key"
    finally:
        reset_credentials(token)


def test_complete_falls_back_to_server_env_when_visitor_has_no_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "server-key")
    token = set_credentials(Credentials())  # visitor configured nothing
    try:
        fake_response = MagicMock()
        fake_response.raise_for_status = lambda: None
        fake_response.json = lambda: {"model": "m", "choices": [{"message": {"content": "hi"}}]}
        with patch("httpx.post", return_value=fake_response) as mock_post:
            complete("rewrite", [{"role": "user", "content": "hi"}])
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer server-key"
    finally:
        reset_credentials(token)


def test_provider_has_key_true_from_visitor_credentials_even_with_no_server_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    token = set_credentials(Credentials(openrouter="visitor-key"))
    try:
        assert provider_has_key("openrouter") is True
    finally:
        reset_credentials(token)


def test_complete_model_override_bypasses_role_resolution(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.delenv("RAG_MODEL_REWRITE", raising=False)

    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "choices": [{"message": {"content": "fallback response"}}],
    }
    override = ModelBinding(provider="openrouter", model_id="meta-llama/llama-3.1-8b-instruct:free")

    with patch("httpx.post", return_value=fake_response) as mock_post:
        result = complete("rewrite", [{"role": "user", "content": "hi"}], model_override=override)

    # role=rewrite normally resolves to groq — model_override must win
    assert mock_post.call_args.kwargs["json"]["model"] == "meta-llama/llama-3.1-8b-instruct:free"
    assert result.provider == "openrouter"
