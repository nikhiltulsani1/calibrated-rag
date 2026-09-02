from unittest.mock import MagicMock, patch

import pytest

from src.index.embed_toggle import active_embed_index_name, get_active_embed_provider, set_active_embed_provider

pytestmark = pytest.mark.unit


def test_defaults_to_jina_and_production_index(monkeypatch):
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)
    fake_client = MagicMock()
    fake_client.get.return_value = None
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        assert get_active_embed_provider() == "jina"
        assert active_embed_index_name() == "rag_chunks"


def test_env_var_sets_the_baseline_when_no_redis_override(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "mistral")
    fake_client = MagicMock()
    fake_client.get.return_value = None
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        assert get_active_embed_provider() == "mistral"
        assert active_embed_index_name() == "rag_chunks_mistral_embed"


def test_redis_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "jina")
    fake_client = MagicMock()
    fake_client.get.return_value = "mistral"
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        assert get_active_embed_provider() == "mistral"


def test_invalid_env_var_falls_back_to_jina(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "not_a_real_provider")
    fake_client = MagicMock()
    fake_client.get.return_value = None
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        assert get_active_embed_provider() == "jina"


def test_invalid_redis_value_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "mistral")
    fake_client = MagicMock()
    fake_client.get.return_value = "garbage"
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        assert get_active_embed_provider() == "mistral"


def test_redis_unavailable_fails_open_to_env_var(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "mistral")
    fake_client = MagicMock()
    fake_client.get.side_effect = RuntimeError("connection refused")
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        assert get_active_embed_provider() == "mistral"


def test_set_active_embed_provider_rejects_unknown_provider():
    with pytest.raises(ValueError):
        set_active_embed_provider("nonsense")


def test_set_active_embed_provider_writes_to_redis():
    fake_client = MagicMock()
    with patch("src.index.embed_toggle.get_client", return_value=fake_client):
        set_active_embed_provider("mistral")
    fake_client.set.assert_called_once_with("embed_provider:active", "mistral")
