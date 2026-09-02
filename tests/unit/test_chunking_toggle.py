from unittest.mock import MagicMock, patch

import pytest

from src.reason.chunking_toggle import active_index_name, get_active_strategy, set_active_strategy

pytestmark = pytest.mark.unit


def test_defaults_to_default_strategy_and_production_index(monkeypatch):
    monkeypatch.delenv("CHUNKING_STRATEGY", raising=False)
    fake_client = MagicMock()
    fake_client.get.return_value = None
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        assert get_active_strategy() == "default"
        assert active_index_name() == "rag_chunks"


def test_env_var_sets_the_baseline_when_no_redis_override(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "winner")
    fake_client = MagicMock()
    fake_client.get.return_value = None
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        assert get_active_strategy() == "winner"
        assert active_index_name() == "rag_chunks_winner"


def test_redis_override_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "winner")
    fake_client = MagicMock()
    fake_client.get.return_value = "efficient"
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        assert get_active_strategy() == "efficient"


def test_invalid_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "not_a_real_strategy")
    fake_client = MagicMock()
    fake_client.get.return_value = None
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        assert get_active_strategy() == "default"


def test_invalid_redis_value_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "median")
    fake_client = MagicMock()
    fake_client.get.return_value = "garbage"
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        assert get_active_strategy() == "median"


def test_redis_unavailable_fails_open_to_env_var(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "median")
    fake_client = MagicMock()
    fake_client.get.side_effect = RuntimeError("connection refused")
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        assert get_active_strategy() == "median"


def test_set_active_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        set_active_strategy("nonsense")


def test_set_active_strategy_writes_to_redis():
    fake_client = MagicMock()
    with patch("src.reason.chunking_toggle.get_client", return_value=fake_client):
        set_active_strategy("winner")
    fake_client.set.assert_called_once_with("chunking_strategy:active", "winner")
