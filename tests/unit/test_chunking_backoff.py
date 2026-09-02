from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.ingest.chunking_strategies import _with_rate_limit_backoff

pytestmark = pytest.mark.unit


def _fake_429_error() -> httpx.HTTPStatusError:
    response = MagicMock(status_code=429)
    return httpx.HTTPStatusError("429", request=MagicMock(), response=response)


def test_retries_on_429_and_eventually_succeeds():
    fn = MagicMock(side_effect=[_fake_429_error(), "ok"])
    with patch("src.ingest.chunking_strategies.time.sleep") as mock_sleep:
        result = _with_rate_limit_backoff(fn, ["text"])
    assert result == "ok"
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(65.0)


def test_gives_up_after_max_retries():
    fn = MagicMock(side_effect=_fake_429_error())
    with patch("src.ingest.chunking_strategies.time.sleep"):
        with pytest.raises(httpx.HTTPStatusError):
            _with_rate_limit_backoff(fn, ["text"])
    assert fn.call_count == 4  # 1 initial + 3 retries


def test_non_429_errors_are_not_retried():
    response = MagicMock(status_code=500)
    error = httpx.HTTPStatusError("500", request=MagicMock(), response=response)
    fn = MagicMock(side_effect=error)
    with patch("src.ingest.chunking_strategies.time.sleep") as mock_sleep:
        with pytest.raises(httpx.HTTPStatusError):
            _with_rate_limit_backoff(fn, ["text"])
    assert fn.call_count == 1
    assert not mock_sleep.called
