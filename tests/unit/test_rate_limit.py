from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.app.rate_limit import _resolve_caller_ip, check_rate_limit, enforce_rate_limit

pytestmark = pytest.mark.unit


def _fake_client(counts: dict[str, int]):
    client = MagicMock()

    def incr(key):
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    client.incr.side_effect = incr
    return client


def test_under_limit_passes():
    counts: dict[str, int] = {}
    with patch("src.app.rate_limit.get_client", return_value=_fake_client(counts)):
        for _ in range(5):
            assert check_rate_limit("1.2.3.4", limit=10, window_seconds=60) is True


def test_over_limit_fails():
    counts: dict[str, int] = {}
    with patch("src.app.rate_limit.get_client", return_value=_fake_client(counts)):
        for _ in range(10):
            check_rate_limit("1.2.3.4", limit=10, window_seconds=60)
        assert check_rate_limit("1.2.3.4", limit=10, window_seconds=60) is False


def test_different_callers_have_independent_budgets():
    counts: dict[str, int] = {}
    with patch("src.app.rate_limit.get_client", return_value=_fake_client(counts)):
        for _ in range(10):
            check_rate_limit("1.2.3.4", limit=10, window_seconds=60)
        # a different caller's budget is untouched by 1.2.3.4 exhausting theirs
        assert check_rate_limit("5.6.7.8", limit=10, window_seconds=60) is True


def test_expire_is_only_set_on_the_first_request_in_a_window():
    client = MagicMock()
    client.incr.return_value = 1
    with patch("src.app.rate_limit.get_client", return_value=client):
        check_rate_limit("1.2.3.4", limit=10, window_seconds=60)
    assert client.expire.called

    client2 = MagicMock()
    client2.incr.return_value = 2
    with patch("src.app.rate_limit.get_client", return_value=client2):
        check_rate_limit("1.2.3.4", limit=10, window_seconds=60)
    assert not client2.expire.called


def test_enforce_raises_429_over_limit():
    request = MagicMock()
    request.client.host = "1.2.3.4"
    with patch("src.app.rate_limit.check_rate_limit", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            enforce_rate_limit(request)
    assert exc_info.value.status_code == 429


def test_enforce_passes_silently_under_limit():
    request = MagicMock()
    request.client.host = "1.2.3.4"
    with patch("src.app.rate_limit.check_rate_limit", return_value=True):
        enforce_rate_limit(request)  # must not raise


# ---------------------------------------------------------------------
# _resolve_caller_ip — plan D2, horizontal scaling: request.client.host
# alone becomes the LOAD BALANCER's IP behind a proxy, collapsing every
# real caller into one shared bucket. X-Forwarded-For must be honored,
# but only from a proxy this deployment actually trusts — blind trust
# lets any caller spoof a fresh identity every request and bypass the
# limit outright.
# ---------------------------------------------------------------------


def _fake_request(direct_ip: str, forwarded_for: str | None = None):
    request = MagicMock()
    request.client.host = direct_ip
    request.headers = {"x-forwarded-for": forwarded_for} if forwarded_for else {}
    return request


def test_untrusted_direct_peer_uses_its_own_ip_even_with_forwarded_header(monkeypatch):
    # An untrusted caller setting X-Forwarded-For itself must not be
    # able to claim a different identity and dodge the real limit.
    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)
    request = _fake_request("203.0.113.9", forwarded_for="1.1.1.1")
    assert _resolve_caller_ip(request) == "203.0.113.9"


def test_trusted_proxy_forwarded_header_is_honored(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1")
    request = _fake_request("10.0.0.1", forwarded_for="203.0.113.9")
    assert _resolve_caller_ip(request) == "203.0.113.9"


def test_trusted_proxy_with_multiple_hops_uses_the_last_value(monkeypatch):
    # Single-trusted-hop model: the last entry is the one the trusted
    # proxy itself appended, not an earlier, unverified client claim.
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1")
    request = _fake_request("10.0.0.1", forwarded_for="1.1.1.1, 203.0.113.9")
    assert _resolve_caller_ip(request) == "203.0.113.9"


def test_trusted_proxy_allowlist_supports_multiple_ips(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1, 10.0.0.2")
    request = _fake_request("10.0.0.2", forwarded_for="203.0.113.9")
    assert _resolve_caller_ip(request) == "203.0.113.9"


def test_no_forwarded_header_falls_back_to_direct_ip_even_when_trusted(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1")
    request = _fake_request("10.0.0.1")
    assert _resolve_caller_ip(request) == "10.0.0.1"
