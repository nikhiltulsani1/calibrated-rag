import pytest

from src.app.session import (
    encrypt_credentials,
    get_or_create_session_id,
    read_credentials,
    sign_session_id,
)
from src.platform.credentials import Credentials

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------
# Session id — signed (tamper-resistant), not secret
# ---------------------------------------------------------------------


def test_no_cookie_mints_a_new_session_id():
    session_id, is_new = get_or_create_session_id(None)
    assert session_id
    assert is_new is True


def test_valid_signed_cookie_is_verified_and_reused():
    session_id, _ = get_or_create_session_id(None)
    signed = sign_session_id(session_id)
    recovered, is_new = get_or_create_session_id(signed)
    assert recovered == session_id
    assert is_new is False


def test_tampered_cookie_mints_a_fresh_session_id_instead_of_trusting_it():
    signed = sign_session_id("real-session-id")
    tampered = signed[:-1] + ("A" if signed[-1] != "A" else "B")
    recovered, is_new = get_or_create_session_id(tampered)
    assert recovered != "real-session-id"
    assert is_new is True


# ---------------------------------------------------------------------
# Credentials cookie — encrypted, real secrets must not be recoverable
# by inspection even though the value is base64-ish text
# ---------------------------------------------------------------------


def test_encrypt_then_read_round_trips(monkeypatch):
    monkeypatch.setenv("COOKIE_ENCRYPTION_KEY", "a-real-test-secret")
    creds = Credentials(groq="gsk_real123", jina="jina_real456")
    encrypted = encrypt_credentials(creds)
    assert encrypted is not None
    # The real key must not be readable in the encrypted payload itself.
    assert "gsk_real123" not in encrypted
    assert "jina_real456" not in encrypted

    recovered = read_credentials(encrypted)
    assert recovered.groq == "gsk_real123"
    assert recovered.jina == "jina_real456"
    assert recovered.mistral is None


def test_no_encryption_key_configured_disables_byok_gracefully(monkeypatch):
    monkeypatch.delenv("COOKIE_ENCRYPTION_KEY", raising=False)
    creds = Credentials(groq="gsk_real123")
    assert encrypt_credentials(creds) is None
    # A cookie that somehow exists anyway (e.g. key was rotated away)
    # must not crash — just resolve to "no credentials."
    assert read_credentials("some-stale-value") == Credentials()


def test_missing_cookie_returns_all_none(monkeypatch):
    monkeypatch.setenv("COOKIE_ENCRYPTION_KEY", "a-real-test-secret")
    assert read_credentials(None) == Credentials()


def test_tampered_credentials_cookie_fails_closed_not_open(monkeypatch):
    # Real security property: a corrupted/forged credentials cookie must
    # decrypt to "no credentials," never to something that happens to
    # parse as valid — fail-closed, matching the input-guardrail
    # convention used elsewhere in this codebase for security-relevant
    # checks (as opposed to the fail-open convention for availability).
    monkeypatch.setenv("COOKIE_ENCRYPTION_KEY", "a-real-test-secret")
    encrypted = encrypt_credentials(Credentials(groq="gsk_real123"))
    tampered = encrypted[:-2] + "xx"
    assert read_credentials(tampered) == Credentials()


def test_key_rotation_makes_old_cookies_unreadable_not_crash(monkeypatch):
    monkeypatch.setenv("COOKIE_ENCRYPTION_KEY", "old-secret")
    encrypted = encrypt_credentials(Credentials(groq="gsk_real123"))
    monkeypatch.setenv("COOKIE_ENCRYPTION_KEY", "new-secret-after-rotation")
    assert read_credentials(encrypted) == Credentials()
