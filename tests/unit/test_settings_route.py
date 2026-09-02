import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.app.session import CREDENTIALS_COOKIE_NAME
from src.platform.credentials import Credentials

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _cookie_encryption_key(monkeypatch):
    monkeypatch.setenv("COOKIE_ENCRYPTION_KEY", "a-real-test-secret-for-settings")


def test_get_settings_page_renders_with_no_keys_set():
    client = TestClient(app, base_url="https://testserver")
    response = client.get("/settings")
    assert response.status_code == 200
    assert b"Groq" in response.content
    assert b"not set" in response.content


def test_post_settings_saves_a_key_and_sets_the_cookie():
    client = TestClient(app, base_url="https://testserver", follow_redirects=False)
    response = client.post("/settings", data={"groq": "gsk_real_test_key"})
    assert response.status_code == 303
    assert response.headers["location"] == "/settings"
    assert CREDENTIALS_COOKIE_NAME in response.cookies
    # The real key must never appear in plaintext anywhere in the cookie.
    assert "gsk_real_test_key" not in response.cookies[CREDENTIALS_COOKIE_NAME]


def test_local_http_dev_still_round_trips_the_cookie_correctly():
    # Real bug found live (Phase 2): a hardcoded secure=True cookie is
    # genuinely dropped by real browsers over plain HTTP, confirmed
    # directly in a browser (document.cookie came back empty after
    # save) — local `docker compose` dev (plain HTTP) must still work,
    # just without the Secure flag, rather than BYOK silently never
    # persisting locally.
    client = TestClient(app, base_url="http://testserver", follow_redirects=False)
    response = client.post("/settings", data={"groq": "gsk_real_test_key"})
    assert CREDENTIALS_COOKIE_NAME in response.cookies
    assert "secure" not in response.headers["set-cookie"].lower()


def test_https_deployment_sets_the_secure_flag():
    client = TestClient(app, base_url="https://testserver", follow_redirects=False)
    response = client.post("/settings", data={"groq": "gsk_real_test_key"})
    assert "secure" in response.headers["set-cookie"].lower()


def test_saved_key_shows_masked_not_in_plaintext_on_the_next_page_load():
    client = TestClient(app, base_url="https://testserver")
    client.post("/settings", data={"groq": "gsk_real_test_key_123456"})
    response = client.get("/settings")
    assert b"gsk_real_test_key_123456" not in response.content
    assert b"\xe2\x80\xa2" in response.content  # the masking bullet character


def test_blank_field_keeps_the_existing_saved_key_not_clear_it():
    client = TestClient(app, base_url="https://testserver")
    client.post("/settings", data={"groq": "gsk_original_key"})
    # Second save only sets jina, leaves groq blank
    client.post("/settings", data={"jina": "jina_key"})

    from src.app.session import read_credentials

    cookie_value = client.cookies.get(CREDENTIALS_COOKIE_NAME)
    creds = read_credentials(cookie_value)
    assert creds.groq == "gsk_original_key"
    assert creds.jina == "jina_key"


def test_clear_removes_only_the_named_provider():
    client = TestClient(app, base_url="https://testserver")
    client.post("/settings", data={"groq": "gsk_key", "jina": "jina_key"})
    client.post("/settings/clear", data={"provider": "groq"})

    from src.app.session import read_credentials

    cookie_value = client.cookies.get(CREDENTIALS_COOKIE_NAME)
    creds = read_credentials(cookie_value)
    assert creds.groq is None
    assert creds.jina == "jina_key"  # untouched


def test_clear_rejects_an_unknown_provider_name_without_crashing():
    client = TestClient(app, base_url="https://testserver", follow_redirects=False)
    response = client.post("/settings/clear", data={"provider": "not_a_real_provider"})
    assert response.status_code == 303  # redirects harmlessly, no 500


def test_settings_unavailable_gracefully_when_no_encryption_key_configured(monkeypatch):
    monkeypatch.delenv("COOKIE_ENCRYPTION_KEY", raising=False)
    client = TestClient(app, base_url="https://testserver", follow_redirects=False)
    response = client.post("/settings", data={"groq": "gsk_real_key"})
    assert response.status_code == 303  # doesn't crash
    assert CREDENTIALS_COOKIE_NAME not in response.cookies  # but also doesn't pretend to have saved it
