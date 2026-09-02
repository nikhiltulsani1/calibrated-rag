import pytest

from src.platform.credentials import Credentials, get_credentials, reset_credentials, set_credentials

pytestmark = pytest.mark.unit

# Phase 2 (BYOK) — the contextvar is the one seam every provider-calling
# function reads through instead of os.environ, specifically to avoid a
# real cross-request race under concurrent visitors (see this module's
# own docstring). These tests cover the isolation/reset discipline
# directly, mirroring RequestIdMiddleware's own _REQUEST_ID pattern.


def test_default_is_all_none():
    creds = get_credentials()
    assert creds == Credentials()
    assert creds.groq is None
    assert creds.jina is None


def test_set_then_get_returns_the_set_value():
    token = set_credentials(Credentials(groq="real-key"))
    try:
        assert get_credentials().groq == "real-key"
    finally:
        reset_credentials(token)


def test_reset_restores_the_previous_value():
    token = set_credentials(Credentials(groq="real-key"))
    reset_credentials(token)
    assert get_credentials() == Credentials()


def test_nested_set_then_reset_restores_the_outer_value():
    # Real regression this guards: a nested set/reset (e.g. a test
    # helper, or a future retry path) must not permanently clobber an
    # outer scope's credentials.
    outer_token = set_credentials(Credentials(groq="outer"))
    try:
        inner_token = set_credentials(Credentials(groq="inner"))
        assert get_credentials().groq == "inner"
        reset_credentials(inner_token)
        assert get_credentials().groq == "outer"
    finally:
        reset_credentials(outer_token)
