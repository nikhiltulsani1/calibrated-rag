import pytest


@pytest.fixture(autouse=True)
def _no_real_cohere_key_in_unit_tests(monkeypatch):
    # tests/conftest.py loads the real .env for the whole session (unit
    # AND integration — integration genuinely needs real keys to hit
    # real services). Unit tests must not: a real COHERE_API_KEY landing
    # here (added 2026-08-22 for the automatic Jina->Cohere rerank
    # fallback) already caused one real cross-file bug — a test in
    # test_telemetry.py expecting a Jina degrade silently made a real,
    # unmocked network call to Cohere instead, because it never
    # anticipated a second provider's key being present. Scoped to
    # tests/unit/ only (this file, not the shared root conftest.py) so
    # integration tests are unaffected. Tests that specifically want to
    # exercise the fallback re-set this key themselves (see
    # test_reranker.py's own fallback tests).
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
