from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import app

pytestmark = pytest.mark.unit


def test_health_always_returns_ok_regardless_of_dependencies():
    # Liveness must not depend on any backing service — see health.py's
    # docstring for why (an orchestrator restarting a healthy process
    # over a neighbour's hiccup is the failure mode this avoids).
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_200_when_every_dependency_is_reachable():
    client = TestClient(app)
    with patch("src.app.routes.health._check_postgres", return_value=None), patch(
        "src.app.routes.health._check_opensearch", return_value=None
    ), patch("src.app.routes.health._check_redis", return_value=None):
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readyz_returns_503_when_one_dependency_is_down():
    # Plan D1's real gap: a replica with a dead OpenSearch connection
    # must be taken out of rotation, not keep accepting traffic.
    client = TestClient(app)
    with patch("src.app.routes.health._check_postgres", return_value=None), patch(
        "src.app.routes.health._check_opensearch", return_value="opensearch: connection refused"
    ), patch("src.app.routes.health._check_redis", return_value=None):
        response = client.get("/readyz")
    assert response.status_code == 503
    assert "opensearch: connection refused" in response.json()["failures"]


def test_readyz_names_every_failure_not_just_the_first():
    client = TestClient(app)
    with patch("src.app.routes.health._check_postgres", return_value="postgres: down"), patch(
        "src.app.routes.health._check_opensearch", return_value="opensearch: down"
    ), patch("src.app.routes.health._check_redis", return_value=None):
        response = client.get("/readyz")
    failures = response.json()["failures"]
    assert len(failures) == 2
    assert any("postgres" in f for f in failures)
    assert any("opensearch" in f for f in failures)


def test_check_postgres_catches_a_real_exception(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    with patch("psycopg.connect", side_effect=RuntimeError("connection refused")):
        from src.app.routes.health import _check_postgres

        result = _check_postgres()
    assert result is not None
    assert "postgres" in result


def test_check_postgres_uses_a_short_connect_timeout_not_the_production_default(monkeypatch):
    # Real bug this locks in: reusing the shared production client's
    # generous timeout (found live: OpenSearch's is 30s) let /readyz
    # hang instead of failing fast on a truly down dependency.
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    with patch("psycopg.connect") as mock_connect:
        from src.app.routes.health import _CHECK_TIMEOUT_SECONDS, _check_postgres

        mock_connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        _check_postgres()
    assert mock_connect.call_args.kwargs["connect_timeout"] == _CHECK_TIMEOUT_SECONDS
    assert _CHECK_TIMEOUT_SECONDS <= 5  # a readiness probe must fail fast, not eventually


def test_check_opensearch_reports_ping_failure():
    fake_client = MagicMock()
    fake_client.ping.return_value = False
    with patch("opensearchpy.OpenSearch", return_value=fake_client):
        from src.app.routes.health import _check_opensearch

        result = _check_opensearch()
    assert result == "opensearch: ping returned false"


def test_check_opensearch_catches_a_real_exception():
    with patch("opensearchpy.OpenSearch", side_effect=RuntimeError("connection refused")):
        from src.app.routes.health import _check_opensearch

        result = _check_opensearch()
    assert result is not None
    assert "opensearch" in result


def test_check_redis_catches_a_real_exception():
    with patch("redis.Redis", side_effect=RuntimeError("connection refused")):
        from src.app.routes.health import _check_redis

        result = _check_redis()
    assert result is not None
    assert "redis" in result
