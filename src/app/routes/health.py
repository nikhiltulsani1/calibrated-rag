from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Response

logger = logging.getLogger(__name__)

router = APIRouter()

# A real readiness probe must fail FAST — an orchestrator reacting to a
# down dependency needs an answer in low single-digit seconds, not
# whatever timeout the production clients use for real queries (found
# live: OpenSearch's shared client defaults to a 30s timeout, meant for
# genuine slow queries under load — reusing it here meant /readyz could
# hang for up to 30s instead of reporting unhealthy, defeating the
# entire point of the probe). Each check below builds its own short-
# timeout connection rather than reusing get_client()/get_engine().
_CHECK_TIMEOUT_SECONDS = 2.0


@router.get("/health")
def health() -> dict:
    """Liveness only — is the process itself up and able to answer HTTP
    at all. Deliberately checks nothing else: a liveness probe that
    depends on Postgres/OpenSearch/Redis would make an orchestrator
    restart a perfectly healthy process just because a neighbour
    service hiccuped. See /readyz below for the dependency check."""
    return {"status": "ok"}


def _check_postgres() -> str | None:
    try:
        import psycopg

        # Phase 2: this probe used to only understand the discrete
        # POSTGRES_USER/PASSWORD/DB/HOST vars compose.yml's local stack
        # sets — on the live deployment only DATABASE_URL is set (Neon's
        # connection string), so this always failed there with a
        # KeyError, same class of gap _database_url() itself had (see
        # src/store/relational.py) before that was fixed. Reuses that
        # same fallback logic rather than a second hand-rolled copy.
        from src.store.relational import _database_url

        url = _database_url().replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(url, connect_timeout=_CHECK_TIMEOUT_SECONDS) as conn:
            conn.execute("SELECT 1")
        return None
    except Exception as exc:
        return f"postgres: {exc}"


def _check_opensearch() -> str | None:
    # Phase 2: RETRIEVAL_BACKEND=postgres deployments (the live
    # free-tier one) have no OpenSearch at all — see hybrid_postgres.py
    # and the Phase 2 plan §2. Checking for it unconditionally would
    # make this probe permanently report "not ready" on that
    # deployment, which is exactly the kind of false-negative readiness
    # check A1/D1's own design (see this file's module docstring
    # intent) exists to avoid. The default RETRIEVAL_BACKEND=opensearch
    # path (local clone-and-run) is completely unaffected — this check
    # still runs there exactly as before.
    if os.environ.get("RETRIEVAL_BACKEND", "opensearch") == "postgres":
        return None
    try:
        from opensearchpy import OpenSearch

        host = os.environ.get("OPENSEARCH_HOST", "localhost")
        port = int(os.environ.get("OPENSEARCH_PORT", "9200"))
        password = os.environ["OPENSEARCH_ADMIN_PASSWORD"]
        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=("admin", password),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=_CHECK_TIMEOUT_SECONDS,
        )
        if not client.ping():
            return "opensearch: ping returned false"
        return None
    except Exception as exc:
        return f"opensearch: {exc}"


def _check_redis() -> str | None:
    try:
        import redis

        # Same Upstash TLS/password gap already found and fixed in
        # src/platform/cache.py::get_client() — this is a separate,
        # hand-rolled connection (deliberately not reusing the shared
        # client, see this file's own timeout-isolation comment above),
        # so it needed the identical fix applied here too.
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        password = os.environ.get("REDIS_PASSWORD") or None
        ssl = os.environ.get("REDIS_SSL", "false").lower() == "true"
        client = redis.Redis(
            host=host, port=port, password=password, ssl=ssl,
            socket_connect_timeout=_CHECK_TIMEOUT_SECONDS,
            socket_timeout=_CHECK_TIMEOUT_SECONDS,
        )
        client.ping()
        return None
    except Exception as exc:
        return f"redis: {exc}"


@router.get("/readyz")
def readyz(response: Response) -> dict:
    """Real readiness — plan D1. Genuinely checks Postgres, OpenSearch,
    and Redis reachability rather than returning a hardcoded 200 like
    /health did before this. Without this, a replica whose OpenSearch
    connection died would keep accepting traffic from a load balancer
    and fail every real request instead of being taken out of rotation
    — this is exactly the gap that made /health alone insufficient for
    horizontal scaling. Runs all three checks even after an early
    failure so one bad response names every real problem, not just the
    first one hit. Each check uses its own short timeout (see
    _CHECK_TIMEOUT_SECONDS) rather than the production clients' — found
    live that reusing those let this probe hang for up to 30s on a
    truly down dependency instead of failing fast.
    """
    failures = [f for f in (_check_postgres(), _check_opensearch(), _check_redis()) if f]
    if failures:
        logger.warning("readyz failing: %s", failures, extra={"event": "readyz_failed"})
        response.status_code = 503
        return {"status": "not_ready", "failures": failures}
    return {"status": "ready"}
