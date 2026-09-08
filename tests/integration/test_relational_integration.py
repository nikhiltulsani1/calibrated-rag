import uuid

import pytest
from sqlalchemy import create_engine, inspect, text

import src.store.relational as relational
from src.store.relational import init_db

pytestmark = pytest.mark.integration

# Real gap found in review (Phase 3): init_db() was never called
# automatically anywhere, so a genuinely fresh clone-and-run had no
# schema at all — and even calling it manually would have hard-failed,
# since chunks.embedding is a Vector(1024) column regardless of which
# retrieval backend is active, and Postgres has no `vector` type until
# the pgvector extension is enabled. This creates a REAL throwaway
# database on the local Postgres server (the same one this project's
# other integration tests already use) and proves init_db() alone,
# starting from nothing, produces a working schema — not mocked.
#
# Created once per test session (not per test) and reset between tests
# via DROP SCHEMA/CREATE SCHEMA rather than DROP DATABASE — dropping the
# whole database requires zero other sessions holding it open, which
# raced with the engine's own pooled connections even after
# Engine.dispose(); dropping the public schema has no such requirement
# and is exactly as "genuinely empty" a starting point for init_db().
_DB_NAME = f"scratch_init_db_test_{uuid.uuid4().hex[:12]}"
_DB_URL = f"postgresql+psycopg://rag:localdevpass@localhost:5432/{_DB_NAME}"


@pytest.fixture(scope="session", autouse=True)
def _scratch_database():
    admin_engine = create_engine("postgresql+psycopg://rag:localdevpass@localhost:5432/rag")
    with admin_engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"DROP DATABASE IF EXISTS {_DB_NAME}"))
        conn.execute(text(f"CREATE DATABASE {_DB_NAME} TEMPLATE template0"))
    admin_engine.dispose()
    yield
    if relational._engine is not None:
        relational._engine.dispose()
        relational._engine = None
    admin_engine = create_engine("postgresql+psycopg://rag:localdevpass@localhost:5432/rag")
    with admin_engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"DROP DATABASE IF EXISTS {_DB_NAME}"))
    admin_engine.dispose()


@pytest.fixture
def fresh_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    relational._engine = None  # force get_engine() to pick up the new DATABASE_URL
    engine = create_engine(_DB_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    yield _DB_URL
    # Nulling the reference alone leaves the pool's sockets open until
    # garbage collection — dispose() explicitly so the session-scoped
    # teardown's DROP DATABASE isn't racing a connection this test
    # itself left open.
    if relational._engine is not None:
        relational._engine.dispose()
        relational._engine = None


def test_init_db_builds_a_working_schema_from_a_genuinely_empty_database(fresh_database_url):
    init_db()

    engine = relational.get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"papers", "chunks", "runs", "chunk_variants"} <= tables

    # The specific column that would have hard-failed create_all()
    # without the extension enabled first — proof this isn't just
    # "some tables got created," it's the real, full schema.
    columns = {c["name"] for c in inspector.get_columns("chunks")}
    assert "embedding" in columns
    assert "page_number" in columns


def test_init_db_is_a_safe_no_op_when_run_twice(fresh_database_url):
    init_db()
    init_db()  # must not raise — every real deployment calls this on every startup
