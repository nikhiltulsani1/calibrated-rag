from unittest.mock import MagicMock, patch

import pytest

from src.store.relational import _database_url, init_db

pytestmark = pytest.mark.unit

# Real bug found live during the Neon deploy check (Phase 2): a bare
# `postgresql://` DATABASE_URL — exactly what Neon (and most managed
# Postgres providers) hand out by default — resolves to psycopg2 in
# SQLAlchemy, which isn't installed here (only psycopg[binary]/psycopg3
# is in requirements.txt). Every connection attempt using the
# provider-given string verbatim failed with ModuleNotFoundError until
# this was normalized.


def test_bare_postgresql_url_gets_normalized_to_psycopg_driver(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host.neon.tech/db?sslmode=require")
    assert _database_url() == "postgresql+psycopg://user:pass@host.neon.tech/db?sslmode=require"


def test_url_with_driver_already_specified_is_left_alone(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@host/db")
    assert _database_url() == "postgresql+psycopg://user:pass@host/db"


def test_falls_back_to_discrete_postgres_env_vars_when_no_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    assert _database_url() == "postgresql+psycopg://u:p@localhost:5432/d"


# ---------------------------------------------------------------------
# init_db — real gap found in review (Phase 3): nothing ever called this
# automatically, so a genuinely fresh clone-and-run had no schema at
# all. Confirmed directly against a real scratch Postgres database (not
# assumed): chunks.embedding is a Vector(1024) column regardless of
# which retrieval backend is active, so create_all() hard-fails with
# "type vector does not exist" unless the pgvector extension is enabled
# first. See tests/integration/test_relational_integration.py for the
# real end-to-end proof against a real fresh database; these are the
# fast unit-level checks of the call ordering itself.
# ---------------------------------------------------------------------


def test_init_db_enables_the_pgvector_extension_before_create_all():
    call_order = []
    fake_engine = MagicMock()
    fake_conn = MagicMock()
    fake_engine.begin.return_value.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = lambda *a, **k: call_order.append("create_extension")

    with patch("src.store.relational.get_engine", return_value=fake_engine), patch(
        "src.store.relational.Base"
    ) as mock_base:
        mock_base.metadata.create_all.side_effect = lambda *a, **k: call_order.append("create_all")
        init_db()

    assert call_order == ["create_extension", "create_all"]


def test_init_db_extension_statement_is_idempotent():
    fake_engine = MagicMock()
    fake_conn = MagicMock()
    fake_engine.begin.return_value.__enter__.return_value = fake_conn

    with patch("src.store.relational.get_engine", return_value=fake_engine), patch("src.store.relational.Base"):
        init_db()

    executed_sql = str(fake_conn.execute.call_args.args[0])
    assert "IF NOT EXISTS" in executed_sql
    assert "vector" in executed_sql
