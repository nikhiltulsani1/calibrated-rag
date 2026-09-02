import pytest

from src.store.relational import _database_url

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
