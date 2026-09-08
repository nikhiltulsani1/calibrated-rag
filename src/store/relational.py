from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.store.schema import Base

_engine: Engine | None = None


def _database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        # Phase 2: Neon (and most managed Postgres providers) hand out a
        # bare `postgresql://` connection string. SQLAlchemy resolves
        # that to psycopg2 by default, which isn't installed here
        # (requirements.txt only has psycopg[binary], psycopg3) — a real
        # bug found live: every provided-DATABASE_URL connection
        # silently failed with ModuleNotFoundError until this was
        # normalized. `postgresql+psycopg://` forces the driver that's
        # actually installed, without requiring the operator to know to
        # edit the connection string Neon gave them verbatim.
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()


def init_db() -> None:
    # No migration tool yet — Alembic lands with the ingest pipeline, once
    # the schema needs to evolve under real data rather than be created
    # once from nothing. create_all is honest about the current state:
    # idempotent, additive-only, no upgrade path yet.
    #
    # Real gap found in review (Phase 3): nothing in this codebase ever
    # called init_db() automatically — not at app startup, not in the
    # Dockerfile, not in compose.yml — so a genuinely fresh clone-and-run
    # had no schema at all. Worse, `chunks.embedding` is a Vector(1024)
    # column REGARDLESS of which retrieval backend is active (the
    # OpenSearch path just never reads/writes it), so create_all() below
    # would fail creating even the `chunks` table with a hard "type
    # vector does not exist" error — confirmed directly against a real
    # scratch database, not assumed — unless the pgvector extension is
    # enabled first. `CREATE EXTENSION IF NOT EXISTS` is idempotent, a
    # no-op on this project's existing local/Neon databases (already
    # enabled via scripts/add_phase2_postgres_backend_columns.py) — this
    # only matters for a database that has never seen this project's
    # schema at all.
    with get_engine().begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(get_engine())


if __name__ == "__main__":
    init_db()
    print("tables created (or already existed)")
