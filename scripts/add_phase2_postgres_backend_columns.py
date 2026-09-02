"""One-time schema migration for Phase 2's RETRIEVAL_BACKEND=postgres path
(src/retrieve/hybrid_postgres.py). No migration tool exists yet
(src/store/relational.py::init_db()'s own comment says Alembic lands
later) — Base.metadata.create_all() is additive for whole tables but does
NOT add columns to tables that already exist, and the real local/live
Postgres already has real ingested data (papers/chunks) this must not
touch. So this runs explicit, idempotent ALTER TABLE statements instead.

Every statement here is IF NOT EXISTS / additive-only:
- enables the pgvector extension (required before the `embedding` column
  type can exist at all)
- widens papers.arxiv_id / chunks.paper_id / chunk_variants.paper_id from
  varchar(32) to varchar(64) — real arXiv ids are ~10 chars and fit
  either way; this only matters for the new synthetic upload ids
  ("upload-<uuid4().hex>", 39 chars) the OpenSearch path never produces
- adds every new nullable column from src/store/schema.py's Paper/Chunk/Run
  models (source, owner_session_id, embedding, embedding_provider,
  text_tsv, plus the GIN index on text_tsv)

Every new column is nullable and defaults to NULL for existing rows —
the RETRIEVAL_BACKEND=opensearch path never reads or writes any of them,
so existing arXiv-ingested data and the default backend's behavior are
both completely unaffected by running this.

Run once against whichever Postgres will serve RETRIEVAL_BACKEND=postgres
(local dev Postgres to build/test hybrid_postgres.py, or the live Neon
instance before the first deploy):
    python -m scripts.add_phase2_postgres_backend_columns
"""
from __future__ import annotations

from sqlalchemy import text

from src.store.relational import get_engine

_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    "ALTER TABLE papers ALTER COLUMN arxiv_id TYPE VARCHAR(64);",
    "ALTER TABLE chunks ALTER COLUMN paper_id TYPE VARCHAR(64);",
    "ALTER TABLE chunk_variants ALTER COLUMN paper_id TYPE VARCHAR(64);",
    "ALTER TABLE papers ADD COLUMN IF NOT EXISTS source VARCHAR(16);",
    "ALTER TABLE papers ADD COLUMN IF NOT EXISTS owner_session_id VARCHAR(64);",
    "CREATE INDEX IF NOT EXISTS ix_papers_owner_session_id ON papers (owner_session_id);",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS owner_session_id VARCHAR(64);",
    "CREATE INDEX IF NOT EXISTS ix_chunks_owner_session_id ON chunks (owner_session_id);",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(1024);",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_provider VARCHAR(32);",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;",
    "CREATE INDEX IF NOT EXISTS ix_chunks_text_tsv ON chunks USING gin (text_tsv);",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS owner_session_id VARCHAR(64);",
    "CREATE INDEX IF NOT EXISTS ix_runs_owner_session_id ON runs (owner_session_id);",
]


def run() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for statement in _STATEMENTS:
            conn.execute(text(statement))
            print(f"ok: {statement}")


if __name__ == "__main__":
    run()
    print("Phase 2 postgres-backend columns present (or already were).")
