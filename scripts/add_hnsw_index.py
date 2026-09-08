"""One-time migration: builds the HNSW ANN index on chunks.embedding
(see src/store/schema.py's Chunk.__table_args__) on a Postgres instance
that already has the table — same reason
scripts/add_phase2_postgres_backend_columns.py exists instead of relying
on Base.metadata.create_all(): create_all() only creates whole tables
that don't exist yet, never adds an index to a table that's already
there, and the real local/live `chunks` table already has real data.

Real gap this closes: RETRIEVAL_BACKEND=postgres's dense search
(hybrid_postgres.py's `.cosine_distance()` query) had NO index on
`embedding` at all until this existed — every dense query was a full
sequential scan over every row. Invisible at hundreds of chunks;
becomes the actual scaling bottleneck once the corpus grows.

Uses `CREATE INDEX CONCURRENTLY`, not a plain `CREATE INDEX` — the
concurrent form doesn't take the exclusive lock that would otherwise
block every write against `chunks` for the build's whole duration
(harmless today at this corpus's size, but that's exactly the property
that matters once "scale" is the point). CONCURRENTLY cannot run inside
a transaction block, so this runs on an autocommit connection rather
than reusing the transactional pattern the columns migration script
uses.

Run once against whichever Postgres serves RETRIEVAL_BACKEND=postgres
(local dev Postgres, or the live Neon instance):
    python -m scripts.add_hnsw_index
"""
from __future__ import annotations

from sqlalchemy import text

from src.store.relational import get_engine

_STATEMENT = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_embedding_hnsw "
    "ON chunks USING hnsw (embedding vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64);"
)


def run() -> None:
    engine = get_engine()
    # autocommit: CREATE INDEX CONCURRENTLY errors ("cannot run inside a
    # transaction block") under the engine's normal transactional
    # execution — this is the standard, documented way around that, not
    # a workaround for a bug.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(_STATEMENT))
    print(f"ok: {_STATEMENT}")


if __name__ == "__main__":
    run()
    print("HNSW index on chunks.embedding present (or already was).")
