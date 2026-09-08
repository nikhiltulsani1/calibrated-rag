"""One-time migration: adds chunks.page_number (see src/store/schema.py's
Chunk.page_number) on a Postgres instance that already has the table —
same reason scripts/add_phase2_postgres_backend_columns.py and
scripts/add_hnsw_index.py exist instead of relying on
Base.metadata.create_all() (which only creates whole tables that don't
exist yet, never adds a column to a table that's already there, and the
real local/live `chunks` table already has real data).

Nullable and additive-only: NULL for every existing row (nothing has
ever set it before this), and only src/ingest/document_parser.py's
parse_pdf ever populates it going forward (Phase 3 stage 2 is
deliberately PDF-only — see schema.py's own comment) — every other
format and the default RETRIEVAL_BACKEND=opensearch path are completely
unaffected by running this.

Run once against whichever Postgres serves RETRIEVAL_BACKEND=postgres
(local dev Postgres, or the live Neon instance):
    python -m scripts.add_page_number_column
"""
from __future__ import annotations

from sqlalchemy import text

from src.store.relational import get_engine

_STATEMENT = "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number INTEGER;"


def run() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(_STATEMENT))
    print(f"ok: {_STATEMENT}")


if __name__ == "__main__":
    run()
    print("chunks.page_number present (or already was).")
