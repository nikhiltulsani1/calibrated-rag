"""One-time backfill for RETRIEVAL_BACKEND=postgres (Phase 2): embeds
every existing chunk's text and writes it into the new
chunks.embedding/embedding_provider columns (see
scripts/add_phase2_postgres_backend_columns.py, which must run first).

The default RETRIEVAL_BACKEND=opensearch path is completely untouched —
this only ever reads Chunk.text and writes Chunk.embedding/
embedding_provider, columns that path never looks at.

Processes chunks in small, paced batches (not one giant embed_passages()
call) and commits after every batch, so a real rate-limit hit partway
through (this project's own documented, recurring free-tier reality)
loses at most one batch's progress, not the whole run — re-running is
safe and idempotent (only chunks with embedding IS NULL are picked up).

Run (needs the active EMBED_PROVIDER's real API key in .env):
    python -m scripts.backfill_postgres_embeddings
"""
from __future__ import annotations

import time

from sqlalchemy import select

from src.index.embed_toggle import get_active_embed_provider
from src.index.embedder import embed_passages
from src.store.relational import get_session
from src.store.schema import Chunk

_BATCH_SIZE = 32
_PACING_SECONDS = 2.0


def run() -> None:
    provider = get_active_embed_provider()
    print(f"backfilling with provider={provider}")

    session = get_session()
    try:
        pending_ids = [
            row[0]
            for row in session.execute(select(Chunk.chunk_id).where(Chunk.embedding.is_(None))).all()
        ]
    finally:
        session.close()

    print(f"{len(pending_ids)} chunks need embedding")
    done = 0
    for start in range(0, len(pending_ids), _BATCH_SIZE):
        batch_ids = pending_ids[start : start + _BATCH_SIZE]
        session = get_session()
        try:
            rows = session.execute(select(Chunk.chunk_id, Chunk.text).where(Chunk.chunk_id.in_(batch_ids))).all()
            texts_by_id = {row[0]: row[1] for row in rows}
            ordered_ids = [cid for cid in batch_ids if cid in texts_by_id]
            texts = [texts_by_id[cid] for cid in ordered_ids]

            result = embed_passages(texts, provider=provider)
            if len(result.vectors) != len(ordered_ids):
                raise RuntimeError(
                    f"embed_passages returned {len(result.vectors)} vectors for {len(ordered_ids)} chunks"
                )

            for chunk_id, vector in zip(ordered_ids, result.vectors):
                session.execute(
                    Chunk.__table__.update()
                    .where(Chunk.chunk_id == chunk_id)
                    .values(embedding=vector, embedding_provider=provider)
                )
            session.commit()
            done += len(ordered_ids)
            print(f"  {done}/{len(pending_ids)} embedded")
        finally:
            session.close()
        time.sleep(_PACING_SECONDS)

    print(f"done: {done} chunks embedded with provider={provider}")


if __name__ == "__main__":
    run()
