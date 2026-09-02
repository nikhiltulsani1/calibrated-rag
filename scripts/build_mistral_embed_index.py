"""One-time build of the Mistral-embed alternative index — part of the
embed-provider switch (src/index/embed_toggle.py). Re-embeds the SAME
Postgres chunk text (default chunking only) via mistral-embed (1024-d,
confirmed live 2026-08-22) into a new, permanent `rag_chunks_mistral_embed`
index, leaving the production `rag_chunks` (jina) index completely
untouched. Re-runnable: reindex_from_postgres/create_index are both
idempotent-safe (create_index no-ops if the index already exists; a
second run just re-bulk-indexes the same chunk_ids).

Run inside the api container (needs the real src/ package + MISTRAL_API_KEY):
    docker exec -e MISTRAL_API_KEY=... -w /app <container> python scripts/build_mistral_embed_index.py
"""
from __future__ import annotations

from src.index.embed_toggle import _MISTRAL_INDEX_NAME
from src.index.reindex import reindex_from_postgres

if __name__ == "__main__":
    n = reindex_from_postgres(
        embed=True,
        index_name=_MISTRAL_INDEX_NAME,
        embed_provider="mistral",
        dimension=1024,
    )
    print(f"indexed {n} documents into {_MISTRAL_INDEX_NAME} (provider=mistral)")
