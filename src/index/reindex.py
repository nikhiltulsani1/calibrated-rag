from __future__ import annotations

import logging

from opensearchpy import helpers

from src.index.client import create_index, get_client
from src.index.mapping import INDEX_NAME

logger = logging.getLogger(__name__)


def reindex_from_postgres(
    *, embed: bool = True, index_name: str = INDEX_NAME, embed_provider: str | None = None, dimension: int | None = None
) -> int:
    """Rebuilds an index from Postgres, the source of truth — see
    architecture.md: "every vector and every lexical field in [OpenSearch]
    can be regenerated from Postgres's chunk text." This was documented as
    a property of the design but had no actual runnable implementation
    until this function — found missing while building the A3 retrieval
    eval harness, when `rag_chunks` turned out to hold 0 documents despite
    Postgres correctly holding the full ingested corpus.

    With embed=False (no JINA_API_KEY available), indexes lexical and
    metadata fields only — BM25 search works immediately, but dense/kNN
    search over these documents finds nothing until a real embed=True
    pass runs. Never silently claims embeddings exist when they don't:
    `embedding_model` is only set on documents that actually got a vector.

    `index_name`/`embed_provider`/`dimension` (added 2026-08-22, embed
    provider switch) let this same function build the alternate
    `rag_chunks_mistral_embed` index too — see
    scripts/build_mistral_embed_index.py. Every existing caller omits
    them and rebuilds today's production `rag_chunks` via jina exactly
    as before.
    """
    from src.store.relational import get_session
    from src.store.schema import Chunk, Paper

    client = get_client()
    create_index(client, index_name, dimension=dimension)

    session = get_session()
    try:
        rows = session.query(Chunk, Paper).join(Paper, Chunk.paper_id == Paper.arxiv_id).all()
    finally:
        session.close()

    if not rows:
        return 0

    served_model = None
    served_dim = None
    if embed:
        from src.index.embedder import embed_passages

        # The embed call's own result — not chunk.embedding_model, which
        # was NULL on every row here (the original ingestion run never
        # got far enough to record it). Using the stale Postgres column
        # instead of this real value was a bug caught while writing this
        # function: it silently wrote embedding_model=None into OpenSearch
        # even on documents that had just received a real vector.
        embed_result = embed_passages([chunk.text for chunk, _paper in rows], provider=embed_provider)
        vectors = embed_result.vectors
        served_model = embed_result.model
        served_dim = embed_result.dimension
    else:
        vectors = [None] * len(rows)
        logger.warning(
            "reindex_from_postgres(embed=False): lexical-only rebuild — "
            "dense/kNN search will find nothing until a real embed=True pass runs",
            extra={"event": "reindex_lexical_only"},
        )

    def actions():
        for (chunk, paper), vector in zip(rows, vectors):
            source = {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "title": paper.title,
                "text": chunk.text,
                "section": chunk.section,
                "authors": paper.authors,
                "category": paper.category,
                "published_date": paper.published_date.isoformat() if paper.published_date else None,
                "ingested_at": paper.ingested_at.isoformat() if paper.ingested_at else None,
            }
            if vector is not None:
                source["embedding"] = vector
                source["embedding_model"] = served_model
            yield {"_index": index_name, "_id": chunk.chunk_id, "_source": source}

    success, errors = helpers.bulk(client, actions(), raise_on_error=False)
    if errors:
        raise RuntimeError(f"{len(errors)} documents failed to index: {errors[:3]}")
    client.indices.refresh(index=index_name)

    if embed and served_model and index_name == INDEX_NAME:
        # Backfill Postgres too, per architecture.md's documented
        # provenance design ("which model/width produced the vector...
        # is what a rebuild checks against models.lock.yaml") — the
        # column existing but never being populated is exactly the kind
        # of gap that design assumes doesn't exist. Only backfilled for
        # the production index — the mistral alternative index's
        # provenance lives in OpenSearch's own embedding_model field on
        # each document, not in Postgres, since Postgres's single
        # embedding_model/embedding_dim columns describe the ONE
        # production embedding, not every alternative index built from
        # the same chunk text.
        session = get_session()
        try:
            session.query(Chunk).filter(Chunk.chunk_id.in_([c.chunk_id for c, _p in rows])).update(
                {"embedding_model": served_model, "embedding_dim": served_dim}, synchronize_session=False
            )
            session.commit()
        finally:
            session.close()

    return success


if __name__ == "__main__":
    import os
    import sys

    embed = "--no-embed" not in sys.argv and bool(os.environ.get("JINA_API_KEY"))
    n = reindex_from_postgres(embed=embed)
    print(f"indexed {n} documents (embed={embed})")
