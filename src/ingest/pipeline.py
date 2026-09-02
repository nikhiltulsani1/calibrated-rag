from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.ingest.chunker import chunk_document
from src.ingest.document_parser import fetch_pdf_bytes, parse_pdf
from src.ingest.paper_source import PaperMetadata, fetch_papers_paginated
from src.index.client import create_index, get_client as get_opensearch_client
from src.index.embedder import embed_passages
from src.index.mapping import INDEX_NAME
from src.platform.telemetry import get_tracer
from src.store.relational import get_session
from src.store.schema import Chunk as ChunkRow
from src.store.schema import Paper as PaperRow

logger = logging.getLogger(__name__)


def _persist_paper(session, paper: PaperMetadata) -> bool:
    """Returns True if this paper is new."""
    if session.get(PaperRow, paper.arxiv_id) is not None:
        return False
    session.add(
        PaperRow(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            category=paper.category,
            published_date=paper.published_date,
            url=paper.pdf_url,
        )
    )
    session.commit()
    return True


def ingest_category(category: str, *, count: int, embed: bool = True) -> dict:
    """source -> parse -> chunk -> persist.

    Postgres is written first and is authoritative (A0) — the OpenSearch
    write happens after and is what makes the index a derived, rebuildable
    artifact rather than a second source of truth. Content-addressed
    chunk_ids make both the paper and chunk writes naturally idempotent:
    re-running this over the same papers is a no-op, not a duplicate.

    Wrapped in a real OTel span (R2's background-job tracing gap) — this
    is a scheduled/background job today invoked manually, not a live
    request, and it previously emitted zero spans despite `get_tracer()`
    being trivially available. Making the run itself traceable is what a
    future "did the ingestion job even run" alert would key off; a per-
    paper child span means one slow or failing paper is individually
    visible instead of only showing up in the aggregate.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("ingest.ingest_category") as span:
        span.set_attribute("ingest.category", category)
        span.set_attribute("ingest.count_requested", count)
        span.set_attribute("ingest.embed", embed)

        papers = fetch_papers_paginated(category, total=count)
        session = get_session()
        opensearch = None
        if embed:
            opensearch = get_opensearch_client()
            create_index(opensearch)  # idempotent — no-op if it already exists

        stats = {"papers_seen": 0, "papers_new": 0, "chunks_new": 0, "chunks_indexed": 0}

        for paper in papers:
            with tracer.start_as_current_span("ingest.paper") as paper_span:
                paper_span.set_attribute("ingest.arxiv_id", paper.arxiv_id)
                stats["papers_seen"] += 1
                if _persist_paper(session, paper):
                    stats["papers_new"] += 1

                pdf_bytes = fetch_pdf_bytes(paper.pdf_url)
                document = parse_pdf(pdf_bytes)
                raw_chunks = chunk_document(paper.arxiv_id, document)

                # Content-addressing means identical text (a repeated figure
                # caption, a page number) collides on chunk_id by design —
                # see A3b. Found by actually running this: two chunks with
                # the same id in the *same* paper's own output triggered a
                # primary-key violation on insert, since the DB-only
                # "already exists" check below doesn't catch a collision
                # within the batch being built. First occurrence wins;
                # that's a dedup, not data loss — the text is byte-identical.
                chunks = list({c.chunk_id: c for c in raw_chunks}.values())

                new_chunks = [c for c in chunks if session.get(ChunkRow, c.chunk_id) is None]
                for chunk in new_chunks:
                    session.add(
                        ChunkRow(
                            chunk_id=chunk.chunk_id,
                            paper_id=paper.arxiv_id,
                            section=chunk.section,
                            text=chunk.text,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                        )
                    )
                session.commit()
                stats["chunks_new"] += len(new_chunks)
                paper_span.set_attribute("ingest.chunks_new_for_paper", len(new_chunks))

                if embed and new_chunks:
                    result = embed_passages([c.text for c in new_chunks])
                    for chunk, vector in zip(new_chunks, result.vectors):
                        opensearch.index(
                            index=INDEX_NAME,
                            id=chunk.chunk_id,
                            body={
                                "chunk_id": chunk.chunk_id,
                                "paper_id": paper.arxiv_id,
                                "title": paper.title,
                                "text": chunk.text,
                                "section": chunk.section,
                                "authors": paper.authors,
                                "category": paper.category,
                                "published_date": paper.published_date.isoformat() if paper.published_date else None,
                                "embedding_model": result.model,
                                "embedding": vector,
                                "ingested_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        row = session.get(ChunkRow, chunk.chunk_id)
                        if row is not None:
                            row.embedding_model = result.model
                            row.embedding_dim = result.dimension
                    session.commit()
                    stats["chunks_indexed"] += len(new_chunks)

                logger.info(
                    "ingested %s: %d chunks (%d new)",
                    paper.arxiv_id,
                    len(chunks),
                    len(new_chunks),
                )

        session.close()
        span.set_attribute("ingest.papers_seen", stats["papers_seen"])
        span.set_attribute("ingest.papers_new", stats["papers_new"])
        span.set_attribute("ingest.chunks_new", stats["chunks_new"])
        span.set_attribute("ingest.chunks_indexed", stats["chunks_indexed"])
        return stats


if __name__ == "__main__":
    import sys

    category = sys.argv[1] if len(sys.argv) > 1 else "cs.IR"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    embed = "--no-embed" not in sys.argv
    print(ingest_category(category, count=count, embed=embed))
