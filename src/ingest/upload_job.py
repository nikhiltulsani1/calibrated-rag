from __future__ import annotations

from src.app.errors import friendly_error_message
from src.index.embed_toggle import get_active_embed_provider
from src.index.embedder import embed_passages
from src.ingest.chunker import chunk_document
from src.ingest.document_parser import parse_document
from src.platform.credentials import Credentials, reset_credentials, set_credentials
from src.store.relational import get_session
from src.store.schema import Chunk as ChunkRow
from src.store.schema import Paper as PaperRow


def process_upload(
    document_id: str, filename: str, raw_bytes: bytes, session_id: str, credentials: Credentials, title: str
) -> dict:
    """Phase 3 stage 3: the RQ job body — the exact parse -> chunk_document
    -> embed_passages -> Paper/Chunk write sequence that used to run
    inline inside routes/upload.py's POST handler, moved verbatim so a
    large file's processing happens in the background worker thread
    (started in src/app/main.py) instead of blocking the request.

    Two things needed explicit handling here that don't survive into a
    worker thread automatically:

    `credentials`: get_credentials() normally reads a ContextVar set by
    CredentialsMiddleware per HTTP request — that context doesn't exist
    in this worker thread at all. The route captures get_credentials()
    at enqueue time (while still inside the real request) and passes it
    as this explicit argument; set_credentials/reset_credentials here
    make it visible to embed_passages exactly the way the middleware
    would for a normal request.

    Never raises on a known, expected failure (missing extractable
    text, missing BYOK key) — those are caught and returned as a
    structured `{"status": "error", "message": ...}` result via
    friendly_error_message, the same message shape the old synchronous
    route showed inline. This is deliberately NOT left to RQ's own
    failed-job/exc_info mechanism: RQ's exc_info is a raw formatted
    traceback string, and rendering that anywhere a visitor could see it
    would violate this codebase's "never show raw traceback" rule
    (routes/upload.py's status endpoint reads this dict via job.result,
    never job.exc_info, for exactly that reason). An unexpected bug
    still propagates as a real exception and shows up as a failed RQ
    job for the operator to investigate — this only intercepts the
    already-known, already-handled error shapes.
    """
    token = set_credentials(credentials)
    try:
        document = parse_document(filename, raw_bytes)
        raw_chunks = chunk_document(document_id, document)
        chunks = list({c.chunk_id: c for c in raw_chunks}.values())

        if not chunks:
            raise RuntimeError(
                "Couldn't find any extractable text in that file — a PDF may be scanned images "
                "without a text layer, or the document may genuinely be empty."
            )

        embed_result = embed_passages([c.text for c in chunks])
        embedding_provider = get_active_embed_provider()

        session = get_session()
        try:
            session.add(
                PaperRow(
                    arxiv_id=document_id,
                    title=title,
                    authors=[],
                    abstract="",
                    category=[],
                    published_date=None,
                    url="",
                    source="upload",
                    owner_session_id=session_id,
                )
            )
            for chunk, vector in zip(chunks, embed_result.vectors):
                session.add(
                    ChunkRow(
                        chunk_id=chunk.chunk_id,
                        paper_id=document_id,
                        section=chunk.section,
                        text=chunk.text,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        page_number=chunk.page,
                        embedding_model=embed_result.model,
                        embedding_dim=embed_result.dimension,
                        owner_session_id=session_id,
                        embedding=vector,
                        embedding_provider=embedding_provider,
                    )
                )
            session.commit()
        finally:
            session.close()

        return {"status": "ok", "document_id": document_id}
    except Exception as exc:
        return {"status": "error", "message": friendly_error_message(exc)}
    finally:
        reset_credentials(token)
