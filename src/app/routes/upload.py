from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.app.deps import templates
from src.app.errors import friendly_error_message
from src.app.rate_limit import enforce_rate_limit
from src.index.embed_toggle import get_active_embed_provider
from src.index.embedder import embed_passages
from src.ingest.chunker import chunk_document
from src.ingest.document_parser import parse_pdf
from src.store.relational import get_session
from src.store.schema import Chunk as ChunkRow
from src.store.schema import Paper as PaperRow

router = APIRouter()

# Phase 2 §5 — free-tier resource-fit math scoped these to something the
# 512MB container and Neon's 500MB free budget can absorb without a real
# per-request size check being anything but a cheap len() comparison.
_MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
# Opportunistic TTL, not a scheduled job (Airflow is out of scope for
# this deployment, see the Phase 2 plan's "Explicitly out of scope") —
# run on every GET /upload instead.
_UPLOAD_TTL_DAYS = int(os.environ.get("UPLOAD_TTL_DAYS", "7"))


def _is_postgres_backend() -> bool:
    return os.environ.get("RETRIEVAL_BACKEND", "opensearch") == "postgres"


def _cleanup_expired_uploads(session: Session) -> None:
    """ORM-level delete, not a bulk SQL DELETE — Paper's cascade="all,
    delete-orphan" relationship only fires when a Paper object is deleted
    through the session; there's no ON DELETE CASCADE at the DB level (see
    Chunk.paper_id's plain ForeignKey). Free-tier upload volume is small
    enough that loading the expired rows first is genuinely cheap, not a
    scalability shortcut that will bite later.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=_UPLOAD_TTL_DAYS)
    expired = session.execute(
        select(PaperRow).where(PaperRow.source == "upload", PaperRow.ingested_at < cutoff)
    ).scalars().all()
    for paper in expired:
        session.delete(paper)
    if expired:
        session.commit()


def _own_documents(session: Session, session_id: str) -> list[dict]:
    rows = session.execute(
        select(PaperRow.arxiv_id, PaperRow.title, PaperRow.ingested_at)
        .where(PaperRow.source == "upload", PaperRow.owner_session_id == session_id)
        .order_by(PaperRow.ingested_at.desc())
    ).all()
    return [{"document_id": r[0], "title": r[1], "ingested_at": r[2]} for r in rows]


@router.get("/upload")
def upload_page(request: Request):
    # Private uploads need the postgres backend's owner_session_id
    # isolation (see hybrid_postgres.py's _owner_predicate) — the default
    # OpenSearch path has no such concept, so this stays a dormant,
    # clearly-labeled page there rather than silently ingesting into the
    # shared corpus with no privacy guarantee at all.
    if not _is_postgres_backend():
        return templates.TemplateResponse(request, "upload.html", {"active": "upload", "unavailable": True})

    session = get_session()
    try:
        _cleanup_expired_uploads(session)
        documents = _own_documents(session, request.state.session_id)
    finally:
        session.close()

    return templates.TemplateResponse(
        request,
        "upload.html",
        {"active": "upload", "unavailable": False, "documents": documents, "max_mb": _MAX_UPLOAD_MB, "ttl_days": _UPLOAD_TTL_DAYS},
    )


@router.post("/upload", dependencies=[Depends(enforce_rate_limit)])
async def upload_submit(request: Request, file: UploadFile = File(...)):
    if not _is_postgres_backend():
        return templates.TemplateResponse(request, "upload.html", {"active": "upload", "unavailable": True})

    session_id = request.state.session_id
    error = None
    document_id = None

    pdf_bytes = await file.read()
    if len(pdf_bytes) > _MAX_UPLOAD_MB * 1024 * 1024:
        error = f"That file is larger than the {_MAX_UPLOAD_MB} MB limit for this deployment."
    else:
        try:
            # Bytes come straight from the upload, not an HTTP fetch — no
            # fetch_pdf_bytes/arXiv User-Agent involved, unlike
            # ingest/pipeline.py's arXiv path. parse_pdf itself is
            # already generic (just takes bytes), reused as-is.
            document = parse_pdf(pdf_bytes)
            document_id = f"upload-{uuid.uuid4().hex}"
            title = file.filename or document_id
            # Content-addressed chunk_ids collide across genuinely
            # identical text within one document (a repeated header,
            # e.g.) — same dedup rule as ingest/pipeline.py's identical
            # comment on the arXiv path, not data loss.
            raw_chunks = chunk_document(document_id, document)
            chunks = list({c.chunk_id: c for c in raw_chunks}.values())

            if not chunks:
                error = "Couldn't find any extractable text in that PDF — it may be scanned images without a text layer."
            else:
                # Uses the visitor's own BYOK key automatically —
                # embed_passages -> _embed_jina/_embed_mistral both check
                # get_credentials() first (see src/index/embedder.py) — a
                # shared owner embed key would either rate-limit across
                # every visitor or cost the deployment owner money per
                # upload, which is exactly why BYOK had to exist before
                # this feature (Phase 2 plan §5).
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
        except Exception as exc:
            # Same friendly-message convention as ask.py/pipeline.py — a
            # missing BYOK key surfaces here as "needs an API key that
            # isn't configured", not a raw traceback.
            error = friendly_error_message(exc)
            document_id = None

    if error is None and document_id:
        return RedirectResponse(url=f"/ask?document_id={document_id}", status_code=303)

    session = get_session()
    try:
        _cleanup_expired_uploads(session)
        documents = _own_documents(session, session_id)
    finally:
        session.close()

    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "active": "upload",
            "unavailable": False,
            "documents": documents,
            "max_mb": _MAX_UPLOAD_MB,
            "ttl_days": _UPLOAD_TTL_DAYS,
            "error": error,
        },
    )
