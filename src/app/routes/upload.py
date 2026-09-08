from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import RedirectResponse
from rq.exceptions import NoSuchJobError
from rq.job import Job
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.app.deps import is_postgres_backend, templates
from src.app.errors import friendly_error_message
from src.app.rate_limit import enforce_rate_limit
from src.index.embed_toggle import get_active_embed_provider
from src.index.embedder import embed_passages
from src.ingest.chunker import chunk_document
from src.ingest.document_parser import parse_document
from src.ingest.upload_job import process_upload
from src.platform.credentials import get_credentials
from src.platform.queue import get_queue, get_redis_conn
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


def _job_status_context(job_id: str, session_id: str) -> dict:
    """Phase 3 stage 3: shared by GET /upload (first page load, server-
    rendered) and GET /upload/status/{job_id} (htmx polling) so the two
    never compute this differently. Session-id is embedded in the job's
    own meta at enqueue time and checked here — a job_id belonging to a
    different visitor is treated as unknown, never confirmed to exist,
    same not-found-not-403 convention as everywhere else in this file.
    """
    try:
        job = Job.fetch(job_id, connection=get_redis_conn())
    except NoSuchJobError:
        return {"job_status": "unknown"}
    if job.meta.get("session_id") != session_id:
        return {"job_status": "unknown"}
    if job.is_finished:
        result = job.result or {}
        if result.get("status") == "ok":
            return {"job_status": "done", "document_id": result["document_id"]}
        return {"job_status": "error", "job_error": result.get("message", "Something went wrong.")}
    if job.is_failed:
        # A genuinely unexpected bug, not one of process_upload's own
        # caught cases (those return a normal "error" result instead of
        # failing the job — see upload_job.py's own docstring for why).
        # Never surface job.exc_info directly — it's a raw traceback
        # string, and this codebase never shows those to a visitor.
        return {"job_status": "error", "job_error": "Something went wrong while processing this upload."}
    return {"job_status": "processing", "job_id": job_id}


@router.get("/upload")
def upload_page(request: Request, job_id: str | None = None):
    # Private uploads need the postgres backend's owner_session_id
    # isolation (see hybrid_postgres.py's _owner_predicate) — the default
    # OpenSearch path has no such concept, so this stays a dormant,
    # clearly-labeled page there rather than silently ingesting into the
    # shared corpus with no privacy guarantee at all.
    if not is_postgres_backend():
        return templates.TemplateResponse(request, "upload.html", {"active": "upload", "unavailable": True})

    session = get_session()
    try:
        _cleanup_expired_uploads(session)
        documents = _own_documents(session, request.state.session_id)
    finally:
        session.close()

    context = {
        "active": "upload",
        "unavailable": False,
        "documents": documents,
        "max_mb": _MAX_UPLOAD_MB,
        "ttl_days": _UPLOAD_TTL_DAYS,
    }
    if job_id:
        context.update(_job_status_context(job_id, request.state.session_id))
    return templates.TemplateResponse(request, "upload.html", context)


@router.get("/upload/status/{job_id}")
def upload_status(request: Request, job_id: str):
    if not is_postgres_backend():
        return templates.TemplateResponse(request, "upload.html", {"active": "upload", "unavailable": True})
    context = _job_status_context(job_id, request.state.session_id)
    return templates.TemplateResponse(request, "_upload_status.html", context)


@router.post("/upload", dependencies=[Depends(enforce_rate_limit)])
def upload_submit(request: Request, file: UploadFile = File(...)):
    if not is_postgres_backend():
        return templates.TemplateResponse(request, "upload.html", {"active": "upload", "unavailable": True})

    session_id = request.state.session_id
    error = None

    # `file.file.read()` (sync, not `await file.read()`) is still correct
    # here even though the heavy work moved to a background job — this
    # plain `def` handler still benefits from FastAPI's threadpool for
    # the read itself, and rate limiting (enforce_rate_limit above) must
    # gate on ENQUEUE, not on job execution, so it stays exactly here.
    raw_bytes = file.file.read()
    if len(raw_bytes) > _MAX_UPLOAD_MB * 1024 * 1024:
        error = f"That file is larger than the {_MAX_UPLOAD_MB} MB limit for this deployment."
    else:
        # Phase 3 stage 3: parse/chunk/embed/write now happen in
        # process_upload (src/ingest/upload_job.py), run by the
        # background worker thread (src/app/main.py) instead of inline
        # here — this request returns as soon as the job is queued,
        # regardless of how large the file is. get_credentials() is
        # captured NOW, inside the real request, and passed explicitly:
        # the worker thread has no CredentialsMiddleware-set ContextVar
        # of its own to read from.
        document_id = f"upload-{uuid.uuid4().hex}"
        title = file.filename or document_id
        job = get_queue().enqueue(
            process_upload, document_id, file.filename, raw_bytes, session_id, get_credentials(), title
        )
        job.meta["session_id"] = session_id
        job.save_meta()
        return RedirectResponse(url=f"/upload?job_id={job.id}", status_code=303)

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


@router.post("/upload/{document_id}/pages/{page_number}", dependencies=[Depends(enforce_rate_limit)])
def upload_page_replace(request: Request, document_id: str, page_number: int, file: UploadFile = File(...)):
    """Phase 3 stage 2: replace just one page of an already-uploaded PDF
    without re-processing the whole document. Deliberately PDF-only (see
    schema.py's Chunk.page_number comment) — the visitor re-uploads a
    small file containing just the replacement page's content, reusing
    the exact same parse/chunk/embed pipeline as a normal upload; every
    resulting chunk is tagged with the URL's page_number (the page being
    replaced), not whatever page the snippet's own content would imply.
    """
    if not is_postgres_backend():
        return templates.TemplateResponse(request, "upload.html", {"active": "upload", "unavailable": True})

    session_id = request.state.session_id
    error = None

    session = get_session()
    try:
        # Ownership check follows store/runs.py::load_run's established
        # pattern exactly: missing OR owned by a different session is
        # indistinguishable "no such document," never a 403 that would
        # confirm the id exists. A shared/arXiv paper (owner_session_id
        # is None) is never replaceable through this endpoint either —
        # None != session_id fails the check the same as a real mismatch.
        paper = session.get(PaperRow, document_id)
        if paper is None or paper.owner_session_id != session_id:
            error = f"No document found with id {document_id!r} for this session."
        else:
            raw_bytes = file.file.read()
            if len(raw_bytes) > _MAX_UPLOAD_MB * 1024 * 1024:
                error = f"That file is larger than the {_MAX_UPLOAD_MB} MB limit for this deployment."
            else:
                try:
                    document = parse_document(file.filename, raw_bytes)
                    raw_chunks = chunk_document(document_id, document)
                    chunks = list({c.chunk_id: c for c in raw_chunks}.values())

                    if not chunks:
                        error = "Couldn't find any extractable text in that file."
                    else:
                        embed_result = embed_passages([c.text for c in chunks])
                        embedding_provider = get_active_embed_provider()

                        # Scoped delete — every other page's chunks are
                        # untouched by construction, not by convention.
                        session.execute(
                            delete(ChunkRow).where(
                                ChunkRow.paper_id == document_id, ChunkRow.page_number == page_number
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
                                    page_number=page_number,
                                    embedding_model=embed_result.model,
                                    embedding_dim=embed_result.dimension,
                                    owner_session_id=session_id,
                                    embedding=vector,
                                    embedding_provider=embedding_provider,
                                )
                            )
                        session.commit()
                except Exception as exc:
                    error = friendly_error_message(exc)
    finally:
        session.close()

    if error is None:
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
