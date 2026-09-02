from __future__ import annotations

import dataclasses
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.store.schema import Run


def serialize_trace(obj: Any) -> Any:
    """Recursively turns a StageTrace (dataclasses nesting Pydantic
    models nesting more dataclasses) into plain JSON-safe dicts/lists, so
    it can be stored in a JSON column and, on replay, fed straight back
    into pipeline.html — Jinja resolves `trace.answer.text` on a plain
    dict via its usual getattr-then-getitem fallback, so nothing on the
    read side needs to reconstruct real StageTrace/Answer/etc objects.

    Public since 2026-08-23 — reused as-is by
    src/reason/answer_cache.py (A6's semantic answer cache) for the exact
    same serialize-once-replay-as-a-dict shape, rather than duplicating
    this recursion a second time.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: serialize_trace(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: serialize_trace(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_trace(v) for v in obj]
    return obj


def save_run(session: Session, trace: Any, *, owner_session_id: str | None = None) -> str:
    """Persists a StageTrace as plain JSON. Returns the new run's id —
    what /pipeline?run_id=... loads back via load_run() below.

    `owner_session_id` (Phase 2, stage 6) is the requesting visitor's
    session id, but callers only actually pass it on the
    RETRIEVAL_BACKEND=postgres path (see routes/ask.py, routes/pipeline.py)
    — a run made against the default OpenSearch path's shared corpus has
    no private data to protect, and tagging it with an owner would make
    today's "anyone can replay any run_id" behavior a regression there.
    """
    run_id = str(uuid.uuid4())
    session.add(
        Run(id=run_id, query=trace.original_query, trace_json=serialize_trace(trace), owner_session_id=owner_session_id)
    )
    session.commit()
    return run_id


def load_run(session: Session, run_id: str, *, requester_session_id: str | None = None) -> dict | None:
    """Returns the trace dict, or None if the run doesn't exist OR
    belongs to a different session than the requester (Phase 2 §5: a
    guessed/shared run_id from a private-upload session must not become
    a cross-visitor data-exposure vector — treated identically to "not
    found" rather than a distinct "forbidden" response, so a caller can't
    even confirm the run_id exists). A run with no owner (NULL —
    everything on the default OpenSearch path, and any postgres-path run
    against only the shared corpus) is visible to every requester, same
    as today.
    """
    run = session.get(Run, run_id)
    if run is None:
        return None
    if run.owner_session_id is not None and run.owner_session_id != requester_session_id:
        return None
    return run.trace_json
