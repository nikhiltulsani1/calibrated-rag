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


def save_run(session: Session, trace: Any) -> str:
    """Persists a StageTrace as plain JSON. Returns the new run's id —
    what /pipeline?run_id=... loads back via load_run() below."""
    run_id = str(uuid.uuid4())
    session.add(Run(id=run_id, query=trace.original_query, trace_json=serialize_trace(trace)))
    session.commit()
    return run_id


def load_run(session: Session, run_id: str) -> dict | None:
    run = session.get(Run, run_id)
    return run.trace_json if run is not None else None
