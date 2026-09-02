from __future__ import annotations

from src.index.mapping import INDEX_NAME
from src.platform.backend import is_postgres_backend
from src.reason.state import fetch_metadata
from src.retrieve.hybrid import FusionTrace, retrieve_with_trace
from src.retrieve.reranker import Candidate, RerankResult, rerank
from src.schemas.query_plan import QueryPlan

_RECALL_SIZE = 50


def run_retrieve(
    plan: QueryPlan,
    index_name: str = INDEX_NAME,
    *,
    session_id: str | None = None,
    document_id: str | None = None,
) -> tuple[list[Candidate], FusionTrace]:
    """The raw hybrid-search fetch — once per query, not once per retry
    attempt. Only reranking's `top_n` widens on retry (see nodes/refine.py);
    re-running the fetch itself on retry would just re-score the same 50
    candidates, not find new ones. `index_name` defaults to the production
    index; graph.py passes A7's active toggle selection.

    Phase 2: dispatches to hybrid.py (default, OpenSearch-backed, every
    existing caller/behavior unchanged) or hybrid_postgres.py
    (RETRIEVAL_BACKEND=postgres, the free-tier live deployment only) —
    the two backends never run in the same process's request path at
    once, so a single env-var check here is enough, not a per-call
    parameter threaded from every caller. `session_id`/`document_id` are
    meaningless on the OpenSearch path (that corpus has no private
    uploads) and are only forwarded when the postgres backend is
    actually active.
    """
    if is_postgres_backend():
        from src.retrieve.hybrid_postgres import retrieve_with_trace as retrieve_with_trace_postgres

        return retrieve_with_trace_postgres(plan, top_n=_RECALL_SIZE, session_id=session_id, document_id=document_id)
    return retrieve_with_trace(plan, top_n=_RECALL_SIZE, index_name=index_name)


def run_rerank_and_metadata(
    query: str, candidates: list[Candidate], top_n: int, index_name: str = INDEX_NAME
) -> tuple[RerankResult, dict[str, dict]]:
    """Rerank, then look up the winning candidates' titles/paper_id/section
    — cheap, deterministic, and needed by both the assess node (context
    display) and the answer node (citations), so it happens once here
    right after reranking rather than being duplicated in either.
    """
    reranked = rerank(query, candidates, top_n=top_n)
    ids = [item.id for item in reranked.items]
    if is_postgres_backend():
        metadata = _fetch_metadata_postgres(ids)
    else:
        metadata = fetch_metadata(ids, index_name=index_name)
    return reranked, metadata


def _fetch_metadata_postgres(candidate_ids: list[str]) -> dict[str, dict]:
    """Postgres-backend counterpart of state.py::fetch_metadata — a plain
    Chunk JOIN Paper instead of an OpenSearch mget, same return shape.
    """
    if not candidate_ids:
        return {}
    from sqlalchemy import select

    from src.store.relational import get_session
    from src.store.schema import Chunk, Paper

    session = get_session()
    try:
        rows = session.execute(
            select(Chunk.chunk_id, Paper.title, Chunk.paper_id, Chunk.section)
            .join(Paper, Chunk.paper_id == Paper.arxiv_id)
            .where(Chunk.chunk_id.in_(candidate_ids))
        ).all()
    finally:
        session.close()
    return {row[0]: {"title": row[1], "paper_id": row[2], "section": row[3]} for row in rows}
