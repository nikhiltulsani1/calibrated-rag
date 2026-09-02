from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from opentelemetry import context as otel_context
from sqlalchemy import Text, cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB

from src.index.embed_toggle import get_active_embed_provider
from src.index.embedder import embed_queries
from src.platform.cache import get_json, set_json
from src.platform.telemetry import get_tracer, run_with_otel_context
from src.retrieve.hybrid import ArmResult, FusionTrace, rrf_fuse_with_scores
from src.retrieve.reranker import Candidate
from src.schemas.query_plan import QueryPlan
from src.store.relational import get_session
from src.store.schema import Chunk, Paper

# Real bug found live during the Render deploy check: graph.py's
# hallucinated-category guard (added earlier this session — see
# project-docs/result.md §23) unconditionally called hybrid.py's
# OpenSearch-backed real_categories(), which crashed with
# KeyError('OPENSEARCH_ADMIN_PASSWORD') on RETRIEVAL_BACKEND=postgres
# deployments that have no OpenSearch at all. This is the Postgres
# counterpart, same cache key prefix/TTL discipline as the OpenSearch
# version, so both share Redis without colliding.
_VALID_CATEGORIES_CACHE_PREFIX = "retrieval:valid_categories:postgres"
_VALID_CATEGORIES_TTL_SECONDS = 24 * 60 * 60


def real_categories() -> set[str]:
    """The real, distinct `category` values actually present in the
    shared corpus right now — `category` is a JSON array per paper
    (e.g. ["cs.IR", "cs.CL"]), so this flattens across all papers in
    Python rather than a more elaborate SQL unnest, since the corpus
    size here is small enough that this is genuinely simpler and just
    as fast. Cached 24h, same discipline as the OpenSearch version.

    Scoped to the SHARED corpus only (`owner_session_id IS NULL`) — real
    gap found in review: this result is cached globally, visible to every
    visitor via the query planner. Uploads always set `category=[]`
    today so nothing has actually leaked yet, but without this filter a
    private upload's category would enter the 24h globally-cached set the
    moment that stopped being true, the same class of bug as the
    /corpus leak this stage's review already found and fixed.
    """
    cached = get_json(_VALID_CATEGORIES_CACHE_PREFIX)
    if cached is not None:
        return set(cached)

    session = get_session()
    try:
        rows = session.execute(select(Paper.category).where(Paper.owner_session_id.is_(None))).all()
    finally:
        session.close()
    categories: set[str] = set()
    for (cats,) in rows:
        if cats:
            categories.update(cats)
    set_json(_VALID_CATEGORIES_CACHE_PREFIX, sorted(categories), _VALID_CATEGORIES_TTL_SECONDS)
    return categories

# Phase 2 — the RETRIEVAL_BACKEND=postgres path, used only by the
# free-tier live deployment (src/reason/nodes/retrieve.py picks this
# module over hybrid.py based on that env var). The default
# RETRIEVAL_BACKEND=opensearch path (hybrid.py) is completely untouched
# by this file's existence — see project-docs/architecture.md and the
# Phase 2 plan for why this is a parallel module, not a replacement.
#
# Reuses hybrid.py's rrf_fuse_with_scores/ArmResult/FusionTrace directly
# rather than duplicating the fusion math — that arithmetic doesn't care
# what produced the ranked lists it fuses.

_RECALL_SIZE = 50  # matches hybrid.py's own constant, kept independent on purpose — the two backends are allowed to tune separately


def _owner_predicate(session_id: str | None):
    """The literal private-per-uploader isolation enforcement point (see
    Phase 2 plan §5). Unconditional on every call — NULL rows (the shared
    corpus) are always visible; a non-NULL owner_session_id row is only
    visible to the matching session. Never optional, never caller-toggled.
    """
    if session_id:
        return or_(Chunk.owner_session_id.is_(None), Chunk.owner_session_id == session_id)
    return Chunk.owner_session_id.is_(None)


def _filter_conditions(filters: dict):
    """QueryPlan.filters -> SQLAlchemy predicates against Paper, mirroring
    hybrid.py::build_filter_clauses's semantics on the OpenSearch path —
    same three keys, same "applied identically to every arm" rule.
    """
    conditions = []

    author = filters.get("author")
    if author:
        # No Postgres equivalent of the OpenSearch path's analyzed
        # authors.text partial-name match without also building a
        # generated tsvector over the authors JSON array (out of scope
        # for Phase 2, see the plan) — a coarser ILIKE substring match on
        # the JSON's text representation is an explicit, accepted
        # shortcut on this backend only.
        conditions.append(cast(cast(Paper.authors, JSONB), Text).ilike(f"%{author}%"))

    category = filters.get("category")
    if category:
        conditions.append(cast(Paper.category, JSONB).contains([category]))

    date_from = filters.get("date_from")
    date_to = filters.get("date_to")
    if date_from:
        conditions.append(Paper.published_date >= date_from)
    if date_to:
        conditions.append(Paper.published_date <= date_to)

    return conditions


def _document_conditions(document_id: str | None):
    """Phase 2 (stage 6, uploads): optional single-document scoping —
    defense in depth ON TOP OF, not instead of, `_owner_predicate` above.
    A caller passing someone else's document_id still gets nothing back,
    since `_owner_predicate` is applied unconditionally alongside this,
    not replaced by it — see the Phase 2 plan's §5.
    """
    if document_id:
        return [Chunk.paper_id == document_id]
    return []


def _lexical_search(
    query_text: str, filters: dict, size: int, session_id: str | None, document_id: str | None = None
) -> list[str]:
    session = get_session()
    try:
        tsquery = func.plainto_tsquery("english", query_text)
        stmt = (
            select(Chunk.chunk_id)
            .join(Paper, Chunk.paper_id == Paper.arxiv_id)
            .where(
                Chunk.text_tsv.op("@@")(tsquery),
                _owner_predicate(session_id),
                *_filter_conditions(filters),
                *_document_conditions(document_id),
            )
            .order_by(func.ts_rank(Chunk.text_tsv, tsquery).desc())
            .limit(size)
        )
        return [row[0] for row in session.execute(stmt).all()]
    finally:
        session.close()


def _dense_search(
    vector: list[float],
    filters: dict,
    size: int,
    session_id: str | None,
    embedding_provider: str,
    document_id: str | None = None,
) -> list[str]:
    session = get_session()
    try:
        stmt = (
            select(Chunk.chunk_id)
            .join(Paper, Chunk.paper_id == Paper.arxiv_id)
            .where(
                Chunk.embedding.is_not(None),
                Chunk.embedding_provider == embedding_provider,
                _owner_predicate(session_id),
                *_filter_conditions(filters),
                *_document_conditions(document_id),
            )
            .order_by(Chunk.embedding.cosine_distance(vector))
            .limit(size)
        )
        return [row[0] for row in session.execute(stmt).all()]
    finally:
        session.close()


def retrieve_with_trace(
    plan: QueryPlan, *, top_n: int = 50, session_id: str | None = None, document_id: str | None = None
) -> tuple[list[Candidate], FusionTrace]:
    """The RETRIEVAL_BACKEND=postgres counterpart of hybrid.py's function
    of the same name — same contract (Candidate list + FusionTrace), same
    fan-out-across-variants-then-RRF-fuse shape, different storage
    backend underneath. `session_id`, unlike the OpenSearch path, is a
    real, explicit, load-bearing parameter here: it's the private-
    per-uploader isolation filter, not an afterthought — see
    _owner_predicate above and the Phase 2 plan's §5.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("hybrid_postgres.retrieve") as span:
        span.set_attribute("hybrid_postgres.has_filters", bool(plan.filters))
        span.set_attribute("hybrid_postgres.session_scoped", session_id is not None)

        query_variants = [plan.normalized, *plan.expansions]
        span.set_attribute("hybrid_postgres.num_query_variants", len(query_variants))

        embedding_provider = get_active_embed_provider()
        embed_result = embed_queries(query_variants, provider=embedding_provider)
        variant_vectors = embed_result.vectors

        current_ctx = otel_context.get_current()
        with ThreadPoolExecutor(max_workers=8) as executor:
            lexical_futures = [
                executor.submit(
                    run_with_otel_context, current_ctx, _lexical_search, variant, plan.filters, _RECALL_SIZE, session_id, document_id
                )
                for variant in query_variants
            ]
            dense_futures = [
                executor.submit(
                    run_with_otel_context,
                    current_ctx,
                    _dense_search,
                    vector,
                    plan.filters,
                    _RECALL_SIZE,
                    session_id,
                    embedding_provider,
                    document_id,
                )
                for vector in variant_vectors
            ]
            lexical_results = [f.result() for f in lexical_futures]
            dense_results = [f.result() for f in dense_futures]

        arms: list[ArmResult] = []
        for variant, lexical_ids, dense_ids in zip(query_variants, lexical_results, dense_results):
            arms.append(ArmResult(variant=variant, arm="lexical", ranked_ids=lexical_ids))
            arms.append(ArmResult(variant=variant, arm="dense", ranked_ids=dense_ids))

        fused_scores = rrf_fuse_with_scores([arm.ranked_ids for arm in arms])
        fused_order = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]
        trace = FusionTrace(arms=arms, fused_scores=fused_scores, fused_order=fused_order, dense_index_name="postgres:chunks")

        fused_ids = fused_order[:top_n]
        if not fused_ids:
            span.set_attribute("hybrid_postgres.num_results", 0)
            return [], trace

        # Defense in depth, found in review: fused_ids were already
        # correctly owner/document-scoped by _lexical_search/_dense_search
        # above, but re-applying the same predicates here means this final
        # hydration step can never return another session's text even if
        # a future change to chunk_id generation (e.g. a content-only hash
        # shared across documents) ever produced a cross-session id
        # collision — the isolation boundary isn't allowed to depend on
        # chunk_id uniqueness being perfect.
        session = get_session()
        try:
            rows = session.execute(
                select(Chunk.chunk_id, Chunk.text).where(
                    Chunk.chunk_id.in_(fused_ids),
                    _owner_predicate(session_id),
                    *_document_conditions(document_id),
                )
            ).all()
        finally:
            session.close()
        by_id = {row[0]: row[1] for row in rows}
        results = [Candidate(id=doc_id, text=by_id[doc_id]) for doc_id in fused_ids if doc_id in by_id]
        span.set_attribute("hybrid_postgres.num_results", len(results))
        return results, trace
