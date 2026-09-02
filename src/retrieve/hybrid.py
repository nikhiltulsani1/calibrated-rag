from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from opentelemetry import context as otel_context

from src.index.client import get_client
from src.index.embed_toggle import provider_to_index
from src.index.embedder import embed_queries, embed_queries_with_fallback
from src.index.mapping import INDEX_NAME
from src.platform.cache import get_json, set_json
from src.platform.telemetry import get_tracer, run_with_otel_context
from src.retrieve.reranker import Candidate
from src.schemas.query_plan import QueryPlan

_RRF_K = 60
_RECALL_SIZE = 50  # per arm, per query variant, before fusion — see A2

# Real bug found live 2026-08-24 (A8's first genuinely complete run): the
# rewrite LLM sometimes extracts a plausible-*sounding* arXiv category
# that was never actually assigned to the real paper it's talking about
# (e.g. "cs.LG" for a retrieval paper actually filed as cs.IR/cs.CL) —
# `build_filter_clauses`'s category clause is a hard, exact `term` match
# by design (arXiv codes are exact strings, no fuzziness wanted — see
# that function's own docstring), so a hallucinated category zeroes out
# every single retrieval arm, producing a real, silent 0-candidate
# result that then abstains "correctly" given genuinely empty context.
# Two real over-refusal cases (q025: real category cs.IR/cs.CL,
# extracted cs.LG; q073: real category cs.CV/cs.AI/cs.CL, extracted
# cs.SE) were traced to exactly this. Neither is a B1-B4/assess.py
# abstention-logic defect — the abstention decision itself was correct
# given the (wrongly) empty context it was handed.
_VALID_CATEGORIES_CACHE_PREFIX = "retrieval:valid_categories:"
_VALID_CATEGORIES_TTL_SECONDS = 24 * 60 * 60


def real_categories(client, index_name: str) -> set[str]:
    """The real, distinct `category` values actually present in this
    index right now — cached 24h (same discipline as query_planner's
    plan cache) since the corpus changes rarely and this would otherwise
    add a real extra OpenSearch round-trip to every single query.
    """
    cache_key = _VALID_CATEGORIES_CACHE_PREFIX + index_name
    cached = get_json(cache_key)
    if cached is not None:
        return set(cached)

    response = client.search(
        index=index_name,
        body={"size": 0, "aggs": {"cats": {"terms": {"field": "category", "size": 200}}}},
    )
    categories = {bucket["key"] for bucket in response["aggregations"]["cats"]["buckets"]}
    set_json(cache_key, sorted(categories), _VALID_CATEGORIES_TTL_SECONDS)
    return categories


def _sanitize_filters(client, index_name: str, filters: dict) -> dict:
    """Drops a `category` filter that doesn't correspond to any real
    category in this corpus — a hallucinated filter must not silently
    zero out every arm of retrieval. Every other filter key
    (author/date_from/date_to) is untouched: those aren't matched
    against a closed real vocabulary the way arXiv category codes are,
    so there's no equivalent cheap "is this real" check for them.
    """
    category = filters.get("category")
    if not category:
        return filters
    if category in real_categories(client, index_name):
        return filters
    sanitized = dict(filters)
    del sanitized["category"]
    return sanitized


@dataclass(frozen=True)
class ArmResult:
    """One lexical-or-dense search over one query variant, ranked."""

    variant: str
    arm: str  # "lexical" or "dense"
    ranked_ids: list[str]


@dataclass(frozen=True)
class FusionTrace:
    """Everything A5's stages 4-5 need to show: what each arm returned,
    and the RRF arithmetic that combined them — captured directly from a
    real retrieve() call, not reconstructed after the fact.

    Note this can be MORE than the plan's illustrative "two columns"
    (lexical vs dense) when the query plan has expansions (A1): each
    expansion is its own variant, each variant has both arms, so there
    can be more than 2 ArmResults here. The plan's 2-column sketch
    describes a single-query pipeline; this one fans out across
    expansions too, per A1's own stated design.
    """

    arms: list[ArmResult]
    fused_scores: dict[str, float]
    fused_order: list[str]
    # Which index the dense/kNN arm actually queried — may differ from
    # the caller's `index_name` if embed-provider failover engaged (see
    # retrieve_with_trace). Exposed so a caller (e.g. the answer cache)
    # can detect a real fallback and react to it, rather than only the
    # OTel span attribute knowing this happened.
    dense_index_name: str


def build_filter_clauses(filters: dict) -> list[dict]:
    """QueryPlan.filters -> OpenSearch bool `filter` clauses.

    This is the piece that makes A1's extracted filters do anything at
    all — verified live that without it, `filters` was dead data the
    rewrite model produces and nothing consumes (grepped the whole src/
    tree; zero query-construction code referenced it). Applied identically
    to every lexical and dense sub-query below, not just one arm.
    """
    clauses: list[dict] = []

    author = filters.get("author")
    if author:
        # authors.text (analyzed) rather than the plain authors keyword —
        # an extracted "LeCun" needs to match a stored "Yann LeCun", which
        # a keyword-exact filter cannot do. See the mapping.py comment.
        clauses.append({"match": {"authors.text": author}})

    category = filters.get("category")
    if category:
        # arXiv category codes (cs.CL, cs.IR, ...) are exact strings —
        # keyword term match is correct here, no fuzziness wanted.
        clauses.append({"term": {"category": category}})

    date_from = filters.get("date_from")
    date_to = filters.get("date_to")
    if date_from or date_to:
        date_range: dict = {}
        if date_from:
            date_range["gte"] = date_from
        if date_to:
            date_range["lte"] = date_to
        clauses.append({"range": {"published_date": date_range}})

    return clauses


def rrf_fuse_with_scores(ranked_lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion, k=60 per the plan's full-pipeline spec —
    returns the per-doc score dict, which rrf_fuse() below reduces to
    just the ordering. Separated so A5's fusion-arithmetic panel can show
    the actual score, not just the resulting rank.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def rrf_fuse(ranked_lists: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Pure function — no network, no OpenSearch — deliberately, so the
    fusion arithmetic is testable without a live index."""
    scores = rrf_fuse_with_scores(ranked_lists, k)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _lexical_search(client, query_text: str, filter_clauses: list[dict], size: int, index_name: str = INDEX_NAME) -> list[str]:
    body = {
        "size": size,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query_text, "fields": ["text", "title^2"]}}],
                "filter": filter_clauses,
            }
        },
        "_source": False,
    }
    response = client.search(index=index_name, body=body)
    return [hit["_id"] for hit in response["hits"]["hits"]]


def _dense_search(client, vector: list[float], filter_clauses: list[dict], size: int, index_name: str = INDEX_NAME) -> list[str]:
    # Takes an already-computed vector, not query text — plan A3: the
    # caller (retrieve_with_trace) embeds every query variant in ONE
    # embed_queries() call instead of one embed_query() call per
    # variant here. embed_queries already existed (built for A7's
    # ablation, same real N-calls-per-strategy problem), it just wasn't
    # wired into the live query path until this pipeline's own latency
    # was actually measured (A3's report: full pipeline p50 7173ms).
    body = {
        "size": size,
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": vector, "k": size}}}],
                "filter": filter_clauses,
            }
        },
        "_source": False,
    }
    response = client.search(index=index_name, body=body)
    return [hit["_id"] for hit in response["hits"]["hits"]]


def retrieve_with_trace(plan: QueryPlan, *, top_n: int = 50, index_name: str = INDEX_NAME) -> tuple[list[Candidate], FusionTrace]:
    """The real implementation — retrieve() below is a thin wrapper that
    discards the trace, kept so every existing caller/test that only
    wants the final candidate list is unaffected by this addition.

    `index_name` defaults to the production `rag_chunks` index — every
    existing caller is unaffected. A7's chunking-strategy toggle (see
    src/reason/nodes/retrieve.py) is the one caller that passes a
    different, permanently-indexed strategy variant here.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("hybrid.retrieve") as span:
        client = get_client()
        sanitized_filters = _sanitize_filters(client, index_name, plan.filters)
        dropped_category = "category" in plan.filters and "category" not in sanitized_filters
        span.set_attribute("hybrid.dropped_hallucinated_category", dropped_category)
        if dropped_category:
            span.set_attribute("hybrid.hallucinated_category", plan.filters["category"])
        filter_clauses = build_filter_clauses(sanitized_filters)
        span.set_attribute("hybrid.has_filters", bool(filter_clauses))
        span.set_attribute("hybrid.index_name", index_name)

        query_variants = [plan.normalized, *plan.expansions]
        span.set_attribute("hybrid.num_query_variants", len(query_variants))

        # One embed call for every variant, not one per variant — see
        # _dense_search's docstring. embed_queries_with_fallback
        # preserves input order (verified by embedder.py's own
        # reorder-by-index test), so vectors[i] belongs to
        # query_variants[i].
        #
        # Automatic Jina->Mistral embed failover, added 2026-08-23 —
        # dense_index_name is whichever index actually matches the
        # provider that served this embed call, which may differ from
        # `index_name` if a fallback engaged. Lexical search and the
        # later mget both stay on `index_name` unchanged — safe ONLY
        # among the embed-provider variant indices of the SAME corpus
        # (rag_chunks vs rag_chunks_mistral_embed — same real Postgres
        # chunk text, same real chunk_ids).
        #
        # Real bug found live 2026-08-24: that safety does NOT extend to
        # A7's chunking-strategy variant indices (rag_chunks_winner/
        # _median/_efficient — DIFFERENT chunk boundaries, DIFFERENT
        # chunk_ids, built jina-embedded only). Letting embed failover
        # silently redirect the dense step to rag_chunks_mistral_embed
        # while `index_name` was e.g. rag_chunks_winner meant every
        # dense-arm hit failed the final mget's `found` check and was
        # silently dropped — the whole dense arm evaporated with no
        # error. `index_name in (INDEX_NAME, mistral's default-corpus
        # index)` is exactly the "default corpus" test: true for the two
        # embed-provider variants of the default corpus (failover is
        # safe there), false for any A7 chunking-strategy corpus (which
        # is jina-only — embed there must fail loud on a real Jina
        # failure, per A0, not silently mismatch).
        if index_name in (INDEX_NAME, provider_to_index("mistral")):
            embed_result, dense_index_name = embed_queries_with_fallback(query_variants)
        else:
            embed_result = embed_queries(query_variants, provider="jina")
            dense_index_name = index_name
        variant_vectors = embed_result.vectors
        span.set_attribute("hybrid.dense_index_name", dense_index_name)

        # Plan A4: lexical and dense search are independent per variant,
        # and independent across variants — 2N sequential round-trips
        # collapsed into one wave of concurrent calls. max_workers caps
        # at a small, fixed number rather than 2N unbounded threads;
        # this project's real query plans have at most ~4 variants
        # (1 normalized + up to 3 A1 expansions), so 8 is already
        # generous headroom, not a load-bearing tuned value.
        current_ctx = otel_context.get_current()
        with ThreadPoolExecutor(max_workers=8) as executor:
            lexical_futures = [
                executor.submit(
                    run_with_otel_context, current_ctx, _lexical_search, client, variant, filter_clauses, _RECALL_SIZE, index_name
                )
                for variant in query_variants
            ]
            dense_futures = [
                executor.submit(
                    run_with_otel_context, current_ctx, _dense_search, client, vector, filter_clauses, _RECALL_SIZE, dense_index_name
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
        trace = FusionTrace(
            arms=arms, fused_scores=fused_scores, fused_order=fused_order, dense_index_name=dense_index_name
        )

        fused_ids = fused_order[:top_n]
        if not fused_ids:
            span.set_attribute("hybrid.num_results", 0)
            return [], trace

        docs = client.mget(index=index_name, body={"ids": fused_ids})["docs"]
        by_id = {doc["_id"]: doc["_source"] for doc in docs if doc.get("found")}

        results = [Candidate(id=doc_id, text=by_id[doc_id]["text"]) for doc_id in fused_ids if doc_id in by_id]
        span.set_attribute("hybrid.num_results", len(results))
        return results, trace


def retrieve(plan: QueryPlan, *, top_n: int = 50, index_name: str = INDEX_NAME) -> list[Candidate]:
    """rewrite -> **hybrid retrieve** -> rerank.

    BM25 + dense recall over the normalized query and every A1 expansion,
    RRF-fused (k=60) into one ranked list, top `top_n` handed to the
    reranker. The metadata filters A1 extracted are applied identically to
    every lexical and dense sub-query across every query variant — a
    filtered query stays filtered through every arm of the fan-out, not
    just the first one.
    """
    results, _trace = retrieve_with_trace(plan, top_n=top_n, index_name=index_name)
    return results
