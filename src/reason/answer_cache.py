from __future__ import annotations

import os

from src.index.embed_toggle import get_active_embed_provider, provider_to_index
from src.platform.cache import build_cache_key, get_json, set_json
from src.reason.chunking_toggle import get_active_strategy
from src.reason.state import StageTrace
from src.store.runs import serialize_trace

# A6: a semantic answer cache, wrapping the HTTP routes only — never
# run_traced_query/run_graph itself. Every eval script (run_abstention_
# eval.py, run_generation_eval.py, ...) calls run_traced_query directly
# for a genuinely fresh measurement every time; caching inside that
# function would silently corrupt every one of those real numbers. So
# this lives entirely in src/app/routes/ask.py and pipeline.py, which
# call get_cached_trace()/set_cached_trace() around the unchanged
# run_traced_query call — zero risk to evals, zero change to run_graph's
# contract.

_CACHE_PREFIX = "answer_cache:"
# Shorter than query_planner's 24h plan-cache TTL: an *answer* can go
# stale faster than a query plan — a live guardrail-mode toggle flip or
# a corpus edit mid-TTL would otherwise keep serving a now-stale answer.
# Env-overridable like CONTEXT_SUFFICIENCY_OVERLAP_THRESHOLD, so it's
# tunable without a code change. A guardrail-mode flip during the TTL
# window can still serve a stale-but-bounded answer — an accepted
# tradeoff, same class as A5's documented one, not hidden.
_ANSWER_CACHE_TTL_SECONDS = int(os.environ.get("ANSWER_CACHE_TTL_SECONDS", "3600"))


def _cache_key(query: str, *, session_id: str | None = None) -> str:
    # Includes the active chunking strategy and embed provider, not just
    # the query text — the same query against a different toggle is a
    # genuinely different corpus/vector-space answer (exactly the
    # failure class embed_toggle.py exists to prevent elsewhere).
    # Resolved fresh per call, not bound at import time, so a live
    # toggle flip is reflected on the very next request.
    #
    # Phase 2 (stage 6, uploads): `session_id`, when given, is folded into
    # the key too — a real cross-visitor leak found while wiring up
    # private uploads, not a hypothetical one: this cache is a single
    # shared Redis keyspace, and a session-scoped answer (one that may
    # cite a visitor's own private upload) cached under the *plain* query
    # text would be served verbatim to a different visitor who happens to
    # type the identical question — exactly the kind of leak the Phase 2
    # plan's multi-visitor isolation test is designed to catch. Callers
    # only pass session_id on the postgres/BYOK path (see routes/ask.py,
    # routes/pipeline.py); the default OpenSearch path has no private
    # data at all, so its cache key is byte-for-byte unchanged.
    normalized = query.strip().lower()
    strategy = get_active_strategy()
    provider = get_active_embed_provider()
    if session_id:
        return build_cache_key(_CACHE_PREFIX, normalized, strategy, provider, session_id)
    return build_cache_key(_CACHE_PREFIX, normalized, strategy, provider)


def get_cached_trace(query: str, *, session_id: str | None = None) -> dict | None:
    """Returns the cached, serialized trace dict for this exact query +
    active toggles (+ session, on the postgres path), or None on a miss.
    The dict shape matches store.runs.load_run()'s return value exactly
    (same serialize_trace), so callers can treat a cache hit identically
    to a Postgres run_id replay — pipeline.html already renders that
    shape with zero changes.
    """
    return get_json(_cache_key(query, session_id=session_id))


def set_cached_trace(query: str, trace: StageTrace, *, session_id: str | None = None) -> None:
    """Real bug found live 2026-08-24: `_cache_key` reads the *nominal*
    `get_active_embed_provider()` toggle, but `embed_queries_with_fallback`
    (src/index/embedder.py) can silently serve a request off the OTHER
    provider — its own designed failover, no error surfaced, the toggle
    itself never flips. Caching that trace under the nominal key would
    poison it: a later identical query, once the nominal provider is
    healthy again, would hit this entry and get an answer built from the
    *other* provider's vector space — exactly the cross-corpus
    contamination the toggle-aware cache key exists to prevent.

    `trace.fusion.dense_index_name` (set by retrieve_with_trace) is the
    index the dense arm actually queried — compared here against what
    the nominal provider *should* have produced. A mismatch means a real
    fallback engaged for this specific request; skip caching it rather
    than caching it under a key that no longer describes its own content.
    A trace with no fusion data (a short-circuited guardrail decline,
    e.g. scope_screening) never touched embed at all, so there's nothing
    to mismatch — always safe to cache.

    Real bug found in review (Phase 2 stage 6): this guard only makes
    sense on the RETRIEVAL_BACKEND=opensearch path — `provider_to_index`
    only ever returns an OpenSearch index name, but
    `hybrid_postgres.retrieve_with_trace` always sets
    `dense_index_name="postgres:chunks"`, which can never equal an
    OpenSearch index name. Ungated, this `return` fired on every single
    postgres-backend query that reached retrieval, silently disabling the
    entire answer cache (including this stage's own session-scoped
    isolation fix) on the one backend BYOK/uploads actually runs on.
    hybrid_postgres.py never uses embed_queries_with_fallback — every
    call there passes an explicit provider — so there's no silent
    cross-provider fallback to guard against on that path at all; the
    check is simply inapplicable there, not just differently-keyed.
    """
    if trace.fusion is not None and os.environ.get("RETRIEVAL_BACKEND", "opensearch") != "postgres":
        expected_index = provider_to_index(get_active_embed_provider())
        if trace.fusion.dense_index_name != expected_index:
            return
    set_json(_cache_key(query, session_id=session_id), serialize_trace(trace), _ANSWER_CACHE_TTL_SECONDS)
