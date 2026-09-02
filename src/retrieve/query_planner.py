from __future__ import annotations

import json
import logging
import os
import time

import httpx

from src.guardrails.base import overlap_ratio
from src.platform.cache import build_cache_key, get_json, set_json
from src.platform.models import complete, is_retryable_error, usable_ladder
from src.platform.telemetry import get_tracer
from src.schemas.query_plan import QueryPlan

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_PREFIX = "query_plan:"

# A3's own report showed this single LLM call costing ~5.5s p50 on top
# of the rest of the pipeline (144ms BM25-only -> 7173ms full pipeline)
# for a real but small quality gain (+0.037 nDCG@10) — and it compounds,
# since every expansion this call emits becomes its own embed + lexical
# + dense round-trip downstream in hybrid.py. QUERY_REWRITE_MODE lets
# this be skipped for queries that don't need it, same toggle
# convention as CHUNKING_STRATEGY/EMBED_PROVIDER — read fresh per call
# (not bound at import time) so it behaves like guardrail_mode(), a
# live-switchable setting, not a fixed-at-startup constant.
_MIN_WORDS_TO_SKIP_REWRITE = 6

# A tiny, deliberately generic anchor vocabulary for the domain this
# corpus covers (CS research papers) — NOT the corpus's own vocabulary,
# which would need a live OpenSearch/Postgres call defeating the point
# of a cheap pre-check. Same deterministic-first-escalate-when-
# inconclusive shape as assess_context's overlap_ratio use: this is not
# meant to be a precise classifier, only to fail toward "run the real
# call" whenever the query doesn't obviously already look like a
# specific, in-domain, unambiguous question.
_DOMAIN_ANCHOR_TERMS = (
    "paper model dataset algorithm training evaluation benchmark method "
    "approach system study research results analysis experiment "
    "architecture retrieval embedding reasoning agent language corpus "
    "baseline metric accuracy performance framework pipeline task"
)


def _should_skip_rewrite(query: str) -> bool:
    """Deterministic, conservative pre-check — skip only when there is
    real reason to believe the query doesn't need normalization,
    expansion, or filter extraction. Fails toward NOT skipping (i.e.
    toward running the real LLM call) on any doubt, matching this
    project's guardrail-safety convention of failing toward the safer
    behavior rather than the cheaper one.

    Three independent reasons not to skip, any one of which is enough:
    1. Too short/sparse to already be a specific, well-formed question.
    2. Contains a bare short all-caps token (a likely unexpanded
       acronym) — expanding those is exactly rewrite's job.
    3. Shares no vocabulary at all with the domain anchor terms — this
       is also the LLM call's only signal for out_of_scope
       classification (screen_scope depends on plan.intent), so a query
       that doesn't even weakly resemble the domain must still go
       through the real call rather than silently default to "factual".

       A full-codebase review (2026-08-25) flagged this exact `> 0.0`
       bar as too permissive — a genuinely off-topic query sharing one
       ordinary English word with the anchor list could clear it. Tried
       raising it to "at least 2 distinct shared tokens" and checked
       that fix against real data before shipping it, per this
       project's own discipline: it made things WORSE, not better. A
       real off-topic probe ("What's the best method for training a dog
       to sit calmly?") shares 2 anchor words ("method", "training") —
       still clears a >=2 bar — while the existing true-positive test
       case ("What method does AnnoIndex use to build its annotation
       index for structured filtering?") shares only 1 ("method") and
       would have stopped skipping, silently losing A2's real measured
       latency win for a genuinely on-topic, specific question. The
       anchor-word-overlap signal itself is too weak/noisy on this small
       generic list to reliably separate domain-relevance by any simple
       threshold — proven by the off-topic probe scoring a *higher*
       overlap ratio (0.33) than the real true positive (0.1). Reverted
       to the original `> 0.0` gate rather than ship a fix that provably
       trades one real gap for a different, real regression; the actual
       downstream blast radius of a rare off-topic false-skip is bounded
       anyway — `context_sufficiency`/`groundedness` still catch a
       genuinely irrelevant retrieval later in the pipeline. Left open
       as a real, lower-priority item — a fix here needs a stronger
       signal than generic-word overlap (e.g. real embedding similarity
       against the domain), not another threshold guess.
    """
    words = query.strip().split()
    if len(words) < _MIN_WORDS_TO_SKIP_REWRITE:
        return False
    if any(len(w) in (2, 3, 4) and w.isupper() for w in words):
        return False
    if overlap_ratio(query, _DOMAIN_ANCHOR_TERMS) <= 0.0:
        return False
    return True

# Authored from scratch for this system's schema and domain — see the IP
# posture note on prompts in the plan.
_SYSTEM_PROMPT = """You are the query-understanding stage of a retrieval system \
over a corpus of arXiv computer-science papers. Given a user's question, \
respond with a single JSON object with exactly these fields:

- "normalized": the query rewritten with standard spelling, with common \
  domain acronyms expanded in parentheses on first use (e.g. "LLM" -> \
  "LLM (large language model)"), and obvious typos fixed. Do not change \
  the question's meaning.
- "expansions": a JSON array of 2 to 3 alternative phrasings of the same \
  question, each a full standalone question a search engine could run \
  independently. Vary vocabulary and phrasing, not intent.
- "filters": a JSON object with any of these keys the question implies, \
  omitted entirely if none are implied: "author" (string), "category" \
  (string, an arXiv category like "cs.CL"), "date_from" and "date_to" \
  (ISO 8601 dates).
- "intent": exactly one of "factual", "comparative", "multi_hop", \
  "out_of_scope". Use "out_of_scope" only if the question has nothing to \
  do with computer-science research papers.

Respond with only the JSON object, no other text."""


def _build_system_prompt(known_categories: frozenset[str] | None) -> str:
    """Constrains the "category" filter at extraction time, not just
    downstream — real bug found live 2026-08-24 (see Results.md §23):
    the base prompt above never told the model what a *real* category in
    this corpus actually is, so it would guess a plausible-sounding but
    wrong one (e.g. "cs.LG" for a paper really filed as cs.IR/cs.CL),
    and `build_filter_clauses`'s exact-match category clause then zeroed
    out every retrieval arm silently. `hybrid.py`'s `_sanitize_filters`
    already catches this downstream and is left completely unchanged —
    defense in depth, not replaced — but preventing the guess in the
    first place is strictly better than dropping it after the fact: a
    dropped category still means the model's *other* extraction (e.g. a
    misread author or date range) went untouched, and the model itself
    never gets a chance to omit the filter cleanly instead of guessing.

    `known_categories` is the caller's responsibility to fetch (see
    `hybrid.real_categories`, already Redis-cached 24h for this exact
    purpose) — this function only formats it into the prompt, matching
    `metadata`'s injection pattern in assess.py: real corpus data belongs
    to the caller that already has it, not re-fetched here.
    """
    if not known_categories:
        return _SYSTEM_PROMPT
    categories_line = ", ".join(sorted(known_categories))
    return _SYSTEM_PROMPT + (
        f'\n\nThe "category" filter, if you include one, MUST be exactly one of these '
        f"real arXiv categories actually present in this corpus: {categories_line}. "
        'If you are not confident which one applies, omit "category" from filters '
        "entirely rather than guessing — an omitted filter still lets normal retrieval "
        "run; a wrong one silently returns nothing."
    )


def _cache_key(query: str) -> str:
    # Keyed on the raw query text (normalized just enough for cache
    # hygiene: trimmed, lowercased), NOT on QueryPlan.normalized — that
    # field is the LLM call's *output*, so keying on it would require the
    # call this cache exists to avoid.
    return build_cache_key(_CACHE_PREFIX, query.strip().lower())


def _parse_plan(original: str, raw_content: str) -> QueryPlan:
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"rewrite model did not return valid JSON: {raw_content!r}") from exc

    data["original"] = original
    data.setdefault("filters", {})
    try:
        return QueryPlan.model_validate(data)
    except Exception as exc:
        raise ValueError(f"rewrite model returned JSON that doesn't match QueryPlan: {data!r}") from exc


def plan_query(query: str, *, use_cache: bool = True, known_categories: frozenset[str] | None = None) -> QueryPlan:
    """The A1 pre-retrieval stage: one LLM call producing the whole
    QueryPlan (normalization, expansions, filters, intent), cached by the
    query text so a repeat query costs no second round-trip.

    Distinct from the agentic graph's corrective rewrite_query_node, which
    fires only after retrieval grading fails — this one runs before
    OpenSearch is ever touched. Routing on `intent` (short-circuiting
    out_of_scope straight to the decline node) is the caller's job, not
    this function's — this only classifies.

    `known_categories`, when given, is injected into the system prompt so
    the model can't hallucinate a plausible-but-wrong category (see
    `_build_system_prompt`'s docstring). Deliberately NOT part of the
    cache key: it changes at most once per corpus ingestion (same 24h
    cache window as `hybrid.real_categories` itself), so a cached plan
    from just before a category set changed staying valid for the
    remainder of its own 24h TTL is the same bounded, already-accepted
    staleness as every other toggle-independent cache in this codebase —
    and `hybrid._sanitize_filters` remains as an unconditional downstream
    safety net regardless.
    """
    tracer = get_tracer()
    # No manual try/except — start_as_current_span records an uncaught
    # exception and sets ERROR status by default; see models.py::complete.
    with tracer.start_as_current_span("query_planner.plan_query") as span:
        span.set_attribute("query_planner.use_cache", use_cache)

        cache_key = _cache_key(query)
        if use_cache:
            cached = get_json(cache_key)
            if cached is not None:
                logger.info("query_planner cache hit")
                span.set_attribute("query_planner.cache_hit", True)
                plan = QueryPlan.model_validate(cached)
                span.set_attribute("query_planner.intent", plan.intent)
                return plan
            logger.info("query_planner cache miss")
            span.set_attribute("query_planner.cache_hit", False)

        rewrite_mode = os.environ.get("QUERY_REWRITE_MODE", "conditional")
        skip_rewrite = rewrite_mode == "conditional" and _should_skip_rewrite(query)
        span.set_attribute("query_planner.skipped_rewrite", skip_rewrite)
        if skip_rewrite:
            # Not "degraded" — degraded=True means the real call was
            # attempted and failed (see below); this is a deliberate,
            # working decision to not attempt it. Still cached like any
            # other plan (use_cache governs this normally) so a repeat
            # of the same query doesn't re-run even the cheap heuristic.
            plan = QueryPlan(original=query, normalized=query, expansions=[], filters={}, intent="factual")
            span.set_attribute("query_planner.intent", plan.intent)
            span.set_attribute("query_planner.num_expansions", 0)
            if use_cache:
                set_json(cache_key, plan.model_dump(), _CACHE_TTL_SECONDS)
            return plan

        messages = [
            {"role": "system", "content": _build_system_prompt(known_categories)},
            {"role": "user", "content": query},
        ]
        # Groq's json_mode guarantees valid JSON, not a matching schema,
        # and its own server-side generation can flat-out fail. Verified
        # live (2026-08-16): one real query dropped "intent" in 3 of 4
        # identical temperature=0.0 calls, and a *different* real query
        # returned Groq's own `json_validate_failed` 400 in 4 of 5 calls —
        # a genuine per-query reliability floor for this model, not a
        # rare fluke. Same-model retries were found to often reproduce
        # the identical failure rather than being independent tries, so
        # retries here descend the ladder to a genuinely different model
        # (only among rungs with a configured key — see usable_ladder)
        # instead of re-asking the one that just failed the same way.
        ladder = usable_ladder("rewrite")
        last_error: Exception | None = None
        plan: QueryPlan | None = None
        for attempt in range(3):
            rung = ladder[min(attempt, len(ladder) - 1)]
            try:
                result = complete("rewrite", messages=messages, temperature=0.0, json_mode=True, model_override=rung)
            except httpx.HTTPError as exc:
                # httpx.HTTPError (not just HTTPStatusError) — real bug
                # found live 2026-08-24: a bare connection reset/timeout
                # (httpx.TransportError, a SIBLING of HTTPStatusError, not
                # a subclass) used to propagate straight out uncaught
                # instead of retrying/degrading, despite the exact real
                # evidence below (10 of 14 real failures in one run) being
                # precisely that failure class. See is_retryable_error's
                # own docstring in platform/models.py.
                last_error = exc
                if not is_retryable_error(exc) or attempt == 2:
                    break
                time.sleep(2**attempt)
                continue
            try:
                plan = _parse_plan(query, result.content)
                break
            except ValueError as exc:
                last_error = exc

        if plan is None:
            # Every attempt failed. Verified live (2026-08-16) this isn't
            # always a content problem: in one real 80-question run, 10 of
            # 14 failures on this stage were plain network connection
            # resets talking to the fallback provider's free tier, not bad
            # model output at all. No amount of picking "the right" free
            # model changes that a free tier can be transiently down.
            # Degrading to a safe, unrewritten plan instead of crashing
            # the whole request matches RerankResult's own
            # degrade-instead-of-fail design (see reranker.py). `intent`
            # defaults to "factual", never "out_of_scope" — the same
            # fail-open reasoning the input guardrails already use: a
            # broken classifier must not itself become a
            # denial-of-service or silently decline a legitimate
            # question. A genuinely unanswerable question still gets
            # caught downstream by generation's own abstention — it just
            # loses the benefit of query expansion for this one request.
            logger.warning(
                "query_planner degraded to an unrewritten plan: %s",
                last_error,
                extra={"event": "query_planner_degraded"},
            )
            plan = QueryPlan(
                original=query,
                normalized=query,
                expansions=[],
                filters={},
                intent="factual",
                degraded=True,
                degrade_reason=str(last_error),
            )

        span.set_attribute("query_planner.intent", plan.intent)
        span.set_attribute("query_planner.num_expansions", len(plan.expansions))
        span.set_attribute("query_planner.degraded", plan.degraded)

        # A degraded plan is never cached — it's a one-request fallback
        # for a transient failure, not a real classification worth
        # reusing for 24h on every repeat of this query.
        if use_cache and not plan.degraded:
            set_json(cache_key, plan.model_dump(), _CACHE_TTL_SECONDS)

        return plan
