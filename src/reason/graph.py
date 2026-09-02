from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from opentelemetry import context as otel_context

from src.guardrails.base import guardrail_mode
from src.platform.credentials import get_credentials, reset_credentials, set_credentials
from src.platform.telemetry import get_tracer, run_with_otel_context
from src.index.client import get_client
from src.index.embed_toggle import active_embed_index_name, get_active_embed_provider
from src.reason.chunking_toggle import active_index_name, get_active_strategy
from src.reason.nodes.answer import run_answer
from src.reason.nodes.assess import assess_ambiguity, assess_context
from src.reason.nodes.decline import ungrounded, out_of_scope, query_too_large
from src.reason.nodes.refine import widen
from src.reason.nodes.retrieve import run_retrieve, run_rerank_and_metadata
from src.reason.nodes.screen import check_size_and_shape, plan_query, screen_scope
from src.reason.state import AttemptTrace, StageTrace
from src.retrieve.hybrid import real_categories

# Self-correction loop: how many total passes (first attempt + retries) to
# allow, and how much to widen the context window on each retry. Starting
# points, not tuned values — no real traffic yet to tune them against, same
# honesty rule as GROUNDEDNESS_OVERLAP_THRESHOLD in output_guardrails.py.
_MAX_ATTEMPTS = int(os.environ.get("RAG_MAX_ATTEMPTS", "2"))
_CONTEXT_WIDEN_STEP = 4
_CONTEXT_MAX = 16


@contextmanager
def _timed(timings: dict[str, float], name: str):
    """Real wall-clock timing for one stage, accumulated into `timings`
    (plan A1 — see StageTrace.stage_timings' docstring). Summed rather
    than overwritten so a stage that runs more than once across the
    refine/retry loop (rerank, both assess calls, generate) reports its
    real total cost for the run, not just its last attempt's cost —
    what actually explains a slow response is the sum, not one sample.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings[name] = timings.get(name, 0.0) + elapsed_ms


def run_graph(query: str, *, top_n_context: int = 8, session_id: str | None = None, document_id: str | None = None) -> StageTrace:
    """The real orchestrator: screen -> retrieve -> (rerank -> assess ->
    [refine loop] -> answer) -> decline-on-final-groundedness-failure.

    This is the same sequence `run_traced_query` has always run — moved
    here out of pipeline.py, decomposed into the node functions imported
    above, plus one genuinely new step: `assess_context` runs on the
    reranked context *before* generate is called, closing the gap between
    today's retry-only loop (which only reacts after generate already
    failed) and the plan's intended screen->retrieve->assess->refine->
    answer shape.

    `assess_context`'s effect is guarded by `guardrail_mode("context_
    sufficiency")` exactly like every other guardrail here — in the
    default `monitor` mode it only traces (`reason.assess.sufficient`),
    it never skips a generate call, so today's exact behavior is
    unchanged until that guardrail is explicitly promoted to `enforce`.

    `assess_ambiguity` (added for A8's false-premise/ambiguity finding)
    runs alongside it, same rollout discipline: in `monitor` mode it
    computes and traces the real signal but generate never sees it; only
    in `enforce` mode does its note actually reach generate_answer. This
    is a real, per-query LLM call with no cheap deterministic shortcut
    (unlike assess_context) — a real, accepted quota cost, not hidden.

    Index selection composes TWO independent toggles: A7's chunking
    strategy (src/reason/chunking_toggle.py) and the embed-provider
    switch (src/index/embed_toggle.py, added 2026-08-22). Both need a
    matching index (different chunk boundaries / different vector spaces
    aren't interchangeable), and building the full 4x2 cross-product of
    indices is out of scope — so a non-default chunking strategy always
    wins and stays on its own (jina-embedded) index; the embed-provider
    switch only applies to the "default" strategy's corpus, which is
    what every eval this session has actually run against.

    `session_id`/`document_id` (Phase 2, stage 6: uploads): forwarded to
    run_retrieve, which only actually uses them on the
    RETRIEVAL_BACKEND=postgres path (private-per-uploader isolation +
    optional single-document scoping) — meaningless, and ignored, on the
    default OpenSearch path, which has no private-upload concept at all.
    """
    active_strategy = get_active_strategy()
    if active_strategy == "default":
        index_name = active_embed_index_name()
    else:
        index_name = active_index_name()

    tracer = get_tracer()
    timings: dict[str, float] = {}
    with tracer.start_as_current_span("reason.answer_query") as span:
        span.set_attribute("reason.chunking_strategy", active_strategy)
        span.set_attribute("reason.embed_provider", get_active_embed_provider())

        size_result = check_size_and_shape(query)
        span.set_attribute("guardrail.size_and_shape.passed", size_result.passed)
        if not size_result.passed and guardrail_mode("size_and_shape") == "enforce":
            return StageTrace(
                original_query=query,
                stopped_at="size_and_shape",
                answer=query_too_large(),
                chunking_strategy=active_strategy,
                stage_timings=timings,
            )

        with _timed(timings, "plan_query"):
            # Real corpus categories, fetched here (not inside plan_query
            # itself, which stays free of any OpenSearch/Postgres
            # dependency — see _build_system_prompt's docstring) so the
            # rewrite LLM can be told what a real category actually is,
            # closing the hallucinated-category gap at its source rather
            # than only downstream in hybrid.py's _sanitize_filters.
            # Redis-cached 24h inside real_categories itself, so this
            # adds one cheap cache lookup per call, not a real round-trip
            # on every request.
            #
            # Real bug found live during the Render deploy check: this
            # used to call hybrid.py's OpenSearch-backed real_categories
            # unconditionally, crashing with
            # KeyError('OPENSEARCH_ADMIN_PASSWORD') on
            # RETRIEVAL_BACKEND=postgres deployments — dispatches the
            # same way src/reason/nodes/retrieve.py already does.
            if os.environ.get("RETRIEVAL_BACKEND", "opensearch") == "postgres":
                from src.retrieve.hybrid_postgres import real_categories as real_categories_postgres

                known_categories = frozenset(real_categories_postgres())
            else:
                known_categories = frozenset(real_categories(get_client(), index_name))
            plan = plan_query(query, known_categories=known_categories)
        span.set_attribute("reason.intent", plan.intent)
        span.set_attribute("reason.query_plan_degraded", plan.degraded)

        scope_result = screen_scope(plan)
        span.set_attribute("guardrail.scope_screening.passed", scope_result.passed)
        if not scope_result.passed and guardrail_mode("scope_screening") == "enforce":
            span.set_attribute("reason.short_circuited", True)
            return StageTrace(
                original_query=query,
                stopped_at="scope_screening",
                query_plan=plan,
                answer=out_of_scope(),
                chunking_strategy=active_strategy,
                stage_timings=timings,
            )
        span.set_attribute("reason.short_circuited", False)

        with _timed(timings, "retrieve"):
            candidates, fusion = run_retrieve(plan, index_name=index_name, session_id=session_id, document_id=document_id)
        span.set_attribute("reason.num_retrieved", len(candidates))

        attempts: list[AttemptTrace] = []
        attempt_n = 1
        current_top_n = top_n_context
        attempts_saved = 0
        assess_sufficient = True
        ambiguity_flagged = False

        while True:
            with _timed(timings, "rerank"):
                reranked, metadata = run_rerank_and_metadata(query, candidates, current_top_n, index_name=index_name)

            # Plan A5: assess_context and assess_ambiguity take identical
            # inputs (query, reranked), are mutually independent, and
            # each may fire a real `grade`-role LLM call — run them
            # concurrently rather than back-to-back, same
            # thread-pool-plus-context-propagation shape as A4's search
            # arms (see run_with_otel_context's docstring).
            #
            # Real, accepted tradeoff: assess_ambiguity always escalates
            # (no deterministic shortcut — see its own docstring), so
            # when context turns out insufficient under `enforce` (about
            # to widen and retry, never reaching generate this attempt),
            # running both in parallel means assess_ambiguity's call
            # already happened and its result is simply unused, where
            # the old sequential order skipped it outright. Bounded to
            # at most one extra `grade` call per query (only possible on
            # a non-final attempt), traded for a real latency win on the
            # more common sufficient-context path. Not hidden — this is
            # exactly the kind of tradeoff this project documents rather
            # than silently accepts.
            def _call_assess_context():
                with _timed(timings, "assess_context"):
                    return assess_context(query, reranked)

            def _call_assess_ambiguity():
                with _timed(timings, "assess_ambiguity"):
                    return assess_ambiguity(query, reranked, metadata)

            # Real bug found in review (Phase 2, BYOK): ThreadPoolExecutor
            # worker threads do NOT inherit the submitting thread's
            # contextvars (unlike asyncio task creation) — confirmed
            # empirically, not assumed. run_with_otel_context only
            # re-attaches the OTel context; it says nothing about
            # src/platform/credentials.py's Credentials contextvar, which
            # CredentialsMiddleware sets on the request's own task. Without
            # this, a BYOK visitor's key is invisible inside
            # assess_ambiguity's real `complete()` call (it always fires,
            # no deterministic shortcut) — silently falls back to any
            # server-side env key, or raises "not set" if there is none,
            # exactly the "not set" error a BYOK-only deployment is built
            # to never hit. Captured on the submitting thread (correct
            # value for THIS request) and explicitly re-set inside each
            # worker before it runs, mirroring run_with_otel_context's own
            # attach-in-worker/detach-after pattern for the same reason.
            request_credentials = get_credentials()

            def _with_credentials(fn):
                token = set_credentials(request_credentials)
                try:
                    return fn()
                finally:
                    reset_credentials(token)

            current_ctx = otel_context.get_current()
            with ThreadPoolExecutor(max_workers=2) as executor:
                context_future = executor.submit(run_with_otel_context, current_ctx, _with_credentials, _call_assess_context)
                ambiguity_future = executor.submit(run_with_otel_context, current_ctx, _with_credentials, _call_assess_ambiguity)
                assess_result = context_future.result()
                ambiguity_signal = ambiguity_future.result()
            assess_sufficient = assess_result.passed

            skip_generate = (
                not assess_result.passed
                and not assess_result.errored
                and guardrail_mode("context_sufficiency") == "enforce"
                and attempt_n < _MAX_ATTEMPTS
            )
            if skip_generate:
                attempts_saved += 1
                attempt_n += 1
                current_top_n = widen(current_top_n, step=_CONTEXT_WIDEN_STEP, cap=_CONTEXT_MAX)
                continue

            ambiguity_flagged = ambiguity_signal.flagged
            note_for_generate = (
                ambiguity_signal.note
                if ambiguity_signal.flagged and guardrail_mode("ambiguity_detection") == "enforce"
                else None
            )

            with _timed(timings, "generate"):
                answer, citation_result, grounded_result = run_answer(
                    query, reranked.items, metadata, ambiguity_note=note_for_generate
                )

            attempts.append(
                AttemptTrace(
                    attempt=attempt_n,
                    top_n_context=current_top_n,
                    reranked=reranked,
                    answer=answer,
                    groundedness_passed=grounded_result.passed,
                    groundedness_reason=grounded_result.reason,
                )
            )

            needs_retry = answer.abstained or (not grounded_result.passed and not grounded_result.errored)
            if not needs_retry or attempt_n >= _MAX_ATTEMPTS:
                break
            attempt_n += 1
            current_top_n = widen(current_top_n, step=_CONTEXT_WIDEN_STEP, cap=_CONTEXT_MAX)

        span.set_attribute("reason.attempts", len(attempts))
        span.set_attribute("reason.retried", len(attempts) > 1)
        span.set_attribute("reason.num_context", len(reranked.items))
        span.set_attribute("guardrail.citation_integrity.passed", citation_result.passed)
        span.set_attribute("guardrail.groundedness.passed", grounded_result.passed)
        span.set_attribute("reason.assess.sufficient", assess_sufficient)
        span.set_attribute("reason.assess.attempts_saved", attempts_saved)
        span.set_attribute("reason.assess.ambiguity_flagged", ambiguity_flagged)
        for stage_name, ms in timings.items():
            span.set_attribute(f"reason.timing.{stage_name}_ms", ms)

        stopped_at = None
        if not grounded_result.passed and guardrail_mode("groundedness") == "enforce":
            stopped_at = "groundedness"
            answer = ungrounded()

        return StageTrace(
            original_query=query,
            stopped_at=stopped_at,
            query_plan=plan,
            retrieved=candidates,
            fusion=fusion,
            reranked=reranked,
            metadata=metadata,
            answer=answer,
            attempts=attempts,
            chunking_strategy=active_strategy,
            stage_timings=timings,
        )
