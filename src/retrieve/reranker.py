from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from src.platform.models import get_model
from src.platform.telemetry import get_tracer

logger = logging.getLogger(__name__)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_HOSTED_TIMEOUT_SECONDS = 2.0
_DEFAULT_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Added 2026-08-22 as a second real hosted rerank provider — Jina's
# account balance ran out repeatedly this session (4 separate trial
# keys exhausted within about an hour), and local CPU cross-encoder
# reranking was measured and ruled out as a fallback on this machine
# (BAAI/bge-reranker-v2-m3: ~140-165s/call; even the tiny default
# MiniLM: ~10-12s/call — both far too slow, see Results.md). Cohere's
# rerank-v3.5 verified live: correct relevance ordering (0.87 for an
# actually-relevant doc vs 0.006 for an irrelevant one) and real
# steady-state latency of 0.9-3.6s for 20 documents — comparable to
# Jina, not Jina-fast. Given that real variance, a slightly more
# generous timeout than Jina's strict 2.0s avoids spurious degrades on
# normal response-time swings rather than genuine unavailability.
_COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
_COHERE_TIMEOUT_SECONDS = 5.0
_DEFAULT_COHERE_MODEL = "rerank-v3.5"


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str


@dataclass(frozen=True)
class RankedCandidate:
    id: str
    text: str
    score: float | None  # None when degraded — the input order carries no reranker score


@dataclass(frozen=True)
class RerankResult:
    items: list[RankedCandidate]
    degraded: bool
    reason: str | None
    model_served: str | None


def _degrade(candidates: list[Candidate], top_n: int, reason: str) -> RerankResult:
    # Un-reranked RRF order: truncate the input as-is, never reorder on a
    # degrade path. Never silent, per A2 — full telemetry.py doesn't exist
    # yet, so this goes through logging until it does; the event name is
    # deliberately stable so it's greppable once real tracing lands.
    logger.warning(
        "reranker degraded to un-reranked order: %s",
        reason,
        extra={"event": "rerank_degraded", "reason": reason},
    )
    items = [RankedCandidate(id=c.id, text=c.text, score=None) for c in candidates[:top_n]]
    return RerankResult(items=items, degraded=True, reason=reason, model_served=None)


def _parse_rerank_results(payload: dict, candidates: list[Candidate]) -> list[RankedCandidate]:
    # Shared by _rerank_hosted (Jina) and _rerank_cohere — both APIs return
    # the identical {"results": [{"index": int, "relevance_score": float}, ...]}
    # shape, just under different top-level response envelopes.
    return [
        RankedCandidate(
            id=candidates[result["index"]].id,
            text=candidates[result["index"]].text,
            score=result["relevance_score"],
        )
        for result in payload["results"]
    ]


def _rerank_hosted(query: str, candidates: list[Candidate], top_n: int) -> RerankResult:
    binding = get_model("rerank")
    if binding.provider != "jina":
        # A config/integration gap, not a transient unavailability — raise
        # rather than degrade, same split as the embedder's unsupported-
        # provider case in A0.
        raise NotImplementedError(
            f"reranker only implements the jina provider so far, got {binding.provider!r}"
        )

    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        return _degrade(candidates, top_n, "JINA_API_KEY is not set")

    body = {
        "model": binding.model_id,
        "query": query,
        "top_n": top_n,
        "documents": [c.text for c in candidates],
    }
    try:
        response = httpx.post(
            _JINA_RERANK_URL,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_HOSTED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # Covers the 2s timeout, connection failures, and 5xx/429 —
        # exactly the "2s timeout, degrading" case A2 calls for. A 200
        # with an unexpected body shape is NOT caught here on purpose: that
        # indicates our integration is wrong, not that the service is
        # unavailable, and should fail loud rather than degrade silently.
        return _degrade(candidates, top_n, f"{type(exc).__name__}: {exc}")

    payload = response.json()
    items = _parse_rerank_results(payload, candidates)
    return RerankResult(
        items=items,
        degraded=False,
        reason=None,
        model_served=payload.get("model", binding.model_id),
    )


def _rerank_local(query: str, candidates: list[Candidate], top_n: int) -> RerankResult:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "RERANKER_BACKEND=local requires sentence-transformers, which is not a "
            "default dependency (hosted is primary, to keep the container small — "
            "see A2). Install it with `pip install sentence-transformers` to use "
            "this backend."
        ) from exc

    model_name = os.environ.get("RERANKER_LOCAL_MODEL", _DEFAULT_LOCAL_MODEL)
    model = CrossEncoder(model_name)
    scores = model.predict([(query, c.text) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
    items = [RankedCandidate(id=c.id, text=c.text, score=float(s)) for c, s in ranked]
    return RerankResult(items=items, degraded=False, reason=None, model_served=f"local:{model_name}")


def _rerank_cohere(query: str, candidates: list[Candidate], top_n: int) -> RerankResult:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        return _degrade(candidates, top_n, "COHERE_API_KEY is not set")

    model_name = os.environ.get("RERANKER_COHERE_MODEL", _DEFAULT_COHERE_MODEL)
    body = {
        "model": model_name,
        "query": query,
        "documents": [c.text for c in candidates],
        "top_n": top_n,
    }
    try:
        response = httpx.post(
            _COHERE_RERANK_URL,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_COHERE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # Same degrade-not-fail contract as _rerank_hosted — a Cohere
        # outage/timeout just leaves fusion order in place, it doesn't
        # fail the request (reranking only reorders already-valid
        # candidates).
        return _degrade(candidates, top_n, f"{type(exc).__name__}: {exc}")

    payload = response.json()
    items = _parse_rerank_results(payload, candidates)
    return RerankResult(items=items, degraded=False, reason=None, model_served=f"cohere:{model_name}")


def rerank(
    query: str,
    candidates: list[Candidate],
    *,
    top_n: int = 8,
    backend: str | None = None,
) -> RerankResult:
    """A2: rewrite -> hybrid retrieve -> **rerank** -> top 8 to the generator.

    Degrades to the input order (unchanged, truncated to top_n) rather than
    failing the request on a hosted-call problem — reranking only reorders
    already-valid candidates, so skipping a bad reorder is safe. This is
    the opposite failure mode from `embed` (A0), which must fail loud: a
    wrong embedding silently corrupts what gets retrieved at all, a
    skipped rerank just leaves fusion order in place.

    backend="hosted" (the default) automatically falls back to Cohere
    when Jina fails and COHERE_API_KEY is configured — see the fallback
    block below. backend="local"/"cohere" are direct, no-fallback
    choices: a person or script asking for one specific provider
    explicitly shouldn't have it silently substituted.
    """
    tracer = get_tracer()
    # No manual try/except for genuine errors (unsupported provider,
    # unknown backend) — start_as_current_span records those and sets
    # ERROR status by default; see models.py::complete. A *degrade* is
    # different: it's expected, successful behaviour (see the docstring
    # above), so it's recorded as ordinary span attributes below, never
    # as a span error.
    with tracer.start_as_current_span("reranker.rerank") as span:
        span.set_attribute("reranker.num_candidates", len(candidates))
        span.set_attribute("reranker.top_n", top_n)

        if not candidates:
            return RerankResult(items=[], degraded=False, reason=None, model_served=None)

        chosen_backend = backend or os.environ.get("RERANKER_BACKEND", "hosted")
        span.set_attribute("reranker.backend", chosen_backend)

        if chosen_backend == "hosted":
            result = _rerank_hosted(query, candidates, top_n)
            if result.degraded and os.environ.get("COHERE_API_KEY"):
                # Real, automatic hosted->hosted fallback — added
                # 2026-08-22 after Jina's account balance ran out
                # repeatedly this session (four separate keys within
                # about an hour). A degraded Jina call means "we're
                # about to answer with un-reranked RRF order," which is
                # strictly worse than "we're about to answer with a
                # DIFFERENT real reranker's order" whenever a working
                # alternative exists — so try Cohere before accepting
                # the degrade, not after someone notices and flips
                # RERANKER_BACKEND by hand. Only engages when a Cohere
                # key is actually configured (a no-op otherwise, exactly
                # today's behavior). If Cohere also fails, its own
                # result is already a valid degrade (same un-reranked
                # fallback, same contract) — nothing further to do.
                jina_reason = result.reason
                span.set_attribute("reranker.jina_failed_reason", jina_reason or "")
                fallback_result = _rerank_cohere(query, candidates, top_n)
                if not fallback_result.degraded:
                    logger.info(
                        "reranker fell back from jina to cohere: %s",
                        jina_reason,
                        extra={"event": "rerank_fallback_engaged", "jina_reason": jina_reason},
                    )
                    span.set_attribute("reranker.fallback_engaged", True)
                result = fallback_result
        elif chosen_backend == "local":
            result = _rerank_local(query, candidates, top_n)
        elif chosen_backend == "cohere":
            result = _rerank_cohere(query, candidates, top_n)
        else:
            raise ValueError(
                f"unknown RERANKER_BACKEND {chosen_backend!r}, expected 'hosted', 'local', or 'cohere'"
            )

        span.set_attribute("reranker.degraded", result.degraded)
        if result.degraded:
            span.set_attribute("reranker.degrade_reason", result.reason or "")
        if result.model_served:
            span.set_attribute("reranker.model_served", result.model_served)
        return result
