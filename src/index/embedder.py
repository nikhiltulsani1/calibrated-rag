from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import httpx

from src.index.embed_toggle import get_active_embed_provider, provider_to_index
from src.platform.models import get_model
from src.platform.telemetry import get_tracer

_JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
_MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"
_MISTRAL_MODEL_ID = "mistral-embed"
_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    dimension: int
    vectors: list[list[float]]
    prompt_tokens: int


def _embed_jina(texts: list[str], task: str, dimensions: int | None) -> EmbeddingResult:
    binding = get_model("embed")
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY is not set")

    body: dict = {
        "model": binding.model_id,
        "input": texts,
        "task": task,
        "normalized": True,
        "embedding_type": "float",
    }
    if dimensions is not None:
        body["dimensions"] = dimensions

    response = httpx.post(
        _JINA_EMBEDDINGS_URL,
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    vectors = [item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]
    # Real bug found in a full-codebase review, fixed 2026-08-25: a 200
    # response with a shorter `data` array than `texts` sent (a plausible
    # partial-batch failure, not just a full-request error) used to sail
    # through unchecked, despite embed_queries_with_fallback's own
    # docstring claiming embedding "fails LOUD" on a wrong/missing
    # vector. src/ingest/pipeline.py does
    # `for chunk, vector in zip(new_chunks, result.vectors)` — zip
    # silently truncates to the shorter iterable, so a dropped item would
    # misalign every subsequent chunk_id<->vector pair in the batch
    # rather than erroring. A0's fail-loud rule, actually enforced here.
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Jina embeddings returned {len(vectors)} vectors for {len(texts)} input texts — "
            "a partial-batch response would silently corrupt chunk_id<->vector alignment downstream"
        )
    return EmbeddingResult(
        model=payload["model"],
        dimension=len(vectors[0]) if vectors else 0,
        vectors=vectors,
        # Jina's actual response shape uses "total_tokens", not
        # "prompt_tokens" (the OpenAI convention this was apparently
        # copied from) — never exercised against a real response until
        # a real JINA_API_KEY existed to test with. Verified directly
        # against a live API call, not assumed.
        prompt_tokens=payload["usage"]["total_tokens"],
    )


# Real, live-discovered constraint (2026-08-22, building the
# rag_chunks_mistral_embed index): mistral-embed's /v1/embeddings
# rejects a full 571-chunk batch with 400 "Too many tokens overall,
# split into more batches" (code 3210) — Jina's endpoint has no
# equivalent limit hit by the same corpus. Verified 64 texts/request
# succeeds at this corpus's real max chunk length (1600 chars); kept as
# a fixed batch size rather than a token-counting scheme since it's a
# real, tested number, not a computed guess at where the boundary is.
_MISTRAL_BATCH_SIZE = 64


def _embed_mistral(texts: list[str], dimensions: int | None) -> EmbeddingResult:
    # No task-asymmetry parameter (unlike Jina) — mistral-embed's own docs
    # don't distinguish a query/passage mode, and no `dimensions`
    # truncation param either (unverified whether MRL truncation is
    # supported; not requested here to avoid a real 400 on an unverified
    # param). Real 1024-d output, confirmed live 2026-08-22.
    if dimensions is not None:
        raise NotImplementedError("dimensions truncation is not verified for the mistral embed provider")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    all_vectors: list[list[float]] = []
    model_served = _MISTRAL_MODEL_ID
    total_tokens = 0
    for start in range(0, len(texts), _MISTRAL_BATCH_SIZE):
        batch = texts[start : start + _MISTRAL_BATCH_SIZE]
        response = httpx.post(
            _MISTRAL_EMBEDDINGS_URL,
            json={"model": _MISTRAL_MODEL_ID, "input": batch},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        batch_vectors = [item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]
        # See _embed_jina's identical check for the real bug this fixes.
        # Checked per-batch (not just once at the end) so a mismatch is
        # caught at the exact batch that produced it, not blamed on the
        # wrong one once results are already concatenated.
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"Mistral embeddings returned {len(batch_vectors)} vectors for {len(batch)} input "
                "texts in one batch — a partial-batch response would silently corrupt "
                "chunk_id<->vector alignment downstream"
            )
        all_vectors.extend(batch_vectors)
        model_served = payload["model"]
        total_tokens += payload["usage"]["total_tokens"]

    return EmbeddingResult(
        model=model_served,
        dimension=len(all_vectors[0]) if all_vectors else 0,
        vectors=all_vectors,
        prompt_tokens=total_tokens,
    )


def _embed(
    texts: list[str],
    task: Literal["retrieval.query", "retrieval.passage"],
    dimensions: int | None = None,
    provider: str | None = None,
) -> EmbeddingResult:
    """`provider` defaults to the live embed_toggle selection (jina or
    mistral — see src/index/embed_toggle.py) so ordinary query/passage
    calls automatically follow whatever's active. Pass it explicitly to
    bypass the toggle — the one-time index-building script
    (scripts/build_mistral_embed_index.py) always requests "mistral"
    regardless of what's currently active, since building that index
    isn't itself a live query.

    Added 2026-08-22: this used to hard-fail with NotImplementedError on
    any provider but jina ("no fallback provider is wired up for embed").
    That was a deliberate honest gap, not a bug — but real, repeated live
    testing hit Jina's account-balance limit hard enough (multiple trial
    keys exhausted within an hour) that having no alternative blocked all
    further verification. Unlike the chat-model ladder (RAG_MODEL_<ROLE>,
    automatic fallback-on-failure), this is a deliberate, explicit switch
    a person or script chooses — automatic silent fallback would mean a
    query silently gets served from a totally different vector space
    against an index built for another, which is exactly the "wrong
    model/width silently returns nonsense" failure A0's no-fallback rule
    was protecting against. The switch has to also pick the matching
    index (see embed_toggle.active_embed_index_name) for the same reason.
    """
    tracer = get_tracer()
    # No manual try/except here — start_as_current_span already records an
    # uncaught exception and sets ERROR status by default. See the note
    # in models.py::complete for why this isn't done manually here too.
    with tracer.start_as_current_span("embed.request") as span:
        span.set_attribute("embed.task", task)
        span.set_attribute("embed.num_texts", len(texts))
        if dimensions is not None:
            span.set_attribute("embed.requested_dimensions", dimensions)

        active_provider = provider or get_active_embed_provider()
        span.set_attribute("embed.provider", active_provider)

        if active_provider == "jina":
            result = _embed_jina(texts, task, dimensions)
        elif active_provider == "mistral":
            result = _embed_mistral(texts, dimensions)
        else:
            raise NotImplementedError(
                f"embedder only implements jina and mistral, got {active_provider!r}"
            )

        span.set_attribute("embed.model_served", result.model)
        span.set_attribute("embed.dimension", result.dimension)
        span.set_attribute("embed.prompt_tokens", result.prompt_tokens)
        return result


def embed_passages(texts: list[str], dimensions: int | None = None, provider: str | None = None) -> EmbeddingResult:
    """Embed chunk text at indexing time (asymmetric retrieval: passage side)."""
    return _embed(texts, task="retrieval.passage", dimensions=dimensions, provider=provider)


def embed_query(text: str, dimensions: int | None = None, provider: str | None = None) -> EmbeddingResult:
    """Embed a user query at search time (asymmetric retrieval: query side).

    Queries and passages are NOT interchangeable inputs to the jina
    provider — jina-embeddings-v3 is trained asymmetrically, so using the
    wrong task string doesn't error, it just retrieves worse. Always go
    through this function for queries, embed_passages for chunks.
    """
    return _embed([text], task="retrieval.query", dimensions=dimensions, provider=provider)


def embed_queries(texts: list[str], dimensions: int | None = None, provider: str | None = None) -> EmbeddingResult:
    """Batch form of embed_query — one HTTP call for many query texts,
    same "retrieval.query" task. Added for A7's chunking ablation
    (evals/run_chunking_eval.py), which scores the SAME 80 queries
    against 5 different chunking strategies: the query vector doesn't
    depend on which strategy's index is being searched, so embedding
    each query once here and reusing the vector across all 5 strategies
    is a real 5x reduction in Jina calls, not just batching for its own
    sake — discovered live after an unbatched version (one embed_query
    call per query per strategy, 400 total) hit Jina's rate limit.
    """
    return _embed(texts, task="retrieval.query", dimensions=dimensions, provider=provider)


def embed_queries_with_fallback(texts: list[str]) -> tuple[EmbeddingResult, str]:
    """Automatic embed-provider failover — added 2026-08-23 after a real
    gap bit: a full 61-question Groq run scored 0/61 because Jina's
    embed capability ran dry mid-run with no fallback, unlike rerank
    (src/retrieve/reranker.py's real, verified Jina->Cohere fallback),
    which recovers automatically.

    Embedding has one real extra constraint reranking doesn't: which
    provider embeds a query determines which INDEX the resulting vector
    is even comparable against (embed_toggle.py's whole reason for
    existing — a Mistral vector means nothing scored against Jina's
    stored vectors). So this returns the vectors AND the index that
    actually matches them, not just the vectors — the caller must query
    that index for the dense/kNN step, not whatever index_name it may
    have been passed for lexical search (which stays valid regardless,
    since every embed-provider variant index is built from the same
    real Postgres chunk text with the same real chunk_ids — only the
    stored vectors differ).

    Tries the currently active provider first (respecting the live
    toggle exactly like every other call), falls back to the other real
    provider on any failure. If BOTH fail, raises — embedding still
    fails LOUD by design (A0): a silently wrong or missing vector
    corrupts retrieval, unlike a skipped rerank which just leaves order
    unchanged. The raised error names both real failures, not just the
    first, for real debuggability.
    """
    primary = get_active_embed_provider()
    fallback = "mistral" if primary == "jina" else "jina"

    try:
        result = embed_queries(texts, provider=primary)
        return result, provider_to_index(primary)
    except Exception as primary_exc:
        try:
            result = embed_queries(texts, provider=fallback)
            return result, provider_to_index(fallback)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"embedding failed on both providers — {primary}: {primary_exc}; {fallback}: {fallback_exc}"
            ) from fallback_exc
