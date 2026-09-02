from __future__ import annotations

import os

from src.index.mapping import INDEX_NAME
from src.platform.cache import get_client

# A real, reversible embedding-provider switch — mirrors A7's chunking
# toggle exactly (src/reason/chunking_toggle.py): same Redis-override-wins,
# then env var, then default precedence; same reason it exists at all
# (embed_provider.py's docstring on _embed used to say plainly "no
# fallback provider is wired up for embed... add a branch here when a
# second provider is actually bound"). Added 2026-08-22 after repeatedly
# hitting Jina's real balance limits mid-eval with no way to keep testing.
#
# Vector spaces from different embedding models are NOT interchangeable —
# a query embedded by Mistral means nothing scored against Jina-embedded
# vectors. So switching provider must also switch which index gets
# queried, exactly like switching chunking strategy must. "jina" points
# at the untouched production `rag_chunks` index; "mistral" points at a
# separate, permanent index built once by
# scripts/build_mistral_embed_index.py from the SAME Postgres chunk text
# (default chunking only — this doesn't cross the chunking-strategy axis;
# see get_index_name_for_query in graph.py for how the two toggles compose).

_REDIS_KEY = "embed_provider:active"
_VALID_PROVIDERS = {"jina", "mistral"}
_MISTRAL_INDEX_NAME = "rag_chunks_mistral_embed"


def provider_to_index(provider: str) -> str:
    """Public since 2026-08-23 — reused by embedder.py's automatic
    embed-provider failover (embed_queries_with_fallback) to determine
    which index actually matches whatever provider ends up serving a
    given embed call, not just whichever provider was originally active.
    """
    if provider == "jina":
        return INDEX_NAME
    return _MISTRAL_INDEX_NAME


def get_active_embed_provider() -> str:
    """Redis override wins over the EMBED_PROVIDER env var, which wins
    over "jina" — same precedence shape as get_active_strategy() and
    RAG_MODEL_<ROLE>.
    """
    try:
        override = get_client().get(_REDIS_KEY)
    except Exception:
        # Redis being briefly unavailable must not break every query —
        # fall through to the env var / default exactly like today's
        # behavior with no toggle at all.
        override = None
    if override and override in _VALID_PROVIDERS:
        return override

    env_value = os.environ.get("EMBED_PROVIDER", "jina")
    return env_value if env_value in _VALID_PROVIDERS else "jina"


def set_active_embed_provider(provider: str) -> None:
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"unknown embed provider {provider!r} — must be one of {sorted(_VALID_PROVIDERS)}")
    get_client().set(_REDIS_KEY, provider)


def active_embed_index_name() -> str:
    return provider_to_index(get_active_embed_provider())
