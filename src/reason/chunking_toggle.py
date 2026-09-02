from __future__ import annotations

import os

from src.index.mapping import INDEX_NAME
from src.platform.cache import get_client

# A7's live, reversible chunking-strategy toggle. Three named roles map
# to permanent OpenSearch indices built by evals/run_chunking_eval.py's
# persist_selected_variants() — "default" (today's production `rag_chunks`,
# untouched by A7 at all) plus "winner"/"median"/"efficient" (the 3
# strategies selected from the 5-way ablation). No new store: the live
# override reuses Redis (src/platform/cache.py), already a hard
# dependency of this stack (A1's query-plan cache).

_REDIS_KEY = "chunking_strategy:active"
_VALID_STRATEGIES = {"default", "winner", "median", "efficient"}


def _strategy_to_index(strategy: str) -> str:
    if strategy == "default":
        return INDEX_NAME
    return f"rag_chunks_{strategy}"


def get_active_strategy() -> str:
    """Redis override wins over the CHUNKING_STRATEGY env var, which wins
    over "default" — same precedence shape as this project's other
    env-var-first, override-able settings (e.g. RAG_MODEL_<ROLE>).
    """
    try:
        override = get_client().get(_REDIS_KEY)
    except Exception:
        # Redis being briefly unavailable must not break every query —
        # fall through to the env var / default exactly like today's
        # behavior with no toggle at all.
        override = None
    if override and override in _VALID_STRATEGIES:
        return override

    env_value = os.environ.get("CHUNKING_STRATEGY", "default")
    return env_value if env_value in _VALID_STRATEGIES else "default"


def set_active_strategy(strategy: str) -> None:
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r} — must be one of {sorted(_VALID_STRATEGIES)}")
    get_client().set(_REDIS_KEY, strategy)


def active_index_name() -> str:
    return _strategy_to_index(get_active_strategy())
