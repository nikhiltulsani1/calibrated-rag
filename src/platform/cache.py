from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import redis

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        _client = redis.Redis(host=host, port=port, decode_responses=True)
    return _client


def get_json(key: str) -> Any | None:
    raw = get_client().get(key)
    return json.loads(raw) if raw is not None else None


def set_json(key: str, value: Any, ttl_seconds: int) -> None:
    get_client().set(key, json.dumps(value), ex=ttl_seconds)


def build_cache_key(prefix: str, *parts: str) -> str:
    """Real, minimal duplication fix (full-codebase review, 2026-08-25):
    query_planner.py's plan cache and reason/answer_cache.py's answer
    cache each independently hand-rolled the identical "join parts with
    |, sha256, prefix" pattern. Centralizing it here means a future
    change to the hashing/normalization scheme only needs to happen
    once, not two places that can silently drift.

    Callers normalize their own parts before passing them in (e.g.
    `.strip().lower()` on a free-text query string) — this function only
    joins and hashes, since what counts as "normalized" differs by
    caller: a user-typed query needs case-folding, a toggle value like
    "jina" already doesn't.
    """
    raw = "|".join(parts)
    return prefix + hashlib.sha256(raw.encode()).hexdigest()
