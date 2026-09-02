from __future__ import annotations

import os
import time

from fastapi import HTTPException, Request

from src.platform.cache import get_client

# R6's "resource abuse" gap: per-caller request limits so one actor can't
# exhaust shared quota that every other caller also depends on (this
# project's free-tier Groq/Jina caps are real and already hit more than
# once). Redis-backed rather than an in-memory counter — this stack
# already hard-depends on Redis (src/platform/cache.py, A1's query-plan
# cache) and already runs it, so an in-memory counter would just be
# *simpler code* while being wrong the moment this ever runs as more than
# one worker process (each process would get its own counter, silently
# multiplying the effective limit).
#
# caller_key = request.client.host (or X-Forwarded-For, see
# _resolve_caller_ip below): the only real caller identity this codebase
# has today — there is no auth system. This is a coarse proxy (shared
# NAT/proxy defeats it) and should be replaced with real caller identity
# once one exists, not treated as a solved problem.

_LIMIT = int(os.environ.get("RATE_LIMIT_REQUESTS", "20"))
_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))


def _trusted_proxy_ips() -> set[str]:
    raw = os.environ.get("TRUSTED_PROXY_IPS", "")
    return {ip.strip() for ip in raw.split(",") if ip.strip()}


def _resolve_caller_ip(request: Request) -> str:
    """Plan D2 — horizontal scaling gap: behind a load balancer,
    request.client.host alone becomes the BALANCER's IP for every
    caller, collapsing every real user into one bucket and making the
    limiter useless exactly when multiple replicas need it most.

    X-Forwarded-For is only honored when the immediate connecting peer
    (request.client.host) is in TRUSTED_PROXY_IPS — blindly trusting
    the header from anyone would let any caller spoof a fresh identity
    on every request and bypass the limit outright, which is worse than
    the coarse-IP status quo this replaces. Single-trusted-hop model:
    takes the LAST value in the header (the one the trusted proxy
    itself appended), not the first (which the original, unverified
    client could have set to anything). No TRUSTED_PROXY_IPS configured
    -> the header is never consulted, exact same behavior as before
    this existed.
    """
    direct_ip = request.client.host if request.client else "unknown"
    trusted = _trusted_proxy_ips()
    if direct_ip in trusted:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
            if hops:
                return hops[-1]
    return direct_ip


def check_rate_limit(caller_key: str, *, limit: int = _LIMIT, window_seconds: int = _WINDOW_SECONDS) -> bool:
    """True if `caller_key` is still under `limit` requests in the current
    `window_seconds`-wide bucket. INCR+EXPIRE on a windowed key, not a
    sliding log — an approximation (a caller can burst near a window
    boundary) that's the same one this project's own free-tier providers
    themselves use, and cheap enough to run on every request.
    """
    bucket = int(time.time()) // window_seconds
    key = f"ratelimit:{caller_key}:{bucket}"
    client = get_client()
    count = client.incr(key)
    if count == 1:
        client.expire(key, window_seconds)
    return count <= limit


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency — raises 429 when the caller is over budget.
    Applied selectively (see routes/ask.py, routes/pipeline.py) to the
    two routes that actually spend LLM-provider quota, not blanket
    middleware — /health and static assets have no reason to be limited.
    """
    caller_key = _resolve_caller_ip(request)
    if not check_rate_limit(caller_key):
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests from this address — try again in under {_WINDOW_SECONDS} seconds.",
        )
