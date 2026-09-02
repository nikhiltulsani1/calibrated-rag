from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

# Phase 2 (BYOK) — the real bug this design exists to avoid: every
# provider-calling function in this codebase (models.py::complete,
# embedder.py::_embed_jina/_embed_mistral, reranker.py::_rerank_hosted/
# _rerank_cohere) reads its API key via os.environ.get(...) fresh on
# every call. For a single-operator deployment that's fine — one shared
# environment, one owner. For a public BYOK deployment where concurrent
# visitors each supply their own keys, mutating process-global
# os.environ per request would be a genuine race condition: uvicorn's
# async event loop can interleave two different visitors' requests
# within the same process (an `await` in one yields control to
# another's), so one visitor's env write could leak into or get
# clobbered by another's concurrent request.
#
# contextvars.ContextVar is coroutine-safe by design — each
# concurrently-running request's context is isolated even within one
# event loop/process, which is exactly the isolation os.environ can't
# give here. This is the one seam every provider-calling function reads
# through instead of a dozen threaded-parameter call sites.


@dataclass(frozen=True)
class Credentials:
    groq: str | None = None
    openrouter: str | None = None
    nvidia: str | None = None
    mistral: str | None = None
    jina: str | None = None
    cohere: str | None = None


_EMPTY = Credentials()
_current: ContextVar[Credentials] = ContextVar("credentials", default=_EMPTY)


def get_credentials() -> Credentials:
    """The active request's visitor-supplied keys, or an all-None
    Credentials if unset (outside a request, or a visitor who hasn't
    configured any keys yet) — every call site falls back to the
    server's own os.environ value in that case, so this is purely
    additive: nothing changes for a deployment that never sets this.
    """
    return _current.get()


def set_credentials(creds: Credentials):
    """Returns the ContextVar token — callers (CredentialsMiddleware)
    must reset() it in a finally block, same discipline as
    RequestIdMiddleware's _REQUEST_ID, so a request's credentials never
    leak into whatever runs after it in the same task.
    """
    return _current.set(creds)


def reset_credentials(token) -> None:
    _current.reset(token)
