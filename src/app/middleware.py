from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# R2's "correlation across components" gap: a single user request touches
# guardrails, retrieval, generation, and Postgres/OpenSearch — without a
# shared identifier, reconstructing one slow or failed request means
# correlating by timestamp across separate tools. This generates one id
# per request, at the edge, and threads it through logs, the current OTel
# span, and the response — so a user-reported problem maps to exactly one
# trace instead of a timestamp guess.
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

HEADER_NAME = "X-Request-ID"


def current_request_id() -> str:
    """The active request's id, or "-" outside any request (e.g. a
    background script) — never raises, so call sites never need a
    try/except just to log."""
    return _REQUEST_ID.get()


class _RequestIdLogFilter(logging.Filter):
    """Injects `request_id` into every log record's attributes, so every
    existing `logger.info(...)`/`logger.exception(...)` call site in this
    codebase gains request correlation with zero changes to any call
    site — only the log FORMAT (not shown here; this project doesn't yet
    configure a custom formatter) would need to reference `%(request_id)s`
    to display it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True


def install_request_id_log_filter() -> None:
    logging.getLogger().addFilter(_RequestIdLogFilter())


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Honors an inbound `X-Request-ID` header if the caller already has
    one (e.g. a load balancer or another service), otherwise generates a
    fresh UUID4. Sets it as the active span's attribute using this
    project's existing tracing convention (see src/platform/telemetry.py
    — dotted attribute names, `span.set_attribute`), stores it in
    `request.state` for route handlers to read, and always returns it on
    the response so a user can report "X-Request-ID: ..." and that maps
    to exactly one trace/log correlation key.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(HEADER_NAME) or str(uuid.uuid4())
        request.state.request_id = request_id
        token = _REQUEST_ID.set(request_id)
        try:
            span = trace.get_current_span()
            span.set_attribute("request_id", request_id)
            response: Response = await call_next(request)
        finally:
            _REQUEST_ID.reset(token)
        response.headers[HEADER_NAME] = request_id
        return response
