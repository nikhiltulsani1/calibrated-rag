from __future__ import annotations

import os

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Verified against Opik's own OpenTelemetry docs, not assumed — the HTTP
# exporter's `endpoint=` constructor arg needs the full /v1/traces path
# (unlike the OTEL_EXPORTER_OTLP_ENDPOINT env-var convention, which
# auto-appends it).
_OPIK_OTEL_ENDPOINT = "https://www.comet.com/opik/api/v1/private/otel/v1/traces"

_tracer: trace.Tracer | None = None


def _build_provider() -> TracerProvider:
    project = os.environ.get("OPIK_PROJECT_NAME", "production-rag-system")
    provider = TracerProvider(resource=Resource.create({"service.name": project}))

    api_key = os.environ.get("OPIK_API_KEY")
    workspace = os.environ.get("OPIK_WORKSPACE")
    if api_key and workspace:
        exporter = OTLPSpanExporter(
            endpoint=_OPIK_OTEL_ENDPOINT,
            headers={
                "Authorization": api_key,
                "projectName": project,
                "Comet-Workspace": workspace,
            },
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    # No OPIK_API_KEY/OPIK_WORKSPACE yet: the provider is still returned,
    # spans are still created — instrumented code never needs an "if
    # telemetry is configured" branch — they just have nowhere to export
    # to. Same honest-gap shape as every other missing key in this
    # project: correct code, unverified network path, until the key
    # exists.
    return provider


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        project = os.environ.get("OPIK_PROJECT_NAME", "production-rag-system")
        _tracer = _build_provider().get_tracer(project)
    return _tracer


def reset_tracer() -> None:
    """Test-only: forces get_tracer() to rebuild on next call, and is how
    tests inject an in-memory exporter to assert on real recorded spans
    instead of mocking the tracing calls themselves."""
    global _tracer
    _tracer = None


def run_with_otel_context(ctx, fn, *args):
    """Run `fn(*args)` inside `ctx` — for use as a ThreadPoolExecutor
    task. A span started inside a bare worker thread has no parent by
    default (each thread gets its own empty OTel context), so it shows
    up in Opik as an orphaned trace rather than correctly nested under
    the caller's span — silently breaking the tracing this project
    relies on throughout. Explicitly attaching the caller's context
    (from otel_context.get_current(), captured on the calling thread
    before submitting) before running, and detaching after, fixes that.
    Shared by both parallelization sites this project has (A4's search
    arms in retrieve/hybrid.py, A5's assess calls in reason/graph.py) so
    there is one correct pattern, not two independently-written ones.
    """
    token = otel_context.attach(ctx)
    try:
        return fn(*args)
    finally:
        otel_context.detach(token)
