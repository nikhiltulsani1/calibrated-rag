import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from unittest.mock import MagicMock, patch

import src.platform.telemetry as telemetry
from src.platform.models import CompletionResult

pytestmark = pytest.mark.unit


@pytest.fixture
def traced_spans():
    """Injects a real OTel tracer backed by an in-memory exporter — spans
    are genuinely created and recorded, just never sent anywhere. This
    tests the actual instrumentation (span names, attributes), not a
    mock standing in for it.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry._tracer = provider.get_tracer("test")
    yield exporter
    telemetry.reset_tracer()


def test_get_tracer_builds_without_opik_key(monkeypatch):
    monkeypatch.delenv("OPIK_API_KEY", raising=False)
    monkeypatch.delenv("OPIK_WORKSPACE", raising=False)
    telemetry.reset_tracer()
    tracer = telemetry.get_tracer()
    # Must not raise, and must be usable as a real context manager even
    # with no exporter attached — the whole point of the honest-gap
    # design: instrumented code never needs an "if configured" branch.
    with tracer.start_as_current_span("smoke.test"):
        pass
    telemetry.reset_tracer()


def test_get_tracer_is_a_singleton():
    telemetry.reset_tracer()
    t1 = telemetry.get_tracer()
    t2 = telemetry.get_tracer()
    assert t1 is t2
    telemetry.reset_tracer()


def test_complete_records_span_with_model_served(traced_spans, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    from src.platform.models import complete

    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {
        "model": "openai/gpt-oss-20b",
        "choices": [{"message": {"content": "hi"}}],
    }
    with patch("httpx.post", return_value=fake_response):
        complete("rewrite", [{"role": "user", "content": "hi"}])

    spans = traced_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "llm.complete"
    assert spans[0].attributes["llm.role"] == "rewrite"
    assert spans[0].attributes["llm.provider"] == "groq"
    assert spans[0].attributes["llm.model_served"] == "openai/gpt-oss-20b"


def test_complete_records_exception_on_failure(traced_spans, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src.platform.models import complete

    with pytest.raises(RuntimeError):
        complete("rewrite", [{"role": "user", "content": "hi"}])

    spans = traced_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
    assert len(spans[0].events) == 1  # the recorded exception


def test_rerank_records_degraded_attribute_not_error(traced_spans, monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    from src.retrieve.reranker import Candidate, rerank

    rerank("q", [Candidate(id="c0", text="t")], top_n=1)

    spans = traced_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "reranker.rerank"
    assert spans[0].attributes["reranker.degraded"] is True
    # A degrade is expected/successful behaviour, not a span error.
    assert spans[0].status.status_code.name != "ERROR"


def test_plan_query_span_is_parent_of_complete_span(traced_spans, monkeypatch):
    # query_planner.plan_query() calls models.complete() internally — this
    # verifies real OTel context propagation nests them correctly (child
    # span's parent_span_id matches the parent's span_id), which matters
    # for reading an actual trace tree in Opik later, not just that spans
    # exist independently.
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    from src.retrieve.query_planner import plan_query

    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json = lambda: {
        "model": "openai/gpt-oss-20b",
        "choices": [
            {
                "message": {
                    "content": '{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}'
                }
            }
        ],
    }
    with patch("src.retrieve.query_planner.get_json", return_value=None), patch(
        "src.retrieve.query_planner.set_json"
    ), patch("httpx.post", return_value=fake_response):
        plan_query("a test query")

    spans = {s.name: s for s in traced_spans.get_finished_spans()}
    assert set(spans) == {"query_planner.plan_query", "llm.complete"}
    assert spans["llm.complete"].parent.span_id == spans["query_planner.plan_query"].context.span_id
