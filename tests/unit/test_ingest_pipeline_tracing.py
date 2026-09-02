from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import src.platform.telemetry as telemetry
from src.ingest.document_parser import ParsedDocument, ParsedSection
from src.ingest.paper_source import PaperMetadata

pytestmark = pytest.mark.unit

# R2's background-job tracing gap, closed: ingest_category previously
# emitted zero OTel spans despite get_tracer() being trivially available
# project-wide. This tests the real instrumentation (span names,
# attributes) the same way test_telemetry.py already does for the
# request-serving path — an in-memory exporter records genuine spans,
# nothing here is mocked away.

_PAPER = PaperMetadata(
    arxiv_id="9999.00001",
    title="Test Paper",
    abstract="an abstract",
    authors=["A. Author"],
    category=["cs.IR"],
    published_date=date(2026, 1, 1),
    pdf_url="https://example.invalid/9999.00001.pdf",
)
_DOCUMENT = ParsedDocument(sections=[ParsedSection(heading="Intro", text="some section text " * 20)])


@pytest.fixture
def traced_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry._tracer = provider.get_tracer("test")
    yield exporter
    telemetry.reset_tracer()


def test_ingest_category_records_run_and_per_paper_spans(traced_spans):
    from src.ingest.pipeline import ingest_category

    fake_session = MagicMock()
    fake_session.get.return_value = None  # nothing pre-exists: paper and chunks are all new

    with patch("src.ingest.pipeline.fetch_papers_paginated", return_value=[_PAPER]), patch(
        "src.ingest.pipeline.get_session", return_value=fake_session
    ), patch("src.ingest.pipeline.fetch_pdf_bytes", return_value=b"%PDF-fake"), patch(
        "src.ingest.pipeline.parse_pdf", return_value=_DOCUMENT
    ):
        stats = ingest_category("cs.IR", count=1, embed=False)

    spans = {s.name: s for s in traced_spans.get_finished_spans()}
    assert "ingest.ingest_category" in spans
    assert "ingest.paper" in spans

    run_span = spans["ingest.ingest_category"]
    assert run_span.attributes["ingest.category"] == "cs.IR"
    assert run_span.attributes["ingest.count_requested"] == 1
    assert run_span.attributes["ingest.embed"] is False
    assert run_span.attributes["ingest.papers_seen"] == stats["papers_seen"] == 1
    assert run_span.attributes["ingest.papers_new"] == stats["papers_new"] == 1
    assert run_span.attributes["ingest.chunks_new"] == stats["chunks_new"]

    paper_span = spans["ingest.paper"]
    assert paper_span.attributes["ingest.arxiv_id"] == "9999.00001"
    assert paper_span.parent.span_id == run_span.context.span_id


def test_ingest_category_records_a_run_span_even_with_zero_papers(traced_spans):
    from src.ingest.pipeline import ingest_category

    fake_session = MagicMock()
    with patch("src.ingest.pipeline.fetch_papers_paginated", return_value=[]), patch(
        "src.ingest.pipeline.get_session", return_value=fake_session
    ):
        stats = ingest_category("cs.IR", count=1, embed=False)

    spans = {s.name: s for s in traced_spans.get_finished_spans()}
    assert "ingest.ingest_category" in spans
    assert "ingest.paper" not in spans  # nothing to iterate, no child span emitted
    assert stats["papers_seen"] == 0
