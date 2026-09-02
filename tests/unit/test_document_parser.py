import pytest

from src.ingest.document_parser import _line_text_and_size, _looks_like_heading

pytestmark = pytest.mark.unit


def _fake_line(text: str, size: float) -> dict:
    return {"spans": [{"text": text, "size": size}]}


def test_line_text_and_size_joins_spans_and_takes_max_size():
    line = {"spans": [{"text": "Hello ", "size": 10.0}, {"text": "World", "size": 12.0}]}
    text, size = _line_text_and_size(line)
    assert text == "Hello World"
    assert size == 12.0


def test_line_text_strips_embedded_nul_bytes():
    # Found by actually running this against a real arXiv PDF and hitting
    # Postgres's "text fields cannot contain NUL bytes" — see the
    # document_parser.py comment and the ingestion pipeline README entry.
    line = {"spans": [{"text": "bad\x00text", "size": 10.0}]}
    text, _ = _line_text_and_size(line)
    assert "\x00" not in text
    assert text == "badtext"


def test_heading_needs_meaningfully_larger_size():
    assert _looks_like_heading("Introduction", size=11.5, body_size=10.0) is True
    assert _looks_like_heading("Introduction", size=10.2, body_size=10.0) is False


def test_heading_rejects_long_text():
    long_text = "x" * 100
    assert _looks_like_heading(long_text, size=20.0, body_size=10.0) is False


def test_heading_rejects_sentence_trailing_punctuation():
    assert _looks_like_heading("This looks like a sentence.", size=20.0, body_size=10.0) is False


def test_heading_rejects_empty_text():
    assert _looks_like_heading("", size=20.0, body_size=10.0) is False


def test_heading_rejects_zero_body_size():
    assert _looks_like_heading("Short", size=20.0, body_size=0.0) is False
