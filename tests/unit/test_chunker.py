import pytest

from src.ingest.chunker import chunk_document
from src.ingest.document_parser import ParsedDocument, ParsedSection

pytestmark = pytest.mark.unit


def test_short_section_becomes_one_chunk():
    doc = ParsedDocument(sections=[ParsedSection(heading="Intro", text="short text")])
    chunks = chunk_document("1234.5678", doc)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len("short text")
    assert chunks[0].section == "Intro"


def test_long_section_windows_with_correct_overlap():
    text = "x" * 5000
    doc = ParsedDocument(sections=[ParsedSection(heading="Body", text=text)])
    chunks = chunk_document("1234.5678", doc)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert a.char_end - b.char_start == 200  # _CHUNK_OVERLAP_CHARS
    assert chunks[-1].char_end == len(text)


def test_chunk_ids_are_unique_and_content_addressed():
    # Non-repetitive text on purpose: a homogeneous "x" * n string makes
    # windows byte-identical by construction, which correctly collapses
    # to the same chunk_id (see test_identical_text_same_paper_... below)
    # and would make this test pass for the wrong reason.
    text = " ".join(f"word{i}" for i in range(2000))
    doc = ParsedDocument(sections=[ParsedSection(heading=None, text=text)])
    chunks = chunk_document("1234.5678", doc)
    ids = [c.chunk_id for c in chunks]
    assert len(chunks) > 1
    assert len(set(ids)) == len(ids)
    assert all(len(i) == 64 for i in ids)  # sha256 hex digest length


def test_same_text_different_paper_gives_different_chunk_id():
    doc = ParsedDocument(sections=[ParsedSection(heading=None, text="identical text")])
    chunks_a = chunk_document("paper.a", doc)
    chunks_b = chunk_document("paper.b", doc)
    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id


def test_identical_text_same_paper_gives_identical_chunk_id():
    # The property the ingestion pipeline's dedup step relies on — see
    # the pipeline.py comment on the within-batch collision it caused.
    doc = ParsedDocument(
        sections=[
            ParsedSection(heading="A", text="repeated caption"),
            ParsedSection(heading="B", text="repeated caption"),
        ]
    )
    chunks = chunk_document("1234.5678", doc)
    assert chunks[0].chunk_id == chunks[1].chunk_id


def test_empty_document_yields_no_chunks():
    assert chunk_document("1234.5678", ParsedDocument(sections=[])) == []


def test_offsets_accumulate_across_sections():
    doc = ParsedDocument(
        sections=[
            ParsedSection(heading="A", text="12345"),
            ParsedSection(heading="B", text="67890"),
        ]
    )
    chunks = chunk_document("1234.5678", doc)
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 5)
    assert (chunks[1].char_start, chunks[1].char_end) == (5, 10)
