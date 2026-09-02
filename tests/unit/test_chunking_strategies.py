from unittest.mock import patch

import pytest

from src.index.embedder import EmbeddingResult
from src.ingest.chunking_strategies import (
    chunk_fixed_no_overlap,
    chunk_fixed_overlap,
    chunk_recursive_separator,
    chunk_semantic,
    chunk_structure_aware,
)
from src.ingest.document_parser import ParsedDocument, ParsedSection

pytestmark = pytest.mark.unit

_LONG_TEXT = "This is a real sentence about retrieval. " * 100  # well over 1600 chars


def _doc(*sections: tuple[str | None, str]) -> ParsedDocument:
    return ParsedDocument(sections=[ParsedSection(heading=h, text=t) for h, t in sections])


def test_fixed_no_overlap_produces_non_overlapping_windows():
    chunks = chunk_fixed_no_overlap("9999.00001", _doc(("Intro", _LONG_TEXT)))
    assert len(chunks) > 1
    # consecutive windows are back-to-back, not overlapping
    for a, b in zip(chunks, chunks[1:]):
        assert a.char_end == b.char_start


def test_fixed_overlap_matches_production_chunk_document():
    from src.ingest.chunker import chunk_document

    doc = _doc(("Intro", _LONG_TEXT))
    assert chunk_fixed_overlap("9999.00001", doc) == chunk_document("9999.00001", doc)


def test_recursive_separator_splits_on_paragraph_breaks_first():
    text = ("A short paragraph. " * 5 + "\n\n") * 20  # many small paragraphs, well over 1600 chars total
    chunks = chunk_recursive_separator("9999.00001", _doc(("Body", text)))
    assert len(chunks) > 1
    assert all(len(c.text) <= 1600 or "\n\n" not in c.text for c in chunks)


def test_recursive_separator_merges_small_pieces_up_toward_target_size():
    # real PDF-extracted text line-wraps every ~60-80 chars — the
    # regression this test guards: a purely-dividing (never merging)
    # splitter would emit one tiny chunk per line instead of merging
    # consecutive short lines up toward the 1600-char target.
    line = "A single wrapped line of body text here.\n"
    text = line * 100  # 100 short lines, ~4200 chars total
    chunks = chunk_recursive_separator("9999.00001", _doc(("Body", text)))
    assert len(chunks) < 10  # nowhere near one-chunk-per-line (100 lines)
    # most chunks should be substantially larger than a single line
    assert sum(1 for c in chunks if len(c.text) > 500) >= len(chunks) - 1


def test_recursive_separator_char_offsets_match_original_text():
    text = ("Sentence one. Sentence two. Sentence three. " * 50) + "\n\n" + ("Another paragraph. " * 50)
    chunks = chunk_recursive_separator("9999.00001", _doc(("Body", text)))
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text


def test_recursive_separator_falls_back_to_fixed_windows_with_no_separators():
    # no paragraph/line/sentence separators at all — must not infinite-loop
    # or raise, must fall back to a fixed-size split.
    text = "x" * 5000
    chunks = chunk_recursive_separator("9999.00001", _doc(("Body", text)))
    assert len(chunks) > 1
    assert all(len(c.text) <= 1600 for c in chunks)


def test_structure_aware_keeps_short_sections_whole():
    chunks = chunk_structure_aware("9999.00001", _doc(("Abstract", "a short abstract")))
    assert len(chunks) == 1
    assert chunks[0].text == "a short abstract"
    assert chunks[0].section == "Abstract"


def test_structure_aware_subsplits_oversized_sections():
    chunks = chunk_structure_aware("9999.00001", _doc(("Body", _LONG_TEXT)))
    assert len(chunks) > 1
    assert all(c.section == "Body" for c in chunks)


def test_semantic_keeps_single_sentence_section_whole_with_zero_embed_calls():
    with patch("src.ingest.chunking_strategies.embed_passages") as mock_embed:
        chunks = chunk_semantic("9999.00001", _doc(("Abstract", "Just one sentence here.")))
    assert len(chunks) == 1
    assert not mock_embed.called


def test_semantic_splits_on_similarity_drop():
    # three sentences: two "similar" (high cosine), one "different" —
    # boundary should land between the similar pair and the different one.
    fake_vectors = EmbeddingResult(
        model="fake",
        dimension=2,
        vectors=[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
        prompt_tokens=0,
    )
    text = "First sentence here. Second sentence here. Totally different topic here."
    with patch("src.ingest.chunking_strategies.embed_passages", return_value=fake_vectors):
        chunks = chunk_semantic("9999.00001", _doc(("Body", text)))
    assert len(chunks) == 2
    assert "First" in chunks[0].text and "Second" in chunks[0].text
    assert "Totally different" in chunks[1].text


def test_all_strategies_produce_content_addressed_chunk_ids():
    from src.ingest.chunking_strategies import STRATEGIES

    doc = _doc(("Intro", "a repeatable short section"))
    for name, fn in STRATEGIES.items():
        if name == "semantic":
            continue  # needs a real/mocked embedding call, covered above
        chunks_a = fn("9999.00001", doc)
        chunks_b = fn("9999.00001", doc)
        assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b], name
