from __future__ import annotations

import re
import time

import httpx

from src.ingest.chunker import Chunk, _chunk_id, _windows
from src.ingest.document_parser import ParsedDocument
from src.index.embedder import embed_passages

# A7's five candidate chunking strategies, all sharing chunk_document's
# exact interface `(arxiv_id: str, document: ParsedDocument) -> list[Chunk]`
# so the ablation script (evals/run_chunking_eval.py) can iterate them
# uniformly. Reuses Chunk/_chunk_id/_windows from chunker.py rather than
# duplicating them — chunker.py's own chunk_document (the production
# default) is strategy #2 below, re-exported, not reimplemented.

_DEFAULT_SIZE = 1600
_DEFAULT_OVERLAP = 200


def chunk_fixed_no_overlap(arxiv_id: str, document: ParsedDocument) -> list[Chunk]:
    """Fixed-size windows, zero overlap — the cheapest strategy (fewest
    chunks for a given size), and the natural baseline to compare
    overlap's benefit against.
    """
    chunks: list[Chunk] = []
    offset = 0
    for section in document.sections:
        for window_text, rel_start, rel_end in _windows(section.text, size=_DEFAULT_SIZE, overlap=0):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(arxiv_id, window_text),
                    section=section.heading,
                    text=window_text,
                    char_start=offset + rel_start,
                    char_end=offset + rel_end,
                )
            )
        offset += len(section.text)
    return chunks


def chunk_fixed_overlap(arxiv_id: str, document: ParsedDocument) -> list[Chunk]:
    """Today's actual production chunker — re-exported under the
    ablation's naming convention, not reimplemented."""
    from src.ingest.chunker import chunk_document

    return chunk_document(arxiv_id, document)


_SEPARATORS = ["\n\n", "\n", ". "]


def _recursive_split(text: str, size: int, separators: list[str]) -> list[tuple[str, int, int]]:
    """Splits on a separator, then MERGES consecutive pieces back up to
    `size` — the actual point of a recursive character splitter: stay
    close to (not just under) the target size, on natural boundaries.
    A piece still too large on its own recurses into the next separator;
    once separators are exhausted, falls back to a fixed-size window.

    First version of this function only divided, never merged — real
    PDF-extracted text line-wraps every ~60-80 chars, so splitting on
    "\\n" alone (before ever merging back) produced thousands of
    single-line fragments per paper instead of ~size-sized chunks,
    caught by comparing this strategy's real chunk count (4,233) against
    every other strategy's (500-4,300 range... all in the low thousands)
    during an actual ablation run, not assumed correct from reading the
    code. A standard, independently-implemented technique (see the
    plan's IP posture note) — not adapted from any specific library.
    """
    if len(text) <= size:
        return [(text, 0, len(text))]

    if not separators:
        return _windows(text, size=size, overlap=0)

    sep, rest = separators[0], separators[1:]
    pieces = text.split(sep)
    if len(pieces) == 1:
        # this separator doesn't occur in the text at all — try the next
        return _recursive_split(text, size, rest)

    result: list[tuple[str, int, int]] = []
    buffer_start: int | None = None
    buffer_end = 0
    offset = 0
    for i, piece in enumerate(pieces):
        piece_start = offset
        piece_end = offset + len(piece)
        offset = piece_end + len(sep)

        if not piece:
            continue

        candidate_end = piece_end  # merging this piece into the current buffer
        if buffer_start is None:
            candidate_len = len(piece)
        else:
            candidate_len = candidate_end - buffer_start

        if candidate_len <= size:
            if buffer_start is None:
                buffer_start = piece_start
            buffer_end = piece_end
            continue

        # this piece doesn't fit in the current buffer — flush the buffer
        if buffer_start is not None:
            result.append((text[buffer_start:buffer_end], buffer_start, buffer_end))
            buffer_start = None

        if piece_end - piece_start > size:
            # the piece alone is still too big — recurse with the next separator
            for sub_text, rel_start, rel_end in _recursive_split(piece, size, rest):
                result.append((sub_text, piece_start + rel_start, piece_start + rel_end))
        else:
            buffer_start = piece_start
            buffer_end = piece_end

    if buffer_start is not None:
        result.append((text[buffer_start:buffer_end], buffer_start, buffer_end))

    return result


def chunk_recursive_separator(arxiv_id: str, document: ParsedDocument) -> list[Chunk]:
    """Splits on paragraph breaks first, then line breaks, then sentence
    boundaries, recursing until every piece is under the target size —
    tries to keep chunks on natural text boundaries rather than cutting
    mid-sentence the way fixed windows do.
    """
    chunks: list[Chunk] = []
    offset = 0
    for section in document.sections:
        for window_text, rel_start, rel_end in _recursive_split(section.text, _DEFAULT_SIZE, list(_SEPARATORS)):
            if not window_text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(arxiv_id, window_text),
                    section=section.heading,
                    text=window_text,
                    char_start=offset + rel_start,
                    char_end=offset + rel_end,
                )
            )
        offset += len(section.text)
    return chunks


def chunk_structure_aware(arxiv_id: str, document: ParsedDocument) -> list[Chunk]:
    """One chunk per section — section boundaries already exist in
    document_parser.py's output, so this is genuinely cheap to add.
    Only sections larger than the target size get sub-split (via the
    same fixed-window logic as the other strategies), never merged
    across sections.
    """
    chunks: list[Chunk] = []
    offset = 0
    for section in document.sections:
        text = section.text
        if len(text) <= _DEFAULT_SIZE:
            if text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(arxiv_id, text),
                        section=section.heading,
                        text=text,
                        char_start=offset,
                        char_end=offset + len(text),
                    )
                )
        else:
            for window_text, rel_start, rel_end in _windows(text, size=_DEFAULT_SIZE, overlap=_DEFAULT_OVERLAP):
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(arxiv_id, window_text),
                        section=section.heading,
                        text=window_text,
                        char_start=offset + rel_start,
                        char_end=offset + rel_end,
                    )
                )
        offset += len(text)
    return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SIMILARITY_DROP_THRESHOLD = 0.35  # starting point, not tuned — see the plan's honesty rule


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_SEMANTIC_SECTION_PACING_SECONDS = 1.0
# Real, verified limit (2026-08-20): Jina's free tier caps at 100,000
# tokens/minute — a TOKEN-VOLUME limit, not a request-count one, so
# fixed per-call pacing alone doesn't prevent it if a handful of large
# section batches land in the same rolling minute (confirmed live via
# the 429 body: "Token rate limit exceeded: 123,608/100,000 tokens per
# minute"). 65s is comfortably past the 60s window, not guessed.
_RATE_LIMIT_BACKOFF_SECONDS = 65.0
_RATE_LIMIT_MAX_RETRIES = 3


def _with_rate_limit_backoff(embed_fn, texts: list[str]):
    """Wraps any Jina embed_*-style call (embed_passages, embed_queries)
    with real retry-on-429 — see the module constants above for why
    fixed pacing alone isn't enough against a token-VOLUME limit.
    """
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return embed_fn(texts)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempt == _RATE_LIMIT_MAX_RETRIES:
                raise
            time.sleep(_RATE_LIMIT_BACKOFF_SECONDS)


def _embed_passages_with_backoff(texts: list[str]):
    return _with_rate_limit_backoff(embed_passages, texts)


def chunk_semantic(arxiv_id: str, document: ParsedDocument) -> list[Chunk]:
    """Sentence-level split per section, placing a chunk boundary where
    consecutive-sentence embedding similarity drops below threshold —
    the costliest strategy: it spends embedding calls on *boundary
    decisions*, not just the final chunk text, via the existing
    embed_passages (batched once per section, not once per sentence
    pair, to keep the real Jina call count bounded).

    One real embed_passages call per multi-sentence section, paced —
    a real 429 from Jina during the ablation (many sections across 8
    papers, fired back-to-back with zero pacing) is why this exists;
    zero pacing worked fine for the single-call-per-strategy strategies
    but not for this one, which makes O(sections) calls.
    """
    chunks: list[Chunk] = []
    offset = 0
    for section in document.sections:
        text = section.text
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) <= 1:
            if text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(arxiv_id, text),
                        section=section.heading,
                        text=text,
                        char_start=offset,
                        char_end=offset + len(text),
                    )
                )
            offset += len(text)
            continue

        time.sleep(_SEMANTIC_SECTION_PACING_SECONDS)
        vectors = _embed_passages_with_backoff(sentences).vectors

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = _cosine(vectors[i - 1], vectors[i])
            current_len = sum(len(s) for s in groups[-1])
            if sim < _SIMILARITY_DROP_THRESHOLD or current_len >= _DEFAULT_SIZE:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        cursor = 0
        for group in groups:
            group_text = " ".join(group)
            start = text.find(group[0], cursor)
            if start == -1:
                start = cursor
            end = start + len(group_text)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(arxiv_id, group_text),
                    section=section.heading,
                    text=group_text,
                    char_start=offset + start,
                    char_end=offset + end,
                )
            )
            cursor = end
        offset += len(text)
    return chunks


STRATEGIES = {
    "fixed_no_overlap": chunk_fixed_no_overlap,
    "fixed_overlap": chunk_fixed_overlap,
    "recursive_separator": chunk_recursive_separator,
    "structure_aware": chunk_structure_aware,
    "semantic": chunk_semantic,
}
