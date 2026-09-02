from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.ingest.document_parser import ParsedDocument

# Provisional default: fixed-size character windows with overlap, applied
# within each detected section so a chunk never straddles a section
# boundary. This is ONE candidate chunking strategy, not THE decision —
# A7's ablation is what actually decides the chunker, on measured
# evidence. Picking a reasonable default now is what lets ingestion (and
# therefore everything downstream of it) happen before that ablation runs
# — see the plan's Build order, Phase 2.
_CHUNK_SIZE_CHARS = 1600  # ~400 tokens at a rough 4 chars/token
_CHUNK_OVERLAP_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section: str | None
    text: str
    char_start: int
    char_end: int


def _chunk_id(arxiv_id: str, text: str) -> str:
    # Content-addressed, per A3b: a chunk keeps its identity across
    # re-chunking of unrelated papers, which is what lets qrels labeled
    # against a chunk_id outlive routine re-ingestion elsewhere in the
    # corpus.
    return hashlib.sha256((arxiv_id + text).encode()).hexdigest()


def _windows(text: str, size: int = _CHUNK_SIZE_CHARS, overlap: int = _CHUNK_OVERLAP_CHARS) -> list[tuple[str, int, int]]:
    """Generalized for A7's chunking-strategy ablation (src/ingest/
    chunking_strategies.py) to accept arbitrary size/overlap — but
    `chunk_document` below still calls this with the module's own
    `_CHUNK_SIZE_CHARS`/`_CHUNK_OVERLAP_CHARS` defaults, so production
    chunking behavior is byte-for-byte unchanged by this generalization.
    """
    if not text:
        return []
    if len(text) <= size:
        return [(text, 0, len(text))]

    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        result.append((text[start:end], start, end))
        if end == len(text):
            break
        start = end - overlap
    return result


def chunk_document(arxiv_id: str, document: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    offset = 0
    for section in document.sections:
        for window_text, rel_start, rel_end in _windows(section.text):
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
