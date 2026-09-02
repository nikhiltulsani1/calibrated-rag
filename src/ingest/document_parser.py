from __future__ import annotations

import statistics
from dataclasses import dataclass

import httpx
import pymupdf

from src.ingest.paper_source import contact_header

_MIN_HEADING_SIZE_RATIO = 1.15
_MAX_HEADING_CHARS = 80


@dataclass(frozen=True)
class ParsedSection:
    heading: str | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    sections: list[ParsedSection]


def fetch_pdf_bytes(pdf_url: str) -> bytes:
    response = httpx.get(
        pdf_url,
        headers={"User-Agent": contact_header()},
        timeout=60.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.content


def _line_text_and_size(line: dict) -> tuple[str, float]:
    # PDF text extraction can yield embedded NUL bytes (certain embedded
    # font/ligature encodings decode to \x00) — found by actually running
    # this against a real paper and hitting Postgres's "text fields cannot
    # contain NUL bytes" at the write step. Strip at the source so nothing
    # downstream has to know about it.
    text = "".join(span["text"] for span in line["spans"]).replace("\x00", "").strip()
    size = max((span["size"] for span in line["spans"]), default=0.0)
    return text, size


def _looks_like_heading(text: str, size: float, body_size: float) -> bool:
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    if text.endswith((".", ",", ";")):
        return False
    return body_size > 0 and size >= body_size * _MIN_HEADING_SIZE_RATIO


def parse_pdf(pdf_bytes: bytes) -> ParsedDocument:
    """Layout-aware extraction: section headings survive as section
    boundaries rather than being flattened into one text blob (see the
    Infrastructure note on structured parsing).

    Heading detection is a font-size heuristic: a line whose text is
    short, doesn't trail off mid-sentence, and is meaningfully larger
    than the document's own body text size. This is a standard, publicly
    documented PDF-parsing technique — not a specific implementation
    copied from anywhere (see the plan's IP posture note) — deliberately
    simpler than a trained layout model. Revisit if A7's chunking
    ablation shows section boundaries landing wrong.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        lines: list[tuple[str, float]] = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  # 0 = text block; skip images/drawings
                    continue
                for line in block["lines"]:
                    text, size = _line_text_and_size(line)
                    if text:
                        lines.append((text, size))
    finally:
        doc.close()

    if not lines:
        return ParsedDocument(sections=[])

    body_size = statistics.mode(size for _, size in lines)

    sections: list[ParsedSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for text, size in lines:
        if _looks_like_heading(text, size, body_size):
            if current_lines:
                sections.append(ParsedSection(heading=current_heading, text=" ".join(current_lines)))
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)

    if current_lines:
        sections.append(ParsedSection(heading=current_heading, text=" ".join(current_lines)))

    return ParsedDocument(sections=sections)
