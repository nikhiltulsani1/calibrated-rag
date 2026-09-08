import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.ingest.document_parser import ParsedDocument, ParsedSection
from src.store.relational import get_session
from src.store.schema import Chunk, Paper

pytestmark = pytest.mark.integration

# Phase 3, Stage 2: real Postgres round trip for the per-page update
# endpoint — the scoped delete (only the target page's chunks change,
# every other page's are untouched) is exactly the kind of thing a mock
# can't meaningfully prove; it needs the real SQL to actually run.


@pytest.fixture
def session():
    s = get_session()
    yield s
    s.close()


def test_replacing_one_page_only_touches_that_pages_chunks(session):
    document_id = f"upload-pagetest-{uuid.uuid4().hex}"
    session.add(
        Paper(
            arxiv_id=document_id,
            title="Multi-page test doc",
            authors=[],
            abstract="",
            category=[],
            url="",
            source="upload",
            owner_session_id="visitor-a",
        )
    )
    session.add(
        Chunk(
            chunk_id=f"{document_id}-page1-chunk",
            paper_id=document_id,
            section=None,
            text="original page 1 content",
            char_start=0,
            char_end=10,
            page_number=1,
            owner_session_id="visitor-a",
        )
    )
    session.add(
        Chunk(
            chunk_id=f"{document_id}-page2-chunk",
            paper_id=document_id,
            section=None,
            text="original page 2 content",
            char_start=0,
            char_end=10,
            page_number=2,
            owner_session_id="visitor-a",
        )
    )
    session.commit()

    try:
        parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="brand new page 1 content")])
        fake_embed_result = MagicMock(vectors=[[0.1] * 1024], model="jina-embeddings-v3", dimension=1024)

        client = TestClient(app, follow_redirects=False)
        with patch("src.app.routes.upload.parse_document", return_value=parsed), patch(
            "src.app.routes.upload.embed_passages", return_value=fake_embed_result
        ), patch("src.app.routes.upload.get_active_embed_provider", return_value="jina"), patch(
            "src.app.rate_limit.check_rate_limit", return_value=True
        ), patch(
            "src.app.middleware.get_or_create_session_id", return_value=("visitor-a", False)
        ), patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}):
            response = client.post(
                f"/upload/{document_id}/pages/1",
                files={"file": ("page1.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

        assert response.status_code == 303
        assert response.headers["location"] == f"/ask?document_id={document_id}"

        session.expire_all()
        remaining = session.query(Chunk).filter(Chunk.paper_id == document_id).all()

        page1_chunks = [c for c in remaining if c.page_number == 1]
        page2_chunks = [c for c in remaining if c.page_number == 2]

        # Page 2's original chunk is completely untouched.
        assert len(page2_chunks) == 1
        assert page2_chunks[0].chunk_id == f"{document_id}-page2-chunk"
        assert page2_chunks[0].text == "original page 2 content"

        # Page 1's old chunk is gone, replaced by new content.
        assert all(c.chunk_id != f"{document_id}-page1-chunk" for c in page1_chunks)
        assert any("brand new page 1 content" in c.text for c in page1_chunks)
    finally:
        for chunk in session.query(Chunk).filter(Chunk.paper_id == document_id).all():
            session.delete(chunk)
        paper = session.get(Paper, document_id)
        if paper:
            session.delete(paper)
        session.commit()


def test_replacing_a_page_on_a_different_visitors_document_is_rejected(session):
    document_id = f"upload-pagetest-{uuid.uuid4().hex}"
    session.add(
        Paper(
            arxiv_id=document_id,
            title="Someone else's doc",
            authors=[],
            abstract="",
            category=[],
            url="",
            source="upload",
            owner_session_id="visitor-a",
        )
    )
    session.commit()

    try:
        client = TestClient(app, follow_redirects=False)
        with patch(
            "src.app.rate_limit.check_rate_limit", return_value=True
        ), patch(
            "src.app.middleware.get_or_create_session_id", return_value=("visitor-b", False)
        ), patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}):
            response = client.post(
                f"/upload/{document_id}/pages/1",
                files={"file": ("page1.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )

        # Not-found-shaped, not a 403 — same convention as load_run.
        assert response.status_code == 200
        assert b"No document found" in response.content
    finally:
        paper = session.get(Paper, document_id)
        if paper:
            session.delete(paper)
        session.commit()
