import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.app.routes.upload import upload_submit
from src.ingest.document_parser import ParsedDocument, ParsedSection

pytestmark = pytest.mark.unit


def test_upload_submit_is_a_plain_sync_route_not_a_coroutine():
    # Real bug found in review: this route used to be `async def`, but
    # every step inside it (parse_pdf, embed_passages's synchronous
    # httpx.post, the DB writes) is blocking code with no real `await` —
    # an `async def` route with only blocking work inside runs directly
    # on the single event loop thread instead of FastAPI's automatic
    # threadpool (which every OTHER route in this codebase gets for free
    # by being a plain `def` handler). Being sync is what lets FastAPI
    # thread-pool it like ask.py/pipeline.py/corpus.py.
    assert not inspect.iscoroutinefunction(upload_submit)

# Phase 2 §5 (stage 6): private uploads only exist on the
# RETRIEVAL_BACKEND=postgres path. parse_pdf/chunk_document/embed_passages
# and every DB call are mocked here (unit, not integration — no real PDF
# parsing, embed API, or Postgres round trip); the actual isolation SQL
# (owner_session_id filtering) is covered directly by test_hybrid_postgres.py.


@pytest.fixture(autouse=True)
def _skip_real_rate_limiting():
    with patch("src.app.rate_limit.check_rate_limit", return_value=True):
        yield


def test_upload_page_unavailable_on_default_opensearch_backend(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    client = TestClient(app)
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Not available on this deployment" in response.content


def test_upload_post_unavailable_on_default_opensearch_backend(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    client = TestClient(app)
    response = client.post("/upload", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert response.status_code == 200
    assert b"Not available on this deployment" in response.content


def test_upload_page_available_on_postgres_backend(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app)
    with patch("src.app.routes.upload.get_session") as mock_get_session:
        fake_session = MagicMock()
        fake_session.execute.return_value.scalars.return_value.all.return_value = []
        fake_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = fake_session
        response = client.get("/upload")
    assert response.status_code == 200
    assert b"Not available on this deployment" not in response.content
    assert b"Nothing uploaded yet this session" in response.content


def test_upload_rejects_a_file_over_the_size_limit(monkeypatch):
    # _MAX_UPLOAD_MB is resolved once at import time (same convention as
    # rate_limit.py's _LIMIT/_WINDOW_SECONDS) — patch the module constant
    # directly rather than the env var, which a fresh setenv can't reach.
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    monkeypatch.setattr("src.app.routes.upload._MAX_UPLOAD_MB", 1)
    client = TestClient(app)
    oversized = b"x" * (2 * 1024 * 1024)
    with patch("src.app.routes.upload.get_session") as mock_get_session:
        fake_session = MagicMock()
        fake_session.execute.return_value.scalars.return_value.all.return_value = []
        fake_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = fake_session
        response = client.post("/upload", files={"file": ("paper.pdf", oversized, "application/pdf")})
    assert response.status_code == 200
    assert b"larger than the 1 MB limit" in response.content


def test_upload_rejects_a_pdf_with_no_extractable_text(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app)
    with patch("src.app.routes.upload.parse_pdf", return_value=ParsedDocument(sections=[])), patch(
        "src.app.routes.upload.get_session"
    ) as mock_get_session:
        fake_session = MagicMock()
        fake_session.execute.return_value.scalars.return_value.all.return_value = []
        fake_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = fake_session
        response = client.post("/upload", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert response.status_code == 200
    assert b"any extractable text" in response.content


def test_upload_surfaces_a_missing_byok_key_as_a_friendly_message(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app)
    parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="a" * 200)])
    with patch("src.app.routes.upload.parse_pdf", return_value=parsed), patch(
        "src.app.routes.upload.embed_passages", side_effect=RuntimeError("JINA_API_KEY is not set")
    ), patch("src.app.routes.upload.get_session") as mock_get_session:
        fake_session = MagicMock()
        fake_session.execute.return_value.scalars.return_value.all.return_value = []
        fake_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = fake_session
        response = client.post("/upload", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert response.status_code == 200
    assert b"needs an API key" in response.content


def test_successful_upload_redirects_to_ask_scoped_to_the_new_document(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app, follow_redirects=False)
    parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="a" * 200)])
    fake_embed_result = MagicMock(vectors=[[0.1] * 1024], model="jina-embeddings-v3", dimension=1024)
    with patch("src.app.routes.upload.parse_pdf", return_value=parsed), patch(
        "src.app.routes.upload.embed_passages", return_value=fake_embed_result
    ), patch("src.app.routes.upload.get_active_embed_provider", return_value="jina"), patch(
        "src.app.routes.upload.get_session"
    ) as mock_get_session:
        fake_session = MagicMock()
        mock_get_session.return_value = fake_session
        response = client.post("/upload", files={"file": ("my paper.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert response.status_code == 303
    assert response.headers["location"].startswith("/ask?document_id=upload-")
    # Both the Paper and Chunk rows were added before commit.
    assert fake_session.add.call_count == 2
    fake_session.commit.assert_called_once()
