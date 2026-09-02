import pytest
from fastapi.testclient import TestClient

from src.app.main import app

pytestmark = pytest.mark.unit

# Real bug found in review: GET /ask used to render document_id into the
# "scoped to your document" privacy banner unconditionally. On the
# default OpenSearch backend — where /upload is disabled and ask_submit
# already silently drops document scoping — a stale bookmark or shared
# /ask?document_id=... link would show a privacy claim the POST handler
# doesn't actually keep.


def test_document_id_banner_is_suppressed_on_the_default_opensearch_backend(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    client = TestClient(app)
    response = client.get("/ask", params={"document_id": "upload-xyz"})
    assert response.status_code == 200
    assert b"Scoped to your uploaded document" not in response.content


def test_document_id_banner_shows_on_the_postgres_backend(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app)
    response = client.get("/ask", params={"document_id": "upload-xyz"})
    assert response.status_code == 200
    assert b"Scoped to your uploaded document" in response.content
