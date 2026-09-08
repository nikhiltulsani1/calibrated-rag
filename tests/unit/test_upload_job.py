from unittest.mock import MagicMock, patch

import pytest

from src.ingest.document_parser import ParsedDocument, ParsedSection
from src.ingest.upload_job import process_upload
from src.platform.credentials import Credentials, get_credentials

pytestmark = pytest.mark.unit

# Phase 3, Stage 3: process_upload is the exact parse/chunk/embed/write
# sequence that used to run inline in routes/upload.py's POST handler,
# now run by the background worker thread instead. These tests mock
# DB/embed calls (unit, not integration) but exercise the real function
# body — the actual Postgres round trip is the same code path already
# covered by test_upload_route.py's older synchronous-era assertions
# and the Phase 3 integration tests.


def test_process_upload_writes_paper_and_chunk_rows_and_returns_ok():
    parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="a" * 200)])
    fake_embed_result = MagicMock(vectors=[[0.1] * 1024], model="jina-embeddings-v3", dimension=1024)
    with patch("src.ingest.upload_job.parse_document", return_value=parsed), patch(
        "src.ingest.upload_job.embed_passages", return_value=fake_embed_result
    ), patch("src.ingest.upload_job.get_active_embed_provider", return_value="jina"), patch(
        "src.ingest.upload_job.get_session"
    ) as mock_get_session:
        fake_session = MagicMock()
        mock_get_session.return_value = fake_session

        result = process_upload(
            "upload-abc", "paper.pdf", b"%PDF-1.4 fake", "visitor-a", Credentials(jina="fake-key"), "paper.pdf"
        )

    assert result == {"status": "ok", "document_id": "upload-abc"}
    assert fake_session.add.call_count == 2  # Paper + one Chunk
    fake_session.commit.assert_called_once()


def test_process_upload_returns_a_friendly_error_for_no_extractable_text():
    with patch("src.ingest.upload_job.parse_document", return_value=ParsedDocument(sections=[])):
        result = process_upload("upload-abc", "paper.pdf", b"fake", "visitor-a", Credentials(), "paper.pdf")

    assert result["status"] == "error"
    assert "extractable text" in result["message"]


def test_process_upload_returns_a_friendly_error_for_a_missing_byok_key():
    parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="a" * 200)])
    with patch("src.ingest.upload_job.parse_document", return_value=parsed), patch(
        "src.ingest.upload_job.embed_passages", side_effect=RuntimeError("JINA_API_KEY is not set")
    ):
        result = process_upload("upload-abc", "paper.pdf", b"fake", "visitor-a", Credentials(), "paper.pdf")

    assert result["status"] == "error"
    assert "needs an API key" in result["message"]


def test_process_upload_makes_credentials_visible_to_embed_passages_in_the_worker_thread():
    # Real bug this design exists to avoid: get_credentials() normally
    # reads a ContextVar set by CredentialsMiddleware per HTTP request —
    # that context doesn't exist in an RQ worker thread at all. This
    # confirms set_credentials/reset_credentials inside process_upload
    # itself is what makes the passed-in Credentials actually visible.
    parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="a" * 200)])
    seen_creds = {}

    def _fake_embed_passages(texts):
        seen_creds["jina"] = get_credentials().jina
        return MagicMock(vectors=[[0.1] * 1024], model="m", dimension=1024)

    with patch("src.ingest.upload_job.parse_document", return_value=parsed), patch(
        "src.ingest.upload_job.embed_passages", side_effect=_fake_embed_passages
    ), patch("src.ingest.upload_job.get_active_embed_provider", return_value="jina"), patch(
        "src.ingest.upload_job.get_session"
    ):
        process_upload(
            "upload-abc", "paper.pdf", b"fake", "visitor-a", Credentials(jina="visitor-jina-key"), "paper.pdf"
        )

    assert seen_creds["jina"] == "visitor-jina-key"
    # And it's cleaned up afterward — a leaked credential here would
    # bleed into whatever job this same worker thread processes next.
    assert get_credentials().jina is None


def test_process_upload_resets_credentials_even_when_the_job_fails():
    with patch("src.ingest.upload_job.parse_document", side_effect=RuntimeError("boom")):
        result = process_upload("upload-abc", "paper.pdf", b"fake", "visitor-a", Credentials(jina="k"), "paper.pdf")

    assert result["status"] == "error"
    assert get_credentials().jina is None


def test_process_upload_propagates_page_numbers_from_pdf_chunks():
    from src.ingest.chunker import Chunk as IngestChunk

    parsed = ParsedDocument(sections=[ParsedSection(heading=None, text="short", page=3)])
    fake_embed_result = MagicMock(vectors=[[0.1] * 1024], model="m", dimension=1024)
    with patch("src.ingest.upload_job.parse_document", return_value=parsed), patch(
        "src.ingest.upload_job.embed_passages", return_value=fake_embed_result
    ), patch("src.ingest.upload_job.get_active_embed_provider", return_value="jina"), patch(
        "src.ingest.upload_job.get_session"
    ) as mock_get_session:
        fake_session = MagicMock()
        mock_get_session.return_value = fake_session
        process_upload("upload-abc", "paper.pdf", b"fake", "visitor-a", Credentials(), "paper.pdf")

    chunk_row = [call.args[0] for call in fake_session.add.call_args_list if hasattr(call.args[0], "page_number")][0]
    assert chunk_row.page_number == 3
