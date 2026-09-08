import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.app.routes.upload import upload_submit

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


# Phase 3 stage 3: the "no extractable text" / "missing BYOK key" cases
# used to be synchronous route-level checks — now they happen INSIDE the
# background job (process_upload), not at enqueue time, so they're
# covered by tests/unit/test_upload_job.py directly against the job
# function instead of the route. The route itself no longer knows or
# cares what process_upload will find.


def test_successful_upload_enqueues_a_job_and_redirects_to_its_status(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app, follow_redirects=False)
    fake_job = MagicMock(id="fake-job-id-123")
    fake_job.meta = {}
    with patch("src.app.routes.upload.get_queue") as mock_get_queue:
        mock_get_queue.return_value.enqueue.return_value = fake_job
        response = client.post("/upload", files={"file": ("my paper.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert response.status_code == 303
    assert response.headers["location"] == "/upload?job_id=fake-job-id-123"
    # session_id was recorded on the job so the status endpoint can
    # later verify the same visitor is the one polling it.
    assert fake_job.meta["session_id"]
    fake_job.save_meta.assert_called_once()
    mock_get_queue.return_value.enqueue.assert_called_once()


# ---------------------------------------------------------------------
# Phase 3, Stage 2: per-page update. Ownership check mirrors
# store/runs.py::load_run's established pattern exactly — the real
# scoped-delete SQL is covered by an integration test against real
# Postgres (tests/integration/test_upload_page_replace_integration.py).
# ---------------------------------------------------------------------


def test_page_replace_unavailable_on_default_opensearch_backend(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_BACKEND", raising=False)
    client = TestClient(app)
    response = client.post(
        "/upload/upload-xyz/pages/1", files={"file": ("page1.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )
    assert response.status_code == 200
    assert b"Not available on this deployment" in response.content


def test_page_replace_returns_not_found_for_a_missing_document(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app)
    with patch("src.app.routes.upload.get_session") as mock_get_session:
        fake_session = MagicMock()
        fake_session.get.return_value = None
        fake_session.execute.return_value.scalars.return_value.all.return_value = []
        fake_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = fake_session
        response = client.post(
            "/upload/upload-xyz/pages/1", files={"file": ("page1.pdf", b"%PDF-1.4 fake", "application/pdf")}
        )
    assert response.status_code == 200
    assert b"No document found" in response.content


def test_page_replace_returns_not_found_for_a_different_owners_document(monkeypatch):
    # Same "indistinguishable from not-found" rule as load_run — never a
    # distinguishable 403 that would confirm the document_id exists.
    monkeypatch.setenv("RETRIEVAL_BACKEND", "postgres")
    client = TestClient(app)
    other_owners_paper = MagicMock(owner_session_id="a-different-session")
    with patch("src.app.routes.upload.get_session") as mock_get_session:
        fake_session = MagicMock()
        fake_session.get.return_value = other_owners_paper
        fake_session.execute.return_value.scalars.return_value.all.return_value = []
        fake_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = fake_session
        response = client.post(
            "/upload/upload-xyz/pages/1", files={"file": ("page1.pdf", b"%PDF-1.4 fake", "application/pdf")}
        )
    assert response.status_code == 200
    assert b"No document found" in response.content


# The success path (ownership matches -> scoped delete -> new chunks
# inserted, every other page untouched) needs a real session_id
# consistently threaded through middleware + DB rows to assert
# meaningfully — covered by
# tests/integration/test_upload_page_replace_integration.py against real
# local Postgres instead of fragile mocking here.


# ---------------------------------------------------------------------
# Phase 3, Stage 3: job status polling. RQ's Job.fetch is mocked here —
# the real Redis round trip (including the decode_responses gotcha) is
# covered by tests/unit/test_platform_queue.py.
# ---------------------------------------------------------------------


def test_status_endpoint_reports_processing_while_the_job_is_unfinished():
    monkeypatch_env = {"RETRIEVAL_BACKEND": "postgres"}
    client = TestClient(app)
    fake_job = MagicMock(meta={"session_id": "visitor-a"})
    fake_job.is_finished = False
    fake_job.is_failed = False
    with patch.dict("os.environ", monkeypatch_env), patch(
        "src.app.routes.upload.Job.fetch", return_value=fake_job
    ), patch("src.app.middleware.get_or_create_session_id", return_value=("visitor-a", False)):
        response = client.get("/upload/status/some-job-id")
    assert response.status_code == 200
    assert b"Processing your upload" in response.content


def test_status_endpoint_reports_done_with_a_link_to_ask():
    client = TestClient(app)
    fake_job = MagicMock(meta={"session_id": "visitor-a"})
    fake_job.is_finished = True
    fake_job.result = {"status": "ok", "document_id": "upload-abc"}
    with patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}), patch(
        "src.app.routes.upload.Job.fetch", return_value=fake_job
    ), patch("src.app.middleware.get_or_create_session_id", return_value=("visitor-a", False)):
        response = client.get("/upload/status/some-job-id")
    assert response.status_code == 200
    assert b"/ask?document_id=upload-abc" in response.content


def test_status_endpoint_reports_the_jobs_own_friendly_error_message():
    client = TestClient(app)
    fake_job = MagicMock(meta={"session_id": "visitor-a"})
    fake_job.is_finished = True
    fake_job.result = {"status": "error", "message": "needs an API key that isn't configured"}
    with patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}), patch(
        "src.app.routes.upload.Job.fetch", return_value=fake_job
    ), patch("src.app.middleware.get_or_create_session_id", return_value=("visitor-a", False)):
        response = client.get("/upload/status/some-job-id")
    assert response.status_code == 200
    assert b"needs an API key" in response.content


def test_status_endpoint_never_leaks_a_raw_traceback_on_a_failed_job():
    # A genuinely unexpected bug fails the RQ job itself (job.is_failed),
    # distinct from process_upload's own caught error cases. job.exc_info
    # is a raw formatted traceback string — this codebase never shows
    # that to a visitor, so the endpoint must render a generic message
    # instead, never job.exc_info's actual content.
    client = TestClient(app)
    fake_job = MagicMock(meta={"session_id": "visitor-a"})
    fake_job.is_finished = False
    fake_job.is_failed = True
    fake_job.exc_info = "Traceback (most recent call last):\n  secret internal path\nZeroDivisionError: boom"
    with patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}), patch(
        "src.app.routes.upload.Job.fetch", return_value=fake_job
    ), patch("src.app.middleware.get_or_create_session_id", return_value=("visitor-a", False)):
        response = client.get("/upload/status/some-job-id")
    assert response.status_code == 200
    assert b"secret internal path" not in response.content
    assert b"Traceback" not in response.content


def test_status_endpoint_treats_a_different_owners_job_as_unknown():
    client = TestClient(app)
    fake_job = MagicMock(meta={"session_id": "a-different-visitor"})
    with patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}), patch(
        "src.app.routes.upload.Job.fetch", return_value=fake_job
    ), patch("src.app.middleware.get_or_create_session_id", return_value=("visitor-b", False)):
        response = client.get("/upload/status/some-job-id")
    assert response.status_code == 200
    assert b"No upload found" in response.content


def test_status_endpoint_treats_a_missing_job_as_unknown():
    from rq.exceptions import NoSuchJobError

    client = TestClient(app)
    with patch.dict("os.environ", {"RETRIEVAL_BACKEND": "postgres"}), patch(
        "src.app.routes.upload.Job.fetch", side_effect=NoSuchJobError
    ):
        response = client.get("/upload/status/nonexistent-job-id")
    assert response.status_code == 200
    assert b"No upload found" in response.content
