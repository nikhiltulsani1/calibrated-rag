import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.app.middleware import HEADER_NAME

pytestmark = pytest.mark.unit

# Note: this project has no FastAPI/Starlette OTel auto-instrumentation
# (not in requirements.txt — only opentelemetry-api/sdk/exporter), so
# there is no ambient root span during a plain request like /health. The
# middleware's `span.set_attribute("request_id", ...)` call only lands on
# a *real* recorded span for routes that already open one themselves
# (e.g. /ask, via reason.answer_query) — covered implicitly by those
# routes' own existing tests, not duplicated here against a route that
# creates no span at all.


def test_response_carries_a_generated_request_id():
    client = TestClient(app)
    response = client.get("/health")
    assert HEADER_NAME in response.headers
    assert len(response.headers[HEADER_NAME]) > 0


def test_two_requests_get_two_different_request_ids():
    client = TestClient(app)
    first = client.get("/health").headers[HEADER_NAME]
    second = client.get("/health").headers[HEADER_NAME]
    assert first != second


def test_inbound_request_id_header_is_honored_not_overwritten():
    client = TestClient(app)
    response = client.get("/health", headers={HEADER_NAME: "caller-supplied-id"})
    assert response.headers[HEADER_NAME] == "caller-supplied-id"
