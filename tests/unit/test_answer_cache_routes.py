from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.schemas.answer import Answer

pytestmark = pytest.mark.unit

# A6 route-level tests: verify a second identical /ask or /pipeline
# request is served from the cache without a second real run_traced_query
# call — the actual behavioral contract the cache exists to provide.
# run_traced_query, save_run, and get_session are all mocked here (this
# is a unit test, not integration — no real LLM/Postgres calls), only
# get_cached_trace/set_cached_trace run for real against Redis so the
# round trip itself is genuinely exercised.


def _fake_trace(query: str):
    from src.reason.state import StageTrace

    return StageTrace(
        original_query=query,
        answer=Answer(text="RLHF is reinforcement learning from human feedback.", citations=[], abstained=False),
    )


@pytest.fixture(autouse=True)
def _skip_real_rate_limiting():
    # These are unit tests — no real Redis call belongs here. /ask and
    # /pipeline both depend on enforce_rate_limit, which does hit Redis
    # for real by design (see rate_limit.py); check_rate_limit is the
    # one function it calls, so patching it is enough to isolate these
    # route tests without touching the rate limiter's own real behavior
    # or tests.
    with patch("src.app.rate_limit.check_rate_limit", return_value=True):
        yield


def test_ask_second_identical_request_hits_cache_not_run_traced_query():
    client = TestClient(app)
    query = "what is RLHF in the context of large language models"
    with patch("src.app.routes.ask.get_cached_trace", return_value=None), patch(
        "src.app.routes.ask.run_traced_query", return_value=_fake_trace(query)
    ) as mock_run, patch("src.app.routes.ask.save_run", return_value="run-1"), patch(
        "src.app.routes.ask.get_session"
    ), patch("src.app.routes.ask.set_cached_trace") as mock_set:
        first = client.post("/ask", data={"query": query})
    assert first.status_code == 200
    mock_run.assert_called_once()
    mock_set.assert_called_once()
    cached_trace = mock_set.call_args.args[1]

    with patch("src.app.routes.ask.get_cached_trace", return_value={"answer": {"text": cached_trace.answer.text}}), patch(
        "src.app.routes.ask.run_traced_query"
    ) as mock_run_second:
        second = client.post("/ask", data={"query": query})
    assert second.status_code == 200
    mock_run_second.assert_not_called()
    assert b"Served from cache" in second.content
    assert cached_trace.answer.text.encode() in second.content


def test_pipeline_second_identical_request_hits_cache_not_run_traced_query():
    client = TestClient(app)
    query = "what is RLHF in the context of large language models"
    with patch("src.app.routes.pipeline.get_cached_trace", return_value=None), patch(
        "src.app.routes.pipeline.run_traced_query", return_value=_fake_trace(query)
    ) as mock_run, patch("src.app.routes.pipeline.save_run", return_value="run-1"), patch(
        "src.app.routes.pipeline.get_session"
    ), patch("src.app.routes.pipeline.set_cached_trace") as mock_set:
        first = client.get("/pipeline", params={"query": query})
    assert first.status_code == 200
    mock_run.assert_called_once()
    mock_set.assert_called_once()

    from src.store.runs import serialize_trace

    cached_dict = serialize_trace(_fake_trace(query))
    with patch("src.app.routes.pipeline.get_cached_trace", return_value=cached_dict), patch(
        "src.app.routes.pipeline.run_traced_query"
    ) as mock_run_second:
        second = client.get("/pipeline", params={"query": query})
    assert second.status_code == 200
    mock_run_second.assert_not_called()
    assert b"Replaying a saved run" in second.content
