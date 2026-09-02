import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import scripts.load_test as load_test

pytestmark = pytest.mark.unit


def test_load_questions_returns_real_gold_set_queries():
    questions = load_test._load_questions()
    assert len(questions) > 0
    assert all(isinstance(q, str) and q for q in questions)


def test_one_request_detects_a_passed_through_rate_limit():
    fake_response = MagicMock(status_code=200, text="Too Many Requests from this address")
    fake_client = AsyncMock()
    fake_client.post.return_value = fake_response

    result = asyncio.run(load_test._one_request(fake_client, "some query"))
    assert result["rate_limited"] is True
    assert result["ok"] is False


def test_one_request_detects_this_deployments_own_rate_limiter():
    # Real bug found by reading the actual container logs during a real
    # R5 run: our own R6 limiter returns a genuine HTTP 429 with body
    # `{"detail":"Too many requests..."}` (lowercase 'r') — the original
    # case-sensitive text check for "Too Many Requests" missed this
    # entirely and mislabeled it "application/network error". Status
    # code is the robust signal for this case, not body text.
    fake_response = MagicMock(status_code=429, text='{"detail":"Too many requests from this address — try again in under 60 seconds."}')
    fake_client = AsyncMock()
    fake_client.post.return_value = fake_response

    result = asyncio.run(load_test._one_request(fake_client, "some query"))
    assert result["rate_limited"] is True
    assert result["ok"] is False
    assert result["constraint_kind"] == "app rate limit (this deployment's own R6 limiter)"


def test_one_request_marks_a_clean_200_as_ok():
    fake_response = MagicMock(status_code=200, text="<div class='card'>a real answer</div>")
    fake_client = AsyncMock()
    fake_client.post.return_value = fake_response

    result = asyncio.run(load_test._one_request(fake_client, "some query"))
    assert result["ok"] is True
    assert result["rate_limited"] is False


def test_one_request_captures_connection_errors_without_raising():
    fake_client = AsyncMock()
    fake_client.post.side_effect = ConnectionError("refused")

    result = asyncio.run(load_test._one_request(fake_client, "some query"))
    assert result["ok"] is False
    assert "ConnectionError" in result["error"]


def test_write_report_flags_the_binding_constraint(tmp_path, monkeypatch):
    report_path = tmp_path / "REPORT.md"
    report_path.write_text("# existing content\n", encoding="utf-8")
    monkeypatch.setattr(load_test, "_REPO_ROOT", tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "REPORT.md").write_text("# existing content\n", encoding="utf-8")

    results = [
        {"concurrency": 1, "n_requests": 15, "p50_ms": 500, "p95_ms": 900, "p99_ms": 1000, "mean_ms": 600, "error_rate": 0.0, "binding_constraint": None},
        {"concurrency": 10, "n_requests": 15, "p50_ms": 1200, "p95_ms": 3000, "p99_ms": 4000, "mean_ms": 1500, "error_rate": 0.4, "binding_constraint": "provider rate limit (Groq)"},
    ]
    with patch("scripts.load_test._load_questions", return_value=["q"] * 31):
        load_test.write_report(results)

    written = (tmp_path / "evals" / "REPORT.md").read_text(encoding="utf-8")
    assert "R5" in written
    assert "provider rate limit (Groq)" in written
    assert "concurrency 10" in written
