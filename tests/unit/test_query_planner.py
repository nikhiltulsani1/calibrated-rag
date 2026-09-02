from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.platform.models import CompletionResult, ModelBinding, is_retryable_http_error, provider_has_key, usable_ladder
from src.retrieve.query_planner import _build_system_prompt, _cache_key, _parse_plan, _should_skip_rewrite, plan_query

pytestmark = pytest.mark.unit


def _http_error(status_code: int, error_body: dict | None = None) -> httpx.HTTPStatusError:
    response = MagicMock(status_code=status_code)
    response.json.return_value = error_body or {}
    return httpx.HTTPStatusError("boom", request=MagicMock(), response=response)


def test_parse_plan_valid_json():
    raw = '{"normalized": "n", "expansions": ["e1", "e2"], "filters": {}, "intent": "factual"}'
    plan = _parse_plan("original text", raw)
    assert plan.original == "original text"
    assert plan.normalized == "n"
    assert plan.expansions == ["e1", "e2"]
    assert plan.intent == "factual"


def test_parse_plan_defaults_missing_filters():
    raw = '{"normalized": "n", "expansions": [], "intent": "factual"}'
    plan = _parse_plan("q", raw)
    assert plan.filters == {}


def test_parse_plan_rejects_non_json():
    with pytest.raises(ValueError, match="did not return valid JSON"):
        _parse_plan("q", "this is not json")


def test_parse_plan_rejects_schema_mismatch():
    raw = '{"normalized": "n", "intent": "not_a_real_intent"}'
    with pytest.raises(ValueError, match="doesn't match QueryPlan"):
        _parse_plan("q", raw)


def test_cache_key_is_case_and_whitespace_insensitive():
    assert _cache_key("What Is RLHF") == _cache_key("what is rlhf  ")
    assert _cache_key("a") != _cache_key("b")


# ---------------------------------------------------------------------
# _should_skip_rewrite / QUERY_REWRITE_MODE=conditional — plan A2, added
# after A3's own report showed rewrite costing ~5.5s p50 on top of the
# rest of the pipeline for +0.037 nDCG@10. Deterministic, fails toward
# NOT skipping (running the real call) on any doubt.
# ---------------------------------------------------------------------


def test_skips_when_long_specific_and_in_domain():
    query = "What method does AnnoIndex use to build its annotation index for structured filtering?"
    assert _should_skip_rewrite(query) is True


def test_does_not_skip_short_queries():
    assert _should_skip_rewrite("what is RLHF") is False


def test_does_not_skip_queries_with_a_bare_short_acronym():
    # Expanding an unresolved acronym is exactly rewrite's job — a query
    # this long would otherwise pass the word-count check.
    query = "How does the RLHF process actually work in this training setup"
    assert _should_skip_rewrite(query) is False


def test_does_not_skip_queries_with_no_domain_vocabulary_overlap():
    # No overlap with the domain anchor terms at all — also the only
    # signal available for out_of_scope classification, so this must
    # still go through the real call rather than default to "factual".
    query = "What is a good recipe for baking sourdough bread this weekend"
    assert _should_skip_rewrite(query) is False


def test_plan_query_skips_llm_call_in_conditional_mode(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_MODE", "conditional")
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    query = "What method does AnnoIndex use to build its annotation index for structured filtering?"

    with patch("src.retrieve.query_planner.complete") as mock_complete:
        plan = plan_query(query)

    assert not mock_complete.called
    assert plan.normalized == query
    assert plan.expansions == []
    assert plan.intent == "factual"
    assert plan.degraded is False  # a deliberate skip, not a failure


def test_plan_query_always_mode_never_skips(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_MODE", "always")
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    query = "What method does AnnoIndex use to build its annotation index for structured filtering?"
    fake = CompletionResult(
        provider="groq", model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )

    with patch("src.retrieve.query_planner.complete", return_value=fake) as mock_complete:
        plan_query(query)

    assert mock_complete.called  # even a query that WOULD be skipped in conditional mode isn't here


# ---------------------------------------------------------------------
# _build_system_prompt / known_categories — constrains the hallucinated-
# category gap (Results.md §23) at extraction, not just downstream in
# hybrid.py's _sanitize_filters (unchanged, kept as a safety net).
# ---------------------------------------------------------------------


def test_build_system_prompt_unchanged_when_no_known_categories():
    from src.retrieve.query_planner import _SYSTEM_PROMPT

    assert _build_system_prompt(None) == _SYSTEM_PROMPT
    assert _build_system_prompt(frozenset()) == _SYSTEM_PROMPT


def test_build_system_prompt_lists_real_categories_sorted():
    prompt = _build_system_prompt(frozenset({"cs.CL", "cs.AI", "cs.IR"}))
    assert "cs.AI, cs.CL, cs.IR" in prompt
    assert "omit" in prompt.lower()


def test_plan_query_passes_known_categories_into_the_prompt(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_MODE", "always")
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    fake = CompletionResult(
        provider="groq", model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )

    with patch("src.retrieve.query_planner.complete", return_value=fake) as mock_complete:
        plan_query("what is RLHF", known_categories=frozenset({"cs.CL", "cs.IR"}))

    system_content = mock_complete.call_args.kwargs["messages"][0]["content"]
    assert "cs.CL, cs.IR" in system_content


def test_plan_query_omits_category_constraint_when_categories_not_supplied(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_MODE", "always")
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    fake = CompletionResult(
        provider="groq", model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )

    with patch("src.retrieve.query_planner.complete", return_value=fake) as mock_complete:
        plan_query("what is RLHF")

    system_content = mock_complete.call_args.kwargs["messages"][0]["content"]
    assert "MUST be exactly one of" not in system_content


def test_plan_query_cache_miss_then_hit(monkeypatch):
    """No real Redis, no real LLM — cache and complete() are both mocked,
    verifying plan_query's own orchestration logic (call once on miss,
    never call again on hit) rather than either dependency."""
    store: dict[str, dict] = {}
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: store.get(key))
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: store.update({key: value}))

    fake = CompletionResult(
        provider="groq",
        model_served="openai/gpt-oss-20b",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )

    with patch("src.retrieve.query_planner.complete", return_value=fake) as mock_complete:
        plan1 = plan_query("a test query")
        assert mock_complete.called

    with patch("src.retrieve.query_planner.complete", return_value=fake) as mock_complete_again:
        plan2 = plan_query("a test query")
        assert not mock_complete_again.called

    assert plan1 == plan2


def test_plan_query_use_cache_false_always_calls_llm(monkeypatch):
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: {"should": "never be read"})
    fake = CompletionResult(
        provider="groq",
        model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )
    with patch("src.retrieve.query_planner.complete", return_value=fake) as mock_complete:
        plan_query("q", use_cache=False)
        assert mock_complete.called


def test_plan_query_retries_on_malformed_response(monkeypatch):
    # Groq's json_mode guarantees valid JSON, not a matching schema —
    # seen live (2026-08-16) dropping the "intent" field in 3 of 4
    # identical temperature=0.0 calls for one real query, a genuine
    # per-query reliability floor, not a one-off fluke. Up to 3 attempts
    # give a real (if not guaranteed) chance to recover.
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    bad = CompletionResult(provider="groq", model_served="m", content='{"normalized": "n", "expansions": []}')  # missing intent
    good = CompletionResult(
        provider="groq", model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )
    with patch("src.retrieve.query_planner.complete", side_effect=[bad, bad, good]) as mock_complete:
        plan = plan_query("q")
    assert mock_complete.call_count == 3
    assert plan.intent == "factual"


def test_plan_query_degrades_after_exhausting_all_attempts(monkeypatch):
    # Approved fix (2026-08-16): a total failure no longer crashes the
    # request — it degrades to a safe, unrewritten plan, same pattern as
    # RerankResult.degraded. See plan_query's docstring for why.
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: (_ for _ in ()).throw(AssertionError("degraded plans must not be cached")))
    bad = CompletionResult(provider="groq", model_served="m", content='{"normalized": "n", "expansions": []}')
    with patch("src.retrieve.query_planner.complete", side_effect=[bad, bad, bad]) as mock_complete:
        plan = plan_query("q")
    assert mock_complete.call_count == 3
    assert plan.degraded is True
    assert plan.intent == "factual"
    assert plan.original == "q"
    assert plan.normalized == "q"  # unrewritten — the raw query, verbatim


def test_plan_query_retries_with_backoff_on_429(monkeypatch):
    # Found live (2026-08-16): an unpaced 80-query eval run started
    # hitting Groq's free-tier rate limit mid-run and every one of those
    # queries crashed outright — nothing retried a transient 429. Real
    # /ask traffic under load would hit the identical gap.
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    monkeypatch.setattr("src.retrieve.query_planner.time.sleep", lambda seconds: None)  # don't actually wait in tests
    good = CompletionResult(
        provider="groq", model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )
    with patch("src.retrieve.query_planner.complete", side_effect=[_http_error(429), good]) as mock_complete:
        plan = plan_query("q")
    assert mock_complete.call_count == 2
    assert plan.intent == "factual"


def test_plan_query_degrades_on_non_transient_http_error(monkeypatch):
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    with patch("src.retrieve.query_planner.complete", side_effect=_http_error(401)) as mock_complete:
        plan = plan_query("q", use_cache=False)
    assert mock_complete.call_count == 1  # a 401 (bad key) won't fix itself on retry
    assert plan.degraded is True
    assert plan.degrade_reason  # non-empty — the real exception's str(), not asserted verbatim here


def test_plan_query_degrades_after_exhausting_429_retries(monkeypatch):
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.time.sleep", lambda seconds: None)
    with patch(
        "src.retrieve.query_planner.complete", side_effect=[_http_error(429), _http_error(429), _http_error(429)]
    ) as mock_complete:
        plan = plan_query("q", use_cache=False)
    assert mock_complete.call_count == 3
    assert plan.degraded is True




# ---------------------------------------------------------------------
# Groq's json_validate_failed 400, and ladder-descent on retry
# ---------------------------------------------------------------------


def test_json_validate_failed_400_is_retryable():
    exc = _http_error(400, {"error": {"code": "json_validate_failed", "message": "..."}})
    assert is_retryable_http_error(exc) is True


def test_other_400_errors_are_not_retryable():
    exc = _http_error(400, {"error": {"code": "invalid_api_key", "message": "..."}})
    assert is_retryable_http_error(exc) is False


def test_400_with_unparseable_body_is_not_retryable():
    response = MagicMock(status_code=400)
    response.json.side_effect = ValueError("not json")
    exc = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
    assert is_retryable_http_error(exc) is False


def test_plan_query_retries_json_validate_failed_400(monkeypatch):
    # Reproduced live (2026-08-16): one real query (about Promptriever's
    # p-MRR in Table 3) hit Groq's own server-side json_validate_failed
    # in 4 of 5 identical calls — the model failed to generate anything
    # usable, not a malformed request on our end.
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    monkeypatch.setattr("src.retrieve.query_planner.time.sleep", lambda seconds: None)
    good = CompletionResult(
        provider="groq", model_served="m",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )
    bad = _http_error(400, {"error": {"code": "json_validate_failed"}})
    with patch("src.retrieve.query_planner.complete", side_effect=[bad, good]) as mock_complete:
        plan = plan_query("q")
    assert mock_complete.call_count == 2
    assert plan.intent == "factual"


def test_usable_ladder_skips_rungs_without_a_configured_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    ladder = usable_ladder("rewrite")
    assert all(provider_has_key(binding.provider) for binding in ladder) or len(ladder) == 1
    assert ladder[0].provider == "groq"
    assert all(binding.provider != "openrouter" for binding in ladder)


def test_plan_query_retries_use_different_model_when_fallback_key_present(monkeypatch):
    # With a fallback key actually configured, a retry should genuinely
    # switch models rather than re-asking the one that just failed —
    # same-model retries were found live to often reproduce the exact
    # same failure for a given query.
    monkeypatch.setattr("src.retrieve.query_planner.get_json", lambda key: None)
    monkeypatch.setattr("src.retrieve.query_planner.set_json", lambda key, value, ttl: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    bad = CompletionResult(provider="groq", model_served="m", content='{"normalized": "n", "expansions": []}')
    good = CompletionResult(
        provider="openrouter", model_served="meta-llama/llama-3.1-8b-instruct:free",
        content='{"normalized": "n", "expansions": [], "filters": {}, "intent": "factual"}',
    )
    with patch("src.retrieve.query_planner.complete", side_effect=[bad, good]) as mock_complete:
        plan = plan_query("q")

    first_call_binding = mock_complete.call_args_list[0].kwargs["model_override"]
    second_call_binding = mock_complete.call_args_list[1].kwargs["model_override"]
    assert first_call_binding.provider == "groq"
    assert second_call_binding.provider == "openrouter"
    assert plan.intent == "factual"
