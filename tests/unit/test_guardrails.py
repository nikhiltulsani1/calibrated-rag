from unittest.mock import patch

import pytest

from src.guardrails.base import guardrail_mode
from src.guardrails.input_guardrails import check_size_and_shape, screen_scope
from src.guardrails.output_guardrails import check_citation_integrity, check_groundedness
from src.platform.models import CompletionResult
from src.retrieve.reranker import RankedCandidate
from src.schemas.answer import Answer, Citation
from src.schemas.query_plan import QueryPlan

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------
# guardrail_mode
# ---------------------------------------------------------------------


def test_guardrail_mode_defaults_to_monitor(monkeypatch):
    monkeypatch.delenv("GUARDRAIL_SCOPE_SCREENING_MODE", raising=False)
    assert guardrail_mode("scope_screening") == "monitor"


def test_guardrail_mode_reads_per_guardrail_env_var(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_GROUNDEDNESS_MODE", "enforce")
    assert guardrail_mode("groundedness") == "enforce"


# ---------------------------------------------------------------------
# input guardrails
# ---------------------------------------------------------------------


def test_screen_scope_passes_for_in_scope_intent():
    plan = QueryPlan(original="q", normalized="q", intent="factual")
    result = screen_scope(plan)
    assert result.passed is True


def test_screen_scope_fails_for_out_of_scope():
    plan = QueryPlan(original="q", normalized="q", intent="out_of_scope")
    result = screen_scope(plan)
    assert result.passed is False
    assert "out_of_scope" in result.reason


def test_size_and_shape_rejects_empty_query():
    result = check_size_and_shape("")
    assert result.passed is False


def test_size_and_shape_rejects_oversized_query():
    result = check_size_and_shape("x" * 3000)
    assert result.passed is False


def test_size_and_shape_accepts_normal_query():
    result = check_size_and_shape("what is RLHF?")
    assert result.passed is True


# ---------------------------------------------------------------------
# output guardrails: groundedness
# ---------------------------------------------------------------------


def test_groundedness_abstained_answer_always_passes():
    answer = Answer(text="I don't have enough information", citations=[], abstained=True)
    result = check_groundedness(answer, [])
    assert result.passed is True


def test_groundedness_high_overlap_passes_without_judge_call():
    context = [RankedCandidate(id="c1", text="RLHF trains models using human feedback signals", score=0.9)]
    answer = Answer(text="RLHF trains models using human feedback signals", citations=[], abstained=False)
    with patch("src.guardrails.output_guardrails.complete") as mock_complete:
        result = check_groundedness(answer, context)
    assert result.passed is True
    assert not mock_complete.called  # deterministic check handled it, no escalation needed


def test_groundedness_low_overlap_escalates_to_judge_and_passes(monkeypatch):
    context = [RankedCandidate(id="c1", text="completely unrelated passage about databases", score=0.9)]
    answer = Answer(text="RLHF trains models using reward signals from humans", citations=[], abstained=False)
    fake_judge = CompletionResult(provider="nvidia", model_served="nemotron", content="YES")
    with patch("src.guardrails.output_guardrails.complete", return_value=fake_judge) as mock_complete:
        result = check_groundedness(answer, context)
    assert mock_complete.called
    assert mock_complete.call_args.args[0] == "judge"
    assert result.passed is True


def test_groundedness_low_overlap_judge_says_no_fails():
    context = [RankedCandidate(id="c1", text="completely unrelated passage about databases", score=0.9)]
    answer = Answer(text="RLHF trains models using reward signals from humans", citations=[], abstained=False)
    fake_judge = CompletionResult(provider="nvidia", model_served="nemotron", content="NO")
    with patch("src.guardrails.output_guardrails.complete", return_value=fake_judge):
        result = check_groundedness(answer, context)
    assert result.passed is False
    assert result.errored is False  # a genuine NO verdict, not a broken check — see the retry loop's use of this


def test_groundedness_fails_closed_on_error():
    context = [RankedCandidate(id="c1", text="unrelated", score=0.9)]
    answer = Answer(text="something with zero overlap words", citations=[], abstained=False)
    with patch("src.guardrails.output_guardrails.complete", side_effect=RuntimeError("no judge key")):
        result = check_groundedness(answer, context)
    assert result.passed is False  # fail CLOSED on the output side, unlike input guardrails
    assert "failing closed" in result.reason
    assert result.errored is True  # the check itself broke — the pipeline's retry loop must not retry on this


# ---------------------------------------------------------------------
# output guardrails: citation integrity
# ---------------------------------------------------------------------


def test_citation_integrity_all_valid_unchanged():
    context = [RankedCandidate(id="c1", text="t", score=0.9)]
    answer = Answer(text="a", citations=[Citation(chunk_id="c1", paper_id="p1", title="T")], abstained=False)
    fixed, result = check_citation_integrity(answer, context)
    assert result.passed is True
    assert fixed is answer  # unchanged object, not a needless copy


def test_citation_integrity_strips_invalid_citations():
    context = [RankedCandidate(id="c1", text="t", score=0.9)]
    answer = Answer(
        text="a",
        citations=[
            Citation(chunk_id="c1", paper_id="p1", title="Real"),
            Citation(chunk_id="does_not_exist", paper_id="p2", title="Fake"),
        ],
        abstained=False,
    )
    fixed, result = check_citation_integrity(answer, context)
    assert result.passed is False
    assert len(fixed.citations) == 1
    assert fixed.citations[0].chunk_id == "c1"
    assert "1 citation" in result.reason
