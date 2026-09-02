from unittest.mock import MagicMock, patch

import pytest

from evals.run_abstention_eval import (
    _gate_line,
    _is_infra_failure,
    _refused,
    load_gold,
    load_unanswerable,
    run_all,
)
from src.reason.generate import GENERATION_UNAVAILABLE_TEXT
from src.schemas.answer import Answer

pytestmark = pytest.mark.unit


def test_load_unanswerable_has_30_questions_across_four_kinds():
    rows = load_unanswerable()
    assert len(rows) == 30
    kinds = {r["kind"] for r in rows}
    assert kinds == {"out_of_corpus", "plausible_absent", "false_premise", "under_specified"}
    for row in rows:
        assert row["query_id"]
        assert row["query"]
        assert row["note"]


def test_load_unanswerable_query_ids_are_unique():
    rows = load_unanswerable()
    ids = [r["query_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_load_gold_reused_from_a4():
    rows = load_gold()
    assert len(rows) > 0


def test_refused_true_when_stopped_at_set():
    trace = MagicMock(stopped_at="scope_screening", answer=Answer(text="x", citations=[], abstained=False))
    assert _refused(trace) is True


def test_refused_true_when_abstained():
    trace = MagicMock(stopped_at=None, answer=Answer(text="I don't know", citations=[], abstained=True))
    assert _refused(trace) is True


def test_refused_false_for_a_confident_answer():
    trace = MagicMock(stopped_at=None, answer=Answer(text="the real answer", citations=[], abstained=False))
    assert _refused(trace) is False


def test_refused_true_when_declined_to_guess():
    # Real gap this covers: u027 (genuine ambiguity) got a correct
    # "which one do you mean?" response, verified live, that this
    # function originally scored as a non-refusal because it isn't the
    # literal ABSTAIN_TEXT.
    trace = MagicMock(
        stopped_at=None,
        answer=Answer(text="which entity do you mean?", citations=[], abstained=False, declined_to_guess=True),
    )
    assert _refused(trace) is True


# ---------------------------------------------------------------------
# _is_infra_failure — real gap found live 2026-08-28: generate.py's new
# retry-with-degrade ladder (Results.md §28) stopped a transient error
# from crashing the request, but that meant a real infra hiccup started
# silently counting as a genuine content-based abstention in this eval's
# numbers instead of being excluded like a raw exception always was.
# ---------------------------------------------------------------------


def test_is_infra_failure_true_for_generation_unavailable_text():
    trace = MagicMock(answer=Answer(text=GENERATION_UNAVAILABLE_TEXT, citations=[], abstained=True))
    assert _is_infra_failure(trace) is True


def test_is_infra_failure_false_for_a_real_content_abstention():
    # A genuine "I don't have enough information" abstention is a real
    # content decision, not an infra failure — must not be excluded.
    trace = MagicMock(answer=Answer(text="I don't have enough information in the retrieved context to answer this.", citations=[], abstained=True))
    assert _is_infra_failure(trace) is False


def test_is_infra_failure_false_for_a_real_answer():
    trace = MagicMock(answer=Answer(text="a real cited answer [1]", citations=[], abstained=False))
    assert _is_infra_failure(trace) is False


def test_run_all_excludes_infra_failures_from_scoring(monkeypatch):
    # The real regression this covers: an infra-degraded Answer must land
    # in the same "excluded" bucket as a raw exception, not get scored as
    # a refusal — otherwise a provider outage during the eval run makes
    # the abstention/over-refusal numbers look artificially better/worse
    # than the system's actual content-calibration behavior.
    monkeypatch.setattr("evals.run_abstention_eval.time.sleep", lambda *_: None)

    infra_failed = MagicMock(
        stopped_at=None, answer=Answer(text=GENERATION_UNAVAILABLE_TEXT, citations=[], abstained=True)
    )
    confident = MagicMock(stopped_at=None, answer=Answer(text="a real answer", citations=[], abstained=False))

    unanswerable_fixture = [{"query_id": "u1", "query": "q1", "kind": "out_of_corpus", "note": "n"}]
    gold_fixture = [{"query_id": "q1", "query": "q1", "reference": "r", "relevant_chunk_id": "c1", "paper_id": "p1"}]

    with patch("evals.run_abstention_eval.load_unanswerable", return_value=unanswerable_fixture), patch(
        "evals.run_abstention_eval.load_gold", return_value=gold_fixture
    ), patch("evals.run_abstention_eval.run_traced_query", side_effect=[infra_failed, confident]):
        result = run_all()

    assert result["abstention_rate"] is None  # the only unanswerable-set row was excluded, not scored
    assert result["over_refusal_rate"] == 0.0  # the gold question was scored normally
    assert result["unanswerable_failures"] == [("u1", "generate.py exhausted its retry ladder (real infra failure, not a content decision)")]


def test_gate_line_reports_pass_and_fail_correctly():
    assert "PASS" in _gate_line("abstention", 0.9, 0.80, higher_is_better=True)
    assert "FAIL" in _gate_line("abstention", 0.5, 0.80, higher_is_better=True)
    assert "PASS" in _gate_line("over-refusal", 0.02, 0.05, higher_is_better=False)
    assert "FAIL" in _gate_line("over-refusal", 0.10, 0.05, higher_is_better=False)


def test_gate_line_handles_no_data():
    assert "not run" in _gate_line("abstention", None, 0.80, higher_is_better=True)


def test_gate_line_includes_industry_context_when_provided():
    # Plan B4: a gate FAIL should read with real context, not as an
    # unqualified failure.
    line = _gate_line("abstention", 0.5, 0.80, higher_is_better=True, industry_context="real benchmark note")
    assert "real benchmark note" in line
    assert "Industry context" in line


def test_gate_line_omits_industry_context_when_not_provided():
    line = _gate_line("abstention", 0.9, 0.80, higher_is_better=True)
    assert "Industry context" not in line


def test_run_all_computes_both_gates_from_real_pipeline_calls(monkeypatch):
    monkeypatch.setattr("evals.run_abstention_eval.time.sleep", lambda *_: None)  # skip real pacing in this unit test

    refusal = MagicMock(stopped_at="scope_screening", answer=Answer(text="I don't know", citations=[], abstained=True))
    confident = MagicMock(stopped_at=None, answer=Answer(text="a real answer", citations=[], abstained=False))

    unanswerable_fixture = [{"query_id": "u1", "query": "q1", "kind": "out_of_corpus", "note": "n"}]
    gold_fixture = [{"query_id": "q1", "query": "q1", "reference": "r", "relevant_chunk_id": "c1", "paper_id": "p1"}]

    with patch("evals.run_abstention_eval.load_unanswerable", return_value=unanswerable_fixture), patch(
        "evals.run_abstention_eval.load_gold", return_value=gold_fixture
    ), patch("evals.run_abstention_eval.run_traced_query", side_effect=[refusal, confident]):
        result = run_all()

    assert result["abstention_rate"] == 1.0  # correctly refused the unanswerable question
    assert result["over_refusal_rate"] == 0.0  # correctly answered the answerable question
