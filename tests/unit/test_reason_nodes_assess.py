from unittest.mock import patch

import pytest

from src.platform.models import CompletionResult
from src.reason.nodes.assess import (
    _extract_numbers,
    _numeric_transposition_note,
    _unresolved_reference_note,
    assess_ambiguity,
    assess_context,
)
from src.retrieve.reranker import RankedCandidate, RerankResult

pytestmark = pytest.mark.unit

_EMPTY_RERANKED = RerankResult(items=[], degraded=False, reason=None, model_served=None)


def _reranked(text: str) -> RerankResult:
    return RerankResult(
        items=[RankedCandidate(id="c1", text=text, score=0.9)],
        degraded=False,
        reason=None,
        model_served="m",
    )


def test_empty_context_is_insufficient_with_zero_llm_calls():
    with patch("src.reason.nodes.assess.complete") as mock_complete:
        result = assess_context("what is RLHF", _EMPTY_RERANKED)
    assert result.passed is False
    assert not mock_complete.called


def test_high_overlap_is_sufficient_without_escalating_to_grade():
    reranked = _reranked("RLHF stands for reinforcement learning from human feedback")
    with patch("src.reason.nodes.assess.complete") as mock_complete:
        result = assess_context("what is RLHF", reranked)
    assert result.passed is True
    assert not mock_complete.called  # deterministic overlap check alone decided it


def test_low_overlap_escalates_and_trusts_a_yes_verdict():
    reranked = _reranked("completely unrelated passage about volcanoes")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="YES")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict) as mock_complete:
        result = assess_context("what is RLHF", reranked)
    assert mock_complete.called
    assert mock_complete.call_args.args[0] == "grade"
    assert result.passed is True


def test_low_overlap_escalates_and_trusts_a_no_verdict():
    reranked = _reranked("completely unrelated passage about volcanoes")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_context("what is RLHF", reranked)
    assert result.passed is False


def test_errors_fail_open_exactly_like_todays_default_behavior():
    reranked = _reranked("completely unrelated passage about volcanoes")
    with patch("src.reason.nodes.assess.complete", side_effect=RuntimeError("no key")):
        result = assess_context("what is RLHF", reranked)
    assert result.passed is True
    assert "failing open" in result.reason


# ---------------------------------------------------------------------
# assess_ambiguity — the real A8 fix: a structured signal handed to
# generate, not a block/allow guardrail. No cheap deterministic
# shortcut exists here (unlike assess_context) — always escalates.
# ---------------------------------------------------------------------


def test_empty_context_is_not_flagged_with_zero_llm_calls():
    with patch("src.reason.nodes.assess.complete") as mock_complete:
        result = assess_ambiguity("what is RLHF", _EMPTY_RERANKED, {})
    assert result.flagged is False
    assert result.note is None
    assert not mock_complete.called


def test_always_escalates_even_on_high_overlap_context():
    # The real reason this node exists: a false premise can have HIGH
    # lexical overlap with the correct facts while stating them
    # backwards — overlap_ratio genuinely cannot catch this, so unlike
    # assess_context there is no deterministic shortcut to skip here.
    reranked = _reranked("RLHF stands for reinforcement learning from human feedback")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict) as mock_complete:
        assess_ambiguity("what is RLHF", reranked, {})
    assert mock_complete.called
    assert mock_complete.call_args.args[0] == "grade"


def test_yes_verdict_with_detail_line_is_flagged_with_that_note():
    reranked = _reranked("some real context")
    fake_verdict = CompletionResult(
        provider="groq", model_served="m",
        content="YES\nThe context says X improved, but the question claims X declined.",
    )
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("did X decline?", reranked, {})
    assert result.flagged is True
    assert result.note == "The context says X improved, but the question claims X declined."


def test_no_verdict_is_not_flagged():
    reranked = _reranked("some real context")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a clear question", reranked, {})
    assert result.flagged is False
    assert result.note is None


def test_yes_verdict_with_no_detail_line_still_flags_with_a_fallback_note():
    reranked = _reranked("some real context")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="YES")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a question", reranked, {})
    assert result.flagged is True
    assert result.note == "flagged, no detail given"


def test_errors_fail_open_to_not_flagged():
    reranked = _reranked("some real context")
    with patch("src.reason.nodes.assess.complete", side_effect=RuntimeError("no key")):
        result = assess_ambiguity("a question", reranked, {})
    assert result.flagged is False
    assert result.note is None


# ---------------------------------------------------------------------
# _paper_diversity_note / its use inside assess_ambiguity — the real
# fix added 2026-08-22: the LLM check above caught 0 of 6 real
# under_specified questions live (u025-u030), because it only ever
# sees the already-narrowed top-8 context for one query and has no way
# to notice several equally-plausible papers almost-but-didn't make
# the cut. This deterministic, zero-LLM-cost signal reads that
# diversity straight off the reranked set's own paper_ids instead.
# ---------------------------------------------------------------------


def _reranked_multi(ids: list[str]) -> RerankResult:
    return RerankResult(
        items=[RankedCandidate(id=i, text="passage", score=0.9) for i in ids],
        degraded=False,
        reason=None,
        model_served="m",
    )


def test_diversity_signal_does_not_flag_when_one_paper_dominates():
    reranked = _reranked_multi(["c1", "c2", "c3"])
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    metadata = {cid: {"paper_id": "P1"} for cid in ["c1", "c2", "c3"]}
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a real, specific question", reranked, metadata)
    assert result.flagged is False
    assert result.note is None


def test_diversity_signal_flags_even_when_llm_says_no():
    # The exact real gap this fixes: the LLM alone said NO on every one
    # of 6 live under_specified tests. Reproduced here with a mocked NO
    # verdict — the diversity check must still catch it independently.
    # Query deliberately avoids _unresolved_reference_note's own trigger
    # words ("it"/"the model"/etc) so this test isolates the diversity
    # signal specifically, rather than the (correct, separately tested)
    # note-priority interaction between the two.
    reranked = _reranked_multi(["c1", "c2", "c3"])
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    metadata = {"c1": {"paper_id": "P1"}, "c2": {"paper_id": "P2"}, "c3": {"paper_id": "P3"}}
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("how do these results compare across the corpus", reranked, metadata)
    assert result.flagged is True
    assert "3 different papers" in result.note


def test_llm_note_preferred_over_diversity_note_when_both_fire():
    reranked = _reranked_multi(["c1", "c2"])
    fake_verdict = CompletionResult(
        provider="groq", model_served="m", content="YES\nSpecific contradiction found."
    )
    metadata = {"c1": {"paper_id": "P1"}, "c2": {"paper_id": "P2"}}
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a question", reranked, metadata)
    assert result.flagged is True
    assert result.note == "Specific contradiction found."


def test_two_distinct_papers_no_longer_flags_at_the_raised_default_threshold():
    # Real regression this locks in (2026-08-22): threshold 2 caught 4/6
    # real under_specified cases but also produced 2 new over-refusal
    # false positives in a full production run (q022, q030) — both were
    # exactly-2-paper cases where a second paper just cited the same
    # baseline in passing, not genuine ambiguity. Raised the default to
    # 3 on the evidence that every CONFIRMED true positive spanned 3-5
    # papers, never exactly 2.
    reranked = _reranked_multi(["c1", "c2"])
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    metadata = {"c1": {"paper_id": "P1"}, "c2": {"paper_id": "P2"}}
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a well-defined question", reranked, metadata)
    assert result.flagged is False
    assert result.note is None


def test_diversity_check_failure_fails_open_and_llm_verdict_still_applies():
    # Real bug found live 2026-08-24: _paper_diversity_note used to call
    # its own unindexed fetch_metadata internally — now the caller
    # (graph.py) supplies the already-correctly-indexed metadata dict
    # directly, so the failure mode this test guards is the signal
    # function itself raising, not a metadata-fetch failure.
    reranked = _reranked_multi(["c1", "c2"])
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict), patch(
        "src.reason.nodes.assess._paper_diversity_note", side_effect=RuntimeError("opensearch down")
    ):
        result = assess_ambiguity("a question", reranked, {})
    assert result.flagged is False
    assert result.note is None


# ---------------------------------------------------------------------
# _numeric_transposition_note / its use inside assess_ambiguity — plan
# B1 (2026-08-22): a deterministic, zero-LLM-cost signal for the one
# real failure pattern that survived both the prompt fix and two
# different LLM judges — u018's "900 states from 3,009 questions"
# vs the real "3,009 states from 900 questions".
# ---------------------------------------------------------------------


def test_extract_numbers_handles_commas_and_plain_digits():
    assert _extract_numbers("900 states from 3,009 questions") == [900, 3009]


def test_extract_numbers_ignores_non_numeric_text():
    assert _extract_numbers("no numbers here at all") == []


def test_flags_the_real_u018_transposition_case():
    # The real question and the real paper wording (per the note in
    # evals/datasets/unanswerable.jsonl: "the real numbers are reversed
    # — 3,009 states from 900 disjoint HotpotQA questions").
    query = (
        "Since the Search-R1 stopping-judgment paper's Qwen3.5-2B judge was trained on "
        "900 states from 3,009 disjoint HotpotQA questions, how was such a large question set collected?"
    )
    context_text = (
        "The stopping-judgment classifier was trained on 3,009 states from 900 disjoint "
        "HotpotQA questions, sampled from the training split."
    )
    reranked = _reranked(context_text)
    note = _numeric_transposition_note(query, reranked)
    assert note is not None
    assert "900" in note and "3009" in note


def test_does_not_flag_when_question_order_matches_context():
    query = "The model was trained on 900 states from 3,009 questions, what architecture was used?"
    context_text = "The model was trained on 900 states from 3,009 questions using a transformer."
    reranked = _reranked(context_text)
    assert _numeric_transposition_note(query, reranked) is None


def test_does_not_flag_when_only_one_number_is_real():
    # 3009 never appears in context at all — this is an out-of-corpus /
    # fabricated-number case, which is context_sufficiency's job, not
    # this signal's (it only fires when BOTH numbers are real).
    query = "The model was trained on 900 states from 3,009 questions, what architecture was used?"
    context_text = "The model was trained on 900 states using a transformer."
    reranked = _reranked(context_text)
    assert _numeric_transposition_note(query, reranked) is None


def test_does_not_flag_a_single_number_question():
    query = "How many states were used, 900?"
    reranked = _reranked("The model used 900 states from 3,009 questions.")
    assert _numeric_transposition_note(query, reranked) is None


def test_numeric_transposition_flags_even_when_llm_says_no():
    query = (
        "Since the Search-R1 stopping-judgment paper's Qwen3.5-2B judge was trained on "
        "900 states from 3,009 disjoint HotpotQA questions, how was such a large question set collected?"
    )
    context_text = (
        "The stopping-judgment classifier was trained on 3,009 states from 900 disjoint "
        "HotpotQA questions, sampled from the training split."
    )
    reranked = _reranked(context_text)
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity(query, reranked, {})
    assert result.flagged is True
    assert "900" in result.note and "3009" in result.note


# ---------------------------------------------------------------------
# _has_clear_dominant_paper / B2's score-gap refinement — suppresses
# the diversity flag when the top 3 ranked candidates clearly agree on
# one paper, even though the raw distinct-paper count is still >= 3.
# ---------------------------------------------------------------------


def test_suppresses_diversity_flag_when_top_3_agree_on_one_paper():
    # 3 distinct papers overall (meets the threshold), but the top 3
    # ranked candidates are all the same paper — real evidence that
    # paper dominates despite the raw count.
    reranked = _reranked_multi(["c1", "c2", "c3", "c4", "c5"])
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    metadata = {
        "c1": {"paper_id": "P1"}, "c2": {"paper_id": "P1"}, "c3": {"paper_id": "P1"},
        "c4": {"paper_id": "P2"}, "c5": {"paper_id": "P3"},
    }
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a well-defined question", reranked, metadata)
    assert result.flagged is False


def test_does_not_suppress_when_only_top_2_agree():
    # Real regression this locks in: u028's actual live paper ordering
    # was [P, P, P2, ...] — a genuinely ambiguous case where the top 2
    # (not 3) candidates happened to share a paper. Requiring top-3
    # agreement (not top-2) is what keeps this case correctly flagged.
    reranked = _reranked_multi(["c1", "c2", "c3", "c4"])
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    metadata = {
        "c1": {"paper_id": "P1"}, "c2": {"paper_id": "P1"}, "c3": {"paper_id": "P2"}, "c4": {"paper_id": "P3"},
    }
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a question", reranked, metadata)
    assert result.flagged is True


def test_top_3_agreement_ignored_when_reranker_degraded():
    # Degraded reranking means item ORDER is just RRF-fusion order, not
    # a true relevance ranking — "top 3 agree" wouldn't mean what it
    # claims, so the refinement must not apply, falling back to the
    # pure count-based rule (still flags here).
    reranked = RerankResult(
        items=[RankedCandidate(id=i, text="passage", score=None) for i in ["c1", "c2", "c3", "c4", "c5"]],
        degraded=True,
        reason="hosted reranker unavailable",
        model_served=None,
    )
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    metadata = {
        "c1": {"paper_id": "P1"}, "c2": {"paper_id": "P1"}, "c3": {"paper_id": "P1"},
        "c4": {"paper_id": "P2"}, "c5": {"paper_id": "P3"},
    }
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict):
        result = assess_ambiguity("a question", reranked, metadata)
    assert result.flagged is True


def test_numeric_transposition_check_failure_fails_open():
    query = "900 states from 3,009 questions, how was this collected?"
    reranked = _reranked("3,009 states from 900 questions, sampled randomly.")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict), patch(
        "src.reason.nodes.assess._numeric_transposition_note", side_effect=RuntimeError("boom")
    ):
        result = assess_ambiguity(query, reranked, {})
    assert result.flagged is False
    assert result.note is None


# ---------------------------------------------------------------------
# _unresolved_reference_note — added 2026-08-24 after a real gap found
# in A8's first genuinely complete run: u025 retrieves 7 of 8 reranked
# candidates from ONE paper (AutoDesign) purely on surface relevance,
# so _paper_diversity_note correctly does not fire, even though the raw
# question never names which paper it means. This signal checks the
# query text alone, independent of what retrieval surfaced. Every case
# below is a real question from this project's own datasets, not a
# synthetic example.
# ---------------------------------------------------------------------

# All 6 real under_specified questions (evals/datasets/unanswerable.jsonl)
_REAL_UNDER_SPECIFIED_QUESTIONS = [
    "What score did the framework achieve on its benchmark?",  # u025
    "How many training examples were used to train the model?",  # u026
    "What was the main baseline it was compared against?",  # u027
    "How does the system handle the cold-start problem?",  # u028
    "What embedding dimension does the model use?",  # u029
    "How does the approach compare to the baseline shown in Table 1?",  # u030
]

# Real gold-set questions (evals/datasets/rag_gold.jsonl) that use
# similar generic wording ("it") but DO name a real corpus entity —
# must never be flagged by this check.
_REAL_GOLD_QUESTIONS_NOT_AMBIGUOUS = [
    "What is SchemaLoop's iterative approach called, and across how many granularity levels does it operate?",  # q006
    "What is the official EM difference between Expanded S2G and Native Search-R1 on the confirmatory test set, and does it pass the frozen non-inferiority criterion?",  # q027
    "What is GEM, and how does it unify generation and embedding within a single model according to the abstract?",  # q035
    "What is Promptriever, and how does it allow users to specify relevance criteria?",  # q037
    "On PosterBench's Main Track, what score does AutoDesign achieve, and by how many points does it surpass Claude Design?",  # q070
]


@pytest.mark.parametrize("query", _REAL_UNDER_SPECIFIED_QUESTIONS)
def test_flags_every_real_under_specified_question(query):
    assert _unresolved_reference_note(query) is not None


@pytest.mark.parametrize("query", _REAL_GOLD_QUESTIONS_NOT_AMBIGUOUS)
def test_does_not_flag_real_gold_questions_naming_a_real_entity(query):
    assert _unresolved_reference_note(query) is None


def test_does_not_flag_a_question_with_no_generic_referent_at_all():
    # No "it"/"the model"/etc at all — must never even reach the entity check.
    query = "Why can't vector-based retrieval directly apply algebraic operators like aggregation or multi-document joins, according to AnnoIndex's introduction?"
    assert _unresolved_reference_note(query) is None


def test_arxiv_id_alone_counts_as_a_named_entity():
    query = "What cold-start strategy does the model in 2608.17613 use?"
    assert _unresolved_reference_note(query) is None


def test_entity_name_check_is_word_boundary_safe_not_a_naive_substring_match():
    # Real bug caught before shipping: a naive `"gem" in query.lower()`
    # check would false-positive on ordinary English words that merely
    # contain "gem" as a substring (e.g. "management"), wrongly
    # suppressing a real flag. Word-boundary regex must not do that.
    query = "What risk management framework does the model use?"
    assert _unresolved_reference_note(query) is not None


def test_reference_signal_flags_alongside_the_other_three():
    reranked = _reranked("AutoDesign scored 78.32 on PosterBench Main Track.")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict), patch(
        "src.reason.nodes.assess._paper_diversity_note", return_value=None
    ):
        result = assess_ambiguity("What score did the framework achieve on its benchmark?", reranked, {})
    assert result.flagged is True
    assert "generically" in result.note


def test_reference_check_failure_fails_open():
    reranked = _reranked("some real context")
    fake_verdict = CompletionResult(provider="groq", model_served="m", content="NO")
    with patch("src.reason.nodes.assess.complete", return_value=fake_verdict), patch(
        "src.reason.nodes.assess._unresolved_reference_note", side_effect=RuntimeError("boom")
    ), patch("src.reason.nodes.assess._paper_diversity_note", return_value=None), patch(
        "src.reason.nodes.assess._numeric_transposition_note", return_value=None
    ):
        result = assess_ambiguity("What does the model do?", reranked, {})
    assert result.flagged is False
    assert result.note is None
