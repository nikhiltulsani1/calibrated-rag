from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.platform.models import CompletionResult
from src.reason.generate import ABSTAIN_TEXT, GENERATION_UNAVAILABLE_TEXT, generate_answer
from src.retrieve.reranker import RankedCandidate

pytestmark = pytest.mark.unit


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    response = MagicMock(status_code=status_code)
    response.json.return_value = {}
    return httpx.HTTPStatusError("boom", request=MagicMock(), response=response)


def test_empty_candidates_abstains_without_calling_llm():
    with patch("src.reason.generate.complete") as mock_complete:
        answer = generate_answer("q", [], {})
    assert not mock_complete.called
    assert answer.abstained is True
    assert answer.text == ABSTAIN_TEXT
    assert answer.citations == []


def test_generates_answer_with_citations(monkeypatch):
    candidates = [
        RankedCandidate(id="c1", text="RLHF uses human feedback.", score=0.9),
        RankedCandidate(id="c2", text="Unrelated passage.", score=0.5),
    ]
    metadata = {
        "c1": {"title": "Paper One", "paper_id": "1111.1111", "section": "intro"},
        "c2": {"title": "Paper Two", "paper_id": "2222.2222", "section": "results"},
    }
    fake = CompletionResult(
        provider="groq",
        model_served="openai/gpt-oss-120b",
        content="RLHF trains models using human feedback [1].",
    )
    with patch("src.reason.generate.complete", return_value=fake) as mock_complete:
        answer = generate_answer("what is RLHF", candidates, metadata)

    call_kwargs = mock_complete.call_args
    assert call_kwargs.args[0] == "generate"
    assert answer.abstained is False
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"
    assert answer.citations[0].title == "Paper One"
    assert answer.citations[0].section == "intro"


def test_fullwidth_citation_brackets_are_normalized_and_still_parsed():
    # Real bug found live (Phase 2 verification, 2026-09-02): a model on
    # the retry ladder occasionally emits fullwidth/CJK brackets around a
    # citation marker ("【1】") instead of the ASCII "[1]" every prompt
    # here instructs. Before the fix, the citation-extraction regex only
    # matched ASCII brackets, so this silently produced zero citations —
    # no crash, no error, just a real citation quietly dropped.
    candidates = [RankedCandidate(id="c1", text="RLHF uses human feedback.", score=0.9)]
    metadata = {"c1": {"title": "Paper One", "paper_id": "1111.1111", "section": "intro"}}
    fake = CompletionResult(
        provider="groq", model_served="m", content="RLHF trains models using human feedback【1】."
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("what is RLHF", candidates, metadata)

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"
    # The visible answer text is normalized too, not just extraction —
    # the reader should never see the inconsistent fullwidth marker.
    assert "【" not in answer.text
    assert "[1]" in answer.text


def test_citation_marker_with_trailing_annotation_text_is_still_parsed():
    # Real bug found live (Phase 2 verification, 2026-09-02) — a distinct
    # shape of the same underlying fragility the fullwidth-bracket fix
    # above addressed: a model appended extra annotation text INSIDE
    # plain ASCII brackets ("[1†L1-L4]" instead of the prompted "[1]").
    # Observed directly in a real trace: real retrieval, a real answer,
    # 0 citations extracted. The old regex required the entire bracket
    # content to be digits; matching a leading digit run and tolerating
    # trailing junk is what actually fixes this class of bug.
    candidates = [RankedCandidate(id="c1", text="Octopuses have blue blood.", score=0.9)]
    metadata = {"c1": {"title": "Cephalopod Paper", "paper_id": "1111.1111", "section": "intro"}}
    fake = CompletionResult(
        provider="groq", model_served="m", content="Octopus blood is blue[1†L1-L4]."
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("what color is octopus blood", candidates, metadata)

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"


def test_a_bracket_that_does_not_start_with_a_digit_is_not_mistaken_for_a_citation():
    # The widened regex must still not treat a genuine bracketed aside
    # (no leading digit) as a citation marker.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(provider="groq", model_served="m", content="This is [unclear] from context.")
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert answer.citations == []


def test_only_actually_cited_markers_become_citations():
    # The model was given two passages but only cited one — the unused
    # passage must not appear in the citation list just because it was
    # in context.
    candidates = [
        RankedCandidate(id="c1", text="a", score=0.9),
        RankedCandidate(id="c2", text="b", score=0.5),
    ]
    metadata = {"c1": {"title": "A"}, "c2": {"title": "B"}}
    fake = CompletionResult(provider="groq", model_served="m", content="Fact from passage one [1].")
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("q", candidates, metadata)
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"


def test_out_of_range_citation_marker_is_ignored_not_crashed():
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(provider="groq", model_served="m", content="Claims something [5].")
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert answer.citations == []  # [5] doesn't exist in a 1-candidate context


def test_model_abstaining_produces_no_citations():
    candidates = [RankedCandidate(id="c1", text="unrelated", score=0.9)]
    fake = CompletionResult(provider="groq", model_served="m", content=ABSTAIN_TEXT)
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert answer.abstained is True
    assert answer.citations == []


# ---------------------------------------------------------------------
# ambiguity_note -> declined_to_guess — the real A8 follow-up: a
# structural proxy for "asked for clarification / challenged the
# premise" so the abstention eval scores it as a refusal, not a miss.
# ---------------------------------------------------------------------


def test_flagged_query_with_zero_citations_is_marked_declined_to_guess():
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content="The question is ambiguous — which entity's baseline do you mean?",
    )
    with patch("src.reason.generate.complete", return_value=fake) as mock_complete:
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}},
            ambiguity_note="the context contradicts the premise",
        )
    assert answer.declined_to_guess is True
    assert answer.abstained is False  # distinct from the ABSTAIN_TEXT path
    # the note was actually sent to the model, not just recorded
    sent_content = mock_complete.call_args.kwargs["messages"][1]["content"]
    assert "the context contradicts the premise" in sent_content


def test_flagged_query_that_still_cites_something_is_not_declined_to_guess():
    # If the model was flagged but still committed to a cited claim
    # (e.g. it resolved the ambiguity itself, or partially answered),
    # that's a real answer, not a decline — zero citations is the
    # specific signal, not the flag alone.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(provider="groq", model_served="m", content="It's X [1].")
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="some flag"
        )
    assert answer.declined_to_guess is False


def test_unflagged_query_with_zero_citations_is_not_declined_to_guess():
    # Zero citations alone (no ambiguity_note) is just a short/simple
    # answer — must not be misread as a decline.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(provider="groq", model_served="m", content="A short answer with no marker.")
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert answer.declined_to_guess is False


def test_flagged_query_with_citations_ending_in_a_question_is_declined_to_guess():
    # Real gap found live 2026-08-24: u025's genuinely correct rule-7
    # response cited 4 real passages to explain WHY the question is
    # ambiguous, then asked "Which framework and benchmark are you
    # referring to?" — a real clarifying decline, not a guess, but the
    # zero-citations-only check missed it. This is the exact real
    # response shape (trimmed), not a synthetic example.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content=(
            "The question is ambiguous because the context passages describe multiple "
            "frameworks and benchmarks: the SC2R framework [1] and the AutoDesign framework "
            "[2]. Which framework and benchmark are you referring to?"
        ),
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="referential ambiguity"
        )
    assert answer.declined_to_guess is True
    assert len(answer.citations) > 0  # real citations, still a decline


def test_flagged_query_with_citations_not_ending_in_a_question_stays_not_declined():
    # The widened check is specifically "ends with a real question," not
    # "contains a question mark anywhere" — a normal cited answer that
    # happens to quote a question mid-sentence must not be misread.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content='The paper asks "does scale help?" and concludes yes, citing a 12-point gain [1].',
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="some flag"
        )
    assert answer.declined_to_guess is False


def test_real_rule_6_premise_rejection_is_declined_to_guess():
    # Real live response text (2026-08-24, u021: "Given that AnnoIndex
    # achieved only a 0.45 F1 score, underperforming baselines, what
    # were the main failure modes identified?") — a genuinely correct
    # rule-6 premise rejection, cited, declarative (not a question), so
    # neither the zero-citation nor ends-with-"?" checks caught it.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content=(
            "The premise that AnnoIndex achieved only a 0.45 F1 score conflicts with what "
            "the passages actually say. The passages show that AnnoIndex (in both Economical "
            "and Performance modes) consistently outperformed all baselines, with F1 scores "
            "ranging from 0.74 to 0.96 across datasets [1][2][3][7]. No failure mode analysis "
            "for a 0.45 F1 score is provided in the context."
        ),
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="false premise flagged"
        )
    assert answer.declined_to_guess is True
    assert len(answer.citations) > 0


def test_mentioning_premise_without_a_real_rejection_word_stays_not_declined():
    # "premise" alone isn't enough — must be near an actual rejection
    # word, so a normal answer that legitimately discusses "the
    # experiment's premise" as a topic isn't misread as a decline.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content="The paper's premise is that retrieval quality drives downstream accuracy [1].",
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="some flag"
        )
    assert answer.declined_to_guess is False


def test_real_plausible_absent_acknowledgment_is_declined_to_guess():
    # Real live response text (2026-08-24, u013: "What latency benchmark
    # does AnnoIndex report for its Structured Query Engine under
    # concurrent multi-user load?") — a genuinely correct plausible_
    # absent recognition, cited (to show what the papers DO cover), but
    # neither zero-citation, question-ending, nor a premise rejection.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content=(
            "The passages about AnnoIndex [1] discuss its F1 scores and token consumption, "
            "but the context passages do not report latency benchmarks under concurrent "
            "multi-user load."
        ),
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="plausible absence flagged"
        )
    assert answer.declined_to_guess is True
    assert len(answer.citations) > 0


def test_absence_phrase_alone_without_context_reference_stays_not_declined():
    # Requires BOTH the negation phrase AND a reference to context/
    # passages — a normal answer that says "does not" about something
    # else entirely (not the context's own coverage) must not be misread.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content="The proposed method does not require fine-tuning, unlike prior approaches [1].",
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="some flag"
        )
    assert answer.declined_to_guess is False


# ---------------------------------------------------------------------
# generate's own retry-with-fallback ladder — real bug found live
# 2026-08-27: unlike query_planner.py/assess.py, this call had zero
# retry handling, so a transient network error threw away all of
# retrieval/rerank/assess's already-successful work for nothing.
# ---------------------------------------------------------------------


def test_retries_on_a_transient_http_error_and_succeeds():
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    good = CompletionResult(provider="groq", model_served="m", content="Fact [1].")
    with patch("src.reason.generate.time.sleep"), patch(
        "src.reason.generate.complete", side_effect=[_http_error(429), good]
    ) as mock_complete:
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert mock_complete.call_count == 2
    assert answer.text == "Fact [1]."
    assert answer.abstained is False


def test_degrades_to_a_clear_message_after_exhausting_all_retries():
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    with patch("src.reason.generate.time.sleep"), patch(
        "src.reason.generate.complete", side_effect=[_http_error(429), _http_error(429), _http_error(429)]
    ) as mock_complete:
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert mock_complete.call_count == 3
    assert answer.abstained is True
    assert answer.text == GENERATION_UNAVAILABLE_TEXT
    assert answer.text != ABSTAIN_TEXT  # distinguishable from a real context-insufficiency abstention
    assert answer.citations == []


def test_does_not_retry_a_non_transient_http_error():
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    with patch("src.reason.generate.complete", side_effect=_http_error(401)) as mock_complete:
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert mock_complete.call_count == 1  # a bad key won't fix itself on retry
    assert answer.text == GENERATION_UNAVAILABLE_TEXT


# ---------------------------------------------------------------------
# Rule 8 — structured self-report tag, the new primary declined_to_guess
# signal (2026-08-27), with the heuristics above kept as a fallback net.
# ---------------------------------------------------------------------


def test_self_reported_decline_tag_is_the_primary_signal_and_is_stripped():
    # A response that would NOT trip any of the existing heuristics (has
    # a citation, doesn't end in "?", no premise/absence phrasing) must
    # still be marked declined_to_guess when the model tags it itself —
    # and the tag itself must never reach the user-visible answer text.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content="This is a genuinely uncertain partial note about [1].\n[DECLINED_TO_GUESS]",
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="some flag"
        )
    assert answer.declined_to_guess is True
    assert "[DECLINED_TO_GUESS]" not in answer.text


def test_tag_absent_falls_back_to_existing_heuristics():
    # A provider that drops the tag (real, documented json_mode-style
    # flakiness risk) must not silently lose the signal — the old
    # zero-citation/question/premise/absence checks still apply.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m",
        content="The question is ambiguous — which entity's baseline do you mean?",
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer(
            "q", candidates, {"c1": {"title": "A"}}, ambiguity_note="some flag"
        )
    assert answer.declined_to_guess is True


def test_tag_on_a_normal_answer_without_ambiguity_note_does_not_flip_declined():
    # declined_to_guess still requires ambiguity_note to have been set —
    # the tag alone (e.g. a model false-positive on rule 8 itself) isn't
    # enough on an unflagged query, same precondition as every other
    # signal here.
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(
        provider="groq", model_served="m", content="Fact [1].\n[DECLINED_TO_GUESS]"
    )
    with patch("src.reason.generate.complete", return_value=fake):
        answer = generate_answer("q", candidates, {"c1": {"title": "A"}})
    assert answer.declined_to_guess is False
    assert "[DECLINED_TO_GUESS]" not in answer.text


def test_ambiguity_note_absent_by_default_does_not_alter_the_prompt():
    candidates = [RankedCandidate(id="c1", text="a", score=0.9)]
    fake = CompletionResult(provider="groq", model_served="m", content="Fact [1].")
    with patch("src.reason.generate.complete", return_value=fake) as mock_complete:
        generate_answer("q", candidates, {"c1": {"title": "A"}})
    sent_content = mock_complete.call_args.kwargs["messages"][1]["content"]
    assert "pre-check" not in sent_content
