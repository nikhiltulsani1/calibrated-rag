from __future__ import annotations

import re
import time

import httpx

from src.platform.models import complete, is_retryable_error, usable_ladder
from src.platform.telemetry import get_tracer
from src.retrieve.reranker import RankedCandidate
from src.schemas.answer import Answer, Citation

# Authored from scratch for this system's schema — see the IP posture
# note on prompts in the plan.
_SYSTEM_PROMPT = """You are answering questions about computer-science research \
papers using ONLY the numbered context passages provided below. Rules:

1. Answer using only information in the context passages. Do not use \
outside knowledge, even if you are confident it is correct.
2. Every factual claim must be immediately followed by a citation marker \
like [1] or [2] referencing the passage number it came from.
3. If the passages do not contain enough information to answer the \
question, respond with exactly this sentence and nothing else: \
"I don't have enough information in the retrieved context to answer this."
4. Be concise and direct.
5. Open with a short sentence that directly restates what was asked, \
using similar wording to the question, before listing details or \
supporting points.
6. If the question asserts something that contradicts the context \
passages (a false premise), do not answer around it or invent an \
explanation for it — say plainly that the premise conflicts with what \
the passages actually say, and state what they say instead, with a \
citation.
7. If the question is genuinely ambiguous — it could refer to more than \
one paper, model, or result in the context passages and nothing in the \
question disambiguates which — say what the ambiguity is and ask which \
one is meant, rather than guessing.
8. If this response invokes rule 6 or rule 7 (rejecting a false premise, \
or asking for clarification instead of guessing), end your response with \
a final line containing exactly this tag and nothing else: [DECLINED_TO_GUESS]. \
Do not include this tag for a normal answer, and do not include it for \
the rule-3 abstention sentence."""

# Tried a stricter "answer minimally, no elaboration" version of rule 4
# (2026-08-17) on the hypothesis that Ragas' answer_relevancy metric
# penalizes verbose answers. Measured on a real 25-question sample:
# answer_relevancy dropped from 0.798 to 0.694 — the terser answers
# scored *worse*, not better (Ragas' metric reverse-generates candidate
# questions from the answer text, and bare fact-lists apparently give it
# less to work with than a naturally-phrased sentence does). Reverted.
# Real, tested evidence against a plausible-sounding hypothesis — keep
# this note so the same change isn't tried again without re-deriving why
# it doesn't work.
#
# Rules 6/7 added 2026-08-20 (A8 abstention eval): the original rule 3
# only abstains on MISSING context, so a question with real (but
# contradicted, or ambiguous) context sailed straight through — real
# example caught live: asked whether S2G-RAG's judge was trained on
# "900 states from 3,009 questions" (the true numbers reversed), the
# model didn't just answer around the false premise, it FABRICATED a
# plausible-sounding but entirely invented justification (a "700/100/200
# split" and a "predeclared data-sufficiency gate" that appear nowhere in
# the real paper) rather than catching the contradiction. Narrow-tested
# before/after on the real failing questions (2 false_premise, 2
# under_specified) plus 2 real answerable questions as a regression
# check before rolling this out further — see the A8 section of the
# README for the real before/after numbers.

ABSTAIN_TEXT = "I don't have enough information in the retrieved context to answer this."

# Real bug found live 2026-08-27 while verifying items #3/#4 of the "near
# 100%" punch list: unlike query_planner.py's rewrite call and assess.py's
# grade call (both given a retry-with-fallback ladder in the original
# codebase review, item 5), this call had NO retry handling at all — a
# transient httpx.ReadTimeout here (verified live, see Results.md §27
# follow-up) propagated straight up through run_graph uncaught. ask.py/
# pipeline.py's outer try/except (§26 item 4) stops that from becoming a
# raw 500, but the user still gets nothing — the single most expensive,
# most valuable call in the whole pipeline (retrieval, rerank, and assess
# already succeeded by this point) throwing away all of that work on one
# transient blip that a retry would very plausibly have survived, exactly
# like every other role in this codebase.
GENERATION_UNAVAILABLE_TEXT = (
    "The system is temporarily unable to generate an answer due to a service issue. "
    "Please try again in a moment."
)

# Structured self-report (rule 8) — the primary signal for declined_to_guess,
# added 2026-08-27 to replace guessing at declined_to_guess purely from
# surface phrasing. The regex heuristics below (_PREMISE_REJECTION_RE etc.)
# had already been widened three times in one day (2026-08-24) as each new
# real phrasing slipped through; asking the model to report its own rule
# 6/7 usage directly closes that whole class of gap in one step, rather
# than adding a fourth heuristic for whatever phrasing appears next.
#
# Kept the regex heuristics as a fallback, not removed outright: this
# model family's json_mode has documented, measured unreliability on a
# *stricter* ask (a full schema — see query_planner.py's own real
# evidence, dropped fields in 3/4 calls on one query). A single literal
# trailing tag is a much smaller ask, but hasn't been live-verified
# across every provider on the retry ladder yet, so a provider that
# drops the tag under the same kind of real-world flakiness still falls
# back to the pre-existing heuristics rather than silently losing the
# signal.
_DECLINE_TAG = "[DECLINED_TO_GUESS]"

# Real bug found live (Phase 2 verification, 2026-09-02): a model on the
# retry ladder occasionally emits fullwidth/CJK bracket variants around a
# citation marker (e.g. "【1】" instead of "[1]") instead of the plain
# ASCII brackets every prompt in this file instructs — observed directly,
# not assumed. The citation-extraction regex below only ever matched
# ASCII `[N]`, so a marker in this shape silently produced ZERO parsed
# citations: no "Sources" section, no Citation objects reaching
# citation_integrity/groundedness, even though the model plainly cited a
# real passage. Normalizing to ASCII brackets before any of that runs
# fixes both the invisible extraction miss and the visible answer text
# (which would otherwise show the reader an inconsistent "【1】").
# str.translate over a full dict, not a couple of .replace() calls, so
# adding another observed bracket variant later is a one-line addition
# here rather than a new call site.
_BRACKET_NORMALIZE_TABLE = str.maketrans(
    {
        "【": "[",
        "】": "]",
        "［": "[",  # U+FF3B fullwidth left square bracket
        "］": "]",  # U+FF3D fullwidth right square bracket
    }
)

# Real structural signal for a rule-6 premise rejection — see
# declined_to_guess's own comment for the live evidence (u021) and why
# this is scoped narrowly to rule 6's own instructed phrasing rather
# than a broad guess at every way a rejection might be worded.
_PREMISE_REJECTION_RE = re.compile(
    r"premise.{0,60}(conflict|contradict|incorrect|is wrong|reversed|opposite|false)", re.IGNORECASE
)

# Real structural signal for a "plausible-sounding but never claimed"
# recognition (plausible_absent) — see declined_to_guess's own comment
# for the live evidence (u013: a real, correct response explicitly
# stating the context "do[es] not report" the asked-about detail).
# Requires BOTH a real negation-of-reporting phrase AND a reference to
# the context/passages, rather than either alone — a normal answer that
# legitimately says "the context does not specify an exact date, but
# reports the year as 2026 [1]" while still answering the real question
# must not be misread just because it names one gap in passing; scoping
# to "ambiguity_note already set" (the same precondition every check
# here shares) is what keeps that residual risk bounded.
_ABSENCE_PHRASE_RE = re.compile(
    r"\b(do not|does not|doesn't|no mention|not addressed|not discussed|not report|not claim|not describe)\b",
    re.IGNORECASE,
)
_CONTEXT_REFERENCE_RE = re.compile(r"\b(context|passages?)\b", re.IGNORECASE)


def _format_context(candidates: list[RankedCandidate], metadata: dict[str, dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        title = metadata.get(c.id, {}).get("title", "unknown paper")
        lines.append(f'[{i}] (from "{title}"): {c.text}')
    return "\n\n".join(lines)


def generate_answer(
    query: str,
    candidates: list[RankedCandidate],
    metadata: dict[str, dict],
    *,
    ambiguity_note: str | None = None,
) -> Answer:
    """Reranked context in, grounded answer with citations out.

    `metadata` maps candidate id -> {"title", "paper_id", "section"} —
    supplied by the caller (see reason/pipeline.py) rather than fetched
    here, so this function stays a pure, easily-mocked LLM call rather
    than also owning an OpenSearch lookup.

    `ambiguity_note` is an optional, real signal from the pre-generation
    assess step (src/reason/nodes/assess.py::assess_ambiguity) — a
    specific contradiction or ambiguity it found in THIS SAME reranked
    context, handed to generate explicitly rather than leaving rules 6/7
    to self-diagnose from the passages alone (the real gap A8 found:
    self-diagnosis catches a simple false premise but not a denser one,
    and never catches ambiguity generate itself can't see across
    candidates it wasn't shown). When present, it's injected as a
    directive, not a suggestion — the model is told to follow rules 6/7
    for this specific issue, not just given it as background color.

    This is the direct-call generation step, not the full agentic loop —
    the agent graph's `answer` node (screen -> retrieve -> assess ->
    refine -> answer) wraps this rather than replacing it.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("generate.answer") as span:
        span.set_attribute("generate.num_candidates", len(candidates))
        span.set_attribute("generate.ambiguity_note_present", bool(ambiguity_note))

        if not candidates:
            span.set_attribute("generate.abstained", True)
            span.set_attribute("generate.num_citations", 0)
            return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)

        context = _format_context(candidates, metadata)
        user_content = f"Context passages:\n\n{context}\n\nQuestion: {query}"
        if ambiguity_note:
            user_content += (
                f"\n\nNote from a pre-check of this same context: {ambiguity_note} "
                "Apply rule 6 or 7 above for this specific issue."
            )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # Same retry-with-fallback shape as query_planner.py's rewrite call
        # and assess.py's grade call — descend to a genuinely different
        # model on a retryable failure rather than re-asking the one that
        # just failed the same way (real evidence for that choice is in
        # query_planner.py's own comment). No JSON to re-parse here (this
        # is free text, not a schema), so this loop is simpler than
        # query_planner's: only the HTTP-retry case applies.
        ladder = usable_ladder("generate")
        last_error: Exception | None = None
        result = None
        for attempt in range(3):
            rung = ladder[min(attempt, len(ladder) - 1)]
            try:
                result = complete("generate", messages=messages, temperature=0.0, model_override=rung)
                break
            except httpx.HTTPError as exc:
                last_error = exc
                if not is_retryable_error(exc) or attempt == 2:
                    break
                time.sleep(2**attempt)

        if result is None:
            # Every attempt failed — degrade to a clearly-labeled, honest
            # "try again" message rather than crash the request (ask.py/
            # pipeline.py's outer try/except would otherwise turn this
            # into a generic error page, discarding the fact that
            # retrieval/rerank/assess all already succeeded). Reuses
            # `abstained` rather than adding a new Answer field — same
            # "did not confidently answer" bucket the abstention eval
            # already scores, just with distinguishable text so this is
            # never confused with a genuine context-insufficiency
            # abstention when reading a transcript.
            span.set_attribute("generate.abstained", True)
            span.set_attribute("generate.degraded", True)
            span.set_attribute("generate.degrade_reason", str(last_error))
            span.set_attribute("generate.num_citations", 0)
            return Answer(text=GENERATION_UNAVAILABLE_TEXT, citations=[], abstained=True)

        # Strip rule 8's self-report tag before anything else touches the
        # content — it's an internal signal, never meant to reach the user,
        # and must not be treated as citation text or count toward any of
        # the heuristics below.
        content = result.content.strip()
        self_reported_decline = content.endswith(_DECLINE_TAG)
        if self_reported_decline:
            content = content[: -len(_DECLINE_TAG)].rstrip()
        content = content.translate(_BRACKET_NORMALIZE_TABLE)

        abstained = content.startswith(ABSTAIN_TEXT)
        span.set_attribute("generate.abstained", abstained)
        span.set_attribute("generate.model_served", result.model_served)

        citations: list[Citation] = []
        if not abstained:
            # Only markers the model actually used, not every candidate it
            # was given — an unused passage should not be listed as a
            # citation just because it was in context.
            cited_indices = sorted({int(n) for n in re.findall(r"\[(\d+)\]", content)})
            for idx in cited_indices:
                if 1 <= idx <= len(candidates):
                    candidate = candidates[idx - 1]
                    meta = metadata.get(candidate.id, {})
                    citations.append(
                        Citation(
                            chunk_id=candidate.id,
                            paper_id=meta.get("paper_id", ""),
                            title=meta.get("title", ""),
                            section=meta.get("section"),
                            text=candidate.text,
                        )
                    )
        span.set_attribute("generate.num_citations", len(citations))

        # Real, structural proxy for "asked for clarification / challenged
        # the premise" — see Answer.declined_to_guess's docstring. Only
        # meaningful when assess flagged this query in the first place;
        # zero citations on an unflagged query is just a short/simple
        # answer, not a decline.
        #
        # Widened 2026-08-24 after a real gap found live: u025 ("What
        # score did the framework achieve on its benchmark?"), correctly
        # flagged ambiguous by assess's new _unresolved_reference_note,
        # got a genuinely correct rule-7 response — explained which
        # papers/benchmarks it could mean, ended "Which framework and
        # benchmark are you referring to?" — but it cited 4 real passages
        # to explain the ambiguity, so the zero-citations check alone
        # missed it. Rule 7 explicitly instructs "ask which one is
        # meant," and every real clarifying response observed live ends
        # with a literal question mark — a citation-bearing clarification
        # is not a guess just because it cites evidence for why it's
        # asking.
        #
        # Widened again 2026-08-24, same day, after a related real gap:
        # u021 ("Given that AnnoIndex achieved only a 0.45 F1 score...")
        # got a genuinely correct rule-6 response — "The premise that
        # AnnoIndex achieved only a 0.45 F1 score conflicts with what the
        # passages actually say" — a real, correct premise rejection,
        # cited, but neither zero-citation nor ending in "?" (rule 6's
        # own shape is a declarative correction, not a question, unlike
        # rule 7's). Rule 6's own instructed wording is "say plainly that
        # the premise conflicts with what the passages actually say" —
        # the model echoes that phrasing closely and reliably enough live
        # to check for it directly, rather than guessing a broader
        # pattern. Scoped narrowly (only fires when ambiguity_note was
        # already set, same as every other case here) to avoid
        # misreading an ordinary answered question that happens to use
        # the word "premise" for an unrelated reason.
        #
        # Widened a third time 2026-08-24, same day, after u013 ("What
        # latency benchmark does AnnoIndex report for its Structured
        # Query Engine under concurrent multi-user load?"): a genuinely
        # correct plausible_absent response — "the context passages...
        # do not report latency benchmarks under concurrent load" — cited
        # 2 real passages to establish what the papers DO discuss (to
        # show the asked-about detail isn't among them), so neither
        # zero-citation, question-ending, nor premise-rejection caught
        # it. See _ABSENCE_PHRASE_RE's own comment for the scoping.
        #
        # 2026-08-27: these heuristics are now a fallback, not the primary
        # signal — see _DECLINE_TAG's own comment. Only consulted when the
        # model didn't (or couldn't) produce the rule-8 tag.
        content_ends_with_question = content.endswith("?")
        content_rejects_premise = bool(_PREMISE_REJECTION_RE.search(content))
        content_acknowledges_absence = bool(_ABSENCE_PHRASE_RE.search(content)) and bool(
            _CONTEXT_REFERENCE_RE.search(content)
        )
        declined_to_guess = bool(ambiguity_note) and not abstained and (
            self_reported_decline
            or len(citations) == 0
            or content_ends_with_question
            or content_rejects_premise
            or content_acknowledges_absence
        )
        span.set_attribute("generate.declined_to_guess", declined_to_guess)
        span.set_attribute("generate.self_reported_decline", self_reported_decline)

        return Answer(text=content, citations=citations, abstained=abstained, declined_to_guess=declined_to_guess)
