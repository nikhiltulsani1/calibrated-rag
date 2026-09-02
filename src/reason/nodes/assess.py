from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import httpx

from src.guardrails.base import GuardrailResult, overlap_ratio, record_guardrail_event
from src.platform.models import complete, is_retryable_error, usable_ladder
from src.retrieve.reranker import RerankResult

# Real bug this fixes (found 2026-08-21, diagnosed live): both grade
# escalations below used to call complete("grade", ...) directly, which
# only ever tries the role's top rung (see models.py::complete's own
# docstring — it does no ladder descent itself). A single provider's
# quota exhaustion on that top rung (observed: Groq's grade-role rung
# hit 429 mid A8-eval-run) meant every call here raised, and the
# fail-open except below silently turned that into "not ambiguous" /
# "sufficient" — indistinguishable, from the eval's numbers alone, from
# the guardrail genuinely finding nothing wrong. A fallback rung
# (openrouter) was configured in models.yaml the whole time and simply
# never tried. Mirrors query_planner.py's plan_query retry loop exactly
# (same 3-attempt/clamped-rung shape) so there's one retry idiom in the
# codebase, not two slightly-different ones.


def _complete_with_fallback(role: str, *, messages: list[dict], temperature: float):
    ladder = usable_ladder(role)
    last_error: Exception | None = None
    for attempt in range(3):
        rung = ladder[min(attempt, len(ladder) - 1)]
        try:
            return complete(role, messages=messages, temperature=temperature, model_override=rung)
        except httpx.HTTPError as exc:
            # httpx.HTTPError (not just HTTPStatusError) — real bug found
            # live 2026-08-24: a bare connection reset/timeout
            # (httpx.TransportError, a sibling of HTTPStatusError, not a
            # subclass) used to propagate straight out of this exact
            # ladder-descent loop uncaught — the class of failure this
            # function was built to retry through in the first place (see
            # its own header comment above). See is_retryable_error's
            # docstring in platform/models.py.
            last_error = exc
            if not is_retryable_error(exc) or attempt == 2:
                raise
            time.sleep(2**attempt)
    raise last_error  # pragma: no cover - loop always returns or raises above

# Plan B3 (2026-08-23): tuned against real data, not left as the
# original untuned placeholder (0.15). Computed real overlap_ratio
# scores for 31 real answerable gold questions (ground truth:
# genuinely sufficient context) and 16 real out_of_corpus/
# plausible_absent unanswerable questions (ground truth: genuinely
# insufficient context) against the real retrieved+reranked context
# for each. The two real distributions barely overlap: sufficient
# scored 0.4545-1.0 (median 0.75), insufficient scored 0.0-0.7333
# (median 0.40). 0.4545 is the real threshold that maximizes
# classification accuracy on this data (85.1% on n=47) — the old 0.15
# default was so low it barely ever escalated to the grade LLM at all,
# since nearly every real query's overlap score cleared it regardless
# of true sufficiency. A real, modest sample (n=47, one embed
# provider) — a good, evidenced starting point, not claimed as
# definitively optimal; still overridable via the env var below.
_SUFFICIENCY_OVERLAP_THRESHOLD = float(
    os.environ.get("CONTEXT_SUFFICIENCY_OVERLAP_THRESHOLD", "0.4545")
)

_GRADE_SYSTEM_PROMPT = (
    "You are checking whether the given context passages contain enough "
    "information to at least partially answer the question. Respond with "
    "exactly one word: YES or NO."
)


def assess_context(query: str, reranked: RerankResult) -> GuardrailResult:
    """Pre-generation sufficiency check — the gap the plan's original
    screen->retrieve->assess->refine->answer shape had and the old
    retry-only loop didn't: today's pipeline always calls `generate`
    even when the reranked context is obviously thin, discovering that
    only after the fact. This node lets the orchestrator (graph.py)
    skip straight to widening context instead, when in `enforce` mode.

    Deterministic overlap check first (zero LLM cost); escalates to a
    single `grade`-role call only when that's inconclusive — same shape
    as check_groundedness. Fails OPEN: an error here means "proceed to
    generate exactly like the pipeline does today," not a new risk,
    matching the input-guardrail fail-open convention (this sits on the
    same "should we even attempt this" side as scope_screening, not the
    fail-closed output side).
    """
    if not reranked.items:
        result = GuardrailResult("context_sufficiency", False, "no context retrieved")
        record_guardrail_event(result)
        return result

    try:
        context_text = " ".join(c.text for c in reranked.items)
        score = overlap_ratio(query, context_text)

        if score >= _SUFFICIENCY_OVERLAP_THRESHOLD:
            result = GuardrailResult("context_sufficiency", True, reason=f"overlap={score:.2f}")
        else:
            verdict = _complete_with_fallback(
                "grade",
                messages=[
                    {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\n"
                        "Does the context contain enough information to at least "
                        "partially answer this?",
                    },
                ],
                temperature=0.0,
            )
            sufficient = verdict.content.strip().upper().startswith("YES")
            result = GuardrailResult(
                "context_sufficiency",
                sufficient,
                reason=(
                    None
                    if sufficient
                    else f"deterministic overlap={score:.2f} below threshold, grade escalation also said insufficient"
                ),
            )
    except Exception as exc:
        result = GuardrailResult("context_sufficiency", True, f"guardrail error, failing open: {exc}")

    record_guardrail_event(result)
    return result


# How many distinct papers the reranked top-8 must span before the
# referential-ambiguity signal below fires. Not a tuned value, but a
# directly evidenced one (2026-08-22): checked live against 6 real
# under_specified questions (u025-u030) and 4 real unambiguous
# answerable questions (q001/q003/q011/q017) — every answerable
# question stayed at exactly 1 distinct paper after rerank, while 4 of
# 6 ambiguous ones spanned 3-5.
#
# Started at >= 2 (zero false positives on that narrow 4-question
# sample), but a real 61-question production run at threshold 2 showed
# 2 NEW over-refusal cases (q022, q030) — real, well-defined questions
# that happen to retrieve chunks from a second paper simply because
# that paper CITES the same baseline/method in passing, not because
# the question is ambiguous. Diversity alone can't distinguish "several
# papers equally answer this" from "one paper answers this, another
# just mentions the same term." Raised to >= 3 (2026-08-22) on the
# evidence that every CONFIRMED true positive so far (u026, u027, u028,
# u030) spanned 3-5 papers, never exactly 2 — so 2-paper cases are the
# likely source of the false positives and 3+ is where the real signal
# lives. Not yet re-verified live at the new threshold (blocked by Jina
# balance exhaustion the same day this was raised) — treat as a
# reasoned, evidence-backed adjustment pending that confirmation run,
# not a re-proven number.
_PAPER_DIVERSITY_MIN_DISTINCT = int(os.environ.get("AMBIGUITY_PAPER_DIVERSITY_MIN_DISTINCT", "3"))

# Plan B2 (2026-08-22): the raw distinct-paper COUNT above can't tell
# "several papers genuinely tie for the answer" from "one paper clearly
# answers this, a couple others just cite the same term in passing" —
# exactly the q022/q030 false-positive pattern that motivated raising
# the count threshold to 3. This refines it: when reranking produced
# real scores (not degraded), a clear agreement across the top 3
# highest-ranked candidates on one paper is real evidence that paper
# dominates, even if lower-ranked candidates from other papers still
# push the distinct-paper COUNT over the threshold.
#
# Deliberately top-3, not top-2: checked directly against this
# project's own real recorded data from a confirmed true positive
# (u028's live paper ordering was [P, P, P2, ...] — the top TWO
# candidates already agreed on one paper, even though the question was
# genuinely ambiguous and correctly needed to be flagged). A top-2
# agreement rule would have silently suppressed that real catch;
# requiring 3 is the minimum that survives u028 while still meaning
# something (a single stray top-ranked outlier can't trigger it).
#
# Falls back to the pure count-based rule when reranking degraded
# (RankedCandidate.score is None on every item — the item ORDER is then
# just RRF-fusion order, not a true relevance ranking, so "top 3 agree"
# wouldn't mean what it claims to mean here).
_TOP_N_AGREEMENT_FOR_CLEAR_WINNER = 3


def _has_clear_dominant_paper(reranked: RerankResult, paper_ids_ordered: list[str | None]) -> bool:
    if reranked.degraded or len(paper_ids_ordered) < _TOP_N_AGREEMENT_FOR_CLEAR_WINNER:
        return False
    top_n = paper_ids_ordered[:_TOP_N_AGREEMENT_FOR_CLEAR_WINNER]
    return top_n[0] is not None and len(set(top_n)) == 1


def _paper_diversity_note(reranked: RerankResult, metadata: dict[str, dict]) -> str | None:
    """Deterministic referential-ambiguity signal — zero LLM calls, added
    after real evidence the LLM judge below catches 0 of 6 real
    under_specified cases (u025-u030): assess_ambiguity only ever sees
    the already-reranked top-8 context for THIS query, so a small/fast
    grade-tier model has no way to notice "there were several
    equally-plausible papers here" just by reading 8 passages in isolation
    — it has to infer that other candidates almost-but-didn't make the cut,
    which zero-shot small models are bad at. The reranked set itself
    already contains the evidence though: if it spans multiple distinct
    papers with no one paper dominating, that IS the definition of
    "the question didn't pin down which one it means" for this corpus.
    Verified live at threshold 2 to fire on 4/6 real under_specified
    cases and 0/4 real unambiguous answerable ones — but a full
    production run at that threshold also produced 2 new over-refusal
    false positives, prompting the raise to 3 (see
    _PAPER_DIVERSITY_MIN_DISTINCT's docstring for the full evidence).

    Only covers referential ambiguity, not false premise (a single-paper
    contradiction, invisible to a cross-paper diversity count) — the LLM
    call in assess_ambiguity is still the only check for that half.

    B2 (2026-08-22): also suppresses the flag when the top 3 ranked
    candidates clearly agree on one paper (see
    _has_clear_dominant_paper) — real evidence one paper dominates even
    if the raw count of distinct papers among all 8 candidates is still
    high.

    `metadata` is the caller's responsibility, not fetched here — a real
    bug found live 2026-08-24: this used to call its own
    `fetch_metadata([c.id for c in reranked.items])` with no `index_name`,
    always defaulting to the production `rag_chunks` index regardless of
    which corpus the reranked candidates actually came from. Under a
    non-default A7 chunking-strategy corpus, those candidates' real
    chunk_ids only exist in that corpus's own index — the unindexed mget
    found nothing, `distinct_papers` was permanently empty, and this
    signal went silently dead for as long as the toggle stayed non-default,
    with no error. `graph.py` already computes the correctly-indexed
    metadata dict once via `run_rerank_and_metadata` (same candidates,
    same call) — reusing it here also removes a second, wasted mget
    round-trip for data the caller already had in scope.
    """
    if not reranked.items:
        return None
    paper_ids_ordered = [metadata.get(c.id, {}).get("paper_id") for c in reranked.items]
    distinct_papers = {p for p in paper_ids_ordered if p}
    if len(distinct_papers) >= _PAPER_DIVERSITY_MIN_DISTINCT and not _has_clear_dominant_paper(
        reranked, paper_ids_ordered
    ):
        return (
            f"the top-ranked context spans {len(distinct_papers)} different papers with no single "
            "dominant match — the question may not specify which one it means"
        )
    return None


# Plan B1 (2026-08-22): a deterministic false-premise signal for the one
# specific failure pattern that has now resisted both the prompt fix
# (rules 6/7 in generate.py) and the LLM judge above, across multiple
# models, on the same real case: u018 asks about "900 states from 3,009
# ... questions", but the real paper says 3,009 states from 900
# questions — the numbers are real (both appear in the retrieved
# context) but stated in the wrong pairing/order. Unlike referential
# ambiguity, this needs no diversity signal — it's a single-paper,
# single-passage contradiction, which is exactly the case
# assess_ambiguity's own docstring says overlap_ratio-style checks
# can't distinguish ("relevant" vs "relevant but contradicted"). This
# CAN catch it deterministically because it doesn't need to judge
# meaning — only whether the question's own number ordering appears
# anywhere in the context, or only the reversed ordering does.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _extract_numbers(text: str) -> list[int]:
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        cleaned = match.group().replace(",", "")
        try:
            numbers.append(int(float(cleaned)))
        except ValueError:
            continue
    return numbers


def _has_adjacent_pair(numbers: list[int], first: int, second: int) -> bool:
    return any(numbers[i] == first and numbers[i + 1] == second for i in range(len(numbers) - 1))


def _numeric_transposition_note(query: str, reranked: RerankResult) -> str | None:
    """Deterministic, zero-LLM-cost — flags when the question states two
    numbers adjacently in one order, both numbers are real (present
    somewhere in the retrieved context), but only the REVERSED order of
    that pair appears adjacently in the context, never the question's
    own order. A real, accepted limitation: this has no entity binding
    — it cannot tell "these two numbers describe the same fact,
    transposed" from "these two numbers each appear in the context for
    entirely unrelated reasons, coincidentally in reversed adjacent
    order somewhere else." Narrow by design (adjacency-in-original-
    question-order is the specific pattern u018 exhibits), not a
    general numeric fact-checker.
    """
    if not reranked.items:
        return None
    query_numbers = _extract_numbers(query)
    if len(query_numbers) < 2:
        return None
    context_numbers = _extract_numbers(" ".join(c.text for c in reranked.items))
    context_number_set = set(context_numbers)

    for i in range(len(query_numbers) - 1):
        a, b = query_numbers[i], query_numbers[i + 1]
        if a == b or a not in context_number_set or b not in context_number_set:
            continue
        if _has_adjacent_pair(context_numbers, b, a) and not _has_adjacent_pair(context_numbers, a, b):
            return (
                f"the question states {a} and {b} in that order, but the context states "
                f"{b} and {a} in that order for the same two figures — the question may "
                "have these transposed"
            )
    return None


# Real gap found live 2026-08-24 (A8's first genuinely complete run):
# u025 ("What score did the framework achieve on its benchmark?")
# retrieves 7 of 8 reranked candidates from ONE paper (AutoDesign)
# purely because that paper's content is the strongest surface-level
# match for "score"/"benchmark" — _paper_diversity_note correctly does
# NOT fire (one paper genuinely dominates the reranked set), even
# though the RAW QUESTION is referentially ambiguous across the whole
# corpus (the dataset's own note: at least 3 papers report "a framework
# score on a named benchmark"). Retrieval narrowing hides the ambiguity
# from every check that only looks at the reranked context — including
# _paper_diversity_note itself and the LLM judge, both of which only
# ever see what retrieval already decided to surface. This new signal
# looks at the QUESTION TEXT ONLY, before that narrowing happens —
# same "check the input, not just the narrowed output" idea as
# query_planner.py's own pre-retrieval _should_skip_rewrite check.
#
# Real, corpus-specific anchor list — the actual system/model names
# used across this exact 8-paper corpus, same deliberately-scoped-not-
# general convention as query_planner.py's _DOMAIN_ANCHOR_TERMS. Update
# when Gap C (corpus expansion) actually lands.
_KNOWN_ENTITY_NAME_RE = re.compile(
    r"\b(autodesign|visdocagentbench|gem|ogr|annoindex|dept|sc2r|search-r1|search r1|"
    r"posterbench|promptriever|schemaloop)\b",
    re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")
_GENERIC_REFERENT_RE = re.compile(
    r"\b(the model|the framework|the system|the approach|the method|the baseline|"
    r"the paper|the algorithm|it|its)\b",
    re.IGNORECASE,
)


def _unresolved_reference_note(query: str) -> str | None:
    """Deterministic, zero-LLM-cost, and — unlike _paper_diversity_note —
    computed from the raw query text alone, independent of whatever
    retrieval happens to surface. Verified live against all 6 real
    under_specified cases (u025-u030, all correctly flag) and every real
    gold-set question using similar generic wording but naming a real
    entity (q006, q027, q035, q037, q070 — all correctly do not flag,
    since each names a real corpus entity: SchemaLoop, Search-R1, GEM,
    Promptriever, AutoDesign/PosterBench).

    Word-boundary regex throughout, not naive substring checks — a
    naive `"gem" in query.lower()` would false-positive on ordinary
    words like "management" (which contains "gem" as a bare substring).
    """
    if not _GENERIC_REFERENT_RE.search(query):
        return None
    if _KNOWN_ENTITY_NAME_RE.search(query) or _ARXIV_ID_RE.search(query):
        return None
    return (
        "the question refers to a system/model/framework generically "
        "(e.g. 'it', 'the model') without naming which one — this corpus "
        "has multiple candidates and nothing in the question disambiguates"
    )


@dataclass(frozen=True)
class AmbiguitySignal:
    """A real, structured signal handed to `generate` — not a block/allow
    guardrail verdict like GuardrailResult, because the point here isn't
    to stop the pipeline, it's to inform the answer. `note` is the exact
    text injected into generate's prompt when flagged, so what generate
    sees is traceable back to what assess actually found, not a black box.
    """

    flagged: bool
    note: str | None = None


_NOT_FLAGGED = AmbiguitySignal(flagged=False)

_AMBIGUITY_SYSTEM_PROMPT = (
    "You are checking a question against context passages for two specific "
    "problems before an answer gets written: (1) the question asserts "
    "something the context directly contradicts (a false premise) — this "
    "means the question states a specific fact, number, or claim that the "
    "context shows to be wrong. It does NOT mean the question merely asks "
    "for a list, count, or summary of things without naming them itself — "
    "\"what are the six capabilities\" or \"what are the main components\" "
    "are normal requests for information, not false premises, even though "
    "the question doesn't enumerate the items itself. Only flag problem (1) "
    "if you can point to a specific number, name, or fact the question "
    "states that the context directly contradicts. Or (2) the question "
    "could refer to more than one paper, model, or result in the context "
    "and nothing in the question disambiguates which one. "
    "Respond with exactly two lines: line 1 is YES or NO (YES if either "
    "problem applies); line 2, only if YES, is a one-sentence description "
    "quoting the exact contradicted fact or naming the specific ambiguity."
)


def assess_ambiguity(query: str, reranked: RerankResult, metadata: dict[str, dict]) -> AmbiguitySignal:
    """Pre-generation false-premise/ambiguity check — the real gap A8
    found: rules 6/7 in generate.py's system prompt catch a *simple*
    false premise, but generate only ever sees the already-narrowed
    reranked context, so it has no way to notice the question could
    equally refer to a *different* candidate that isn't the one shown.
    This runs earlier, sees the same reranked set generate will see, and
    hands generate an explicit, traceable note instead of expecting it
    to self-diagnose from the passages alone.

    `metadata` — the same correctly-indexed dict `run_rerank_and_metadata`
    already computed for these exact candidates — is passed in rather
    than re-fetched here; see `_paper_diversity_note`'s docstring for the
    real bug (silent, index-mismatched failure under a non-default
    chunking strategy) this specifically fixes.

    For false premise specifically, there is no cheap deterministic
    pre-filter — a false premise can have HIGH lexical overlap with the
    correct facts while stating them backwards (the real u017/u018
    cases), so overlap_ratio can't distinguish "relevant" from
    "relevant but contradicted." That half always escalates to the LLM
    call below — a real, accepted per-query cost, not an oversight.

    For referential ambiguity ("it", "the model", "the baseline" with
    no named entity), there IS now a cheap deterministic signal — see
    _paper_diversity_note. Added 2026-08-22 after live evidence the LLM
    call alone caught 0 of 6 real under_specified cases (u025-u030) even
    with the grade-role fallback fix in place; the diversity check on
    its own would have caught 4 of 6.

    For the specific "real numbers, wrong order" false-premise pattern
    (u018), there IS also now a cheap deterministic signal — see
    _numeric_transposition_note. Added the same day after that exact
    case survived both the prompt fix and two different LLM judges.

    A fourth signal, _unresolved_reference_note, catches referential
    ambiguity that _paper_diversity_note structurally cannot: when
    retrieval/reranking happens to converge overwhelmingly on ONE paper
    for surface-relevance reasons even though the raw question itself
    never named which paper it meant (real case: u025). Runs on the
    query text alone, independent of what reranking surfaced.

    All four signals are ORed together, each failing open
    independently, so a failure in any one (e.g. an OpenSearch hiccup on
    the diversity check) doesn't take down the others.

    Fails OPEN (same reasoning as assess_context): an error in any
    signal means that signal contributes nothing, not a new risk — at
    worst this behaves exactly like it did before these checks existed.
    """
    if not reranked.items:
        return _NOT_FLAGGED

    diversity_note: str | None = None
    try:
        diversity_note = _paper_diversity_note(reranked, metadata)
    except Exception:
        diversity_note = None

    transposition_note: str | None = None
    try:
        transposition_note = _numeric_transposition_note(query, reranked)
    except Exception:
        transposition_note = None

    reference_note: str | None = None
    try:
        reference_note = _unresolved_reference_note(query)
    except Exception:
        reference_note = None

    llm_flagged = False
    llm_note: str | None = None
    try:
        context_text = " ".join(c.text for c in reranked.items)
        verdict = _complete_with_fallback(
            "grade",
            messages=[
                {"role": "system", "content": _AMBIGUITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
            ],
            temperature=0.0,
        )
        lines = [line.strip() for line in verdict.content.strip().splitlines() if line.strip()]
        llm_flagged = bool(lines) and lines[0].upper().startswith("YES")
        llm_note = lines[1] if llm_flagged and len(lines) > 1 else (
            "flagged, no detail given" if llm_flagged else None
        )
    except Exception:
        llm_flagged, llm_note = False, None

    flagged = llm_flagged or bool(transposition_note) or bool(diversity_note) or bool(reference_note)
    # Prefer the LLM's free-text note when it fired (most specific);
    # then the transposition note (deterministic but names the exact
    # two figures); then the reference note (deterministic, names the
    # real pattern); then the diversity note (templated, least
    # specific) — still concrete and traceable, just the last choice.
    note = llm_note if llm_flagged else (transposition_note or reference_note or diversity_note)
    result = AmbiguitySignal(flagged=flagged, note=note)

    guardrail_result = GuardrailResult(
        "ambiguity_detection", not result.flagged, reason=result.note
    )
    record_guardrail_event(guardrail_result)
    return result
