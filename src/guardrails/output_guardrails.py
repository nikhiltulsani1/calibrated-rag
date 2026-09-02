from __future__ import annotations

import os

from src.guardrails.base import GuardrailResult, overlap_ratio, record_guardrail_event
from src.platform.models import complete
from src.retrieve.reranker import RankedCandidate
from src.schemas.answer import Answer

# A starting point to tune against real failure examples, not a verified
# correct value — same "verify, don't assume" discipline the rest of this
# project applies to free-tier limits, applied here to a guardrail
# threshold with no real data behind it yet.
_OVERLAP_THRESHOLD = float(os.environ.get("GROUNDEDNESS_OVERLAP_THRESHOLD", "0.5"))

_JUDGE_SYSTEM_PROMPT = (
    "You are checking whether an answer is fully supported by the given "
    "context, with no unsupported claims. Respond with exactly one word: "
    "YES or NO."
)


def check_groundedness(answer: Answer, context: list[RankedCandidate]) -> GuardrailResult:
    """Deterministic overlap check first; escalates to a single judge
    call only when the deterministic score is below threshold — rare by
    design, since the judge call is the expensive path (see the plan's
    latency-budget note).

    Fails CLOSED: an error here blocks the response rather than letting
    an unverified answer through — the opposite of the input-side rule,
    because here a broken guardrail failing open would be exactly the
    hallucination-shipping gap this guardrail exists to close.
    """
    if answer.abstained:
        result = GuardrailResult("groundedness", True, reason="abstained — nothing to be ungrounded about")
        record_guardrail_event(result)
        return result

    try:
        context_text = " ".join(c.text for c in context)
        score = overlap_ratio(answer.text, context_text)

        if score >= _OVERLAP_THRESHOLD:
            result = GuardrailResult("groundedness", True, reason=f"overlap={score:.2f}")
        else:
            verdict = complete(
                "judge",
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nAnswer:\n{answer.text}\n\n"
                        "Is the answer fully supported by the context?",
                    },
                ],
                temperature=0.0,
            )
            grounded = verdict.content.strip().upper().startswith("YES")
            result = GuardrailResult(
                "groundedness",
                grounded,
                reason=(
                    None
                    if grounded
                    else f"deterministic overlap={score:.2f} below threshold, judge escalation also failed"
                ),
            )
    except Exception as exc:
        result = GuardrailResult("groundedness", False, f"guardrail error, failing closed: {exc}", errored=True)

    record_guardrail_event(result)
    return result


def check_citation_integrity(answer: Answer, context: list[RankedCandidate]) -> tuple[Answer, GuardrailResult]:
    """Every citation must resolve to a chunk actually present in the
    context the model was given. Invalid citations are stripped and the
    response flagged — never silently left in, and never silently
    rewritten beyond that strip (see "never silently rewrite user-visible
    content" in the plan).
    """
    valid_ids = {c.id for c in context}
    valid_citations = [c for c in answer.citations if c.chunk_id in valid_ids]
    invalid_count = len(answer.citations) - len(valid_citations)

    if invalid_count == 0:
        result = GuardrailResult("citation_integrity", True)
        record_guardrail_event(result)
        return answer, result

    fixed_answer = answer.model_copy(update={"citations": valid_citations})
    result = GuardrailResult(
        "citation_integrity",
        False,
        reason=f"{invalid_count} citation(s) did not resolve to a context chunk, stripped",
    )
    record_guardrail_event(result)
    return fixed_answer, result
