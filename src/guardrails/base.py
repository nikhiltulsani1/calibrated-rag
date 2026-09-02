from __future__ import annotations

import os
import re
from dataclasses import dataclass

from opentelemetry import trace

# Shared word-overlap primitive — originally lived only in
# output_guardrails.py (check_groundedness), moved here so the new
# context_sufficiency guardrail (src/reason/nodes/assess.py) can reuse the
# exact same "cheap deterministic check first, escalate to a judge/grade
# call only when it's inconclusive" shape without duplicating the logic.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to",
    "and", "or", "for", "with", "that", "this", "it", "as", "by", "be",
    "has", "have", "had", "from", "at", "which", "who", "what", "not",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def overlap_ratio(reference_text: str, candidate_text: str) -> float:
    """Fraction of `reference_text`'s (non-stopword) tokens that also
    appear in `candidate_text`. Cheap, deterministic, cannot itself
    hallucinate — the "handle the common case" check both groundedness
    and context-sufficiency run before any judge/grade escalation.
    """
    reference_tokens = tokenize(reference_text)
    if not reference_tokens:
        return 0.0
    candidate_tokens = tokenize(candidate_text)
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


@dataclass(frozen=True)
class GuardrailResult:
    guardrail: str
    passed: bool
    reason: str | None = None
    # True only when the check itself broke (e.g. a missing judge API
    # key), as opposed to the check running fine and genuinely finding a
    # problem. This distinction is what the pipeline's retry loop uses to
    # decide whether retrying could plausibly help — retrying a real
    # infra error just burns another call for the same failure.
    errored: bool = False


def guardrail_mode(name: str) -> str:
    """"monitor" (default) means every new guardrail evaluates and traces
    but never blocks — the tuning state. "enforce" actually blocks on a
    failed check.

    Full three-state (off/monitor/enforce) switching with a TTL
    auto-revert and an audit trail is the control-plane dashboard's job
    (not yet built). This is the minimal version that makes "every new
    guardrail starts in monitor" true today, via one env var per
    guardrail rather than a persisted, switchable state store.
    """
    return os.environ.get(f"GUARDRAIL_{name.upper()}_MODE", "monitor")


def record_guardrail_event(result: GuardrailResult) -> None:
    """Every guardrail *firing* (failing a check) gets a trace event —
    not every guardrail *run*, which would just be noise. A guardrail
    firing frequently is meant to be a visible signal, per the plan, not
    an invisible silent save.
    """
    if result.passed:
        return
    span = trace.get_current_span()
    span.add_event(f"guardrail.{result.guardrail}.fired", {"reason": result.reason or ""})
