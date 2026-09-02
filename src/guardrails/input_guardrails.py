from __future__ import annotations

from src.guardrails.base import GuardrailResult, record_guardrail_event
from src.schemas.query_plan import QueryPlan

_MAX_QUERY_CHARS = 2000


def screen_scope(plan: QueryPlan) -> GuardrailResult:
    """Uses A1's already-computed `intent` — no extra call, per the plan.

    Fails OPEN: an error here lets the request through rather than
    blocking it. Input-side guardrails failing closed would turn a bug in
    the guardrail itself into a denial of service — see "Guardrail
    discipline" in the plan.
    """
    try:
        passed = plan.intent != "out_of_scope"
        result = GuardrailResult(
            guardrail="scope_screening",
            passed=passed,
            reason=None if passed else "classified out_of_scope by the query planner",
        )
    except Exception as exc:
        result = GuardrailResult("scope_screening", True, f"guardrail error, failing open: {exc}")
    record_guardrail_event(result)
    return result


def check_size_and_shape(query: str) -> GuardrailResult:
    """Deterministic length limits — the cheapest possible guard, run
    before anything else so a garbage query never reaches an LLM call.
    Fails open, same reasoning as screen_scope.
    """
    try:
        if len(query) == 0:
            result = GuardrailResult("size_and_shape", False, "query is empty")
        elif len(query) > _MAX_QUERY_CHARS:
            result = GuardrailResult(
                "size_and_shape", False, f"query exceeds {_MAX_QUERY_CHARS} characters"
            )
        else:
            result = GuardrailResult("size_and_shape", True)
    except Exception as exc:
        result = GuardrailResult("size_and_shape", True, f"guardrail error, failing open: {exc}")
    record_guardrail_event(result)
    return result
