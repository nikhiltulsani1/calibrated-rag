from __future__ import annotations

from src.guardrails.base import GuardrailResult
from src.guardrails.output_guardrails import check_citation_integrity, check_groundedness
from src.reason.generate import generate_answer
from src.retrieve.reranker import RankedCandidate
from src.schemas.answer import Answer


def run_answer(
    query: str,
    reranked_items: list[RankedCandidate],
    metadata: dict[str, dict],
    *,
    ambiguity_note: str | None = None,
) -> tuple[Answer, GuardrailResult, GuardrailResult]:
    """generate -> strip invalid citations -> check groundedness, in that
    exact order, moved verbatim out of the old pipeline.py loop. Citation
    stripping is a correctness fix applied unconditionally (not mode-gated
    — see check_citation_integrity's own docstring), so it always runs
    before groundedness is checked against the (possibly corrected) answer.

    `ambiguity_note` is threaded straight through to generate_answer —
    see that function's docstring for what it does.
    """
    answer = generate_answer(query, reranked_items, metadata, ambiguity_note=ambiguity_note)
    answer, citation_result = check_citation_integrity(answer, reranked_items)
    grounded_result = check_groundedness(answer, reranked_items)
    return answer, citation_result, grounded_result
