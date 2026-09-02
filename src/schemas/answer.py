from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str | None = None
    # The retrieved chunk's actual text — without this, a citation is a
    # footnote asking to be trusted, not evidence a reader can check (see
    # the plan's Ask page principle). Populated by generate_answer from
    # the same RankedCandidate the LLM was actually shown.
    text: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    # True when the model explicitly declined to answer from the given
    # context (see A8) rather than fabricating one.
    abstained: bool = False
    # True when the model was flagged (by assess_ambiguity) as facing a
    # false premise or genuine ambiguity, and its response committed to
    # zero cited factual claims — a real, structural proxy for "asked
    # for clarification / challenged the premise" rather than a literal
    # ABSTAIN_TEXT match. Added for A8: the original abstention scoring
    # (_refused() in evals/run_abstention_eval.py) only recognized the
    # exact ABSTAIN_TEXT prefix, so a real, correct "which one do you
    # mean?" response (u027, verified live) was being scored as if it
    # had failed. See generate_answer's docstring for exactly how this
    # is set.
    declined_to_guess: bool = False
