from __future__ import annotations

from src.reason.graph import run_graph
from src.reason.state import AttemptTrace, StageTrace, fetch_metadata
from src.schemas.answer import Answer

# `run_traced_query`/`answer_query` keep their exact original signatures
# and this module keeps re-exporting `StageTrace`/`AttemptTrace`/
# `fetch_metadata` — every existing caller (src/app/routes/ask.py,
# pipeline.py, evals/run_generation_eval.py, and every test that does
# `from src.reason.pipeline import ...`) needs zero changes. The actual
# orchestration now lives in src/reason/graph.py, decomposed into
# src/reason/nodes/* — see graph.py's docstring for why (this file
# importing graph.py, which imports nodes/* that read shared types from
# state.py, is what keeps this a one-directional import graph instead of
# a circular one).

__all__ = ["AttemptTrace", "StageTrace", "fetch_metadata", "run_traced_query", "answer_query"]


def run_traced_query(
    query: str, *, top_n_context: int = 8, session_id: str | None = None, document_id: str | None = None
) -> StageTrace:
    """The full pipeline, and the single source of truth for its
    orchestration: guardrails -> rewrite -> hybrid retrieve -> rerank ->
    assess -> generate -> guardrails. `answer_query()` below is a thin
    wrapper returning just the final answer; this is what actually runs,
    and what A5's visualiser calls to get every real intermediate stage.
    See `src.reason.graph.run_graph` for the real implementation.

    `session_id`/`document_id` (Phase 2, stage 6) are only meaningful on
    the RETRIEVAL_BACKEND=postgres path — see run_graph's docstring.
    """
    return run_graph(query, top_n_context=top_n_context, session_id=session_id, document_id=document_id)


def answer_query(query: str, *, top_n_context: int = 8) -> Answer:
    """The Ask endpoint's actual implementation — just the final answer.
    See run_traced_query for the full stage-by-stage trace."""
    return run_traced_query(query, top_n_context=top_n_context).answer
