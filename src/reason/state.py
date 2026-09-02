from __future__ import annotations

from dataclasses import dataclass, field

from src.index.client import get_client
from src.index.mapping import INDEX_NAME
from src.retrieve.hybrid import FusionTrace
from src.retrieve.reranker import Candidate, RerankResult
from src.schemas.answer import Answer
from src.schemas.query_plan import QueryPlan

# Canonical home for the pipeline's public result types and its one leaf
# data-access helper. Moved here (out of pipeline.py) so nodes/ and graph.py
# can both depend on this module without importing pipeline.py itself —
# pipeline.py imports graph.py, so graph.py (and anything graph.py imports)
# importing back from pipeline.py would be circular. pipeline.py re-exports
# everything below so `from src.reason.pipeline import StageTrace` etc. keeps
# working unchanged for every existing caller and test.


@dataclass(frozen=True)
class AttemptTrace:
    """One pass through rerank -> generate -> guardrails. More than one of
    these on a StageTrace means the pipeline retried after an earlier
    attempt abstained or came back ungrounded. The winning attempt's own
    values are also what land on StageTrace's top-level `reranked`/`answer`
    fields, so any caller that only wants the final result never needs to
    know a retry happened at all.
    """

    attempt: int
    top_n_context: int
    reranked: RerankResult
    answer: Answer
    groundedness_passed: bool
    groundedness_reason: str | None


@dataclass(frozen=True)
class StageTrace:
    """Every intermediate stage's real output from one run — this is
    what A5's pipeline visualiser renders. `stopped_at` names the
    guardrail that short-circuited the run, or None if it reached
    generation. Fields past the stop point stay at their defaults rather
    than being backfilled with placeholder data.
    """

    original_query: str
    stopped_at: str | None = None
    query_plan: QueryPlan | None = None
    retrieved: list[Candidate] = field(default_factory=list)
    fusion: FusionTrace | None = None
    reranked: RerankResult | None = None
    metadata: dict[str, dict] = field(default_factory=dict)
    answer: Answer | None = None
    attempts: list[AttemptTrace] = field(default_factory=list)
    # A7's live chunking-strategy toggle — "default" is today's production
    # rag_chunks index; see src/reason/chunking_toggle.py. Recorded here so
    # A5's pipeline visualiser can show which corpus actually answered.
    chunking_strategy: str = "default"
    # Real per-stage wall-clock time in ms, keyed by stage name (e.g.
    # "plan_query", "retrieve", "rerank", "assess_context",
    # "assess_ambiguity", "generate"). Added for the latency-optimization
    # work (A3's own report showed the full pipeline at 7173ms p50 vs
    # BM25-only's 144ms — this is what makes each later optimization
    # provable against a real number instead of assumed). Populated by
    # graph.py's _timed() helper; empty on any short-circuit path that
    # predates real timing being recorded there, per this project's rule
    # of never backfilling data past the point a run actually stopped.
    stage_timings: dict[str, float] = field(default_factory=dict)


def fetch_metadata(candidate_ids: list[str], index_name: str = INDEX_NAME) -> dict[str, dict]:
    if not candidate_ids:
        return {}
    client = get_client()
    docs = client.mget(index=index_name, body={"ids": candidate_ids})["docs"]
    return {
        doc["_id"]: {
            "title": doc["_source"].get("title", ""),
            "paper_id": doc["_source"].get("paper_id", ""),
            "section": doc["_source"].get("section"),
        }
        for doc in docs
        if doc.get("found")
    }
