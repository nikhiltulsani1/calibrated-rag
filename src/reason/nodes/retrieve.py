from __future__ import annotations

from src.index.mapping import INDEX_NAME
from src.reason.state import fetch_metadata
from src.retrieve.hybrid import FusionTrace, retrieve_with_trace
from src.retrieve.reranker import Candidate, RerankResult, rerank
from src.schemas.query_plan import QueryPlan

_RECALL_SIZE = 50


def run_retrieve(plan: QueryPlan, index_name: str = INDEX_NAME) -> tuple[list[Candidate], FusionTrace]:
    """The raw hybrid-search fetch — once per query, not once per retry
    attempt. Only reranking's `top_n` widens on retry (see nodes/refine.py);
    re-running the OpenSearch fetch itself on retry would just re-score the
    same 50 candidates, not find new ones. `index_name` defaults to the
    production index; graph.py passes A7's active toggle selection.
    """
    return retrieve_with_trace(plan, top_n=_RECALL_SIZE, index_name=index_name)


def run_rerank_and_metadata(
    query: str, candidates: list[Candidate], top_n: int, index_name: str = INDEX_NAME
) -> tuple[RerankResult, dict[str, dict]]:
    """Rerank, then look up the winning candidates' titles/paper_id/section
    — cheap, deterministic, and needed by both the assess node (context
    display) and the answer node (citations), so it happens once here
    right after reranking rather than being duplicated in either.
    """
    reranked = rerank(query, candidates, top_n=top_n)
    metadata = fetch_metadata([item.id for item in reranked.items], index_name=index_name)
    return reranked, metadata
