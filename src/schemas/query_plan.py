from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["factual", "comparative", "multi_hop", "out_of_scope"]


class QueryPlan(BaseModel):
    original: str
    normalized: str
    expansions: list[str] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)
    intent: Intent
    # True only when every rewrite attempt failed and plan_query() fell
    # back to a safe, unrewritten plan (original query used verbatim,
    # intent defaulted to "factual") rather than raising — same
    # degrade-instead-of-crash pattern as RerankResult.degraded. Never
    # cached, so a transient failure doesn't poison repeat queries.
    degraded: bool = False
    degrade_reason: str | None = None
