import pytest
from pydantic import ValidationError

from src.schemas.query_plan import QueryPlan

pytestmark = pytest.mark.unit


def test_valid_query_plan_round_trips():
    plan = QueryPlan(
        original="q",
        normalized="q",
        expansions=["a", "b"],
        filters={"category": "cs.IR"},
        intent="factual",
    )
    assert plan.expansions == ["a", "b"]
    assert plan.intent == "factual"


def test_defaults_are_empty_not_missing():
    plan = QueryPlan(original="q", normalized="q", intent="factual")
    assert plan.expansions == []
    assert plan.filters == {}


def test_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        QueryPlan(original="q", normalized="q", intent="not_a_real_intent")


@pytest.mark.parametrize("intent", ["factual", "comparative", "multi_hop", "out_of_scope"])
def test_accepts_every_documented_intent(intent):
    plan = QueryPlan(original="q", normalized="q", intent=intent)
    assert plan.intent == intent
