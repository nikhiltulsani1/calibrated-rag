import pytest

from src.reason.pipeline import StageTrace
from src.retrieve.hybrid import ArmResult, FusionTrace
from src.retrieve.reranker import Candidate, RankedCandidate, RerankResult
from src.schemas.answer import Answer, Citation
from src.schemas.query_plan import QueryPlan
from src.store.relational import get_session
from src.store.runs import load_run, save_run
from src.store.schema import Run

pytestmark = pytest.mark.integration


@pytest.fixture
def session():
    s = get_session()
    yield s
    s.close()


def _real_trace() -> StageTrace:
    plan = QueryPlan(original="what is RLHF", normalized="what is rlhf", expansions=["reinforcement learning from human feedback"], intent="factual")
    candidates = [Candidate(id="c1", text="RLHF trains models using human feedback")]
    fusion = FusionTrace(
        arms=[ArmResult(variant="what is rlhf", arm="lexical", ranked_ids=["c1"])],
        fused_scores={"c1": 0.5},
        fused_order=["c1"],
        dense_index_name="rag_chunks",
    )
    reranked = RerankResult(
        items=[RankedCandidate(id="c1", text="RLHF trains models using human feedback", score=0.9)],
        degraded=False,
        reason=None,
        model_served="jina-reranker-v3.5",
    )
    answer = Answer(
        text="RLHF trains models using human feedback [1]",
        citations=[Citation(chunk_id="c1", paper_id="p1", title="A Paper", section="intro", text="RLHF trains models using human feedback")],
        abstained=False,
    )
    return StageTrace(
        original_query="what is RLHF",
        stopped_at=None,
        query_plan=plan,
        retrieved=candidates,
        fusion=fusion,
        reranked=reranked,
        metadata={"c1": {"title": "A Paper", "paper_id": "p1", "section": "intro"}},
        answer=answer,
    )


def test_save_and_load_run_round_trips_real_shapes(session):
    trace = _real_trace()
    run_id = save_run(session, trace)

    loaded = load_run(session, run_id)

    assert loaded is not None
    assert loaded["original_query"] == "what is RLHF"
    assert loaded["query_plan"]["intent"] == "factual"
    assert loaded["fusion"]["fused_order"] == ["c1"]
    assert loaded["reranked"]["items"][0]["score"] == 0.9
    assert loaded["answer"]["text"] == "RLHF trains models using human feedback [1]"
    assert loaded["answer"]["citations"][0]["chunk_id"] == "c1"
    # attempts defaults to an empty list on a single-pass trace, and must
    # still round-trip as a (JSON-safe) empty list, not be dropped
    assert loaded["attempts"] == []

    session.delete(session.get(Run, run_id))
    session.commit()


def test_load_run_returns_none_for_unknown_id(session):
    assert load_run(session, "00000000-0000-0000-0000-000000000000") is None
