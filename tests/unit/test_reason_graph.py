from unittest.mock import patch

import pytest

from src.guardrails.base import GuardrailResult
from src.reason.generate import ABSTAIN_TEXT
from src.reason.pipeline import answer_query, run_traced_query
from src.retrieve.hybrid import FusionTrace
from src.retrieve.reranker import Candidate, RankedCandidate, RerankResult
from src.schemas.answer import Answer
from src.schemas.query_plan import QueryPlan

pytestmark = pytest.mark.unit

# Orchestration/branching tests for src/reason/graph.py::run_graph — the
# real implementation behind run_traced_query/answer_query. Migrated from
# the old (pre-agentic-loop-refactor) test_pipeline_orchestration.py:
# patches now target names as graph.py imports them (`src.reason.graph.*`)
# rather than the old flat pipeline.py, and the old fine-grained
# rerank/fetch_metadata/generate_answer/check_groundedness mocks are
# replaced by mocking the two node-level functions that now bundle them
# (run_rerank_and_metadata, run_answer) — see src/reason/nodes/.
# assess_context is mocked to a "sufficient" pass in every test that
# doesn't specifically exercise assess's own branching, so these tests
# stay about wiring order, not about the new guardrail's internals (that
# belongs in test_reason_nodes_assess.py).

_EMPTY_FUSION = FusionTrace(arms=[], fused_scores={}, fused_order=[], dense_index_name="rag_chunks")
_SUFFICIENT = GuardrailResult("context_sufficiency", True, reason="overlap=1.00")


@pytest.fixture(autouse=True)
def _default_chunking_strategy():
    # A7's toggle reads real Redis by default (get_active_strategy) —
    # pinned to "default" here so these orchestration tests stay hermetic
    # and unaffected by whatever strategy happens to be active live. The
    # toggle's own behavior is covered by test_chunking_toggle.py.
    with patch("src.reason.graph.get_active_strategy", return_value="default"), patch(
        "src.reason.graph.active_index_name", return_value="rag_chunks"
    ):
        yield


@pytest.fixture(autouse=True)
def _default_no_real_category_lookup():
    # graph.py fetches the corpus's real category set (real_categories,
    # src/retrieve/hybrid.py) before calling plan_query, added 2026-08-27
    # to close the hallucinated-category-at-extraction gap — a real,
    # unmocked OpenSearch/Redis call that every existing test here would
    # otherwise silently make, since they only ever patch plan_query
    # itself. Pinned to an empty set by default so these orchestration
    # tests stay hermetic; the one test that specifically exercises the
    # wiring (below) overrides this itself.
    with patch("src.reason.graph.real_categories", return_value=set()):
        yield


@pytest.fixture(autouse=True)
def _default_not_ambiguous():
    # assess_ambiguity is a real, unconditional per-attempt LLM call in
    # the real graph — pinned to "not flagged" here by default so these
    # orchestration tests stay hermetic and don't depend on real network
    # access. The signal's own behavior (flagging, note content,
    # fail-open) is covered by test_reason_nodes_assess.py; its
    # monitor/enforce routing into generate is covered explicitly below.
    from src.reason.nodes.assess import AmbiguitySignal

    with patch("src.reason.graph.assess_ambiguity", return_value=AmbiguitySignal(flagged=False)):
        yield


def test_out_of_scope_does_not_short_circuit_in_default_monitor_mode(monkeypatch):
    # "Every new guardrail starts in monitor" — by default a guardrail
    # evaluates and traces but never blocks. scope_screening is
    # switchable, unlike citation_integrity.
    monkeypatch.delenv("GUARDRAIL_SCOPE_SCREENING_MODE", raising=False)
    out_of_scope_plan = QueryPlan(original="q", normalized="q", intent="out_of_scope")
    fake_answer = Answer(text="proceeded anyway", citations=[], abstained=False)
    fake_reranked = RerankResult(items=[], degraded=False, reason=None, model_served=None)

    with patch("src.reason.graph.plan_query", return_value=out_of_scope_plan), patch(
        "src.reason.graph.run_retrieve", return_value=([], _EMPTY_FUSION)
    ) as mock_retrieve, patch(
        "src.reason.graph.run_rerank_and_metadata", return_value=(fake_reranked, {})
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(fake_answer, GuardrailResult("citation_integrity", True), GuardrailResult("groundedness", True))
    ):
        answer = answer_query("some off-topic question")

    assert mock_retrieve.called  # monitor mode: guardrail fired, but did not block
    assert answer == fake_answer


def test_out_of_scope_short_circuits_when_enforced(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_SCOPE_SCREENING_MODE", "enforce")
    out_of_scope_plan = QueryPlan(original="q", normalized="q", intent="out_of_scope")

    with patch("src.reason.graph.plan_query", return_value=out_of_scope_plan), patch(
        "src.reason.graph.run_retrieve"
    ) as mock_retrieve, patch("src.reason.graph.run_rerank_and_metadata") as mock_rerank, patch(
        "src.reason.graph.run_answer"
    ) as mock_answer:
        trace = run_traced_query("some off-topic question")

    assert not mock_retrieve.called
    assert not mock_rerank.called
    assert not mock_answer.called
    assert trace.stopped_at == "scope_screening"
    assert trace.answer.abstained is True
    assert trace.answer.text == ABSTAIN_TEXT


def test_oversized_query_short_circuits_when_enforced(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_SIZE_AND_SHAPE_MODE", "enforce")
    with patch("src.reason.graph.plan_query") as mock_plan_query:
        trace = run_traced_query("x" * 5000)
    assert not mock_plan_query.called  # size check runs before plan_query is ever invoked
    assert trace.stopped_at == "size_and_shape"
    assert trace.answer.abstained is True


def test_in_scope_query_wires_all_stages_in_order(monkeypatch):
    monkeypatch.delenv("GUARDRAIL_GROUNDEDNESS_MODE", raising=False)
    plan = QueryPlan(original="q", normalized="q", intent="factual")
    fake_candidates = [Candidate(id="c1", text="t")]
    fake_fusion = FusionTrace(arms=[], fused_scores={"c1": 0.5}, fused_order=["c1"], dense_index_name="rag_chunks")
    fake_reranked = RerankResult(
        items=[RankedCandidate(id="c1", text="t", score=0.9)],
        degraded=False,
        reason=None,
        model_served="jina-reranker-v3.5",
    )
    fake_answer = Answer(text="the answer", citations=[], abstained=False)
    grounded_pass = GuardrailResult("groundedness", True, reason="overlap=1.00")

    with patch("src.reason.graph.plan_query", return_value=plan), patch(
        "src.reason.graph.run_retrieve", return_value=(fake_candidates, fake_fusion)
    ) as mock_retrieve, patch(
        "src.reason.graph.run_rerank_and_metadata", return_value=(fake_reranked, {"c1": {"title": "T"}})
    ) as mock_rerank, patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(fake_answer, GuardrailResult("citation_integrity", True), grounded_pass)
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    mock_retrieve.assert_called_once_with(plan, index_name="rag_chunks")
    mock_rerank.assert_called_once_with("what is RLHF", fake_candidates, 8, index_name="rag_chunks")
    mock_answer.assert_called_once_with("what is RLHF", fake_reranked.items, {"c1": {"title": "T"}}, ambiguity_note=None)

    # Every stage's real output actually landed on the trace, not just
    # the final answer — this is the whole point of run_traced_query
    # existing separately from answer_query.
    assert trace.query_plan == plan
    assert trace.retrieved == fake_candidates
    assert trace.fusion == fake_fusion
    assert trace.reranked == fake_reranked
    assert trace.metadata == {"c1": {"title": "T"}}
    assert trace.answer.text == fake_answer.text
    assert trace.stopped_at is None
    assert len(trace.attempts) == 1  # passed on the first try, no retry needed


def test_real_categories_are_fetched_and_passed_into_plan_query(monkeypatch):
    # Real bug fix, 2026-08-27 (Results.md §27/item 4): plan_query needs
    # the corpus's real category set to stop the rewrite LLM from
    # hallucinating a plausible-but-wrong one at extraction time — this
    # verifies graph.py actually wires real_categories's result through
    # to plan_query, not just that both are independently callable.
    monkeypatch.delenv("GUARDRAIL_GROUNDEDNESS_MODE", raising=False)
    plan = QueryPlan(original="q", normalized="q", intent="factual")
    fake_answer = Answer(text="the answer", citations=[], abstained=False)
    grounded_pass = GuardrailResult("groundedness", True, reason="overlap=1.00")

    with patch("src.reason.graph.real_categories", return_value={"cs.CL", "cs.IR"}), patch(
        "src.reason.graph.plan_query", return_value=plan
    ) as mock_plan_query, patch(
        "src.reason.graph.run_retrieve", return_value=([], _EMPTY_FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata",
        return_value=(RerankResult(items=[], degraded=False, reason=None, model_served=None), {}),
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(fake_answer, GuardrailResult("citation_integrity", True), grounded_pass)
    ):
        run_traced_query("what is RLHF")

    mock_plan_query.assert_called_once_with("what is RLHF", known_categories=frozenset({"cs.CL", "cs.IR"}))


def test_answer_query_returns_just_the_final_answer():
    plan = QueryPlan(original="q", normalized="q", intent="out_of_scope")
    with patch("src.reason.graph.plan_query", return_value=plan), patch.dict(
        "os.environ", {"GUARDRAIL_SCOPE_SCREENING_MODE": "enforce"}
    ):
        answer = answer_query("off topic")
    assert isinstance(answer, Answer)
    assert answer.abstained is True


# ---------------------------------------------------------------------
# self-correction retry loop
# ---------------------------------------------------------------------

_PLAN = QueryPlan(original="q", normalized="q", intent="factual")
_CANDIDATES = [Candidate(id="c1", text="t")]
_FUSION = FusionTrace(arms=[], fused_scores={"c1": 0.5}, fused_order=["c1"], dense_index_name="rag_chunks")


def _fake_reranked_and_metadata(query, candidates, top_n, index_name=None):
    reranked = RerankResult(items=[RankedCandidate(id="c1", text="t", score=0.9)], degraded=False, reason=None, model_served="m")
    return reranked, {}


def test_retries_after_abstaining_then_succeeds():
    abstained = Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)
    succeeded = Answer(text="the real answer", citations=[], abstained=False)
    grounded_pass = GuardrailResult("groundedness", True)
    citation_pass = GuardrailResult("citation_integrity", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ) as mock_rerank, patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", side_effect=[(abstained, citation_pass, grounded_pass), (succeeded, citation_pass, grounded_pass)]
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    assert mock_answer.call_count == 2
    # second attempt widened the context window rather than repeating
    # the exact same call
    assert mock_rerank.call_args_list[0].args[2] == 8
    assert mock_rerank.call_args_list[1].args[2] == 12
    assert len(trace.attempts) == 2
    assert trace.attempts[0].answer.abstained is True
    assert trace.attempts[1].answer.abstained is False
    assert trace.answer.text == "the real answer"  # the winning attempt, not the first one


def test_retry_gives_up_at_max_attempts_still_failing(monkeypatch):
    monkeypatch.setenv("RAG_MAX_ATTEMPTS", "2")
    abstained = Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)
    grounded_pass = GuardrailResult("groundedness", True)
    citation_pass = GuardrailResult("citation_integrity", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(abstained, citation_pass, grounded_pass)
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    assert mock_answer.call_count == 2  # capped at RAG_MAX_ATTEMPTS, not infinite
    assert len(trace.attempts) == 2
    assert trace.answer.abstained is True


def test_no_retry_when_groundedness_check_itself_errors():
    # A GuardrailResult with errored=True means the *check* broke (e.g. a
    # missing judge key) — retrying generation can't fix that, so this
    # must NOT trigger a second attempt, unlike a genuine "NO" verdict.
    answer = Answer(text="a fine answer", citations=[], abstained=False)
    citation_pass = GuardrailResult("citation_integrity", True)
    errored_result = GuardrailResult("groundedness", False, "guardrail error, failing closed: no key", errored=True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(answer, citation_pass, errored_result)
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    assert mock_answer.call_count == 1
    assert len(trace.attempts) == 1


# ---------------------------------------------------------------------
# assess (context_sufficiency) driving the loop — the genuinely new
# behavior this refactor adds. Monitor mode (default) never skips
# generate; enforce mode does, saving a generate call, but still forces
# a real attempt on the final allowed attempt rather than giving up
# without ever generating.
# ---------------------------------------------------------------------

_INSUFFICIENT = GuardrailResult("context_sufficiency", False, reason="overlap=0.00")


def test_assess_insufficient_does_not_skip_generate_in_default_monitor_mode(monkeypatch):
    monkeypatch.delenv("GUARDRAIL_CONTEXT_SUFFICIENCY_MODE", raising=False)
    answer = Answer(text="an answer anyway", citations=[], abstained=False)
    citation_pass = GuardrailResult("citation_integrity", True)
    grounded_pass = GuardrailResult("groundedness", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ), patch(
        "src.reason.graph.assess_context", return_value=_INSUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(answer, citation_pass, grounded_pass)
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    # monitor mode: assess fired (insufficient) but generate still ran —
    # today's exact behavior, unchanged.
    assert mock_answer.call_count == 1
    assert len(trace.attempts) == 1


def test_assess_insufficient_skips_generate_when_enforced(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_CONTEXT_SUFFICIENCY_MODE", "enforce")
    monkeypatch.setenv("RAG_MAX_ATTEMPTS", "2")
    answer = Answer(text="an answer on the wider context", citations=[], abstained=False)
    citation_pass = GuardrailResult("citation_integrity", True)
    grounded_pass = GuardrailResult("groundedness", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ) as mock_rerank, patch(
        # insufficient on attempt 1, sufficient on attempt 2 (wider context)
        "src.reason.graph.assess_context", side_effect=[_INSUFFICIENT, _SUFFICIENT]
    ), patch(
        "src.reason.graph.run_answer", return_value=(answer, citation_pass, grounded_pass)
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    # attempt 1 was assessed insufficient and skipped generate entirely —
    # the exact call-saving this node exists for.
    assert mock_answer.call_count == 1
    assert mock_rerank.call_count == 2
    assert mock_rerank.call_args_list[1].args[2] == 12  # widened before the second (real) attempt
    assert len(trace.attempts) == 1  # only the real attempt is recorded


def test_ambiguity_flagged_does_not_reach_generate_in_default_monitor_mode(monkeypatch):
    from src.reason.nodes.assess import AmbiguitySignal

    monkeypatch.delenv("GUARDRAIL_AMBIGUITY_DETECTION_MODE", raising=False)
    answer = Answer(text="an answer", citations=[], abstained=False)
    citation_pass = GuardrailResult("citation_integrity", True)
    grounded_pass = GuardrailResult("groundedness", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.assess_ambiguity",
        return_value=AmbiguitySignal(flagged=True, note="the context contradicts the premise"),
    ), patch(
        "src.reason.graph.run_answer", return_value=(answer, citation_pass, grounded_pass)
    ) as mock_answer:
        run_traced_query("what is RLHF", top_n_context=8)

    # monitor mode: the signal was computed (traced) but never reaches generate
    mock_answer.assert_called_once_with("what is RLHF", _fake_reranked_and_metadata(None, None, None)[0].items, {}, ambiguity_note=None)


def test_ambiguity_flagged_note_reaches_generate_when_enforced(monkeypatch):
    from src.reason.nodes.assess import AmbiguitySignal

    monkeypatch.setenv("GUARDRAIL_AMBIGUITY_DETECTION_MODE", "enforce")
    answer = Answer(text="an answer", citations=[], abstained=False)
    citation_pass = GuardrailResult("citation_integrity", True)
    grounded_pass = GuardrailResult("groundedness", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.assess_ambiguity",
        return_value=AmbiguitySignal(flagged=True, note="the context contradicts the premise"),
    ), patch(
        "src.reason.graph.run_answer", return_value=(answer, citation_pass, grounded_pass)
    ) as mock_answer:
        run_traced_query("what is RLHF", top_n_context=8)

    mock_answer.assert_called_once_with(
        "what is RLHF",
        _fake_reranked_and_metadata(None, None, None)[0].items,
        {},
        ambiguity_note="the context contradicts the premise",
    )


# ---------------------------------------------------------------------
# stage_timings — plan A1's per-stage latency instrumentation, added so
# every later latency optimization (conditional rewrite, batched embeds,
# parallel search arms) is provable against a real number instead of
# assumed. A3's own report showed the full pipeline at 7173ms p50 vs
# BM25-only's 144ms with no way to see which stage actually cost that.
# ---------------------------------------------------------------------


def test_stage_timings_records_every_real_stage():
    plan = QueryPlan(original="q", normalized="q", intent="factual")
    fake_reranked = RerankResult(items=[RankedCandidate(id="c1", text="t", score=0.9)], degraded=False, reason=None, model_served="m")
    fake_answer = Answer(text="the answer", citations=[], abstained=False)
    grounded_pass = GuardrailResult("groundedness", True)
    citation_pass = GuardrailResult("citation_integrity", True)

    with patch("src.reason.graph.plan_query", return_value=plan), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", return_value=(fake_reranked, {})
    ), patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(fake_answer, citation_pass, grounded_pass)
    ):
        trace = run_traced_query("what is RLHF", top_n_context=8)

    for stage in ("plan_query", "retrieve", "rerank", "assess_context", "assess_ambiguity", "generate"):
        assert stage in trace.stage_timings
        assert trace.stage_timings[stage] >= 0.0


def test_stage_timings_accumulates_across_retries_not_overwrites():
    # A stage that runs twice (rerank, both assess calls, generate all do
    # on a retry) must report its REAL total cost for the run, not just
    # the last attempt's — that's what _timed sums rather than assigns.
    abstained = Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)
    succeeded = Answer(text="the real answer", citations=[], abstained=False)
    grounded_pass = GuardrailResult("groundedness", True)
    citation_pass = GuardrailResult("citation_integrity", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ) as mock_rerank, patch(
        "src.reason.graph.assess_context", return_value=_SUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", side_effect=[(abstained, citation_pass, grounded_pass), (succeeded, citation_pass, grounded_pass)]
    ):
        trace = run_traced_query("what is RLHF", top_n_context=8)

    assert mock_rerank.call_count == 2
    # generate ran twice; stage_timings still has exactly one "generate"
    # key holding the summed cost, not two separate entries.
    assert "generate" in trace.stage_timings
    assert isinstance(trace.stage_timings["generate"], float)


def test_assess_insufficient_still_forces_a_real_attempt_at_max_attempts(monkeypatch):
    # Even if every assessment says insufficient, the loop must not give
    # up without ever calling generate once — it forces a real attempt
    # on the final allowed pass.
    monkeypatch.setenv("GUARDRAIL_CONTEXT_SUFFICIENCY_MODE", "enforce")
    monkeypatch.setenv("RAG_MAX_ATTEMPTS", "2")
    abstained = Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)
    citation_pass = GuardrailResult("citation_integrity", True)
    grounded_pass = GuardrailResult("groundedness", True)

    with patch("src.reason.graph.plan_query", return_value=_PLAN), patch(
        "src.reason.graph.run_retrieve", return_value=(_CANDIDATES, _FUSION)
    ), patch(
        "src.reason.graph.run_rerank_and_metadata", side_effect=_fake_reranked_and_metadata
    ), patch(
        "src.reason.graph.assess_context", return_value=_INSUFFICIENT
    ), patch(
        "src.reason.graph.run_answer", return_value=(abstained, citation_pass, grounded_pass)
    ) as mock_answer:
        trace = run_traced_query("what is RLHF", top_n_context=8)

    assert mock_answer.call_count == 1  # forced on the final attempt, not skipped forever
    assert len(trace.attempts) == 1
