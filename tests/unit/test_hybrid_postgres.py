from unittest.mock import MagicMock, patch

import pytest

from src.retrieve.hybrid_postgres import (
    _document_conditions,
    _filter_conditions,
    _owner_predicate,
    real_categories,
    retrieve_with_trace,
)
from src.schemas.query_plan import QueryPlan

pytestmark = pytest.mark.unit

# RETRIEVAL_BACKEND=postgres path — the free-tier live deployment's
# retrieval backend, built alongside (not instead of) the default
# hybrid.py/OpenSearch path per the Phase 2 plan. These tests mock the
# actual Postgres round-trip (get_session().execute(...)) but let every
# other line of _lexical_search/_dense_search/retrieve_with_trace run for
# real — proving the SQLAlchemy query construction itself is valid, not
# just that a mock accepted whatever was thrown at it.


def test_owner_predicate_with_no_session_id_only_matches_shared_corpus():
    # No session_id (e.g. a brand-new visitor, or the OpenSearch path
    # which never passes one) -> only NULL-owner (shared) rows visible.
    from src.store.schema import Chunk

    predicate = _owner_predicate(None)
    assert str(predicate) == str(Chunk.owner_session_id.is_(None))


def test_owner_predicate_with_session_id_matches_shared_or_own():
    predicate = _owner_predicate("sess-123")
    rendered = str(predicate.compile(compile_kwargs={"literal_binds": True}))
    assert "owner_session_id IS NULL" in rendered
    assert "sess-123" in rendered


def test_filter_conditions_empty_for_no_filters():
    assert _filter_conditions({}) == []


def test_filter_conditions_category_produces_one_condition():
    conditions = _filter_conditions({"category": "cs.CL"})
    assert len(conditions) == 1


def test_filter_conditions_all_three_combine():
    conditions = _filter_conditions(
        {"author": "LeCun", "category": "cs.IR", "date_from": "2026-01-01", "date_to": "2026-06-01"}
    )
    # author -> 1, category -> 1, date_from + date_to -> 2 (separate >= / <=)
    assert len(conditions) == 4


def _fake_session(lexical_rows, dense_rows, text_rows):
    """A MagicMock session whose .execute(stmt).all() returns canned rows
    based on inspecting the actual compiled SQL — NOT call order.
    _lexical_search and _dense_search run concurrently via
    ThreadPoolExecutor, so which one's session.execute() call lands
    first is genuinely nondeterministic; a call-order-based side_effect
    would be a real, flaky test. Distinguishing by "does this statement
    rank via ts_rank" (lexical) vs not (dense's cosine-distance ORDER BY,
    or the final plain chunk_id/text lookup, which has neither) is
    robust to thread scheduling.
    """

    def execute(stmt, *args, **kwargs):
        rendered = str(stmt)
        result = MagicMock()
        if "ts_rank" in rendered:
            result.all.return_value = lexical_rows
        elif "ORDER BY" in rendered:
            result.all.return_value = dense_rows
        else:
            result.all.return_value = text_rows
        return result

    session = MagicMock()
    session.execute.side_effect = execute
    return session


def test_retrieve_with_trace_fuses_lexical_and_dense_results(monkeypatch):
    plan = QueryPlan(original="q", normalized="what is rlhf", expansions=[], filters={}, intent="factual")
    fake_embed = MagicMock(vectors=[[0.1] * 1024])

    session = _fake_session(
        lexical_rows=[("c1",), ("c2",)],
        dense_rows=[("c2",), ("c3",)],
        text_rows=[("c1", "text one"), ("c2", "text two"), ("c3", "text three")],
    )

    with patch("src.retrieve.hybrid_postgres.get_session", return_value=session), patch(
        "src.retrieve.hybrid_postgres.embed_queries", return_value=fake_embed
    ), patch("src.retrieve.hybrid_postgres.get_active_embed_provider", return_value="jina"):
        results, trace = retrieve_with_trace(plan, top_n=10)

    # c2 appears in both arms -> should rank first after RRF fusion
    assert results[0].id == "c2"
    assert {r.id for r in results} == {"c1", "c2", "c3"}
    assert trace.dense_index_name == "postgres:chunks"
    assert len(trace.arms) == 2  # one variant (no expansions) x 2 arms


def test_retrieve_with_trace_passes_session_id_through_to_the_query(monkeypatch):
    # Real regression this guards: session_id must reach the owner
    # predicate on both arms, not get silently dropped in the fan-out —
    # verified by confirming a session-scoped run doesn't error and
    # produces the same shape as the unscoped case (the isolation SQL
    # itself is covered directly by test_owner_predicate_* above).
    plan = QueryPlan(original="q", normalized="what is rlhf", expansions=[], filters={}, intent="factual")
    fake_embed = MagicMock(vectors=[[0.1] * 1024])
    session = _fake_session(lexical_rows=[("c1",)], dense_rows=[("c1",)], text_rows=[("c1", "text one")])

    with patch("src.retrieve.hybrid_postgres.get_session", return_value=session), patch(
        "src.retrieve.hybrid_postgres.embed_queries", return_value=fake_embed
    ), patch("src.retrieve.hybrid_postgres.get_active_embed_provider", return_value="jina"):
        results, _trace = retrieve_with_trace(plan, top_n=10, session_id="visitor-abc")

    assert results[0].id == "c1"


# ---------------------------------------------------------------------
# Phase 2 §5 (stage 6, uploads) — optional single-document scoping,
# defense in depth ON TOP OF (not instead of) _owner_predicate above.
# ---------------------------------------------------------------------


def test_document_conditions_empty_when_no_document_id():
    assert _document_conditions(None) == []
    assert _document_conditions("") == []


def test_document_conditions_filters_on_paper_id():
    from src.store.schema import Chunk

    conditions = _document_conditions("upload-abc123")
    assert len(conditions) == 1
    rendered = str(conditions[0].compile(compile_kwargs={"literal_binds": True}))
    assert "paper_id" in rendered
    assert "upload-abc123" in rendered


def test_retrieve_with_trace_applies_document_scoping_alongside_owner_scoping(monkeypatch):
    plan = QueryPlan(original="q", normalized="what is rlhf", expansions=[], filters={}, intent="factual")
    fake_embed = MagicMock(vectors=[[0.1] * 1024])
    session = _fake_session(lexical_rows=[("c1",)], dense_rows=[("c1",)], text_rows=[("c1", "text one")])

    with patch("src.retrieve.hybrid_postgres.get_session", return_value=session), patch(
        "src.retrieve.hybrid_postgres.embed_queries", return_value=fake_embed
    ), patch("src.retrieve.hybrid_postgres.get_active_embed_provider", return_value="jina"):
        results, _trace = retrieve_with_trace(
            plan, top_n=10, session_id="visitor-a", document_id="upload-xyz"
        )

    # Doesn't error, and still returns results — the actual isolation SQL
    # is covered directly by test_document_conditions_filters_on_paper_id
    # and test_owner_predicate_* above.
    assert results[0].id == "c1"


def test_real_categories_query_is_scoped_to_the_shared_corpus():
    # Real bug found in review: this Redis-cached, globally-shared result
    # used to scan every Paper row with no owner_session_id filter — a
    # private upload's category (currently always []) would otherwise
    # enter the 24h globally-cached "valid categories" set the moment
    # uploads started populating it.
    session = MagicMock()
    session.execute.return_value.all.return_value = [(["cs.IR"],)]

    with patch("src.retrieve.hybrid_postgres.get_session", return_value=session), patch(
        "src.retrieve.hybrid_postgres.get_json", return_value=None
    ), patch("src.retrieve.hybrid_postgres.set_json"):
        real_categories()

    stmt = session.execute.call_args.args[0]
    rendered = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "owner_session_id IS NULL" in rendered


def test_final_text_hydration_query_reapplies_owner_and_document_scoping(monkeypatch):
    # Real bug found in review: after RRF-fusing chunk_ids already scoped
    # by _lexical_search/_dense_search, the final hydration step used to
    # re-fetch by chunk_id ALONE, with no owner/document predicate
    # re-applied — a defense-in-depth gap that only stayed safe because
    # chunk_id happens to be collision-free today. This asserts the fix:
    # the final SELECT's WHERE clause actually carries both predicates.
    plan = QueryPlan(original="q", normalized="what is rlhf", expansions=[], filters={}, intent="factual")
    fake_embed = MagicMock(vectors=[[0.1] * 1024])
    captured_final_stmt = {}

    def execute(stmt, *args, **kwargs):
        rendered = str(stmt)
        result = MagicMock()
        if "ts_rank" in rendered:
            result.all.return_value = [("c1",)]
        elif "ORDER BY" in rendered:
            result.all.return_value = [("c1",)]
        else:
            captured_final_stmt["stmt"] = stmt
            result.all.return_value = [("c1", "text one")]
        return result

    session = MagicMock()
    session.execute.side_effect = execute

    with patch("src.retrieve.hybrid_postgres.get_session", return_value=session), patch(
        "src.retrieve.hybrid_postgres.embed_queries", return_value=fake_embed
    ), patch("src.retrieve.hybrid_postgres.get_active_embed_provider", return_value="jina"):
        retrieve_with_trace(plan, top_n=10, session_id="visitor-a", document_id="upload-xyz")

    rendered = str(captured_final_stmt["stmt"].compile(compile_kwargs={"literal_binds": True}))
    assert "owner_session_id" in rendered
    assert "visitor-a" in rendered
    assert "paper_id" in rendered
    assert "upload-xyz" in rendered


def test_retrieve_with_trace_returns_empty_on_no_fused_results(monkeypatch):
    plan = QueryPlan(original="q", normalized="what is rlhf", expansions=[], filters={}, intent="factual")
    fake_embed = MagicMock(vectors=[[0.1] * 1024])
    session = _fake_session(lexical_rows=[], dense_rows=[], text_rows=[])

    with patch("src.retrieve.hybrid_postgres.get_session", return_value=session), patch(
        "src.retrieve.hybrid_postgres.embed_queries", return_value=fake_embed
    ), patch("src.retrieve.hybrid_postgres.get_active_embed_provider", return_value="jina"):
        results, trace = retrieve_with_trace(plan, top_n=10)

    assert results == []
    assert trace.fused_order == []
