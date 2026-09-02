import random
from unittest.mock import patch

import pytest

from src.index.client import get_client
from src.index.embedder import EmbeddingResult
from src.index.mapping import INDEX_NAME
from src.retrieve.hybrid import _dense_search, _lexical_search, build_filter_clauses, retrieve, retrieve_with_trace
from src.schemas.query_plan import QueryPlan

pytestmark = pytest.mark.integration

_TEST_IDS = ["test_a", "test_b", "test_c", "test_d"]


@pytest.fixture
def synthetic_docs():
    """Four documents with deliberately overlapping/distinct metadata —
    this is what makes the filter assertions below meaningful rather than
    trivially true. Real vectors (not mocked) so the kNN arm is genuinely
    exercised; dense semantic *quality* isn't the point here, filter
    correctness is."""
    client = get_client()
    random.seed(42)
    docs = {
        "test_a": {
            "chunk_id": "test_a", "text": "transformers for retrieval augmented generation",
            "title": "Paper A", "authors": ["Yann LeCun", "Jane Doe"], "category": ["cs.CL"],
            "published_date": "2026-01-15", "embedding": [random.random() for _ in range(1024)],
        },
        "test_b": {
            "chunk_id": "test_b", "text": "transformers for image classification",
            "title": "Paper B", "authors": ["Alice Smith"], "category": ["cs.CV"],
            "published_date": "2026-03-10", "embedding": [random.random() for _ in range(1024)],
        },
        "test_c": {
            "chunk_id": "test_c", "text": "retrieval augmented generation survey",
            "title": "Paper C", "authors": ["Bob Jones", "Yann LeCun"], "category": ["cs.IR"],
            "published_date": "2026-07-01", "embedding": [random.random() for _ in range(1024)],
        },
        "test_d": {
            "chunk_id": "test_d", "text": "unrelated topic about databases",
            "title": "Paper D", "authors": ["Carol White"], "category": ["cs.DB"],
            "published_date": "2025-05-01", "embedding": [random.random() for _ in range(1024)],
        },
    }
    for doc_id, body in docs.items():
        client.index(index=INDEX_NAME, id=doc_id, body=body, refresh=True)

    yield docs

    for doc_id in _TEST_IDS:
        client.delete(index=INDEX_NAME, id=doc_id, ignore=[404])


# `rag_chunks` now legitimately holds the real ingested corpus (310 real
# chunks) alongside these synthetic fixtures — it is no longer safe to
# assume the index is empty, so assertions below check that the expected
# synthetic docs are present/absent rather than that they are the *only*
# results. `size` is generous so a synthetic doc ranked below real-corpus
# matches on a common word like "transformers" still shows up.
_GENEROUS_SIZE = 500


def test_lexical_search_no_filter_matches_on_text(synthetic_docs):
    client = get_client()
    result = _lexical_search(client, "transformers", [], size=_GENEROUS_SIZE)
    assert "test_a" in result
    assert "test_b" in result
    assert "test_c" not in result  # doesn't mention "transformers"
    assert "test_d" not in result


def test_lexical_search_category_filter_isolates_match(synthetic_docs):
    client = get_client()
    clauses = build_filter_clauses({"category": "cs.CV"})
    result = _lexical_search(client, "transformers", clauses, size=_GENEROUS_SIZE)
    assert "test_b" in result
    assert "test_a" not in result  # right text, wrong category


def test_lexical_search_author_filter_matches_partial_name(synthetic_docs):
    client = get_client()
    clauses = build_filter_clauses({"author": "LeCun"})
    result = _lexical_search(client, "retrieval", clauses, size=_GENEROUS_SIZE)
    assert "test_a" in result and "test_c" in result  # both carry "Yann LeCun"
    assert "test_b" not in result and "test_d" not in result


def test_lexical_search_date_range_filter_excludes_out_of_range(synthetic_docs):
    client = get_client()
    clauses = build_filter_clauses({"date_from": "2026-06-01"})
    result = _lexical_search(client, "retrieval", clauses, size=_GENEROUS_SIZE)
    assert "test_c" in result
    assert "test_a" not in result  # published before the cutoff


def test_lexical_search_combined_filters_apply_as_and(synthetic_docs):
    client = get_client()
    clauses = build_filter_clauses({"author": "LeCun", "category": "cs.IR"})
    result = _lexical_search(client, "retrieval", clauses, size=_GENEROUS_SIZE)
    assert "test_c" in result
    assert "test_a" not in result  # right author, wrong category


def test_dense_search_self_match_is_nearest_neighbor(synthetic_docs):
    # _dense_search now takes an already-computed vector directly (plan
    # A3 — the caller batch-embeds every query variant in one call
    # instead of one embed call per variant), so no embed mock is
    # needed here at all — just pass the real vector straight through.
    client = get_client()
    own_vector = synthetic_docs["test_a"]["embedding"]
    result = _dense_search(client, own_vector, [], size=4)
    assert result[0] == "test_a"


def test_dense_search_respects_filter(synthetic_docs):
    client = get_client()
    own_vector = synthetic_docs["test_a"]["embedding"]
    clauses = build_filter_clauses({"category": "cs.CV"})  # excludes test_a itself
    result = _dense_search(client, own_vector, clauses, size=4)
    assert result == ["test_b"]


def test_retrieve_end_to_end_fans_out_and_filters(synthetic_docs):
    client = get_client()
    query_vector = synthetic_docs["test_c"]["embedding"]
    # Two variants (normalized + 1 expansion) -> embed_queries returns
    # one vector per variant, in order (plan A3: one batched call, not
    # one embed_query() call per variant).
    fake = EmbeddingResult(model="fake", dimension=1024, vectors=[query_vector, query_vector], prompt_tokens=0)

    plan = QueryPlan(
        original="what does LeCun say about retrieval augmented generation",
        normalized="retrieval augmented generation",
        expansions=["RAG survey papers"],
        filters={"author": "LeCun"},
        intent="factual",
    )
    with patch("src.retrieve.hybrid.embed_queries_with_fallback", return_value=(fake, INDEX_NAME)):
        results = retrieve(plan, top_n=10)

    ids = [c.id for c in results]
    assert ids  # non-empty
    assert all(doc_id in ("test_a", "test_c") for doc_id in ids), "filter must exclude test_b/test_d entirely"


def test_retrieve_with_trace_exposes_real_per_arm_and_fusion_data(synthetic_docs):
    # This is what A5's stages 4-5 render — checks the trace against real
    # OpenSearch results, not a mocked stand-in.
    query_vector = synthetic_docs["test_c"]["embedding"]
    fake = EmbeddingResult(model="fake", dimension=1024, vectors=[query_vector], prompt_tokens=0)

    plan = QueryPlan(original="q", normalized="retrieval augmented generation", expansions=[], filters={}, intent="factual")
    with patch("src.retrieve.hybrid.embed_queries_with_fallback", return_value=(fake, INDEX_NAME)):
        results, trace = retrieve_with_trace(plan, top_n=10)

    # One variant (no expansions) x two arms = exactly 2 ArmResults
    assert len(trace.arms) == 2
    assert {arm.arm for arm in trace.arms} == {"lexical", "dense"}
    assert all(arm.variant == "retrieval augmented generation" for arm in trace.arms)

    # Every doc that made it into the final results must have a real,
    # positive RRF score recorded, not a placeholder.
    for candidate in results:
        assert candidate.id in trace.fused_scores
        assert trace.fused_scores[candidate.id] > 0

    # fused_order must actually be sorted by score, descending — the
    # real property the visualiser's ranking depends on.
    scores_in_order = [trace.fused_scores[doc_id] for doc_id in trace.fused_order]
    assert scores_in_order == sorted(scores_in_order, reverse=True)


def test_retrieve_drops_a_hallucinated_category_instead_of_zeroing_every_result(synthetic_docs):
    # Real bug found live 2026-08-24 (A8's first genuinely complete run,
    # q025/q073): the rewrite LLM extracted a plausible category that was
    # never actually assigned to the real paper — the hard exact `term`
    # filter then zeroed out every retrieval arm. "cs.ZZ" here is not a
    # real category on any synthetic_docs fixture doc (cs.CL/cs.CV/
    # cs.IR/cs.DB), exactly mirroring that failure mode. Real results
    # must still come back, not an empty list.
    query_vector = synthetic_docs["test_c"]["embedding"]
    fake = EmbeddingResult(model="fake", dimension=1024, vectors=[query_vector, query_vector], prompt_tokens=0)

    plan = QueryPlan(
        original="what does the retrieval augmented generation survey say",
        normalized="retrieval augmented generation survey",
        expansions=["a survey of retrieval augmented generation"],
        filters={"category": "cs.ZZ"},
        intent="factual",
    )
    with patch("src.retrieve.hybrid.embed_queries_with_fallback", return_value=(fake, INDEX_NAME)):
        results = retrieve(plan, top_n=10)

    assert results, "a hallucinated category must not silently zero out every real result"
    assert any(c.id == "test_c" for c in results)
