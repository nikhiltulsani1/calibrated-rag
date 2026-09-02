from unittest.mock import MagicMock, patch

import pytest

from evals.run_retrieval_eval import _aggregate, _hybrid_fused_order, _score_one, run_dense_only, load_domain_qrels
from src.index.embedder import EmbeddingResult

pytestmark = pytest.mark.unit


def test_load_domain_qrels_has_80_real_entries():
    queries, qrels = load_domain_qrels()
    assert len(queries) == 80
    assert len(qrels) == 80
    for query_id, query_text in queries:
        assert query_id in qrels
        assert query_text  # non-empty
        relevant = qrels[query_id]
        assert relevant  # every question has at least one relevant chunk
        assert all(isinstance(grade, int) and grade > 0 for grade in relevant.values())


def test_score_one_perfect_rank_one():
    scores = _score_one(["c1", "c2", "c3"], {"c1": 2})
    assert scores["recall@5"] == 1.0
    assert scores["mrr@10"] == 1.0
    assert scores["ndcg@10"] == 1.0


def test_score_one_relevant_doc_missing_entirely():
    scores = _score_one(["c1", "c2", "c3"], {"cX": 2})
    assert scores["recall@5"] == 0.0
    assert scores["mrr@10"] == 0.0
    assert scores["ndcg@10"] == 0.0


def test_aggregate_computes_mean_and_latency_percentiles():
    per_query = [{"recall@10": 1.0}, {"recall@10": 0.0}, {"recall@10": 1.0}, {"recall@10": 1.0}]
    latencies = [10.0, 20.0, 30.0, 100.0]
    out = _aggregate(per_query, latencies)
    assert out["recall@10"] == pytest.approx(0.75)
    assert out["p50_ms"] == 30.0  # sorted [10,20,30,100], index len//2=2 -> 30
    assert "p95_ms" in out


def test_aggregate_empty_input_returns_empty_dict():
    assert _aggregate([], []) == {}


# ---------------------------------------------------------------------
# Real regression tests for a real bug this eval's own live run caught
# (2026-08-23): plan A3's latency fix changed _dense_search to take a
# pre-computed vector instead of raw query text (src/retrieve/hybrid.py)
# — retrieve_with_trace was updated to match, but these two call sites
# were not, so they silently passed a raw string as the "vector"
# (Python's duck typing accepts it, a str is iterable), producing a
# real OpenSearch 400 on every single query and scoring three of five
# A3 arms as a flat 0.0000 across every metric. No test existed for
# either function before this — exactly how it went undetected.
# ---------------------------------------------------------------------


def test_run_dense_only_embeds_the_query_before_searching():
    fake_client = MagicMock()
    fake_embedding = EmbeddingResult(model="m", dimension=3, vectors=[[0.1, 0.2, 0.3]], prompt_tokens=5)
    with patch(
        "evals.run_retrieval_eval.embed_queries_with_fallback", return_value=(fake_embedding, "rag_chunks")
    ) as mock_embed, patch("evals.run_retrieval_eval._dense_search", return_value=["c1"]) as mock_dense:
        run_dense_only([("q1", "what is RLHF")], {"q1": {"c1": 1}}, fake_client)
    mock_embed.assert_called_once_with(["what is RLHF"])
    # The real bug: this used to receive the raw query STRING here.
    # Assert it now receives the real vector list instead.
    assert mock_dense.call_args.args[1] == [0.1, 0.2, 0.3]


def test_hybrid_fused_order_embeds_the_query_before_dense_search():
    fake_client = MagicMock()
    fake_embedding = EmbeddingResult(model="m", dimension=3, vectors=[[0.4, 0.5, 0.6]], prompt_tokens=5)
    with patch(
        "evals.run_retrieval_eval.embed_queries_with_fallback", return_value=(fake_embedding, "rag_chunks")
    ), patch("evals.run_retrieval_eval._lexical_search", return_value=["c1"]), patch(
        "evals.run_retrieval_eval._dense_search", return_value=["c2"]
    ) as mock_dense:
        _hybrid_fused_order(fake_client, "a query")
    assert mock_dense.call_args.args[1] == [0.4, 0.5, 0.6]


def test_dense_only_searches_whatever_index_the_fallback_actually_served():
    # Real bug #2 this eval run's own live results caught (2026-08-23):
    # every function here used to implicitly default to the production
    # rag_chunks (Jina-embedded) index regardless of which provider
    # actually embedded the query — comparing e.g. a Mistral query
    # vector against Jina's stored vectors, a real, silent embedding-
    # space mismatch that corrupted every dense-search-involving arm's
    # recall whenever EMBED_PROVIDER wasn't "jina".
    #
    # Real bug #3 (2026-08-23, same day): run_dense_only's dense step now
    # gets BOTH the vector and the matching index from
    # embed_queries_with_fallback() — not from the `index_name` this
    # function was handed — since a fallback engaging for one specific
    # call can differ from whatever was originally active. The
    # `index_name` parameter still exists (used by run_bm25_only/
    # lexical arms elsewhere), it's just not what routes the dense step.
    fake_client = MagicMock()
    fake_embedding = EmbeddingResult(model="m", dimension=3, vectors=[[0.1, 0.2, 0.3]], prompt_tokens=5)
    with patch(
        "evals.run_retrieval_eval.embed_queries_with_fallback",
        return_value=(fake_embedding, "rag_chunks_mistral_embed"),
    ), patch("evals.run_retrieval_eval._dense_search", return_value=["c1"]) as mock_dense:
        run_dense_only([("q1", "a query")], {"q1": {"c1": 1}}, fake_client)
    assert mock_dense.call_args.args[4] == "rag_chunks_mistral_embed"


def test_hybrid_rerank_mget_uses_the_real_active_index_too():
    from evals.run_retrieval_eval import run_hybrid_rerank

    fake_client = MagicMock()
    fake_client.mget.return_value = {"docs": []}
    fake_embedding = EmbeddingResult(model="m", dimension=3, vectors=[[0.1, 0.2, 0.3]], prompt_tokens=5)
    with patch(
        "evals.run_retrieval_eval.embed_queries_with_fallback", return_value=(fake_embedding, "rag_chunks")
    ), patch("evals.run_retrieval_eval._lexical_search", return_value=[]), patch(
        "evals.run_retrieval_eval._dense_search", return_value=[]
    ):
        run_hybrid_rerank([("q1", "a query")], {"q1": {"c1": 1}}, fake_client, index_name="rag_chunks_mistral_embed")
    assert fake_client.mget.call_args.kwargs["index"] == "rag_chunks_mistral_embed"
