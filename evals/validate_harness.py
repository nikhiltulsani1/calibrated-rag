from __future__ import annotations

from opensearchpy import OpenSearch, helpers

from evals.metrics import mean, ndcg_at_k
from evals.scifact_loader import load_corpus, load_qrels, load_queries
from src.index.client import get_client

# A6: validate metrics.py against a published baseline BEFORE trusting any
# number from our own domain-specific (unlabeled-by-me) qrels. SciFact
# (BEIR benchmark) — small (5,183 docs, 300 test queries), binary
# relevance, publicly downloadable, ships WITH real relevance judgments —
# unlike our own 80 domain questions, nothing here required me to
# fabricate expert judgment; the ground truth already exists.
BENCHMARK_INDEX = "eval_benchmark_scifact"

# BEIR paper's widely-cited BM25 baseline for SciFact. "Widely-cited" is
# not the same as "verified against the exact same eval code" — the point
# of this harness is to land NEAR it, not match it to the decimal (see
# the plan: "if our BM25-only nDCG@10 ... does not land near the widely
# reported figure, the bug is in metrics.py, not in the retriever").
PUBLISHED_BM25_NDCG10 = 0.665


def _benchmark_index_body() -> dict:
    # k1=0.9, b=0.4 — not OpenSearch's default (k1=1.2, b=0.75). Verified
    # via search that these are specifically the Anserini/Lucene-tuned
    # parameters the official BEIR BM25 baseline was computed with. Using
    # OpenSearch's stock defaults here would compare our numbers against
    # a baseline computed under different BM25 tuning — not an
    # apples-to-apples check of metrics.py at all.
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"beir_bm25": {"type": "BM25", "k1": 0.9, "b": 0.4}},
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "english", "similarity": "beir_bm25"},
                "text": {"type": "text", "analyzer": "english", "similarity": "beir_bm25"},
                "contents": {"type": "text", "analyzer": "english", "similarity": "beir_bm25"},
            }
        },
    }


def build_benchmark_index(client: OpenSearch) -> int:
    """Its own index, per A6/A3b — never mixed with rag_chunks or any
    other tier. Returns the number of documents indexed."""
    if client.indices.exists(index=BENCHMARK_INDEX):
        client.indices.delete(index=BENCHMARK_INDEX)
    client.indices.create(index=BENCHMARK_INDEX, body=_benchmark_index_body())

    docs = load_corpus()
    actions = (
        {
            "_index": BENCHMARK_INDEX,
            "_id": doc.doc_id,
            "_source": {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                # Anserini/Pyserini's standard BEIR document representation
                # concatenates title+text into one "contents" field rather
                # than searching two separately-weighted fields — BM25's
                # per-field IDF statistics differ materially between the
                # two setups. Indexed alongside title/text (not replacing
                # them) so both retrieval strategies are queryable and
                # comparable from the same index.
                "contents": f"{doc.title}. {doc.text}" if doc.title else doc.text,
            },
        }
        for doc in docs
    )
    success_count, errors = helpers.bulk(client, actions, raise_on_error=False)
    if errors:
        raise RuntimeError(f"{len(errors)} documents failed to index: {errors[:3]}")
    client.indices.refresh(index=BENCHMARK_INDEX)
    return success_count


def _bm25_search(client: OpenSearch, query_text: str, size: int = 10, field: str = "contents") -> list[str]:
    body = {
        "size": size,
        "query": {"match": {field: query_text}},
        "_source": False,
    }
    response = client.search(index=BENCHMARK_INDEX, body=body)
    return [hit["_id"] for hit in response["hits"]["hits"]]


def run_validation(client: OpenSearch | None = None) -> dict:
    """BM25-only, deliberately — this validates metrics.py's arithmetic,
    which has nothing to do with the embedding model, so it needs no
    JINA_API_KEY to run.
    """
    client = client or get_client()
    queries = load_queries()
    qrels = load_qrels("test")
    query_text_by_id = {q.query_id: q.text for q in queries}

    ndcg_scores = []
    for query_id, relevance in qrels.items():
        query_text = query_text_by_id.get(query_id)
        if query_text is None:
            continue
        ranked = _bm25_search(client, query_text, size=10)
        ndcg_scores.append(ndcg_at_k(ranked, relevance, k=10))

    measured = mean(ndcg_scores)
    return {
        "measured_ndcg10": measured,
        "published_ndcg10": PUBLISHED_BM25_NDCG10,
        "delta": measured - PUBLISHED_BM25_NDCG10,
        "n_queries_evaluated": len(ndcg_scores),
        "n_queries_in_qrels": len(qrels),
    }


if __name__ == "__main__":
    _client = get_client()
    print("indexing SciFact benchmark corpus...")
    n_indexed = build_benchmark_index(_client)
    print(f"indexed {n_indexed} documents")

    print("running BM25 validation over test queries...")
    result = run_validation(_client)
    print(result)

    tolerance = 0.05
    if abs(result["delta"]) > tolerance:
        print(
            f"WARNING: measured nDCG@10 ({result['measured_ndcg10']:.4f}) differs from "
            f"the published baseline ({PUBLISHED_BM25_NDCG10}) by more than {tolerance} "
            "— check metrics.py before trusting any other number in this project."
        )
    else:
        print(
            f"harness validated: measured nDCG@10 ({result['measured_ndcg10']:.4f}) is "
            f"within {tolerance} of the published baseline ({PUBLISHED_BM25_NDCG10})."
        )
