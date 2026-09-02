from __future__ import annotations

import json
import time
from pathlib import Path

from evals.metrics import mean, mrr_at_k, ndcg_at_k, percentile, recall_at_k
from src.index.client import get_client
from src.index.embed_toggle import active_embed_index_name
from src.index.embedder import embed_queries_with_fallback
from src.index.mapping import INDEX_NAME
from src.retrieve.hybrid import _dense_search, _lexical_search, retrieve_with_trace, rrf_fuse_with_scores
from src.retrieve.query_planner import plan_query
from src.retrieve.reranker import Candidate, rerank
from src.schemas.query_plan import QueryPlan

# A3: the five-way ablation the plan calls for — BM25-only, dense-only,
# hybrid RRF, hybrid+rerank, and the full pipeline (rewrite+hybrid+rerank).
# Each config's real numbers, or an honest "not run" if the API key it
# needs isn't configured on this machine — never fabricated, never
# silently skipped. See evals/datasets/qrels.jsonl and its README note
# on verification status before trusting these numbers as ground truth.

_QRELS_PATH = Path(__file__).resolve().parent / "datasets" / "qrels.jsonl"
_REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
_RECALL_SIZE = 20


def load_domain_qrels() -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    """Mirrors evals/scifact_loader.py's shape: a list of (query_id,
    query_text) and a query_id -> {chunk_id: relevance} dict. Loads the
    LLM-bootstrapped, not-yet-human-verified domain qrels — see the
    dataset file's own `verified` field and README's caveat.
    """
    rows = [json.loads(line) for line in open(_QRELS_PATH, encoding="utf-8")]
    queries = [(r["query_id"], r["query"]) for r in rows]
    qrels = {r["query_id"]: r["relevant"] for r in rows}
    return queries, qrels


def _score_one(ranked_ids: list[str], relevance: dict[str, int]) -> dict[str, float]:
    relevant_ids = set(relevance)
    return {
        "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
        "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
        "recall@20": recall_at_k(ranked_ids, relevant_ids, 20),
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
        "mrr@10": mrr_at_k(ranked_ids, relevant_ids, 10),
    }


def _aggregate(per_query_scores: list[dict[str, float]], latencies_ms: list[float]) -> dict[str, float]:
    if not per_query_scores:
        return {}
    out = {metric: mean([row[metric] for row in per_query_scores]) for metric in per_query_scores[0]}
    if latencies_ms:
        out["p50_ms"] = percentile(latencies_ms, 0.50)
        out["p95_ms"] = percentile(latencies_ms, 0.95)
    return out


def _run_per_query(queries, qrels, retrieve_fn) -> dict:
    """Shared driver for every config below: times and scores one query
    at a time, and — critically — a single query's failure does NOT
    abort the whole 80-question run. Found necessary live (2026-08-16):
    one real query triggered a persistent query-planner failure (see
    query_planner.py's retry-loop comment) that survived 3 retries, and
    without this, that one query silently zeroed out every number for
    the "full pipeline" config instead of just its own row. Failed
    queries are scored as a full miss (empty ranked list, i.e. 0 on
    every metric) and counted/reported, never dropped silently.
    """
    per_query, latencies, failures = [], [], []
    for query_id, query_text in queries:
        start = time.perf_counter()
        try:
            ranked = retrieve_fn(query_text)
        except Exception as exc:
            ranked = []
            failures.append((query_id, f"{type(exc).__name__}: {exc}"))
        latencies.append((time.perf_counter() - start) * 1000)
        per_query.append(_score_one(ranked, qrels[query_id]))
    result = _aggregate(per_query, latencies)
    if failures:
        result["_failures"] = failures
    return result


def run_bm25_only(queries, qrels, client, index_name: str = INDEX_NAME) -> dict:
    return _run_per_query(queries, qrels, lambda q: _lexical_search(client, q, [], _RECALL_SIZE, index_name))


def run_dense_only(queries, qrels, client, index_name: str = INDEX_NAME) -> dict:
    # _dense_search takes a pre-computed vector, not query text, since
    # plan A3's real latency fix (2026-08-23, src/retrieve/hybrid.py) —
    # embedding here explicitly to match. A real bug this exact eval run
    # caught: this call site (and _hybrid_fused_order below) still
    # passed the raw query string, which Python's duck typing silently
    # accepted (a str is iterable), producing a real OpenSearch 400
    # ("[knn] failed to parse field [vector]") on every single query —
    # scoring three of five arms as a flat 0.0000 across every metric,
    # not a real quality result.
    #
    # `index_name` matters just as much: a query vector from a NON-jina
    # embed provider (e.g. EMBED_PROVIDER=mistral) is meaningless
    # compared against the default rag_chunks index's Jina-embedded
    # vectors — different model, different vector space entirely. A
    # second real bug this same investigation caught: every function
    # here used to implicitly default to INDEX_NAME regardless of which
    # provider actually embedded the query, silently corrupting every
    # dense-search-involving arm's real recall whenever EMBED_PROVIDER
    # wasn't "jina".
    #
    # A third real bug, caught by the 2026-08-23 clean top-to-bottom
    # re-run: this still called embed_query() directly, bypassing
    # embed_queries_with_fallback() (src/index/embedder.py) — the
    # automatic Jina<->Mistral embed failover added the same day but
    # never wired into this eval script. With Jina hard-403ing during
    # that run, every one of 80 queries failed outright here (no
    # fallback), while the "full pipeline" config — which goes through
    # the real production retrieve_with_trace() — succeeded, because
    # THAT path already had the failover. Fixed by using the fallback
    # call and its returned index (which may differ per-query from the
    # `index_name` this function was handed, if a fallback engaged for
    # that specific call) for the dense step, exactly like hybrid.py.
    def retrieve(q: str) -> list[str]:
        embed_result, dense_index = embed_queries_with_fallback([q])
        return _dense_search(client, embed_result.vectors[0], [], _RECALL_SIZE, dense_index)

    return _run_per_query(queries, qrels, retrieve)


def _hybrid_fused_order(client, query_text: str, index_name: str = INDEX_NAME) -> list[str]:
    lexical = _lexical_search(client, query_text, [], _RECALL_SIZE, index_name)
    # See run_dense_only's docstring — dense step uses the fallback-
    # matched index, lexical stays on the caller's index_name (safe
    # regardless, since every embed-provider-variant index shares the
    # same chunk_ids/text).
    embed_result, dense_index = embed_queries_with_fallback([query_text])
    dense = _dense_search(client, embed_result.vectors[0], [], _RECALL_SIZE, dense_index)
    fused_scores = rrf_fuse_with_scores([lexical, dense])
    return [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]


def run_hybrid_rrf(queries, qrels, client, index_name: str = INDEX_NAME) -> dict:
    return _run_per_query(queries, qrels, lambda q: _hybrid_fused_order(client, q, index_name))


def run_hybrid_rerank(queries, qrels, client, index_name: str = INDEX_NAME) -> dict:
    def retrieve(query_text: str) -> list[str]:
        fused_order = _hybrid_fused_order(client, query_text, index_name)
        docs = client.mget(index=index_name, body={"ids": fused_order})["docs"]
        by_id = {d["_id"]: d["_source"]["text"] for d in docs if d.get("found")}
        candidates = [Candidate(id=doc_id, text=by_id[doc_id]) for doc_id in fused_order if doc_id in by_id]
        return [item.id for item in rerank(query_text, candidates, top_n=_RECALL_SIZE).items]

    return _run_per_query(queries, qrels, retrieve)


def run_full_pipeline(queries, qrels, index_name: str = INDEX_NAME) -> dict:
    # Baseline pacing, not just the reactive 429-backoff already inside
    # plan_query() — found live (2026-08-16) that a tight 80-query loop
    # exhausts Groq's free-tier rate limit almost immediately and then
    # spends the whole run cycling backoff/retry at the same too-fast
    # base rate, since a *recovered* retry immediately fires the *next*
    # query at full speed again. ~20 req/min stays under the ~30 RPM
    # ceiling this project has verified for Groq's free tier elsewhere.
    def retrieve(query_text: str) -> list[str]:
        time.sleep(3.0)
        plan = plan_query(query_text)
        candidates, _fusion = retrieve_with_trace(plan, top_n=_RECALL_SIZE, index_name=index_name)
        return [item.id for item in rerank(query_text, candidates, top_n=_RECALL_SIZE).items]

    return _run_per_query(queries, qrels, retrieve)


# `index_name` is bound into each lambda at call time inside run_all()
# (via a default-arg closure trick) rather than hardcoded here, since
# the real active index depends on EMBED_PROVIDER, only known once
# run_all() actually calls active_embed_index_name().
_CONFIGS = [
    ("BM25 only (baseline)", "OPENSEARCH", lambda q, r, c, idx: run_bm25_only(q, r, c, idx)),
    ("Dense only", "JINA_API_KEY", lambda q, r, c, idx: run_dense_only(q, r, c, idx)),
    ("Hybrid RRF", "JINA_API_KEY", lambda q, r, c, idx: run_hybrid_rrf(q, r, c, idx)),
    ("Hybrid + rerank", "JINA_API_KEY", lambda q, r, c, idx: run_hybrid_rerank(q, r, c, idx)),
    ("Hybrid + rewrite + rerank (full)", "GROQ_API_KEY + JINA_API_KEY", lambda q, r, c, idx: run_full_pipeline(q, r, idx)),
]

_COLUMNS = ["recall@5", "recall@10", "recall@20", "ndcg@10", "mrr@10", "p50_ms", "p95_ms"]


def run_all() -> dict[str, dict]:
    queries, qrels = load_domain_qrels()
    client = get_client()
    index_name = active_embed_index_name()
    results = {}
    for name, needs, fn in _CONFIGS:
        try:
            results[name] = {"status": "ran", "metrics": fn(queries, qrels, client, index_name), "needs": needs}
        except RuntimeError as exc:
            results[name] = {"status": "not run", "reason": str(exc), "needs": needs}
        except Exception as exc:
            results[name] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}", "needs": needs}
    return results


def _format_table(results: dict[str, dict]) -> str:
    header = "| config | " + " | ".join(_COLUMNS) + " |"
    sep = "|---" * (len(_COLUMNS) + 1) + "|"
    lines = [header, sep]
    for name, row in results.items():
        if row["status"] != "ran":
            note = f"**not run** — needs `{row['needs']}`" if row["status"] == "not run" else f"**error**: {row['reason']}"
            lines.append(f"| {name} | " + " | ".join([note] + ["—"] * (len(_COLUMNS) - 1)) + " |")
            continue
        m = row["metrics"]
        cells = [f"{m[c]:.4f}" if "ms" not in c else f"{m[c]:.0f}" for c in _COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_failures(results: dict[str, dict]) -> str:
    lines = []
    for name, row in results.items():
        if row["status"] != "ran":
            continue
        failures = row["metrics"].get("_failures")
        if not failures:
            continue
        lines.append(f"**{name}** — {len(failures)} of 80 quer{'y' if len(failures) == 1 else 'ies'} failed after retries, scored as a full miss (not dropped):")
        for query_id, reason in failures:
            lines.append(f"  - `{query_id}`: {reason}")
    return "\n".join(lines)


def write_report(results: dict[str, dict]) -> None:
    n_ran = sum(1 for r in results.values() if r["status"] == "ran")
    failures_block = _format_failures(results)
    failures_section = ("\n" + failures_block + "\n") if failures_block else ""
    body = f"""# Retrieval eval report

Generated by `python -m evals.run_retrieval_eval`. Numbers below are from
runs that actually happened on this machine — a config that couldn't run
(missing API key) is recorded as "not run", never inferred or assumed
passing, per the project's honesty rule.

**{n_ran} of {len(results)} configs ran for real** on this run.

Evaluated against `evals/datasets/qrels.jsonl` — 80 questions, one
relevant chunk each, generated by reading real chunks from the ingested
corpus (4 papers, 310 chunks) and composing a question genuinely
answerable from each one. **Not yet hand-verified by a human** — see
that file's `verified: false` field and the README's A3 section. Treat
these numbers as a first read, not a final gate, until that review
happens.

**Read the BM25 number with this in mind**: every question was written
*by reading its target chunk*, so it naturally reuses that chunk's own
vocabulary — a structural bias toward lexical search that a
naturally-phrased, independently-written question set would not share
this strongly. A high BM25 score here is expected by construction, not
evidence that lexical search alone is sufficient in general. This is
exactly the kind of gap human verification (rephrasing, paraphrasing
away from source wording) is meant to catch — see the README caveat.

{_format_table(results)}
{failures_section}
## Gate (from the plan)

Full pipeline must beat the BM25-only baseline by >=15% relative
nDCG@10, with p95 staying under 800ms. Not evaluated here while most
configs are "not run" — recorded once real numbers exist for every row.
"""
    _REPORT_PATH.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    results = run_all()
    print(_format_table(results))
    write_report(results)
    print(f"\nwrote {_REPORT_PATH}")
