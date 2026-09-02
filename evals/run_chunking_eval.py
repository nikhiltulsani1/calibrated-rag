from __future__ import annotations

import json
import time
from pathlib import Path

from opensearchpy import helpers as os_helpers

from evals.metrics import mean, ndcg_at_k, recall_at_k
from src.index.client import create_index, get_client
from src.index.embedder import embed_queries
from src.index.mapping import INDEX_NAME
from src.ingest.chunking_strategies import STRATEGIES, _embed_passages_with_backoff, _with_rate_limit_backoff
from src.ingest.document_parser import fetch_pdf_bytes, parse_pdf
from src.retrieve.hybrid import _lexical_search, rrf_fuse_with_scores
from src.store.relational import get_session
from src.store.schema import ChunkVariant
from src.store.schema import Paper as PaperRow

# A7: measures all 5 candidate chunking strategies against document-level
# qrels (evals/derive_doc_qrels.py's output), on the real 8-paper corpus,
# each in its OWN scratch OpenSearch index — never touching the production
# `rag_chunks` index or the production `chunks` table. Retrieval
# configuration is held constant across strategies (hybrid BM25+dense RRF,
# no query rewriting, no reranking) so only the chunking variable moves —
# per the plan's "evaluated with retrieval configuration held constant"
# instruction.

_QRELS_DOC_PATH = Path(__file__).resolve().parent / "datasets" / "qrels_doc.jsonl"
_REPORT_PATH = Path(__file__).resolve().parent / "REPORT_chunking.md"
_RESULTS_JSON_PATH = Path(__file__).resolve().parent / "chunking_results.json"

_RECALL_SIZE = 50
_TOP_K = 10
_SCRATCH_PREFIX = "rag_chunks_ablation_"
_PERMANENT_PREFIX = "rag_chunks_"


def load_doc_qrels() -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    rows = [json.loads(line) for line in open(_QRELS_DOC_PATH, encoding="utf-8")]
    queries = [(r["query_id"], r["query"]) for r in rows]
    qrels = {r["query_id"]: r["relevant"] for r in rows}
    return queries, qrels


def _load_real_papers() -> list[PaperRow]:
    session = get_session()
    try:
        return session.query(PaperRow).order_by(PaperRow.arxiv_id).all()
    finally:
        session.close()


def _parse_all_papers(papers: list[PaperRow]) -> dict[str, object]:
    """PDF fetch + parse is identical regardless of chunking strategy —
    doing it once here and reusing across all 5 strategies avoids 5x
    redundant network/CPU cost for zero benefit.
    """
    parsed = {}
    for paper in papers:
        pdf_bytes = fetch_pdf_bytes(paper.url)
        parsed[paper.arxiv_id] = parse_pdf(pdf_bytes)
    return parsed


def _chunks_for_strategy(strategy_fn, papers: list[PaperRow], parsed: dict[str, object]):
    """Returns (chunks_with_paper: list[(Chunk, PaperRow)], chunk_to_paper: dict)."""
    chunks_with_paper = []
    chunk_to_paper: dict[str, str] = {}
    for paper in papers:
        raw_chunks = strategy_fn(paper.arxiv_id, parsed[paper.arxiv_id])
        # same within-batch content-address collision handling as the real
        # ingest pipeline (src/ingest/pipeline.py) — first occurrence wins.
        deduped = list({c.chunk_id: c for c in raw_chunks}.values())
        for c in deduped:
            chunks_with_paper.append((c, paper))
            chunk_to_paper[c.chunk_id] = paper.arxiv_id
    return chunks_with_paper, chunk_to_paper


def _index_strategy(index_name: str, strategy_name: str, chunks_with_paper) -> dict:
    """Embeds and bulk-indexes one strategy's chunks into a scratch
    index. `fixed_overlap` reproduces today's production chunk_ids
    byte-for-byte, so its vectors are fetched directly from the already-
    indexed production data (zero embedding cost for that arm) rather
    than re-embedded — real, verifiable reuse, not an assumed saving.
    """
    client = get_client()
    create_index(client, index_name=index_name)

    embed_calls_made = 0
    embed_calls_reused = 0

    if strategy_name == "fixed_overlap":
        ids = [c.chunk_id for c, _ in chunks_with_paper]
        docs = client.mget(index=INDEX_NAME, body={"ids": ids})["docs"]
        by_id = {d["_id"]: d["_source"] for d in docs if d.get("found")}
        vectors: dict[str, list[float]] = {}
        model_name = None
        missing = []
        for c, _ in chunks_with_paper:
            src = by_id.get(c.chunk_id)
            if src and "embedding" in src:
                vectors[c.chunk_id] = src["embedding"]
                model_name = src.get("embedding_model", model_name)
                embed_calls_reused += 1
            else:
                missing.append(c)
        if missing:
            result = _embed_passages_with_backoff([c.text for c in missing])
            embed_calls_made += len(missing)
            for c, vec in zip(missing, result.vectors):
                vectors[c.chunk_id] = vec
                model_name = model_name or result.model
    else:
        result = _embed_passages_with_backoff([c.text for c, _ in chunks_with_paper])
        embed_calls_made += len(chunks_with_paper)
        vectors = {c.chunk_id: vec for (c, _), vec in zip(chunks_with_paper, result.vectors)}
        model_name = result.model

    def actions():
        for c, paper in chunks_with_paper:
            yield {
                "_index": index_name,
                "_id": c.chunk_id,
                "_source": {
                    "chunk_id": c.chunk_id,
                    "paper_id": paper.arxiv_id,
                    "title": paper.title,
                    "text": c.text,
                    "section": c.section,
                    "authors": paper.authors,
                    "category": paper.category,
                    "published_date": paper.published_date.isoformat() if paper.published_date else None,
                    "embedding_model": model_name,
                    "embedding": vectors[c.chunk_id],
                },
            }

    os_helpers.bulk(client, actions(), raise_on_error=False)
    client.indices.refresh(index=index_name)
    # `refresh` makes documents searchable but doesn't reliably update the
    # store-size stat in the same call — a real 0.0 MB was observed live
    # for a freshly-indexed 571-chunk index (structure_aware's real size
    # was 3.06 MB, confirmed via a direct stats call moments later) until
    # `flush` forces segments to durable storage first.
    client.indices.flush(index=index_name)

    stats = client.indices.stats(index=index_name)
    size_bytes = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]

    return {
        "chunk_count": len(chunks_with_paper),
        "embed_calls_made": embed_calls_made,
        "embed_calls_reused": embed_calls_reused,
        "index_mb": round(size_bytes / (1024 * 1024), 2),
    }


def _dense_search_by_vector(client, vector: list[float], size: int, index_name: str) -> list[str]:
    """Same KNN query hybrid.py's _dense_search builds, but taking an
    already-computed vector instead of embedding query_text itself — the
    embedding is strategy-independent (see _precompute_query_vectors),
    so re-deriving it per strategy would be 5x redundant Jina calls.
    """
    body = {
        "size": size,
        "query": {"bool": {"must": [{"knn": {"embedding": {"vector": vector, "k": size}}}], "filter": []}},
        "_source": False,
    }
    response = client.search(index=index_name, body=body)
    return [hit["_id"] for hit in response["hits"]["hits"]]


def _precompute_query_vectors(queries: list[tuple[str, str]]) -> dict[str, list[float]]:
    """Embeds all 80 doc-qrels queries ONCE, in a single batched Jina
    call (embed_queries), reused across all 5 strategies' dense search.
    Discovered live: the original unbatched version (one embed_query
    call per query PER strategy — 400 total) hit Jina's rate limit
    outright. This is the fix, not just an optimization.
    """
    texts = [text for _, text in queries]
    vectors = _with_rate_limit_backoff(embed_queries, texts).vectors
    return {query_id: vec for (query_id, _), vec in zip(queries, vectors)}


def _score_strategy(index_name: str, queries, qrels, chunk_to_paper: dict[str, str], query_vectors: dict[str, list[float]]) -> dict:
    client = get_client()
    per_query = []
    for query_id, query_text in queries:
        time.sleep(0.3)  # OpenSearch-only load per query now (embedding is precomputed) — light pacing is enough
        lexical_ids = _lexical_search(client, query_text, [], _RECALL_SIZE, index_name)
        dense_ids = _dense_search_by_vector(client, query_vectors[query_id], _RECALL_SIZE, index_name)
        fused_scores = rrf_fuse_with_scores([lexical_ids, dense_ids])
        fused_order = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda i: i[1], reverse=True)]

        # chunk-level ranked ids -> paper-level, first-occurrence-wins dedup
        paper_order: list[str] = []
        seen = set()
        for chunk_id in fused_order:
            paper_id = chunk_to_paper.get(chunk_id)
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                paper_order.append(paper_id)

        relevance = qrels.get(query_id, {})
        relevant_ids = set(relevance)
        per_query.append(
            {
                "recall@10": recall_at_k(paper_order, relevant_ids, _TOP_K),
                "ndcg@10": ndcg_at_k(paper_order, relevance, _TOP_K),
            }
        )

    return {
        "recall@10": mean([q["recall@10"] for q in per_query]),
        "ndcg@10": mean([q["ndcg@10"] for q in per_query]),
    }


def select_variants(results: dict) -> dict:
    """Pure selection logic, separated out so it's directly unit-testable
    without mocking OpenSearch/Postgres/Jina. `winner` = highest nDCG@10;
    `median` = the middle-ranked strategy by nDCG@10 (not necessarily
    close to the mean — a real middle-of-the-pack strategy); `efficient`
    = the best nDCG@10 achieved per chunk, i.e. the best quality/cost
    tradeoff, which need not be the same strategy as `winner`.
    """
    ranked_by_ndcg = sorted(results.items(), key=lambda kv: kv[1]["ndcg@10"], reverse=True)
    winner = ranked_by_ndcg[0][0]
    median = ranked_by_ndcg[len(ranked_by_ndcg) // 2][0]
    efficient = max(
        results.items(),
        key=lambda kv: kv[1]["ndcg@10"] / max(kv[1]["chunk_count"], 1),
    )[0]
    return {"winner": winner, "median": median, "efficient": efficient}


def run_ablation(*, keep_scratch: bool = False) -> dict:
    queries, qrels = load_doc_qrels()
    papers = _load_real_papers()
    parsed = _parse_all_papers(papers)
    query_vectors = _precompute_query_vectors(queries)

    results = {}
    for strategy_name, strategy_fn in STRATEGIES.items():
        time.sleep(2.0)  # real 429 hit here once with zero pacing between strategies' embed_passages calls
        index_name = f"{_SCRATCH_PREFIX}{strategy_name}"
        chunks_with_paper, chunk_to_paper = _chunks_for_strategy(strategy_fn, papers, parsed)
        index_stats = _index_strategy(index_name, strategy_name, chunks_with_paper)
        scores = _score_strategy(index_name, queries, qrels, chunk_to_paper, query_vectors)
        results[strategy_name] = {**index_stats, **scores, "index_name": index_name}

    selection = select_variants(results)

    _write_report(results, selection)
    _RESULTS_JSON_PATH.write_text(json.dumps({"results": results, "selection": selection}, indent=2), encoding="utf-8")

    if not keep_scratch:
        client = get_client()
        for strategy_name, r in results.items():
            client.indices.delete(index=r["index_name"], ignore=[404])

    return {"results": results, "selection": selection}


def _write_report(results: dict, selection: dict) -> None:
    lines = ["# Chunking strategy ablation (A7)", ""]
    lines.append(
        "Generated by `python -m evals.run_chunking_eval`. All 5 strategies run against the "
        "real 8-paper corpus, scored on `evals/datasets/qrels_doc.jsonl` (document-level "
        "labels, derived from the 80 chunk-level qrels — a chunk counts as a hit if it "
        "belongs to a relevant paper). Retrieval config held constant (hybrid BM25+dense "
        "RRF, no rewrite, no rerank) so only chunking varies."
    )
    lines.append("")
    lines.append("| strategy | recall@10 | ndcg@10 | chunks | index MB | embed calls made | embed calls reused |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['recall@10']:.4f} | {r['ndcg@10']:.4f} | {r['chunk_count']} | "
            f"{r['index_mb']} | {r['embed_calls_made']} | {r['embed_calls_reused']} |"
        )
    lines.append("")
    lines.append(
        f"**Selected for the live toggle**: winner = `{selection['winner']}` (highest nDCG@10), "
        f"median = `{selection['median']}` (middle-ranked by nDCG@10), "
        f"efficient = `{selection['efficient']}` (best nDCG@10 per chunk)."
    )
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def persist_selected_variants(selection: dict) -> None:
    """Promotes the 3 selected strategies from scratch ablation indices
    to permanent ones (`rag_chunks_winner`/`rag_chunks_median`/
    `rag_chunks_efficient`) via OpenSearch's server-side reindex API —
    a document copy, not a re-embed, so this costs zero additional Jina
    calls beyond what run_ablation() already spent. Also writes the
    corresponding Postgres `chunk_variants` rows so the corpus is
    inspectable/rebuildable from the authoritative store, same principle
    as the production `chunks` table.
    """
    client = get_client()
    with open(_RESULTS_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    papers = {p.arxiv_id: p for p in _load_real_papers()}
    parsed = _parse_all_papers(list(papers.values()))

    session = get_session()
    try:
        for role, strategy_name in selection.items():
            scratch_index = results[strategy_name]["index_name"]
            permanent_index = f"{_PERMANENT_PREFIX}{role}"

            create_index(client, index_name=permanent_index)
            client.reindex(
                body={"source": {"index": scratch_index}, "dest": {"index": permanent_index}},
                wait_for_completion=True,
            )
            client.indices.refresh(index=permanent_index)

            session.query(ChunkVariant).filter(ChunkVariant.strategy == role).delete()
            chunks_with_paper, _ = _chunks_for_strategy(STRATEGIES[strategy_name], list(papers.values()), parsed)
            for chunk, paper in chunks_with_paper:
                session.add(
                    ChunkVariant(
                        strategy=role,
                        chunk_id=chunk.chunk_id,
                        paper_id=paper.arxiv_id,
                        section=chunk.section,
                        text=chunk.text,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                    )
                )
            session.commit()
            print(f"persisted {role} -> {permanent_index} ({strategy_name}, {len(chunks_with_paper)} chunks)")
    finally:
        session.close()


if __name__ == "__main__":
    import sys

    want_persist = "--persist" in sys.argv
    keep = "--keep-scratch" in sys.argv or want_persist  # persist reindexes FROM scratch, so it must still exist
    output = run_ablation(keep_scratch=keep)
    print(json.dumps(output, indent=2))
    print(f"\nwrote {_REPORT_PATH} and {_RESULTS_JSON_PATH}")

    if want_persist:
        persist_selected_variants(output["selection"])
        if "--keep-scratch" not in sys.argv:
            client = get_client()
            for r in output["results"].values():
                client.indices.delete(index=r["index_name"], ignore=[404])
            print("scratch ablation indices cleaned up")
