from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_RAW_DIR = Path(__file__).resolve().parent / "corpora" / "scifact_raw" / "scifact"


@dataclass(frozen=True)
class BenchmarkDoc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str


def load_corpus() -> list[BenchmarkDoc]:
    docs = []
    with open(_RAW_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(BenchmarkDoc(doc_id=row["_id"], title=row.get("title", ""), text=row["text"]))
    return docs


def load_queries() -> list[BenchmarkQuery]:
    queries = []
    with open(_RAW_DIR / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries.append(BenchmarkQuery(query_id=row["_id"], text=row["text"]))
    return queries


def load_qrels(split: str = "test") -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevance}. SciFact's judgments are binary
    (score always 1) — passed straight through so ndcg_at_k's graded-gain
    formula still runs the real code path, it just degenerates to the
    binary case for this particular benchmark.
    """
    qrels: dict[str, dict[str, int]] = {}
    path = _RAW_DIR / "qrels" / f"{split}.tsv"
    with open(path, encoding="utf-8") as f:
        next(f)  # header: query-id, corpus-id, score
        for line in f:
            query_id, doc_id, score = line.strip().split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels
