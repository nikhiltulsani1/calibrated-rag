from __future__ import annotations

import json
from pathlib import Path

# A7: qrels.jsonl is labeled at chunk_id granularity, which A3b's own
# note explains is why chunking was left fixed for A3/A4 — a re-chunk
# invalidates chunk_id labels. This script derives a document-level
# equivalent from the SAME 80 questions (no new judgment, purely
# mechanical): collapse {chunk_id: grade} to {paper_id: max(grade)}
# using each row's own `paper_id` field. Document-level labels survive
# re-chunking — a chunk counts as a hit if it belongs to a relevant
# document — which is what makes comparing chunking strategies possible
# at all.

_QRELS_PATH = Path(__file__).resolve().parent / "datasets" / "qrels.jsonl"
_OUT_PATH = Path(__file__).resolve().parent / "datasets" / "qrels_doc.jsonl"


def derive() -> list[dict]:
    rows = [json.loads(line) for line in open(_QRELS_PATH, encoding="utf-8")]
    out = []
    for row in rows:
        grades = list(row["relevant"].values())
        out.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "paper_id": row["paper_id"],
                "relevant": {row["paper_id"]: max(grades)},
                "note": row["note"],
                "verified": row["verified"],
            }
        )
    return out


if __name__ == "__main__":
    rows = derive()
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} document-level qrels to {_OUT_PATH}")
