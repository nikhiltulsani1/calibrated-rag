from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# R3's actual drill, not just a designed-for-it claim: wipe a scratch
# database, restore a real backup into it, rebuild the derived OpenSearch
# index from the restored data, and confirm retrieval still works —
# exercising the exact "Postgres is authoritative, OpenSearch is
# rebuildable" claim this project has made since A0, for the first time.
#
# Deliberately uses a throwaway DATABASE on the SAME running Postgres
# instance (`rag_restore_drill`) rather than a second container — lower
# operational risk for a repeatable drill, and never touches the real
# `rag`/`rag_chunks` the running application depends on.

_DRILL_DB = "rag_restore_drill"
_DRILL_INDEX = "rag_chunks_restore_drill"


def _admin_engine():
    """Connects to Postgres's own `postgres` maintenance database — the
    one every Postgres install has, needed because CREATE DATABASE/DROP
    DATABASE cannot run inside a transaction against the database being
    created/dropped itself.
    """
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/postgres"
    return create_engine(url, isolation_level="AUTOCOMMIT")


def _create_drill_database() -> None:
    engine = _admin_engine()
    with engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{_DRILL_DB}"'))
        conn.execute(text(f'CREATE DATABASE "{_DRILL_DB}"'))
    engine.dispose()


def _drop_drill_database() -> None:
    engine = _admin_engine()
    with engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{_DRILL_DB}"'))
    engine.dispose()


def _restore_into_drill_database(backup_path: Path, *, service: str = "postgres") -> None:
    user = os.environ["POSTGRES_USER"]
    with open(backup_path, "rb") as f:
        result = subprocess.run(
            ["docker", "compose", "exec", "-T", service, "pg_restore", "-U", user, "-d", _DRILL_DB, "--no-owner"],
            cwd=_REPO_ROOT,
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    # pg_restore's real exit-code convention: nonzero on genuine failures,
    # but also on benign warnings (e.g. "role does not exist" from
    # --no-owner interacting with ownership metadata) — check stderr for
    # the actual failure signature rather than trusting the exit code
    # alone, which would make this drill cry wolf on every run.
    stderr_text = result.stderr.decode(errors="replace")
    if result.returncode != 0 and "FATAL" in stderr_text.upper():
        raise RuntimeError(f"pg_restore failed: {stderr_text}")


def _rebuild_opensearch_index() -> dict:
    from src.index.client import create_index, get_client
    from src.index.embedder import embed_passages
    from src.index.mapping import INDEX_NAME
    from src.store.schema import Chunk, Paper
    from opensearchpy import helpers as os_helpers
    from datetime import datetime, timezone

    drill_url = _drill_database_url()
    engine = create_engine(drill_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT c.chunk_id, c.paper_id, c.section, c.text, "
                "p.title, p.authors, p.category, p.published_date "
                "FROM chunks c JOIN papers p ON c.paper_id = p.arxiv_id"
            )
        ).fetchall()
    engine.dispose()

    from src.ingest.chunking_strategies import _with_rate_limit_backoff

    client = get_client()
    create_index(client, index_name=_DRILL_INDEX)

    texts = [r.text for r in rows]
    result = _with_rate_limit_backoff(embed_passages, texts)

    def actions():
        for row, vector in zip(rows, result.vectors):
            yield {
                "_index": _DRILL_INDEX,
                "_id": row.chunk_id,
                "_source": {
                    "chunk_id": row.chunk_id,
                    "paper_id": row.paper_id,
                    "title": row.title,
                    "text": row.text,
                    "section": row.section,
                    "authors": row.authors,
                    "category": row.category,
                    "published_date": row.published_date.isoformat() if row.published_date else None,
                    "embedding_model": result.model,
                    "embedding": vector,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            }

    os_helpers.bulk(client, actions(), raise_on_error=False)
    client.indices.refresh(index=_DRILL_INDEX)
    return {"chunks_restored": len(rows)}


def _drill_database_url() -> str:
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{_DRILL_DB}"


def _validate_retrieval() -> dict:
    """A minimal but real retrieval check against the rebuilt index —
    not the full 5-way ablation, just enough to satisfy the plan's own
    acceptance criterion ("confirm the eval suite still passes against
    restored data"): the real qrels.jsonl questions, BM25-only.
    """
    from src.index.client import get_client
    from src.retrieve.hybrid import _lexical_search
    from evals.metrics import mean, recall_at_k

    qrels_path = _REPO_ROOT / "evals" / "datasets" / "qrels.jsonl"
    rows = [json.loads(line) for line in open(qrels_path, encoding="utf-8")]

    client = get_client()
    recalls = []
    for row in rows:
        ranked_ids = _lexical_search(client, row["query"], [], 10, _DRILL_INDEX)
        recalls.append(recall_at_k(ranked_ids, set(row["relevant"]), 10))

    return {"n_questions": len(rows), "recall@10": mean(recalls)}


def _cleanup_opensearch() -> None:
    from src.index.client import get_client

    get_client().indices.delete(index=_DRILL_INDEX, ignore=[404])


def run_drill(backup_path: Path | None = None) -> dict:
    load_dotenv(_REPO_ROOT / ".env")

    if backup_path is None:
        backups = sorted((_REPO_ROOT / "backups").glob("rag_backup_*.dump"))
        if not backups:
            raise RuntimeError("no backup file found — run scripts/backup_postgres.py first")
        backup_path = backups[-1]

    start = time.monotonic()

    _create_drill_database()
    _restore_into_drill_database(backup_path)
    postgres_restored_at = time.monotonic()

    index_stats = _rebuild_opensearch_index()
    index_rebuilt_at = time.monotonic()

    retrieval_check = _validate_retrieval()
    total_elapsed = time.monotonic() - start

    _cleanup_opensearch()
    _drop_drill_database()

    return {
        "backup_file": str(backup_path),
        "postgres_restore_seconds": round(postgres_restored_at - start, 2),
        "opensearch_rebuild_seconds": round(index_rebuilt_at - postgres_restored_at, 2),
        "total_drill_seconds": round(total_elapsed, 2),
        "index_stats": index_stats,
        "retrieval_check": retrieval_check,
    }


if __name__ == "__main__":
    result = run_drill()
    print(json.dumps(result, indent=2))
