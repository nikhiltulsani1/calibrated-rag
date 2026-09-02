from __future__ import annotations

from pathlib import Path

import yaml

_LOCK_PATH = Path(__file__).resolve().parents[1] / "config" / "models.lock.yaml"

INDEX_NAME = "rag_chunks"


def locked_embed_dimension() -> int:
    lock = yaml.safe_load(_LOCK_PATH.read_text(encoding="utf-8")) or {}
    return int(lock["embed"]["dimension"])


def build_index_body(dimension: int | None = None) -> dict:
    """Hybrid index: one lexical field, one dense vector field, fused at
    query time by the retrieval pipeline (RRF) — not by OpenSearch itself.

    MRL truncation (A0) is handled by slicing the stored full-width vector
    at *read* time, not by storing multiple truncated fields — that's the
    entire point of Matryoshka embeddings, and it means this mapping only
    ever needs the one field at full width.
    """
    dim = dimension if dimension is not None else locked_embed_dimension()
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
                "section": {"type": "keyword"},
                # A1 extracts a bare "author" filter from natural language
                # ("papers by LeCun"), which won't exact-match the full
                # stored name ("Yann LeCun") on a plain keyword field.
                # The .text multi-field lets the filter step do an
                # analyzed match instead — added when building the actual
                # filtering logic in hybrid.py surfaced that this field
                # couldn't support it as originally mapped.
                "authors": {"type": "keyword", "fields": {"text": {"type": "text"}}},
                "category": {"type": "keyword"},
                "published_date": {"type": "date"},
                # Provenance, not decoration: the embed no-fallback rule
                # (A0) exists because a vector from the wrong model/width
                # silently returns nonsense rankings instead of failing.
                # Recording the model here is what makes that detectable.
                "embedding_model": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                "ingested_at": {"type": "date"},
            }
        },
    }
