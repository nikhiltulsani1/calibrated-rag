from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant docs that appear in the top k."""
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def dcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Discounted cumulative gain, standard formula (Jarvelin & Kekalainen
    2002): graded gain 2^rel - 1, discounted by log2(rank + 1). Public,
    well-documented formula — not adapted from any specific
    implementation, per the plan's IP posture note.
    """
    score = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = relevance.get(doc_id, 0)
        if rel > 0:
            score += (2**rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """DCG normalized by the ideal (best-possible) ordering's DCG, so the
    score is always in [0, 1] regardless of how many relevant docs exist.
    """
    dcg = dcg_at_k(ranked_ids, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_order) if rel > 0)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 / rank of the first relevant doc in the top k, else 0."""
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (not interpolated) — the same simple
    sorted-index method run_retrieval_eval.py's `_aggregate` already used
    for p50/p95 inline; pulled out here as a real second use (R5's
    scripts/load_test.py) rather than copy-pasted again.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[index]
