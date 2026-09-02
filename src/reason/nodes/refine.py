from __future__ import annotations


def widen(top_n: int, *, step: int, cap: int) -> int:
    """The self-correction retry's only real decision: widen the reranked
    context window by `step`, capped at `cap`, so a persistently thin
    result doesn't grow unbounded. Moved verbatim out of the old
    pipeline.py loop — same arithmetic, same constants owned by graph.py.
    """
    return min(top_n + step, cap)
