import os
from pathlib import Path

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def is_postgres_backend() -> bool:
    """Phase 2, stage 7: a Jinja global (not a per-route context value) so
    base.html's cold-start notice and any other backend-aware chrome can
    read it without every route handler remembering to pass it — resolved
    fresh on each render, same "live toggle, not import-time" discipline
    as get_active_strategy()/get_active_embed_provider().
    """
    return os.environ.get("RETRIEVAL_BACKEND", "opensearch") == "postgres"


templates.env.globals["is_postgres_backend"] = is_postgres_backend


def reranked_items(reranked):
    """`trace.reranked.items` is ambiguous once `reranked` is a plain
    dict (the replay path — see src/store/runs.py): Jinja's attribute
    lookup finds the real, callable `dict.items` method before it ever
    falls back to the `"items"` key, so `{% for item in
    trace.reranked.items %}` silently tries to iterate a bound method
    instead of the list. A real RerankResult object has no such
    collision (it isn't a dict), so this filter is a no-op there —
    needed only because the live and replayed traces are different
    Python shapes carrying the same field name.
    """
    if reranked is None:
        return []
    if isinstance(reranked, dict):
        return reranked.get("items", [])
    return reranked.items


templates.env.filters["reranked_items"] = reranked_items
