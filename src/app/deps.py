from pathlib import Path

from fastapi.templating import Jinja2Templates

from src.platform.backend import is_postgres_backend

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# Re-exported (not just imported for local use) so every route module
# that already does `from src.app.deps import templates` can pull
# `is_postgres_backend` from the same import line — see
# src/platform/backend.py's own docstring for why this single function
# now has one definition instead of 8 independent copies. Registered as
# a Jinja global too so base.html's cold-start notice and any other
# backend-aware chrome can read it without every route handler
# remembering to pass it — resolved fresh on each render, same
# "live toggle, not import-time" discipline as
# get_active_strategy()/get_active_embed_provider().
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
