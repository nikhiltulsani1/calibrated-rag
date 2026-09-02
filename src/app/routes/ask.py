from fastapi import APIRouter, Depends, Form, Request

from src.app.deps import templates
from src.app.errors import friendly_error_message
from src.app.rate_limit import enforce_rate_limit
from src.reason.answer_cache import get_cached_trace, set_cached_trace
from src.reason.pipeline import run_traced_query
from src.store.relational import get_session
from src.store.runs import save_run

router = APIRouter()


@router.get("/ask")
def ask_page(request: Request):
    return templates.TemplateResponse(request, "ask.html", {"active": "ask"})


@router.post("/ask", dependencies=[Depends(enforce_rate_limit)])
def ask_submit(request: Request, query: str = Form(...)):
    run_id = None
    cache_hit = False
    try:
        # Real bug found live 2026-08-24: get_cached_trace/set_cached_trace
        # used to sit unguarded in this same outer try, unlike save_run
        # two lines below — a real Redis hiccup on either call discarded
        # an already-successful answer and showed the user an error page,
        # the exact thing save_run's own adjacent comment says must never
        # happen. Both now get save_run's same local-try treatment: a
        # cache-read failure is just a miss (fall through to a real run),
        # a cache-write failure just means this answer wasn't cached —
        # neither should ever turn a working answer into an error page.
        try:
            cached = get_cached_trace(query)
        except Exception:
            cached = None
        if cached is not None:
            # A6: served from the semantic answer cache — cached shape
            # matches serialize_trace() exactly, so "answer" here is a
            # plain dict, not a real Answer object. Jinja resolves both
            # identically (see store/runs.py's docstring), so
            # _ask_result.html needs no changes for this.
            answer = cached.get("answer")
            cache_hit = True
        else:
            trace = run_traced_query(query)
            answer = trace.answer
            session = get_session()
            try:
                run_id = save_run(session, trace)
            except Exception:
                # Replay persistence is a nice-to-have on top of the answer,
                # not the answer itself — a Postgres hiccup here must not
                # turn a successful answer into an error page.
                run_id = None
            finally:
                session.close()
            try:
                set_cached_trace(query, trace)
            except Exception:
                pass
        error = None
    except Exception as exc:
        answer = None
        error = friendly_error_message(exc)

    return templates.TemplateResponse(
        request,
        "_ask_result.html",
        {
            "answer": answer,
            "error": error,
            "run_id": run_id,
            "cache_hit": cache_hit,
            "request_id": request.state.request_id,
        },
    )
