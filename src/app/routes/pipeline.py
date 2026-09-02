from fastapi import APIRouter, Depends, Request

from src.app.deps import templates
from src.app.errors import friendly_error_message
from src.app.rate_limit import enforce_rate_limit
from src.reason.answer_cache import get_cached_trace, set_cached_trace
from src.reason.pipeline import run_traced_query
from src.store.relational import get_session
from src.store.runs import load_run, save_run

router = APIRouter()


@router.get("/pipeline", dependencies=[Depends(enforce_rate_limit)])
def pipeline_page(request: Request, query: str | None = None, run_id: str | None = None):
    trace = None
    error = None
    context_word_estimate = 0
    replayed = bool(run_id)
    saved_run_id = None

    if run_id:
        session = get_session()
        try:
            trace = load_run(session, run_id)
            if trace is None:
                error = f"No saved run for id {run_id!r} — it may predate this feature, or the id is wrong."
            elif trace.get("reranked"):
                context_word_estimate = sum(len(item["text"].split()) for item in trace["reranked"]["items"])
        except Exception as exc:
            error = friendly_error_message(exc)
        finally:
            session.close()
    elif query:
        try:
            # Real bug found live 2026-08-24: get_cached_trace/
            # set_cached_trace used to sit unguarded here, unlike save_run
            # a few lines below — a real Redis hiccup on either call
            # discarded an already-successful trace and showed an error
            # page, exactly what save_run's own adjacent comment says
            # must never happen. Both get the same local-try treatment
            # now: a read failure is just a miss, a write failure just
            # means this trace wasn't cached.
            try:
                cached = get_cached_trace(query)
            except Exception:
                cached = None
            if cached is not None:
                # A6: cache hit — the cached dict has the exact same
                # shape load_run() returns (both go through
                # store.runs.serialize_trace), so it renders through the
                # identical dict-access template path as a Postgres
                # run_id replay, no template changes needed. Reusing the
                # `replayed` badge is deliberate: "the exact trace from
                # when it originally ran, not a fresh query" is exactly
                # true for a cache hit too.
                trace = cached
                replayed = True
                if trace.get("reranked"):
                    context_word_estimate = sum(len(item["text"].split()) for item in trace["reranked"]["items"])
            else:
                trace = run_traced_query(query)
                if trace.reranked:
                    context_word_estimate = sum(len(item.text.split()) for item in trace.reranked.items)
                session = get_session()
                try:
                    saved_run_id = save_run(session, trace)
                except Exception:
                    # Same "nice-to-have, not the answer itself" rule as /ask
                    # — a save failure must not turn a working trace into an
                    # error page.
                    saved_run_id = None
                finally:
                    session.close()
                try:
                    set_cached_trace(query, trace)
                except Exception:
                    pass
        except Exception as exc:
            error = friendly_error_message(exc)

    return templates.TemplateResponse(
        request,
        "pipeline.html",
        {
            "active": "pipeline",
            "query": query,
            "trace": trace,
            "error": error,
            "context_word_estimate": context_word_estimate,
            "replayed": replayed,
            "run_id": run_id or saved_run_id,
            "request_id": request.state.request_id,
        },
    )
