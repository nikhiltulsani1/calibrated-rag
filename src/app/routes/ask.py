import os

from fastapi import APIRouter, Depends, Form, Request

from src.app.deps import templates
from src.app.errors import friendly_error_message
from src.app.rate_limit import enforce_rate_limit
from src.reason.answer_cache import get_cached_trace, set_cached_trace
from src.reason.pipeline import run_traced_query
from src.store.relational import get_session
from src.store.runs import save_run

router = APIRouter()


def _is_postgres_backend() -> bool:
    return os.environ.get("RETRIEVAL_BACKEND", "opensearch") == "postgres"


@router.get("/ask")
def ask_page(request: Request, document_id: str | None = None):
    return templates.TemplateResponse(request, "ask.html", {"active": "ask", "document_id": document_id})


@router.post("/ask", dependencies=[Depends(enforce_rate_limit)])
def ask_submit(request: Request, query: str = Form(...), document_id: str | None = Form(None)):
    # Phase 2 (stage 6, uploads): session_id/document_id only carry real
    # meaning on the RETRIEVAL_BACKEND=postgres path — see run_graph's
    # docstring. Passed as None on the default OpenSearch path so its
    # cache keys, saved runs, and retrieval behavior stay byte-for-byte
    # what they were before uploads existed.
    session_id = request.state.session_id if _is_postgres_backend() else None
    scoped_document_id = document_id if _is_postgres_backend() else None

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
        #
        # A private, document-scoped question is never served from or
        # written to the cache at all — see answer_cache.py's docstring
        # for the cross-visitor leak this closes; scoping to one document
        # is inherently session-specific in a way the session_id-keyed
        # cache entry alone doesn't fully capture (two different documents
        # asked the same question by the same visitor would otherwise
        # collide on one cache entry).
        if scoped_document_id:
            cached = None
        else:
            try:
                cached = get_cached_trace(query, session_id=session_id)
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
            trace = run_traced_query(query, session_id=session_id, document_id=scoped_document_id)
            answer = trace.answer
            session = get_session()
            try:
                run_id = save_run(session, trace, owner_session_id=session_id)
            except Exception:
                # Replay persistence is a nice-to-have on top of the answer,
                # not the answer itself — a Postgres hiccup here must not
                # turn a successful answer into an error page.
                run_id = None
            finally:
                session.close()
            if not scoped_document_id:
                try:
                    set_cached_trace(query, trace, session_id=session_id)
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
