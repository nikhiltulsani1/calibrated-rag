import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.app.deps import is_postgres_backend
from src.app.errors import friendly_error_message
from src.app.middleware import CredentialsMiddleware, RequestIdMiddleware, install_request_id_log_filter
from src.app.routes import ask, corpus, health, pipeline, settings, upload
from src.platform.queue import ThreadSafeWorker, get_queue, get_redis_conn
from src.store.relational import init_db

logger = logging.getLogger(__name__)
install_request_id_log_filter()


def _run_upload_worker() -> None:
    # Phase 3 stage 3: an RQ Worker loop, run in a daemon thread inside
    # this SAME process rather than a separate Render service — see
    # src/platform/queue.py and the Phase 3 plan for why (Render's free
    # tier has no free dedicated worker service). .work() blocks forever
    # by design, which is exactly why this needs its own thread rather
    # than running at import time. Safe with today's UVICORN_WORKERS=1;
    # if that's ever raised, each process's own thread cooperatively
    # pops from the same Redis-backed queue — Redis list pops are
    # atomic, so N threads is N correct consumers, not duplicate
    # processing, and needs no code change here. ThreadSafeWorker (not
    # RQ's default Worker) — see its own docstring in
    # src/platform/queue.py for two real bugs found live: the default
    # Worker forks a subprocess per job (unsupported on Windows, which
    # this project is developed on locally), and unconditionally
    # installs signal handlers, which only works from the main thread —
    # this one is a background daemon thread.
    ThreadSafeWorker([get_queue()], connection=get_redis_conn()).work(with_scheduler=False)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Real gap found in review: nothing ever called init_db() at all —
    # not here, not in the Dockerfile, not in compose.yml — so a
    # genuinely fresh clone-and-run had no schema. Unconditional (both
    # retrieval backends need papers/chunks/runs), idempotent (a no-op
    # on every database this project already has running — this dev
    # machine's local Postgres, live Neon — see init_db()'s own comment
    # for why), and run before the app starts accepting traffic so a
    # broken DB fails loud at boot instead of on the first real request.
    init_db()
    if is_postgres_backend():
        threading.Thread(target=_run_upload_worker, daemon=True, name="upload-worker").start()
    yield


app = FastAPI(title="Calibrated RAG", lifespan=_lifespan)

# Order matters: Starlette runs middleware in reverse of add order, so
# CredentialsMiddleware (added second) runs OUTERMOST, wrapping
# RequestIdMiddleware — request_id is available for logging inside the
# credentials-handling path too, not just inside routes.
app.add_middleware(RequestIdMiddleware)
app.add_middleware(CredentialsMiddleware)

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(pipeline.router)
app.include_router(corpus.router)
app.include_router(settings.router)
app.include_router(upload.router)


@app.get("/")
def root():
    return RedirectResponse(url="/ask")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Defense in depth, not the primary mechanism — Ask/Pipeline/Corpus
    # each already catch their own known failure modes and render a
    # specific in-page message (see src/app/errors.py). This is what
    # catches anything a future route adds without doing the same,
    # so a raw, unstyled 500 (found live on Ask/Pipeline before those
    # were fixed) can't happen again by omission.
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    message = friendly_error_message(exc)
    request_id = getattr(request.state, "request_id", "-")
    return HTMLResponse(
        f'<html><body style="background:#0b0d10;color:#e6e9ef;font-family:sans-serif;padding:2rem;">'
        f"<h1>Couldn't complete this request.</h1><p>{message}</p>"
        f'<p style="color:#8a93a6;font-size:0.85em;">Reference id: {request_id}</p>'
        f'<p><a href="/ask" style="color:#6ea8fe;">Back to Ask</a></p></body></html>',
        status_code=500,
        headers={"X-Request-ID": request_id},
    )
