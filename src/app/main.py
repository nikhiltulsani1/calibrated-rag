import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.app.errors import friendly_error_message
from src.app.middleware import RequestIdMiddleware, install_request_id_log_filter
from src.app.routes import ask, corpus, health, pipeline

logger = logging.getLogger(__name__)
install_request_id_log_filter()

app = FastAPI(title="Calibrated RAG")

app.add_middleware(RequestIdMiddleware)

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(pipeline.router)
app.include_router(corpus.router)


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
