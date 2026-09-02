FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# torch (a transitive dependency of sentence-transformers, for the local
# reranker fallback — see requirements.txt's own comment) defaults to
# the CUDA-enabled wheel, which is multiple GB and has reliably broken
# the build partway through download on this connection (same
# BrokenPipeError, same byte offset, twice in a row — a real, not
# random, failure). This container has no GPU anyway, so installing the
# CPU-only build FIRST, from PyTorch's own CPU index, satisfies the
# dependency with a wheel roughly an order of magnitude smaller before
# requirements.txt's install ever tries to resolve torch itself.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# A7's Corpus-page toggle reads evals/chunking_results.json at runtime
# (src/app/routes/corpus.py) — the image previously only copied src/, so
# the deployed container could never see real ablation results, only
# ever showing "hasn't been run yet" regardless of what actually ran on
# the host. Caught live by checking the Corpus page in a browser against
# the running container, not assumed to work from reading the code.
COPY evals/ evals/

EXPOSE 8000

# Plan D3 — horizontal scaling: single hardcoded process before this,
# wasting the rest of a multi-core container. UVICORN_WORKERS defaults
# to 1 (today's exact behavior, zero change until deliberately raised)
# but is now genuinely configurable per replica, same env-var-toggle
# convention as CHUNKING_STRATEGY/EMBED_PROVIDER/QUERY_REWRITE_MODE.
# `exec` (not a bare shell command) matters here: without it, uvicorn
# runs as a CHILD of the shell rather than replacing it as PID 1, so a
# real SIGTERM from an orchestrator during a rolling deploy would hit
# the shell, not uvicorn — losing graceful in-flight-request draining.
ENV UVICORN_WORKERS=1
# Phase 2: --proxy-headers/--forwarded-allow-ips='*' — real bug found
# live while testing BYOK's Secure-flagged cookies: Render (and most
# PaaS hosts) terminate TLS at their own edge and forward to this
# container over plain HTTP, so without this, request.url.scheme always
# reads "http" even on the real HTTPS live deployment, which would make
# any scheme-aware Secure-cookie logic wrongly think it's never on
# HTTPS. This tells uvicorn to trust the X-Forwarded-Proto header
# Render's proxy sets, so request.url.scheme reports the real, original
# scheme. '*' is safe here specifically because Render's own edge is
# the only thing that can reach this container's port at all — nothing
# else sits between them for this deployment shape.
CMD ["sh", "-c", "exec uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS} --proxy-headers --forwarded-allow-ips='*'"]
