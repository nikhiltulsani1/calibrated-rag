# Technical details — Calibrated RAG

This is the engineering reference: what each component does, how they fit
together, how to run every deployment profile, and what versions are
pinned and why. If you want plain-language explanations of the terms used
here, see [`glossary.md`](glossary.md). For how a request actually flows
through the system step by step, see [`architecture.md`](architecture.md).
For every real measured number this project has ever produced, see
[`result.md`](result.md) — nothing in that file is estimated.

---

## 1. What's in this repository

```
src/
  app/          FastAPI app — Ask / Pipeline / Corpus pages, routes, templates
  guardrails/   Input and output safety checks (see glossary: "guardrail")
  index/        OpenSearch client, embedding calls, index mapping
  ingest/       arXiv paper fetch, PDF parsing, chunking, ingestion pipeline
  platform/     Shared infrastructure: model routing, caching, telemetry
  reason/       The agentic answer-generation graph and its nodes
  retrieve/     Hybrid (lexical + vector) retrieval, reranking, query planning
  schemas/      Pydantic data contracts shared across the system
  store/        Postgres ORM models and persistence

evals/          Every evaluation script and every real report it produced
tests/
  unit/         Pure logic, no network, no Docker — the cheap tier
  integration/  Needs real Postgres/OpenSearch/Redis, but no real API keys

infra/          Airflow DAGs for scheduled ingestion
docs/           Design notes from earlier in the build
project-docs/   This file, architecture.md/.html, glossary.md, result.md
compose.yml     The whole deployable stack
```

## 2. The moving parts, and what each one is for

| Component | Technology | Role |
|---|---|---|
| Relational store | PostgreSQL 18.6 | Source of truth for papers and chunks. Everything else can be rebuilt from this. |
| Search index | OpenSearch 3.8.0 | A *derived* artifact: BM25 lexical search + kNN vector search over the same chunks, rebuildable from Postgres at any time. |
| Cache | Redis 8.10.0 | Query-plan cache, semantic answer cache, valid-category lookup — all with explicit TTLs, never a source of truth. |
| API | FastAPI (Python) | Serves the Ask/Pipeline/Corpus UI and owns the whole request pipeline. |
| Scheduled ingestion | Apache Airflow 3.3.0 (optional) | Off by default — see §4. Runs the ingestion pipeline on a schedule instead of by hand. |

External services (none of them store anything for this project — every
call is per-request or per-ingest):

| Service | Used for |
|---|---|
| Groq, OpenRouter, Mistral, NVIDIA | LLM calls: query understanding, answer generation, guardrail judging. Multiple providers configured per role so one provider's outage or rate limit doesn't take the system down — see `src/platform/models.py`. |
| Jina AI, Cohere | Embeddings and reranking, with Cohere as an automatic fallback if Jina is unavailable. |
| arXiv | Paper metadata and PDFs, at ingest time only. |
| Opik Cloud | Receives distributed traces for observability (optional). |

## 3. How a question gets answered — the short version

Full step-by-step detail (with exact file/function names) lives in
[`architecture.md`](architecture.md). The short version:

1. A cheap, deterministic check rejects garbage input (empty, absurdly long) before anything touches a network.
2. One LLM call ("query planning") normalizes the question, generates alternative phrasings to search with, and extracts any filters (author, category, date range) it implies.
3. Hybrid retrieval runs both a lexical (keyword) search and a vector (semantic) search across every phrasing, then merges the results.
4. A reranker re-scores and reorders the merged results by how relevant they actually are to the question.
5. Two independent checks run on that context: is there *enough* information here to answer, and is there anything *ambiguous or contradictory* about the question given what was retrieved.
6. If either check fails and the system is configured to act on it, the pipeline retries with a wider context window before giving up.
7. The final LLM call writes the answer, citing exactly which retrieved passages it used, and is instructed — explicitly, with rules the model must follow — to decline rather than guess when the context doesn't support a confident answer.
8. Every stage's LLM calls have automatic retry with fallback to a different provider on a real network failure, and every stage that can safely degrade (rather than must fail loudly) does so cleanly instead of crashing the request.

## 4. Running it

```bash
cp .env.example .env      # fill in the values — see the credentials table below
docker compose up -d postgres opensearch redis api
curl http://localhost:8000/health
```

Everything else is an optional profile, off by default:

```bash
docker compose --profile ingestion up -d    # scheduled ingestion via Airflow
docker compose --profile dashboards up -d   # OpenSearch Dashboards UI
docker compose --profile local-llm up -d    # local model runtime (Ollama)
```

Never disable Postgres or OpenSearch — they are the system. If disk or
RAM is constrained, cut in this order: dashboards first, then Airflow
(replace scheduled ingestion with running `python -m src.ingest.pipeline`
by hand).

### Running the tests

```bash
pip install -r requirements-dev.txt

pytest -m unit          # seconds — no Docker, no API keys, no network at all
pytest -m integration    # needs the real Postgres/OpenSearch/Redis containers running, but still no real third-party API keys — those call sites stay mocked
pytest                   # both
```

## 5. Credentials

| Credential | Required for | Free tier | Card needed? |
|---|---|---|---|
| `POSTGRES_USER/PASSWORD/DB` | Relational store | self-hosted | — |
| `OPENSEARCH_ADMIN_PASSWORD` | Search index | self-hosted | — |
| `GROQ_API_KEY` | Primary text generation | generous free tier | No |
| `OPENROUTER_API_KEY` | Fallback text generation | small free tier | No |
| `MISTRAL_API_KEY` | Fallback text generation + embeddings | small free tier | No |
| `NVIDIA_API_KEY` | Quality-judging role (evaluation only, not live traffic) | one-time free credits | No |
| `JINA_API_KEY` | Embeddings + reranking | ~1M tokens free, non-commercial only | No |
| `COHERE_API_KEY` | Automatic reranking fallback if Jina is unavailable | free tier | No |
| `OPIK_API_KEY`, `OPIK_WORKSPACE` | Trace/eval tracking (optional) | free tier with retention limits | No |
| `ARXIV_CONTACT_EMAIL` | Required by arXiv's own API etiquette | free | — |

Every model-provider role is also overridable per-role via
`RAG_MODEL_GENERATE`, `RAG_MODEL_REWRITE`, `RAG_MODEL_GRADE`,
`RAG_MODEL_JUDGE`, `RAG_MODEL_EMBED`, `RAG_MODEL_RERANK` — set one to
`provider:model_id` to override the default without touching code. This
is how narrow live tests were pinned to a single provider throughout this
project's own development (see `result.md` for real examples).

## 6. Design decisions worth knowing

**Fail loud vs. fail open — deliberately different per component.**
A wrong embedding silently corrupts what gets retrieved at all, so
embedding failures raise immediately rather than continue with bad data.
A failed reranking call, by contrast, just leaves results in their
pre-rerank order — still valid, just less well-sorted — so reranking
degrades gracefully instead of failing the whole request. Guardrails
follow the same logic: a broken guardrail must never itself become a
denial-of-service, so most guardrail checks fail open (let the request
through) rather than fail closed (block it) when the check itself
errors — the one exception is the output-side groundedness check, which
fails closed, because a broken check there is exactly the
hallucination-shipping gap the check exists to close.

**Every LLM-calling stage retries across providers, not just once.**
A transient network error or a rate limit on one provider automatically
falls back to a different, independently-configured provider rather than
failing the request. This was found to matter in practice — see
`result.md` for a real, live-observed case where this exact mechanism
kept the system answering correctly during a genuine multi-provider
outage.

**Postgres is the only source of truth.** The search index is entirely
derived from it and can be rebuilt from scratch at any time (a new
embedding model, a new chunking strategy — neither is data loss, both are
just a rebuild).

**The semantic answer cache never touches evaluation runs.** Every
evaluation script calls the core answer-generation function directly,
bypassing the HTTP-layer cache entirely — an evaluation run must always
measure a fresh answer, never a cached one, or its numbers would be
meaningless.

## 7. Versions pinned

| Service | Image |
|---|---|
| Postgres | `postgres:18.6` |
| OpenSearch | `opensearchproject/opensearch:3.8.0` |
| OpenSearch Dashboards | `opensearchproject/opensearch-dashboards:3.8.0` |
| Redis | `redis:8.10.0-alpine` |
| Airflow | `apache/airflow:3.3.0-python3.12` (LocalExecutor — this is a single-node deployment, Celery's broker/worker containers buy nothing here) |

Re-verify before a production build — these move fast.
