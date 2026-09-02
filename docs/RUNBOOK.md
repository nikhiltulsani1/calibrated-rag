# Runbook

One entry per realistic failure mode — written now, while the design is fresh, not composed during the incident it describes (per the plan's R4 rule). Adapted down to what's real in this codebase today: no ladder-descent, `doctor` tool, or `quota.py` exist yet (see README's R1–R6 audit), so entries reference the actual mechanisms that do exist (`.env` model overrides, `src.index.reindex`, the `evals/REPORT*.md` files) rather than infrastructure this project hasn't built.

Numbers marked **[measured: pending]** get filled in once the corresponding phase of the production-readiness plan (R3's restore drill, R5's load test) actually runs — never estimated in the meantime, per this project's honesty rule.

---

## 1. Groq 429 (rate limit)

**Symptom:** `httpx.HTTPStatusError: Client error '429 Too Many Requests'` raised from `src.platform.models.complete()`, surfacing to the caller as a `RuntimeError`/friendly error message on `/ask` or `/pipeline`, or as a failure row in an `evals/REPORT*.md` run.

**Diagnosis:** Read the response body's `error.message` — Groq reports which limit was hit:
- `"...requests per minute (RPM)..."` or `"...tokens per minute (TPM)..."` — short-lived, resets in under a minute. Real observed TPM ceiling: 8,000 for `openai/gpt-oss-120b` (the `generate` role).
- `"...tokens per day (TPD)..."` — genuinely blocking until UTC day rollover. Real example hit this session: `"Rate limit reached for model \`openai/gpt-oss-120b\`... on tokens per day (TPD): Limit 200000, Used 199391, Requested 3502. Please try again in 20m49.776s"`. This is a hard wall — no code fix, no ladder to fall back to (the `generate` role's second rung, `openrouter:deepseek/deepseek-chat-v3:free`, is a real fallback binding but is not currently wired for automatic retry-on-429 — see the known gap in item 8 below).

**Action:**
- For an eval run: reduce/space out the `time.sleep(...)` pacing between questions, or wait for the daily reset before continuing.
- For live traffic: manually override the affected role via `.env`'s `RAG_MODEL_GENERATE=` or `RAG_MODEL_REWRITE=`/`RAG_MODEL_GRADE=` (see `src/config/models.yaml` and `src/platform/models.py::get_model_ladder`'s env-override rule) to point at a different provider/model until the window resets. This is a manual step today — no automatic ladder descent exists.

## 2. Jina rerank degrades vs. embed fails outright

**Symptom (rerank):** Answers still come back, but `reranker.degraded=true` appears in the trace/span attributes, and the retrieved context is in un-reranked (raw fusion) order.

**Diagnosis:** This is self-healing by design (A2's 2s timeout + degrade path) — check the span event/log for the specific `degrade_reason` (timeout, connection failure, 4xx/5xx/429) to know which.

**Action:** Usually nothing urgent — the system is still answering. If it persists across many requests, check `JINA_API_KEY` validity/quota; the local `RERANKER_BACKEND=local` fallback exists as an offline escape hatch (untested at runtime as of this writing — see README's A2 note).

**Symptom (embed):** Ingestion or a live query fails outright — embedding has no fallback by design (A0's "no ladder" rule: a bad embedding vector is worse than an honest failure).

**Action:** Check `JINA_API_KEY` — this project's Jina free tier has previously been exhausted mid-session (`AUTHZ_INSUFFICIENT_BALANCE`, a real account-balance exhaustion, not a rate limit) and required a manual top-up/key rotation.

**Real, verified Jina limit (2026-08-20)**: separately from account balance, Jina's free tier also caps at **100,000 tokens per minute** — a token-VOLUME limit, not a request-count one. Hit live during A7's chunking ablation (error body: `"Token rate limit exceeded: 123,608/100,000 tokens per minute"`) even with per-call pacing, because pacing controls request *count*, not the total token volume landing in a rolling 60s window. Fixed with real retry-with-backoff (`src/ingest/chunking_strategies.py::_with_rate_limit_backoff` — 65s wait, 3 retries) around every bulk `embed_passages`/`embed_queries` call in the eval scripts; live serving code (`src/index/embedder.py`) does not yet have this backoff, since normal query/answer traffic embeds one query or one small chunk batch at a time and hasn't hit this in practice — worth adding if that assumption ever stops holding.

## 3. NVIDIA judge quota exhausted

**Symptom:** `check_groundedness` fails **closed** — `GuardrailResult(errored=True)`, which (in `enforce` mode) blocks the response with the "can't confirm this answer is well-supported" decline text.

**Diagnosis:** This is correct, intended behavior, not a bug — the judge escalation path fails closed deliberately (see `output_guardrails.py`'s own docstring contrasting it with the input-guardrail fail-open convention).

**Action:** Check `NVIDIA_API_KEY` credits. Important: this key's free allocation (~1,000 credits) is a **one-time grant, not a daily-recurring quota** like Groq's — unlike item 1, there is no "wait for reset," only "the key needs replacing."

## 4. OpenSearch index missing, empty, or corrupt

**Symptom:** Retrieval returns nothing, or an unexplained drop in recall — this is a **real historical bug** already hit once in this project (`rag_chunks` silently held zero documents).

**Diagnosis:**
```bash
curl -sk -u admin:$OPENSEARCH_ADMIN_PASSWORD https://localhost:9200/rag_chunks/_count
```
Compare the count against the real Postgres chunk count (`SELECT COUNT(*) FROM chunks`) — a mismatch confirms the index is stale/incomplete, not that the corpus is actually empty.

**Action:** Postgres is authoritative; OpenSearch is a derived, rebuildable artifact by design (R3). Rebuild it:
```bash
python -m src.index.reindex
```
This is the exact command that fixed the real historical bug.

## 5. A gate regressed after a change

**Symptom:** A number in a freshly-regenerated `evals/REPORT*.md` is worse than the committed baseline.

**Action:** Re-run the specific harness that produced the regressed number (`python -m evals.run_retrieval_eval`, `python -m evals.run_generation_eval`, or the chunking/abstention equivalents once built) and diff the new `REPORT*.md` against the previously committed one. Do not assume a regression is noise — this project's own numbers have moved for real, traceable reasons every time so far (a prompt change, a chunking-quality issue on one specific chunk, real retrieval variance), never silently.

## 6. Restoring from backup

**Action:**
```bash
python -m scripts.backup_postgres   # or use the most recent file in backups/
python -m scripts.restore_drill     # picks the most recent backup automatically
```
`backup_postgres.py` shells out to the running `postgres` compose service's own `pg_dump` (no host-side Postgres client needed — verified none exists on this dev machine). `restore_drill.py` is the real, re-runnable drill: creates a throwaway `rag_restore_drill` database on the same Postgres instance, restores into it, rebuilds a scratch OpenSearch index from the restored data via the same code path as `src/index/reindex.py`, runs a real 80-question BM25 retrieval check, then tears everything down — production `rag`/`rag_chunks` are never touched.

**Measured** (2026-08-20, real run, 571 chunks, 8 papers):
- Postgres restore: **5.59s**
- OpenSearch rebuild (571 chunks re-embedded + bulk-indexed): **17.5s**
- Total drill time: **30.56s**
- Post-restore retrieval check: 80/80 questions scored, recall@10 = **0.925** (BM25-only against `qrels.jsonl` — consistent with A3's own BM25-only baseline, confirms the restored data retrieves correctly, not just that it exists)

"We can recover" now has a real number behind it, not an assumption.

## 7. A compose service is unhealthy

**Symptom:** `docker compose ps` shows a service as `unhealthy` or restarting; the API's own `/health` may or may not reflect it depending on which dependency is down.

**Diagnosis:**
```bash
docker compose ps
docker compose logs <service>
```
Each service's `compose.yml` healthcheck (already defined for postgres/opensearch/redis/api) is itself a diagnostic — a failing healthcheck command run manually often shows the exact error compose only summarizes as "unhealthy."

**Action:** Service-specific — Postgres/OpenSearch/Redis restarts are usually safe (data is in a named volume); the `api` service restart is safe and stateless.

## 8. Rate-limited by this project's own limiter

**Symptom:** HTTP 429 from `/ask` or `/pipeline`, with the message "Too many requests from this address."

**Diagnosis:** This is `src/app/rate_limit.py`'s per-caller Redis-backed limiter (default: `RATE_LIMIT_REQUESTS=20` per `RATE_LIMIT_WINDOW_SECONDS=60`), keyed on `request.client.host` — a coarse proxy, since no real caller-identity/auth system exists yet. A shared NAT/proxy (multiple real users behind one IP) can trigger this for a legitimate caller.

**Action:** Wait for the window to roll over (default 60s), or raise `RATE_LIMIT_REQUESTS`/`RATE_LIMIT_WINDOW_SECONDS` via `.env` if the default is genuinely too tight for expected traffic. If this fires often for legitimate single users behind shared IPs, that's a real signal the IP-based `caller_key` needs upgrading to real identity — not something to silently raise the limit around.

---

## Known gaps this runbook deliberately does not paper over

- **No automatic ladder descent on 429** — `generate`'s second rung (`openrouter:deepseek/deepseek-chat-v3:free`) exists in `models.yaml` but nothing currently retries a failed call on it automatically; only `query_planner.py`'s `rewrite` role has real retry-with-fallback logic today. A Groq TPD wall on `generate` is a genuine full stop until either the window resets or an operator manually overrides `RAG_MODEL_GENERATE`.
- **No scheduled alerting** — this runbook describes what to do once a human notices a problem; nothing pages anyone yet (R2's alerting is deliberately scoped to request/job traceability, not alert delivery — see the production-readiness plan).
