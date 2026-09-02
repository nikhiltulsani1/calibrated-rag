# Production_RAG_System — Architecture & Complete Flow

This is the teaching document: how a question becomes an answer, where
every decision gets made, where every check happens, and where to look
to verify any of it yourself. Every file path, function name, and span
name below is copied directly from the real code (checked on
2026-08-16), not written from memory.

For *what's built vs. not, and every honest gap*, see
[`README.md`](../README.md) — this document explains **how the built parts
work**, not their completeness status.

---

## 1. The shape of the system

Four services, one process talking to all of them:

```
                    ┌─────────────────────────────┐
                    │   FastAPI app (src/app/)     │
                    │   Ask · Pipeline · Corpus     │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  PostgreSQL    │          │  OpenSearch    │          │  Redis        │
│  papers, chunks│          │  hybrid index: │          │  query-plan    │
│  = SOURCE OF   │  ──────▶ │  BM25 + kNN    │          │  cache         │
│  TRUTH         │  rebuild │  = DERIVED,    │          │                │
└───────────────┘  from PG │  REBUILDABLE   │          └───────────────┘
                            └───────────────┘
```

**Why this matters for understanding everything else**: Postgres is
authoritative. OpenSearch is a *derived* artifact — every vector and
every lexical field in it can be regenerated from Postgres's chunk text.
This is why an embedding-model change is a rebuild, not data loss (see
§5), and it's why the ingestion pipeline writes to Postgres first, always
(see §3).

External services this system talks to (none of them store anything —
they're all called per-request or per-ingest):

| Service | Called from | What for |
|---|---|---|
| Groq | `src/platform/models.py::complete()` | `rewrite` role (query understanding) and `generate` role (answer generation) |
| NVIDIA NIM | `src/platform/models.py::complete()` | `judge` role (groundedness-guardrail escalation only) |
| Jina AI | `src/index/embedder.py`, `src/retrieve/reranker.py` | embeddings (`embed` role) and reranking (`rerank` role) |
| arXiv | `src/ingest/paper_source.py` | paper metadata + PDFs, at ingest time only |
| Opik Cloud | `src/platform/telemetry.py` | receives OpenTelemetry traces (§8) |

---

## 2. The complete flow: a question becomes an answer

Everything below is one function: **`run_traced_query()`** in
[`src/reason/pipeline.py`](../src/reason/pipeline.py). This is the single
source of truth for the whole pipeline — `answer_query()` (used by the
`/ask` route) is a one-line wrapper around it that just returns the final
answer; the Pipeline UI page calls `run_traced_query()` directly to show
every stage.

Here is the real sequence, in order, with exactly what happens at each
step:

### Step 0 — Size & shape guardrail
**File**: [`src/guardrails/input_guardrails.py::check_size_and_shape()`](../src/guardrails/input_guardrails.py)
**Guardrail name**: `size_and_shape`

Rejects empty queries and queries over 2,000 characters. Runs *before*
anything else touches the network — a garbage query never reaches an LLM
call. **Fails open**: if this check itself errors, the request proceeds
anyway (a broken guardrail must not become a denial-of-service).

### Step 1 — Query planning (LLM decision #1)
**File**: [`src/retrieve/query_planner.py::plan_query()`](../src/retrieve/query_planner.py)
**Span**: `query_planner.plan_query`
**Model role**: `rewrite` (bound to Groq)

This is the first place an LLM makes a decision. Given the raw question,
one LLM call produces a JSON object with four things:

| Field | What it is |
|---|---|
| `normalized` | Spelling/acronym-normalized version of the question |
| `expansions` | 2-3 alternative phrasings — each gets independently searched |
| `filters` | Extracted `author`/`category`/`date_from`/`date_to`, if the question implies any |
| `intent` | One of `factual`, `comparative`, `multi_hop`, `out_of_scope` |

The exact prompt the model sees is in that file's `_SYSTEM_PROMPT`
constant — read it directly if you want to know exactly what the model
is told to do, rather than trust a paraphrase here.

**Caching**: keyed on `sha256(query.strip().lower())`, stored in Redis,
24h TTL. A repeat question costs zero LLM calls. Deliberately *not* keyed
on the model's own `normalized` output — that would require the call the
cache exists to avoid.

### Step 2 — Scope-screening guardrail
**File**: [`src/guardrails/input_guardrails.py::screen_scope()`](../src/guardrails/input_guardrails.py)
**Guardrail name**: `scope_screening`

Uses the `intent` field Step 1 already computed — no extra LLM call.
If `intent == "out_of_scope"` **and** the guardrail is in `enforce` mode
(env var `GUARDRAIL_SCOPE_SCREENING_MODE`, default `monitor`), the
request stops here and returns a decline. In the default `monitor` mode,
this still runs and still records what *would* have happened, but lets
the request continue — see §4 for why.

### Step 3 — Hybrid retrieval
**File**: [`src/retrieve/hybrid.py::retrieve_with_trace()`](../src/retrieve/hybrid.py)
**Span**: `hybrid.retrieve`

For **every** query variant from Step 1 (the normalized query, plus each
expansion), two searches run against OpenSearch:

- **Lexical** (`_lexical_search`) — BM25 `multi_match` across `text` and
  `title` (title double-weighted).
- **Dense** (`_dense_search`) — the variant is embedded (`embed` role,
  Jina) and searched via OpenSearch's `knn` query against the stored
  1024-dim vectors.

Metadata filters from Step 1 are applied identically to *every* one of
these sub-queries — see §5 for why that needed a real fix partway through
this build.

All the resulting ranked lists (one variant × one arm = one list) are
combined with **Reciprocal Rank Fusion**: `score(doc) = Σ 1/(k + rank +
1)` across every list it appears in, `k=60`. The pure fusion math lives
in `rrf_fuse_with_scores()` — it's a standalone function specifically so
it's testable with no OpenSearch involved at all (see §10).

The full arithmetic (which arm found what, at what rank, and the exact
fused score) is captured in a `FusionTrace` object — this is what the
Pipeline UI page's stages 4-5 render, straight from the real numbers.

### Step 4 — Reranking
**File**: [`src/retrieve/reranker.py::rerank()`](../src/retrieve/reranker.py)
**Span**: `reranker.rerank`
**Model role**: `rerank` (Jina cross-encoder)

The top 50 fused candidates go to a hosted cross-encoder that reorders
them by genuine relevance (not just fusion score) and returns the top 8.

**This is the one stage designed to never hard-fail.** If the hosted
call times out (2s) or errors, it **degrades**: returns the original
fusion order, unchanged, with `degraded=True` and a reason — never
raises. This is the opposite failure mode from embeddings (Step 3),
which fails loud on purpose. Reranking only *reorders* already-valid
candidates, so skipping a bad reorder is safe; a wrong embedding would
silently corrupt what gets retrieved *at all*.

### Step 5 — Metadata fetch
**File**: [`src/reason/pipeline.py::fetch_metadata()`](../src/reason/pipeline.py)

A plain OpenSearch `mget` for the final 8 chunk IDs, pulling `title`,
`paper_id`, `section` — needed for citations in the next step.

### Step 6 — Answer generation (LLM decision #2)
**File**: [`src/reason/generate.py::generate_answer()`](../src/reason/generate.py)
**Span**: `generate.answer`
**Model role**: `generate` (Groq, `openai/gpt-oss-120b`)

The 8 reranked chunks are numbered `[1]` through `[8]` and given to the
model with an explicit instruction: answer *only* from these passages,
cite every claim with its passage number, or say the one exact sentence
that means "I don't have enough information" if the passages don't
support an answer. That exact sentence is the `ABSTAIN_TEXT` constant in
`generate.py` — the abstention check is a literal string match against
it, not a vibe-based classification.

**How citations actually get built**: the code does *not* trust that
every passage it handed the model got cited. It regex-scans the model's
own output for `[N]` markers (`re.findall(r"\[(\d+)\]", ...)`) and only
turns the markers the model *actually wrote* into `Citation` objects. A
passage that was in context but never referenced is not listed as a
source.

### Step 7 — Citation-integrity guardrail
**File**: [`src/guardrails/output_guardrails.py::check_citation_integrity()`](../src/guardrails/output_guardrails.py)
**Guardrail name**: `citation_integrity`

Every citation is checked against the actual set of chunk IDs that were
in context. Anything that doesn't resolve gets silently stripped (never
left in, never presented as real) and the guardrail records that it
fired. **This guardrail cannot be turned off** — no `enforce`/`monitor`
toggle exists for it, on purpose (see §4).

### Step 8 — Groundedness guardrail (LLM decision #3, sometimes)
**File**: [`src/guardrails/output_guardrails.py::check_groundedness()`](../src/guardrails/output_guardrails.py)
**Guardrail name**: `groundedness`
**Model role**: `judge` (NVIDIA — only called if escalation happens)

This is a two-tier check, deliberately cheap-first:

1. **Deterministic overlap** (always runs, no network call): tokenizes
   the answer and the context, strips stopwords, and computes what
   fraction of the answer's words actually appear somewhere in the
   context. Threshold `0.5` by default
   (`GROUNDEDNESS_OVERLAP_THRESHOLD`), explicitly documented as a
   starting point, not a tuned value — there's no real traffic yet to
   tune it against.
2. **Judge escalation** (only if step 1 scores below threshold): one
   call to the `judge` role asking a strict YES/NO — "is this answer
   fully supported by the context?" This is the *only* code path that
   calls the `judge` role at all right now.

If this fails **and** the guardrail is in `enforce` mode, the real
answer is replaced with a decline. **Fails closed**: if the check itself
errors (e.g. no `NVIDIA_API_KEY` during escalation), that counts as a
*failure*, not a pass — the opposite of every input guardrail's
fail-open rule. An error here must never accidentally let an
unverified answer through.

### Step 9 — Self-correction (sometimes)

Steps 4–8 above aren't a strict one-pass sequence — they're a bounded
loop. If the model **abstained** (Step 6) or Step 8 genuinely found the
answer ungrounded (a real "NO" verdict or low overlap — *not* the check
erroring, see below), the pipeline widens the context window by 4 chunks
(capped at 16) and runs Steps 4–8 again, up to `RAG_MAX_ATTEMPTS` total
attempts (default 2). Every attempt — its context size, its answer, its
groundedness verdict — lands in `StageTrace.attempts`, which is what the
Pipeline page's "9. Self-correction" panel renders, and it only appears
when a retry actually happened.

The retry-vs-don't-retry line depends on a real distinction:
`GuardrailResult.errored` is `True` only when the *check itself* broke
(e.g. a missing judge API key), never when the check ran fine and found
a genuine problem. Retrying a real "NO" verdict can plausibly help;
retrying a missing API key just repeats the identical failure, so that
case is excluded from the retry condition on purpose.

That's the whole flow. Every stage's real output — not a summary of it —
lands in a `StageTrace` object, which is exactly what the `/pipeline`
page renders (§9), and every completed run is persisted so it can be
**replayed exactly later** via `/pipeline?run_id=<uuid>` — see §9.5.

---

## 3. Ingestion & chunking (a separate, offline flow)

This does **not** run at question-answering time — it's a batch process
you run manually right now (`python -m src.ingest.pipeline <category>
<count>`; a scheduled Airflow version is designed but not built). One
function, [`src/ingest/pipeline.py::ingest_category()`](../src/ingest/pipeline.py),
runs this sequence per paper:

1. **Source** — [`src/ingest/paper_source.py::fetch_papers_paginated()`](../src/ingest/paper_source.py)
   pages through arXiv's Atom API, honoring their documented 3-second
   courtesy delay between calls (not a guess — it's arXiv's own stated
   rate-limit etiquette).
2. **Parse** — [`src/ingest/document_parser.py::parse_pdf()`](../src/ingest/document_parser.py)
   downloads the PDF and splits it into sections using a font-size
   heuristic (a line meaningfully larger than the document's own body
   text, and shaped like a heading, counts as one). This is a coarse
   first cut, documented as such — see the README for exactly where it
   under-performs and why that wasn't hand-tuned away.
3. **Chunk** — [`src/ingest/chunker.py::chunk_document()`](../src/ingest/chunker.py)
   splits each section into ~1600-character windows with 200 characters
   of overlap, never crossing a section boundary. Every chunk's ID is
   **content-addressed**: `sha256(arxiv_id + chunk_text)` — not a
   position number. This means identical text (a repeated caption, a
   bare page number) legitimately produces the same ID twice; the
   pipeline deduplicates on this by design, not as a bug workaround.
4. **Persist to Postgres first** — `Paper` and `Chunk` rows
   ([`src/store/schema.py`](../src/store/schema.py)) are committed before
   anything touches OpenSearch. If the run fails after this point (e.g.
   no embedding key), the relational data is already safe.
5. **Embed + index** — [`src/index/embedder.py::embed_passages()`](../src/index/embedder.py)
   embeds the new chunks (Jina, `retrieval.passage` mode — see §5 for
   why the task string matters), and the vectors are written into
   OpenSearch alongside the text/metadata fields.

---

## 4. Guardrails — the full picture

Two axes, and they're independent of each other:

**Input vs. output** — where in the flow the check sits (before vs.
after the LLM does anything). **Fail-open vs. fail-closed** — what
happens if the *check itself* breaks.

| Guardrail | Side | Fails... | Why |
|---|---|---|---|
| `size_and_shape` | input | open | A broken input guardrail must not become a denial-of-service |
| `scope_screening` | input | open | Same reasoning |
| `groundedness` | output | closed | A broken output guardrail must not let an unverified answer through |
| `citation_integrity` | output | *(always runs, no mode)* | Correctness fix, not a block/allow decision |

**Three states exist in the design, two are wired up right now**: `off`
/ `monitor` / `enforce`. Every guardrail except `citation_integrity`
reads its mode from an env var (`GUARDRAIL_<NAME>_MODE`, default
`monitor`). In `monitor` mode, the check runs, gets traced, and is
visible in the `StageTrace` — but never changes the response. Switch
to `enforce` and it actually blocks. A persisted, hot-reloadable
three-state switch with a TTL auto-revert (the full design in the plan)
isn't built — this is the minimal version that makes "every new
guardrail starts in monitor" true today.

**How to verify this yourself**: `tests/unit/test_guardrails.py` has a
dedicated test for every row in that table — both the pass and fail
path, and for `groundedness` specifically, both the deterministic-pass
path (no network call, asserted via a mock that must not be called) and
the judge-escalation path (mocked YES and mocked NO). `tests/unit/
test_pipeline_orchestration.py` separately proves the *mode* behavior:
one test confirms `out_of_scope` does **not** stop the pipeline in
default `monitor` mode, a second confirms it **does** stop it once
`GUARDRAIL_SCOPE_SCREENING_MODE=enforce` is set.

---

## 5. Hybrid search & indexing — the mechanics

**Mapping**: [`src/index/mapping.py::build_index_body()`](../src/index/mapping.py).
One OpenSearch index, `rag_chunks`, with lexical and vector fields living
side by side on the same document:

- `text`, `title` — `text` type, English analyzer (stemming, stopwords)
- `embedding` — `knn_vector`, 1024 dimensions, HNSW/Lucene engine,
  cosine similarity
- `authors` — `keyword` **with an analyzed `.text` sub-field**. This
  exists because an LLM-extracted filter like `"LeCun"` needs to match a
  stored `"Yann LeCun"` — a plain keyword field only does exact
  full-string matches, which would silently break every partial-name
  filter. This was a real gap, found and fixed while building the
  filtering logic (see the README's A2/hybrid section for the full
  story).
- `category`, `published_date` — exact-match `keyword` and `date`,
  correct as-is for arXiv category codes and date ranges

**The dimension is a one-way door**: it's locked in
[`src/config/models.lock.yaml`](../src/config/models.lock.yaml), and both
`mapping.py` and the embedder read from that single file — changing the
embedding model later means a full re-embed and re-index, not a config
edit.

**Metadata filtering**: `src/retrieve/hybrid.py::build_filter_clauses()`
turns `QueryPlan.filters` into real OpenSearch `bool.filter` clauses,
applied identically to every lexical and dense sub-query. This is worth
calling out specifically because it did **not** always exist — A1 was
extracting filters from questions for a while before anything actually
consumed them. Verified via `grep` before assuming otherwise.

---

## 6. How the LLM makes decisions — summary table

Every point in this system where a model output changes what happens
next:

| # | Decision | File | Model role | Input it sees | Output that matters |
|---|---|---|---|---|---|
| 1 | Understand the question | `query_planner.py` | `rewrite` | The raw question | `intent`, `expansions`, `filters`, `normalized` — all four change downstream behavior |
| 2 | Write the answer | `generate.py` | `generate` | The question + 8 numbered context passages | The answer text itself, plus which `[N]` markers appear in it (→ citations) |
| 3 | Judge groundedness | `output_guardrails.py` | `judge` | The answer + the context it was given | A binary YES/NO — only called when the cheap deterministic check is ambiguous |

Every one of these calls goes through the *same* function —
`src/platform/models.py::complete()` — which is also where `model_served`
(what the provider actually used, which can differ from what was
requested if a ladder ever descends) gets recorded as a trace attribute.
No call site ever hardcodes a model name; they all ask for a *role* and
the registry resolves it (`src/config/models.yaml`).

---

## 7. Chunking strategy — what's decided vs. what's measured

**Decided now, provisionally**: fixed-size character windows (1600
chars, 200 overlap), never crossing a section boundary. This is written
explicitly as *one candidate strategy*, not *the* answer — the plan
calls for a real ablation (comparing this against recursive-separator,
document-structure-aware, and semantic-boundary chunking) before this
gets treated as a final decision. That ablation needs human-labeled
document-level relevance judgments, which don't exist yet (see the
README's A7 section for exactly why that's a genuine blocker, not
laziness).

**What IS measured**: the chunker's mechanics (unique content-addressed
IDs, correct overlap, correct offsets, correct within-paper
deduplication) — `tests/unit/test_chunker.py`, all against real
`ParsedDocument` structures, not the *quality* of the chunking strategy
itself.

---

## 8. Observability — what's traced and where to look

**Instrumented with OpenTelemetry** (`src/platform/telemetry.py`), not a
vendor SDK directly — the code is portable to any OTel-compatible
backend; only the exporter configuration points at Opik Cloud
specifically.

**Every span that exists right now**, and what's on it:

| Span | Emitted by | Key attributes |
|---|---|---|
| `reason.answer_query` | `pipeline.py` | `reason.intent`, `reason.num_retrieved`, `reason.num_context`, `guardrail.*.passed` for all four guardrails |
| `query_planner.plan_query` | `query_planner.py` | `query_planner.cache_hit`, `query_planner.intent`, `query_planner.num_expansions` |
| `hybrid.retrieve` | `hybrid.py` | `hybrid.has_filters`, `hybrid.num_query_variants`, `hybrid.num_results` |
| `reranker.rerank` | `reranker.py` | `reranker.backend`, `reranker.degraded`, `reranker.model_served` |
| `generate.answer` | `generate.py` | `generate.abstained`, `generate.num_citations`, `generate.model_served` |
| `embed.request` | `embedder.py` | `embed.task`, `embed.dimension`, `embed.prompt_tokens` |
| `llm.complete` | `models.py` | `llm.role`, `llm.provider`, `llm.model_served` |

**These nest correctly** — e.g. `query_planner.plan_query` is a real
parent of `llm.complete` when it calls the model, verified with a
dedicated test that checks the child span's `parent_span_id` actually
matches the parent's `span_id` (not just that both spans happen to
exist). This is what makes a trace a *tree* you can read, not a flat
pile of unrelated events.

**Where to actually watch this**: once `OPIK_API_KEY`/`OPIK_WORKSPACE`
are set, every one of these spans streams to your Opik Cloud project in
real time — that's the real trace viewer, and rebuilding one in this
project's own UI was explicitly avoided (see §9). Until then, spans are
still created (nothing crashes without the key) but have nowhere to
export to — `tests/unit/test_telemetry.py` proves this with a real
in-memory OpenTelemetry exporter, not a mock.

**Guardrail firings** get an additional signal: a *trace event* (not a
full span) is attached to whatever span is currently active, but only
when a guardrail actually fails — a passing guardrail is silent, so a
guardrail firing frequently is visible as a real signal rather than
noise. See `src/guardrails/base.py::record_guardrail_event()`.

---

## 9. Where to watch what — a practical guide

| You want to... | Go here |
|---|---|
| Ask the system a real question | `/ask` |
| See exactly how one question flowed through every stage, with real numbers | `/pipeline?query=...` |
| Replay a *past* answer exactly, not re-run it live | `/pipeline?run_id=<uuid>` — the link is on every `/ask` answer and every live `/pipeline` run |
| See whether the pipeline self-corrected on a given run | The "9. Self-correction" panel on `/pipeline` — only appears when a retry actually happened |
| See what's actually been ingested | `/corpus` |
| See live traces | <https://www.comet.com/opik/ai-eductation/get-started> → project `production-rag-system` (247+ real confirmed traces) |
| Check whether a guardrail is blocking or just watching | The `GUARDRAIL_*_MODE` env vars in `.env` |
| Change how many self-correction attempts are allowed | `RAG_MAX_ATTEMPTS` env var (default 2) |
| See the exact prompt an LLM call uses | `_SYSTEM_PROMPT` in `query_planner.py` or `generate.py` — read the real string, not a summary |
| Verify retrieval math is correct | `tests/unit/test_hybrid.py` (pure RRF/filter-clause tests, no network) |
| Verify the whole pipeline wires together correctly | `tests/unit/test_pipeline_orchestration.py` |
| Verify against a real, published benchmark (not our own assumptions) | `python -m evals.validate_harness` |
| See real ingested data end to end | `docker exec` into Postgres, or just open `/corpus` |

---

## 10. How to verify each piece — mapped to real tests

Run these yourself:

```bash
pytest -m unit                # every guard clause, every pure function, no network, no docker
pytest -m integration         # needs `docker compose up -d postgres opensearch redis` running
pytest                        # both, 180 tests total
```

| Component | Test file | What it actually proves |
|---|---|---|
| Guardrails (all 4) | `test_guardrails.py` | Pass/fail paths, fail-open vs. fail-closed, judge escalation, `errored` vs. genuine-fail distinction |
| Pipeline orchestration | `test_pipeline_orchestration.py` | Stage ordering, monitor-vs-enforce mode switching, self-correction retry/give-up/no-retry-on-error |
| Query planning | `test_query_planner.py` | JSON parsing, cache key stability, cache hit/miss behavior |
| Hybrid search math | `test_hybrid.py` | RRF fusion arithmetic against hand-computed values, filter-clause shapes |
| Hybrid search, live | `test_hybrid_integration.py` | Filters genuinely restrict real OpenSearch results (author/category/date, combined) |
| Reranking | `test_reranker.py` | Degradation on timeout/missing key, correct index-to-candidate mapping on out-of-order responses |
| Chunking | `test_chunker.py` | Content-addressed IDs, overlap math, section boundaries |
| Generation | `test_generate.py` | Citation extraction only includes markers actually present, out-of-range markers ignored not crashed |
| Observability | `test_telemetry.py` | Real spans via a real in-memory OTel exporter, correct parent-child nesting |
| Run persistence | `test_runs_integration.py` | A real `StageTrace` (dataclasses nesting Pydantic models) round-trips through Postgres JSON correctly |
| Benchmark validation | `test_validate_harness.py` + `python -m evals.validate_harness` | Our own retrieval-scoring math checked against a published number (BEIR SciFact), not just our own assumptions |

---

## 10.5. Real eval results (2026-08-19, cleanest run)

Headline numbers only — the full ablation tables, the honesty caveats
about the qrels' vocabulary bias, and the "tried it and it made things
worse" prompt experiment all live in `README.md`'s A3/A4 sections; this
is a pointer, not a duplicate.

| What | Result |
|---|---|
| Retrieval, full pipeline (recall@10 / nDCG@10) | 0.888 / 0.829 — best of all 5 configs, 0 of 80 questions failed |
| Retrieval, plan's own gate (≥15% relative nDCG@10 over BM25, p95 <800ms) | **Not met** — +1.4% relative, 10.3s p95 |
| Ragas faithfulness / context recall / context precision / answer relevancy | 0.982 / 0.994 / 0.902 / 0.780 |
| Guardrail pass rate, 247 genuine real traces | 100% on all 4 |

---

## 11. What's deliberately NOT covered here

This document explains **mechanics** — how the pieces that exist work.
It does not restate:
- What's built vs. scaffolded (→ `README.md`)
- Free-tier limits, cost posture, procurement (→ `PLAN_production_rag_system.md`)
- Production-readiness gaps (→ `README.md`'s R1-R6 audit)

Those all change faster than the mechanics do — this document is meant
to stay accurate even as keys get added and readiness work continues.
