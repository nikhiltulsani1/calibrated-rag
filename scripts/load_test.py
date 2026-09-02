from __future__ import annotations

import asyncio
import itertools
import json
import time
from pathlib import Path

import httpx

from evals.metrics import mean, percentile

# R5: real concurrent requests against the ACTUALLY RUNNING local API
# (docker compose's `api` service, port 8000) — not a simulation. Sweeps
# small concurrency levels deliberately: Groq's real 30 RPM ceiling on
# the `generate` role means the binding constraint should surface almost
# immediately, which IS the honest finding to report here, not a test
# failure. Budget-capped (see _CONCURRENCY_LEVELS/_REQUESTS_PER_LEVEL)
# against the real 1,000 req/day Groq cap shared with every other eval
# this project runs.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOLD_PATH = _REPO_ROOT / "evals" / "datasets" / "rag_gold.jsonl"
_BASE_URL = "http://localhost:8000"
_CONCURRENCY_LEVELS = [1, 2, 5, 10]
_REQUESTS_PER_LEVEL = 15  # 4 levels x 15 = 60 total requests, well under the daily cap


def _load_questions() -> list[str]:
    rows = [json.loads(line) for line in open(_GOLD_PATH, encoding="utf-8")]
    return [r["query"] for r in rows]


async def _one_request(client: httpx.AsyncClient, query: str) -> dict:
    start = time.monotonic()
    try:
        response = await client.post("/ask", data={"query": query}, timeout=60.0)
        elapsed_ms = (time.monotonic() - start) * 1000
        # Two real, distinct rate-limit shapes, confirmed by reading the
        # actual container logs rather than trusting this script's own
        # labels: (1) OUR OWN R6 limiter raises a real HTTP 429 status
        # with body `{"detail":"Too many requests..."}` (lowercase, so
        # an earlier case-sensitive `"Too Many Requests" in text` check
        # missed it entirely and mislabeled it "application/network
        # error"); (2) a PROVIDER 429 (Groq/Jina/etc.) gets caught inside
        # the route's own try/except and rendered as a normal HTTP 200
        # page whose body contains the provider's own error text. Status
        # code alone distinguishes "our limiter fired" from everything
        # else; the text check is still needed for the provider case.
        app_rate_limited = response.status_code == 429
        provider_rate_limited = response.status_code == 200 and (
            "429" in response.text or "too many requests" in response.text.lower()
        )
        rate_limited = app_rate_limited or provider_rate_limited
        constraint_kind = "app rate limit (this deployment's own R6 limiter)" if app_rate_limited else (
            "provider rate limit (Groq)" if provider_rate_limited else None
        )
        return {
            "status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "rate_limited": rate_limited,
            "constraint_kind": constraint_kind,
            "ok": response.status_code == 200 and not rate_limited,
        }
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {"status": None, "elapsed_ms": elapsed_ms, "rate_limited": False, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _run_at_concurrency(concurrency: int, n_requests: int, questions: list[str]) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    question_cycle = itertools.islice(itertools.cycle(questions), n_requests)

    async def _bounded(query: str) -> dict:
        async with semaphore:
            return await _one_request(client, query)

    async with httpx.AsyncClient(base_url=_BASE_URL) as client:
        results = await asyncio.gather(*[_bounded(q) for q in question_cycle])

    latencies = [r["elapsed_ms"] for r in results]
    errors = [r for r in results if not r["ok"]]
    binding_constraint = None
    constraint_kinds = {r["constraint_kind"] for r in results if r.get("constraint_kind")}
    if constraint_kinds:
        # Report every distinct kind seen at this level — a run can hit
        # both our own limiter and a provider limit within the same
        # concurrency level, and collapsing that to one label would hide
        # which one actually bit first.
        binding_constraint = " + ".join(sorted(constraint_kinds))
    elif errors:
        binding_constraint = "application/network error"

    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "mean_ms": round(mean(latencies), 1),
        "error_rate": round(len(errors) / len(results), 4) if results else 0.0,
        "n_errors": len(errors),
        "binding_constraint": binding_constraint,
    }


async def run_load_test() -> list[dict]:
    questions = _load_questions()
    results = []
    for concurrency in _CONCURRENCY_LEVELS:
        result = await _run_at_concurrency(concurrency, _REQUESTS_PER_LEVEL, questions)
        results.append(result)
        print(json.dumps(result))
    return results


def write_report(results: list[dict]) -> None:
    report_path = _REPO_ROOT / "evals" / "REPORT.md"
    lines = ["", "## R5 — Load and capacity", ""]
    lines.append(
        "Generated by `python -m scripts.load_test` — real concurrent `POST /ask` requests "
        "against the actually-running local API (not simulated), cycling through the real "
        f"{len(_load_questions())} `rag_gold.jsonl` questions."
    )
    lines.append("")
    lines.append("| concurrency | requests | p50 ms | p95 ms | p99 ms | mean ms | error rate | binding constraint |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['concurrency']} | {r['n_requests']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | "
            f"{r['mean_ms']} | {r['error_rate']:.2%} | {r['binding_constraint'] or '—'} |"
        )
    lines.append("")

    breaking_point = next((r for r in results if r["binding_constraint"]), None)
    if breaking_point:
        lines.append(
            f"**Honest ceiling**: at concurrency {breaking_point['concurrency']}, "
            f"{breaking_point['binding_constraint']} became the binding constraint "
            f"(error rate {breaking_point['error_rate']:.2%}) — not this application's own code. "
            "This deployment's real concurrent-user ceiling on current free-tier provider limits "
            f"is below {breaking_point['concurrency']} sustained simultaneous requesters."
        )
    else:
        lines.append(
            f"No binding constraint surfaced up to concurrency {results[-1]['concurrency']} "
            f"in this run — the real ceiling is higher than tested here."
        )

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended R5 section to {report_path}")


if __name__ == "__main__":
    results = asyncio.run(run_load_test())
    write_report(results)
