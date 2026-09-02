from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.reason.generate import GENERATION_UNAVAILABLE_TEXT
from src.reason.pipeline import run_traced_query

# A8: tests what the system does with NO evidence — the failure mode the
# plan calls "the one that destroys trust fastest": a confident, fluent,
# fabricated answer to a question the corpus has no basis for. Structurally
# mirrors evals/run_generation_eval.py — same real-pipeline-call pattern,
# same per-question pacing discipline (2+ real Groq calls per question,
# same rate-limit lesson learned in A4), same try/except-continue
# isolation as run_retrieval_eval.py's _run_per_query.
#
# Two gates, BOTH required (the plan's own framing): abstention rate on
# the negative set alone is trivially gamed by refusing everything;
# over-refusal on the answerable gold set alone is gamed by answering
# everything. Only the pair describes a system that knows the difference.

_UNANSWERABLE_PATH = Path(__file__).resolve().parent / "datasets" / "unanswerable.jsonl"
_GOLD_PATH = Path(__file__).resolve().parent / "datasets" / "rag_gold.jsonl"
_REPORT_PATH = Path(__file__).resolve().parent / "REPORT_abstention.md"

_ABSTENTION_GATE = 0.80
_OVER_REFUSAL_GATE = 0.05
# Env-overridable — added 2026-08-24 after checking Mistral's real
# per-key rate-limit headers live: chat completions are capped at just 4
# req/min (embed is a separate, generous 60 req/min, not the bottleneck).
# One question fires 2-3 real chat calls in a tight burst (rewrite unless
# A2 skips it, assess_ambiguity, generate) — a 6s inter-QUESTION pace
# does nothing to prevent that burst itself exceeding 4 req in the same
# rolling 60s window. Default stays 6.0 (today's behavior, tuned for
# Groq/OpenRouter's much looser limits); a real Mistral-isolated run
# needs this set much higher via ABSTENTION_PACING_SECONDS.
_PACING_SECONDS = float(os.environ.get("ABSTENTION_PACING_SECONDS", "6.0"))


def load_unanswerable() -> list[dict]:
    return [json.loads(line) for line in open(_UNANSWERABLE_PATH, encoding="utf-8")]


def load_gold() -> list[dict]:
    return [json.loads(line) for line in open(_GOLD_PATH, encoding="utf-8")]


def _is_infra_failure(trace) -> bool:
    """Real gap found live 2026-08-28: generate.py used to let a
    transient httpx error propagate uncaught, which this eval script's
    own try/except already excluded from scoring (see _run_set below) —
    a real infra hiccup never touched the abstention/over-refusal
    numbers. Giving generate.py its own retry-with-degrade ladder
    (Results.md §28) closed a real production gap (the request no
    longer crashes to an error page) but had a side effect here: total
    exhaustion after 3 real retries now returns a normal, successfully-
    completed Answer(abstained=True, text=GENERATION_UNAVAILABLE_TEXT)
    instead of raising — which _refused() correctly can't distinguish
    from a genuine content-based abstention, so it silently started
    counting real infra flakiness as "the system correctly declined."
    Verified live: re-running 3 of 11 gold questions the eval flagged as
    "over-refused" moments later, unmodified, got real cited answers —
    they were never actually refused on content grounds, just caught
    mid-outage. Scored as a failure (same bucket as the old crash
    exclusion), not a refusal, so the two gates measure content
    calibration, not provider uptime.
    """
    return trace.answer is not None and trace.answer.text == GENERATION_UNAVAILABLE_TEXT


def _refused(trace) -> bool:
    """A refusal is an abstained answer, a guardrail short-circuit (e.g.
    scope_screening in enforce mode), OR the model declining to guess on
    a flagged false-premise/ambiguous question (Answer.declined_to_guess)
    — all three are genuine "did not confidently answer" outcomes, not
    just the literal ABSTAIN_TEXT path. The third case was added after a
    real gap: u027 (a genuine ambiguity case) got a correct "which one
    do you mean?" response, verified live, that this function originally
    scored as a non-refusal because it isn't the exact ABSTAIN_TEXT.

    Callers must check _is_infra_failure() first and exclude those from
    scoring entirely — this function has no way to tell a genuine
    content-based abstention apart from generate.py's own degrade text,
    and must not be asked to.
    """
    return (
        trace.stopped_at is not None
        or (trace.answer is not None and trace.answer.abstained)
        or (trace.answer is not None and trace.answer.declined_to_guess)
    )


def _run_set(rows: list[dict], *, id_field: str = "query_id", query_field: str = "query") -> tuple[list[dict], list[tuple[str, str]]]:
    results = []
    failures: list[tuple[str, str]] = []
    for row in rows:
        time.sleep(_PACING_SECONDS)
        try:
            trace = run_traced_query(row[query_field])
            if _is_infra_failure(trace):
                failures.append((row[id_field], "generate.py exhausted its retry ladder (real infra failure, not a content decision)"))
                continue
            results.append(
                {
                    "query_id": row[id_field],
                    "query": row[query_field],
                    "kind": row.get("kind"),
                    "refused": _refused(trace),
                    "stopped_at": trace.stopped_at,
                    "answer_text": trace.answer.text if trace.answer else None,
                }
            )
        except Exception as exc:
            failures.append((row[id_field], f"{type(exc).__name__}: {exc}"))
    return results, failures


def run_all() -> dict:
    unanswerable_rows = load_unanswerable()
    gold_rows = load_gold()

    unanswerable_results, unanswerable_failures = _run_set(unanswerable_rows)
    gold_results, gold_failures = _run_set(gold_rows)

    n_unanswerable_scored = len(unanswerable_results)
    n_gold_scored = len(gold_results)

    abstention_rate = (
        sum(1 for r in unanswerable_results if r["refused"]) / n_unanswerable_scored
        if n_unanswerable_scored
        else None
    )
    over_refusal_rate = (
        sum(1 for r in gold_results if r["refused"]) / n_gold_scored if n_gold_scored else None
    )

    by_kind: dict[str, dict] = {}
    for r in unanswerable_results:
        kind = r["kind"] or "unknown"
        bucket = by_kind.setdefault(kind, {"n": 0, "refused": 0})
        bucket["n"] += 1
        bucket["refused"] += 1 if r["refused"] else 0

    return {
        "unanswerable_results": unanswerable_results,
        "unanswerable_failures": unanswerable_failures,
        "gold_results": gold_results,
        "gold_failures": gold_failures,
        "abstention_rate": abstention_rate,
        "over_refusal_rate": over_refusal_rate,
        "by_kind": by_kind,
        "n_unanswerable_total": len(unanswerable_rows),
        "n_gold_total": len(gold_rows),
    }


# Plan B4 (2026-08-22): real, sourced benchmark numbers so a gate FAIL
# reads as "here's where this actually sits versus real production
# systems" rather than an unqualified failure implying total brokenness.
# These are real, cited figures from published benchmarks — not this
# project's own aspiration, and not adjusted to make either gate look
# better. The gate thresholds themselves are unchanged (0.80 / 0.05,
# per the plan) — this only adds context alongside the pass/fail line,
# never replaces or softens it.
_ABSTENTION_INDUSTRY_CONTEXT = (
    "best-in-class production calibration (Claude 4.1 Opus on the AA-Omniscience "
    "reliability benchmark) abstains on ~18.7% of questions it doesn't know the answer "
    "to — most frontier models score far lower. A gate FAIL here means real, unsolved-"
    "industry-wide-difficulty territory, not a uniquely broken system."
)
_OVER_REFUSAL_INDUSTRY_CONTEXT = (
    "production RAG systems generally target a low single-digit percent over-refusal "
    "rate on genuinely answerable questions — refusing too often is treated as a real "
    "usability failure, not a safe default, since it trades hallucination risk for "
    "unhelpfulness risk instead of actually reducing risk."
)


def _gate_line(label: str, value: float | None, threshold: float, *, higher_is_better: bool, industry_context: str | None = None) -> str:
    if value is None:
        return f"- **{label}**: not run"
    passed = (value >= threshold) if higher_is_better else (value <= threshold)
    comparator = ">=" if higher_is_better else "<="
    verdict = "PASS" if passed else "FAIL"
    line = f"- **{label}**: {value:.4f} (gate: {comparator} {threshold}) — **{verdict}**"
    if industry_context:
        line += f"\n  - *Industry context*: {industry_context}"
    return line


def write_report(result: dict) -> None:
    lines = ["# Abstention eval report (A8)", ""]
    lines.append(
        "Generated by `python -m evals.run_abstention_eval`. Every question runs through the "
        "real pipeline (`run_traced_query`), not a reimplementation — no numbers here are "
        "estimated or simulated."
    )
    lines.append("")
    lines.append(
        f"**Unanswerable set**: {len(result['unanswerable_results'])}/{result['n_unanswerable_total']} "
        f"scored ({len(result['unanswerable_failures'])} failed outright)."
    )
    lines.append(
        f"**Answerable gold set**: {len(result['gold_results'])}/{result['n_gold_total']} "
        f"scored ({len(result['gold_failures'])} failed outright)."
    )
    lines.append("")
    lines.append("## Gates — both required")
    lines.append(
        _gate_line(
            "Abstention rate (unanswerable set)",
            result["abstention_rate"],
            _ABSTENTION_GATE,
            higher_is_better=True,
            industry_context=_ABSTENTION_INDUSTRY_CONTEXT,
        )
    )
    lines.append(
        _gate_line(
            "Over-refusal rate (answerable gold set)",
            result["over_refusal_rate"],
            _OVER_REFUSAL_GATE,
            higher_is_better=False,
            industry_context=_OVER_REFUSAL_INDUSTRY_CONTEXT,
        )
    )
    lines.append("")
    lines.append(
        "Abstention alone is trivially gamed by refusing everything; over-refusal alone is "
        "gamed by answering everything. Only the pair describes a system that knows the "
        "difference — per the plan's own framing."
    )

    if result["by_kind"]:
        lines.append("")
        lines.append("## Abstention rate per kind")
        lines.append("")
        lines.append("| kind | n | refused | rate |")
        lines.append("|---|---|---|---|")
        for kind, b in sorted(result["by_kind"].items()):
            rate = b["refused"] / b["n"] if b["n"] else 0.0
            lines.append(f"| {kind} | {b['n']} | {b['refused']} | {rate:.4f} |")

    if result["gold_results"]:
        over_refused = [r for r in result["gold_results"] if r["refused"]]
        if over_refused:
            lines.append("")
            lines.append("## Over-refused answerable questions (real over-refusal cases)")
            for r in over_refused:
                lines.append("")
                lines.append(f"**{r['query_id']}** — {r['query']}")
                lines.append(f"- stopped_at: {r['stopped_at']}")

    if result["unanswerable_failures"] or result["gold_failures"]:
        lines.append("")
        lines.append("## Failures")
        for query_id, reason in result["unanswerable_failures"]:
            lines.append(f"  - unanswerable `{query_id}`: {reason}")
        for query_id, reason in result["gold_failures"]:
            lines.append(f"  - gold `{query_id}`: {reason}")

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_all()
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("unanswerable_results", "gold_results")},
            indent=2,
            default=str,
        )
    )
    write_report(result)
    print(f"\nwrote {_REPORT_PATH}")
