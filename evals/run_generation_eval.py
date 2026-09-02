from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from evals.metrics import mean
from src.reason.pipeline import run_traced_query

# A4: generation-quality eval — faithfulness, answer relevancy, context
# precision, context recall. Distinct from A3's retrieval-quality eval:
# that one asks "did we find the right chunk," this one asks "is the
# actual written answer trustworthy." Runs the REAL pipeline
# (run_traced_query — the same function /ask and /pipeline call) per
# question, not a reimplementation, so this measures the system as it
# actually behaves, not an idealized version of it.

_GOLD_PATH = Path(__file__).resolve().parent / "datasets" / "rag_gold.jsonl"
_REPORT_PATH = Path(__file__).resolve().parent / "REPORT_generation.md"


def load_gold() -> list[dict]:
    return [json.loads(line) for line in open(_GOLD_PATH, encoding="utf-8")]


def _build_llm():
    """The judge role's own model (NVIDIA nemotron-3-ultra), reused here
    rather than picking a fifth model — it's already the project's
    designated LLM-judge, and Ragas' metrics are exactly that: an LLM
    judging another LLM's output. Verified live (2026-08-16) that
    ragas' llm_factory needs an *async* OpenAI-compatible client
    (agenerate(), not generate()) — a sync client raises immediately.

    max_tokens=4096 overrides ragas' own default of 1024, which is
    ragas' own documented gap for reasoning models (see llm_factory's
    docstring: "Default max_tokens=1024 may not be sufficient... If
    structured output is truncated, increase max_tokens further").
    Confirmed live: 14 of a real 31-question run hit
    IncompleteOutputException at the 1024 default, since nemotron-3
    spends tokens on its own reasoning before the structured verdict.
    """
    import openai
    from ragas.llms import llm_factory

    client = openai.AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )
    return llm_factory(
        model="nvidia/nemotron-3-ultra-550b-a55b", provider="openai", client=client, max_tokens=4096
    )


def _make_jina_embedding():
    """Thin adapter so Ragas' AnswerRelevancy metric can use this
    project's own Jina embedder (src/index/embedder.py) instead of
    requiring an OpenAI-shaped embeddings client — Jina's real API has a
    different request/response shape (see embedder.py's own history of
    that), so reusing our already-verified client is more honest than
    writing a second, untested Jina integration just for this file.
    Built as a factory (not a module-level class) so the ragas import
    stays lazy/optional, matching this file's other imports.
    """
    from ragas.embeddings.base import BaseRagasEmbedding
    from src.index.embedder import embed_query

    class JinaRagasEmbedding(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return embed_query(text).vectors[0]

        async def aembed_text(self, text, **kwargs):
            return embed_query(text).vectors[0]

    return JinaRagasEmbedding()


async def _score_one(metrics: dict, user_input: str, response: str, retrieved_contexts: list[str], reference: str) -> dict:
    faithfulness = (await metrics["faithfulness"].ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )).value
    answer_relevancy = (await metrics["answer_relevancy"].ascore(
        user_input=user_input, response=response
    )).value
    context_precision = (await metrics["context_precision"].ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )).value
    context_recall = (await metrics["context_recall"].ascore(
        user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference
    )).value
    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


async def run_all() -> dict:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    gold = load_gold()
    llm = _build_llm()
    embeddings = _make_jina_embedding()
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    per_question = []
    failures = []
    abstained = []
    for row in gold:
        query_id, query, reference = row["query_id"], row["query"], row["reference"]
        # Pacing, same lesson as run_retrieval_eval.py's full-pipeline
        # config: an unpaced loop here does TWO real Groq calls per
        # question (rewrite + generate) plus the ragas judge calls below,
        # roughly double the retrieval eval's per-question Groq load —
        # confirmed live: an unpaced run here hit Groq's rate limit on
        # 25 of 31 questions. 6s keeps this comfortably under the ~30
        # RPM ceiling verified elsewhere in this project for Groq's free
        # tier, at double the load per iteration.
        time.sleep(6.0)
        try:
            trace = run_traced_query(query)
            if trace.answer is None:
                failures.append((query_id, "no answer produced (stopped_at=" + str(trace.stopped_at) + ")"))
                continue
            if trace.answer.abstained:
                abstained.append(query_id)
                continue
            response = trace.answer.text
            retrieved_contexts = [item.text for item in trace.reranked.items] if trace.reranked else []
            if not retrieved_contexts:
                failures.append((query_id, "no retrieved context to score against"))
                continue
            scores = await _score_one(metrics, query, response, retrieved_contexts, reference)
            per_question.append({"query_id": query_id, "query": query, "response": response, "reference": reference, **scores})
        except Exception as exc:
            failures.append((query_id, f"{type(exc).__name__}: {exc}"))

    aggregate = {}
    if per_question:
        for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            aggregate[metric_name] = mean([q[metric_name] for q in per_question])

    return {
        "n_total": len(gold),
        "n_scored": len(per_question),
        "n_abstained": len(abstained),
        "n_failed": len(failures),
        "aggregate": aggregate,
        "per_question": per_question,
        "abstained": abstained,
        "failures": failures,
    }


def write_report(result: dict) -> None:
    lines = ["# Generation quality eval report (Ragas)", ""]
    lines.append(
        "Generated by `python -m evals.run_generation_eval`. Every question runs "
        "through the real pipeline (`run_traced_query` — the same function `/ask` "
        "and `/pipeline` call), not a reimplementation. Scored by "
        "`nvidia/nemotron-3-ultra-550b-a55b` (this project's own judge-role model) "
        "via real Ragas metrics — no numbers here are estimated or simulated."
    )
    lines.append("")
    lines.append(
        f"**{result['n_scored']} of {result['n_total']} questions scored** "
        f"({result['n_abstained']} abstained — no answer to score faithfulness of; "
        f"{result['n_failed']} failed outright, see below)."
    )
    lines.append("")
    if result["aggregate"]:
        lines.append("| metric | mean score |")
        lines.append("|---|---|")
        for name, value in result["aggregate"].items():
            lines.append(f"| {name} | {value:.4f} |")
    else:
        lines.append("No questions scored — nothing to aggregate.")
    lines.append("")
    lines.append(
        "**Reference answers are LLM-bootstrapped, not human-verified** — same "
        "caveat as `evals/datasets/qrels.jsonl` (`rag_gold.jsonl` reuses that "
        "file's `note` field as the ground-truth answer). Treat these numbers "
        "as a first real read, not a certified gate."
    )
    if result["abstained"]:
        lines.append("")
        lines.append(f"**Abstained** ({len(result['abstained'])}): " + ", ".join(result["abstained"]))
    if result["failures"]:
        lines.append("")
        lines.append("**Failures:**")
        for query_id, reason in result["failures"]:
            lines.append(f"  - `{query_id}`: {reason}")

    if result["per_question"]:
        lines.append("")
        lines.append("## Per-question detail")
        lines.append("")
        lines.append("| query_id | faithfulness | answer_relevancy | context_precision | context_recall |")
        lines.append("|---|---|---|---|---|")
        for q in sorted(result["per_question"], key=lambda q: q["answer_relevancy"]):
            lines.append(
                f"| {q['query_id']} | {q['faithfulness']:.3f} | {q['answer_relevancy']:.3f} | "
                f"{q['context_precision']:.3f} | {q['context_recall']:.3f} |"
            )
        lines.append("")
        lines.append("### Lowest answer_relevancy — question, real answer, reference")
        for q in sorted(result["per_question"], key=lambda q: q["answer_relevancy"])[:5]:
            lines.append("")
            lines.append(f"**{q['query_id']}** (answer_relevancy={q['answer_relevancy']:.3f})")
            lines.append(f"- Question: {q['query']}")
            lines.append(f"- Real answer: {q['response']}")
            lines.append(f"- Reference: {q['reference']}")

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = asyncio.run(run_all())
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_question",)}, indent=2, default=str))
    write_report(result)
    print(f"\nwrote {_REPORT_PATH}")
