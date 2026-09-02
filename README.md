# Calibrated RAG

A question-answering system that reads a collection of computer-science
research papers and answers questions about them — with one thing built
in from the start that most systems like this skip: **it's designed to
know when it doesn't know.**

Ask it something the papers genuinely answer, and it answers, with
citations back to the exact passage it used. Ask it something the papers
don't cover, or something built on a wrong assumption, and it says so
instead of guessing — because a confident, fluent, made-up answer is far
more dangerous than an honest "I don't have enough information for
that."

Every number below is from a real, measured test run against this exact
system — nothing here is an estimate. Full details, in plain language,
are in [`project-docs/`](project-docs/).

## What it does

1. You ask a question in plain English.
2. It searches the paper collection two different ways at once — by
   keyword and by meaning — to find the most relevant passages.
3. It re-checks and re-sorts those results for genuine relevance.
4. Before writing an answer, it checks: *is there actually enough here to
   answer this properly, and is the question itself asking something
   reasonable?*
5. If yes, it writes an answer and shows exactly which passages it used.
   If no, it says so plainly, instead of guessing.

If any of the outside AI services it depends on has a hiccup along the
way, it automatically tries a different one — and if everything is
genuinely unavailable, it tells you that plainly instead of crashing.

## How well it actually works

| What was measured | Result |
|---|---|
| Correctly recognizes a question it can't answer, and says so | **79%** of the time |
| Correctly answers a question it actually can | **96%** of the time (only declines the wrong questions ~4% of the time) |
| How grounded its answers are in the real source text | **97%+** faithfulness score |
| How well it finds the actually-relevant passages | **99%+** of the time, the right evidence is retrieved |

These come from real, repeatable test runs, including runs deliberately
conducted while the underlying AI services were having a bad day — see
[`project-docs/result.md`](project-docs/result.md) for the full history,
including what didn't work on the first try and what was fixed.

## Guardrails

Built into the pipeline are automated checks that watch for specific
ways an answer could go wrong — a question that contradicts what the
source papers actually say, a question that's genuinely ambiguous
between several different papers, an answer that isn't actually backed up
by the passages it cites. Some of these checks only watch and record what
they see; others can actively stop a bad answer before it ever reaches
you. This is the mechanism behind the "knows when it doesn't know"
behavior described above — it isn't the AI model being cautious on its
own, it's deliberate, testable checks built around it. See
[`project-docs/glossary.md`](project-docs/glossary.md#guardrail) for what
"guardrail" means in plain terms, and
[`project-docs/architecture.md`](project-docs/architecture.md) for
exactly where each check runs.

## Evaluation

Every claim this project makes about its own quality is backed by a real
test run, not a guess or a one-off example. Dedicated evaluation scripts
run the system against real question sets — including questions
deliberately chosen because they *can't* be answered from the paper
collection — and measure, honestly, how often it gets it right. When a
test run is confounded by something outside the system's control (an AI
provider having an outage mid-test, for example), that's recorded
plainly rather than hidden or quietly excluded. See
[`project-docs/result.md`](project-docs/result.md) for every real
evaluation run this project has produced, including the ones that didn't
go as planned.

## Observability

Every question that comes in is traced from start to finish: which steps
ran, what each guardrail decided and why, how long each stage took, and
which passages ended up in the final answer. This isn't just for
debugging — it's what makes every number and every claim in this project
checkable rather than taken on faith. The Pipeline page (see "Try it"
below) shows this trace for any question, live, in the browser.

## Try it

```bash
cp .env.example .env      # add your API keys — see project-docs/technical_details.md
docker compose up -d postgres opensearch redis api
```

Then open **http://localhost:8000** — there's an Ask page for asking a
question directly, a Pipeline page that shows every step the system took
to answer it, and a Corpus page showing what's actually in the paper
collection right now.

## Want to go deeper?

- **[`project-docs/glossary.md`](project-docs/glossary.md)** — every
  technical term used anywhere in this project, explained in plain
  language, no prior knowledge assumed.
- **[`project-docs/architecture.md`](project-docs/architecture.md)** —
  exactly how a question becomes an answer, step by step, with the real
  file and function names, for anyone who wants to read or modify the
  code. A visual version is in
  [`architecture.html`](project-docs/architecture.html).
- **[`project-docs/technical_details.md`](project-docs/technical_details.md)**
  — every moving part, how to run every optional piece, credentials
  needed, versions pinned, and the deliberate engineering decisions
  behind how failures are handled.
- **[`project-docs/result.md`](project-docs/result.md)** — the complete,
  honest record of every test run this project has ever done, including
  the ones that went wrong and what was learned from them. If you want to
  know exactly how the numbers above were produced, this is where they
  come from.
- **[`project-docs/development_history.md`](project-docs/development_history.md)**
  — the story of how this project got built, phase by phase, in plain
  language: what came in what order, what real problems turned up, and
  how they got fixed.

## What this project cares about

Most demo RAG systems stop at "can it answer questions." This one treats
that as the easy half. The harder, more important half is: **can it be
trusted to know the difference between a question it can answer well and
one it can't** — and does it keep working honestly when the outside
services it depends on are having a bad day, instead of silently failing
or silently guessing. Everything in this repository was built and
measured with that as the actual goal.
