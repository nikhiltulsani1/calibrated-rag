# Development history

This is the story of how Calibrated RAG got built — not a raw log, but
the essence of it: what was built in what order, what real bugs turned
up along the way, and the discipline that shaped every decision. For the
complete, unabridged run-by-run record with every real number, see
[`result.md`](result.md). For how the finished system actually works, see
[`architecture.md`](architecture.md) or [`technical_details.md`](technical_details.md).

## The one rule that shaped everything

Every claim in this project had to be checked against something real
before it was trusted — a real API call, a real database round-trip, a
real running container, a real measured number. Nothing was assumed to
work because it looked right on paper, and nothing was reported as done
until it had actually been run. That discipline is why this document (and
`result.md`) is full of sentences like "verified live, not just written
and assumed correct" — it wasn't a slogan, it was how every single
component got built, and it's what caught most of the real bugs described
below before they reached anyone.

## Phase 1 — the foundations (infrastructure, storage, search)

The system started with the three things everything else depends on:
Postgres as the single source of truth, OpenSearch as a rebuildable
search index derived from it, and Redis as a cache. Getting even this far
surfaced two real infrastructure bugs on the very first run — a
Postgres version had changed its data-directory convention, and the
API's health check was shelling out to a tool (`curl`) that didn't
exist in its own container image. Both were only caught by actually
starting the stack, not by reading the config.

From there, the ingestion pipeline came next: pulling real papers from
arXiv, parsing real PDFs, splitting them into chunks, and writing
everything to Postgres first (always) before anything touched the search
index. This surfaced its own real bugs — PDF text can contain characters
Postgres flatly rejects, and two chunks from the very same paper can
legitimately produce an identical ID, which needed handling before the
database would accept them.

## Phase 2 — making the system understand and search

With real documents in place, the system gained the ability to
understand a question before searching for an answer to it: normalizing
the phrasing, generating a few alternative ways to ask the same thing,
and pulling out anything like an author name, category, or date range the
question implied. That extraction step sat unused for a while — a real
gap caught by literally searching the codebase for anything that used it,
and finding nothing. Wiring it up into an actual search filter, and
verifying that filter against a real search index with deliberately
overlapping test data, closed that gap.

Search itself was built to run two different ways at once — matching
literal words and matching meaning — and then a second pass reranks
whatever came back, so the first search can afford to be generous and the
second pass narrows it down to what's genuinely relevant.

## Phase 3 — measuring quality, honestly

Before trusting any of the system's own quality numbers, its measurement
tools were checked against a public, independently-scored benchmark
first — the logic here being that home-grown test questions graded by
nobody but yourself can't tell you whether your grading itself is
trustworthy. Once that passed, the system's own retrieval quality,
answer quality, and — the harder problem — its ability to recognize
questions it genuinely can't answer, all got measured against real
question sets, repeatedly, as the system changed.

That abstention question turned out to be the deepest and most
interesting problem in the whole project: a system that never says "I
don't know" will eventually make something up, and a system that says "I
don't know" too often is just unhelpful. Getting the balance right took
several real rounds — a first version leaned too far toward "unhelpful,"
a fix for that leaned too far the other way, and multiple later rounds
added specific, narrow signals for specific real failure patterns caught
live (a wrong assumption baked into a question, a genuinely ambiguous
question that could mean several different papers, a case where the
system had the right information but still needed to be told plainly its
premise was wrong). Every one of those fixes was tested narrow-then-wide
before being trusted, and at least one promising-looking fix was tried,
measured, found to make things worse on real data, and reverted — kept
in the record specifically so the same mistake wouldn't get made twice.

## Phase 4 — surviving the real world

A system that only works when every outside service is healthy isn't
production-shaped. Real testing kept running into real outages — a
provider's free daily quota running out mid-evaluation, a service
returning a wrong-shaped response, a network timeout mid-request — and
each one became a fix: automatic retry with fallback to a different
provider, degrading gracefully instead of crashing when something truly
can't be recovered, and treating a provider's failure differently
depending on how dangerous silently continuing would be (a broken
embedding step stops the request outright, because continuing would
silently corrupt what gets searched; a broken reranking step just
continues with an unsorted-but-still-valid result, because that's safe).

One of the more subtle bugs in this category didn't show up until much
later: once the system was taught to survive a failed final answer
gracefully instead of crashing, the evaluation script measuring the
system's honesty started silently counting some of those real outages as
if the system had deliberately declined to answer — inflating both the
"good" and "bad" halves of the same number at once. Caught by noticing a
result looked implausibly good, checked directly against a handful of the
flagged cases, confirmed live, and fixed at its actual source (the
measurement script, not the system being measured).

## Phase 5 — a full review, and closing real gaps

Once the core system was working end to end, it went through a complete,
structured line-by-line review looking specifically for real correctness
bugs and unnecessary duplication — not a hypothetical audit, a real one,
with every finding checked directly against the current code before being
trusted. That review found and fixed several real, meaningful bugs: a
search index mismatch that could occur under an uncommon configuration
combination, a safety check that could silently stop working with no
error under that same combination, a cache that could get poisoned with
the wrong provider's data during an automatic failover, and a retry
mechanism that was missing the exact class of network failure its own
documentation said was the most common one in practice.

The same review also flagged a handful of smaller, lower-risk items,
which got worked through afterward with the same discipline: some fixed
and tested, one attempted and deliberately reverted after real data
showed it would trade one rare problem for a more damaging one, and a
couple left open on purpose with the reasoning for that decision written
down rather than silently dropped.

## Where it ended up

By the end, the system's ability to correctly recognize a question it
can't answer had improved substantially and was verified with a real,
statistically meaningful test run — not a lucky one-off — while its
accuracy on questions it *can* answer stayed strong and its tendency to
wrongly refuse a genuinely answerable question stayed low throughout.
Every one of those numbers, and the story of exactly how they were
reached, is in [`result.md`](result.md).
