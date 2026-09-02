# Glossary — plain-language definitions

Every technical term used across this project's docs, explained without
assuming prior knowledge. If a term you're looking for isn't here, it's
missing by oversight, not on purpose — the goal is that nothing in
`result.md` or `architecture.md` should require outside research to
understand.

---

### RAG (Retrieval-Augmented Generation)
A way of answering questions with an AI model where, instead of relying
purely on what the model memorized during training, the system first
*looks up* relevant real documents and hands them to the model as
reference material — like giving someone open-book access to a library
before asking them a question, instead of asking them to answer from
memory alone. This matters because it lets the system answer using
current, specific, verifiable source material (in this case, real arXiv
research papers) rather than whatever the model happened to learn.

### Abstention
The system deciding **not** to answer a question because it doesn't have
enough real information to answer it correctly — saying "I don't know"
instead of guessing. This is treated as a *good* outcome when the
question genuinely can't be answered from the available documents. The
opposite failure — confidently answering with something made up — is
called a **hallucination**, and is treated as much worse than saying "I
don't know."

### Abstention accuracy (this project's docs sometimes call this the
"abstention rate")
Out of all the questions that genuinely *can't* be answered from the
document collection, what percentage did the system correctly recognize
as unanswerable and decline? Higher is better. This is measured
separately from —

### Over-answering / over-refusal
The opposite failure: how often the system incorrectly declines a
question it actually *could* have answered correctly. A system that
refuses everything would score perfectly on abstention accuracy but
terribly on this — which is why both are always measured together. A
system is only actually good at "knowing what it knows" if abstention
accuracy is high **and** over-refusal is low at the same time.

### Hallucination
When an AI model states something confidently and fluently that isn't
actually true or isn't actually supported by the source material it was
given — the single most trust-destroying failure mode for a system like
this, because it looks exactly like a correct answer unless you check the
sources yourself.

### Guardrail
An automated check built into the pipeline that watches for a specific
kind of failure and reacts to it — for example, checking whether a
question contradicts something the retrieved documents actually say, or
checking whether the final answer is genuinely backed up by the sources
it cites. Some guardrails only *observe and log* what they see without
changing behavior (called "monitor mode"); others can actually block a
bad response from reaching the user (called "enforce mode").

### Hybrid retrieval
Searching for relevant documents two different ways at once, then
combining the results:
- **Lexical / keyword search** — finds documents containing the literal
  words in the question (fast, precise, but misses synonyms and
  paraphrasing).
- **Vector / semantic search** (also called "dense retrieval") — finds
  documents that mean something similar to the question, even if they
  don't share exact words, by comparing mathematical representations of
  meaning (see "embedding" below).

Combining both catches more of the genuinely relevant material than
either search alone would.

### Embedding
A way of converting text into a list of numbers (a "vector") that
captures its meaning, such that two pieces of text with similar meaning
end up with mathematically similar number-lists, even if they don't share
any of the same words. This is what makes semantic search possible.

### Reranking
After an initial search returns a batch of candidate documents, a second,
more careful pass re-scores and re-sorts them by how genuinely relevant
each one is to the specific question — the first search is optimized for
speed and recall (finding everything plausibly relevant), reranking is
optimized for precision (putting the truly best matches first).

### Chunk / chunking
Long documents (research papers) are split into smaller pieces
("chunks") before being indexed, because search and the AI model both
work better over focused passages than entire documents at once. How a
document gets split (chunk size, where the boundaries fall) is itself a
real design choice this project measured and tuned.

### Provider fallback / retry ladder
This system calls out to several different AI companies (Groq, Mistral,
OpenRouter, and others) to do its language-understanding and
answer-writing work. If one of them is slow, down, or rate-limited at the
moment a request comes in, the system automatically tries a different one
instead of just failing — the same way a delivery service might try a
backup courier if the first one doesn't pick up.

### Fail loud vs. fail open (or "degrade gracefully")
Two different, deliberate responses to something going wrong internally:
- **Fail loud** — stop immediately and raise a clear error, used when
  continuing anyway would risk silently producing something *wrong*
  (e.g., a broken step that would corrupt what gets searched).
- **Fail open / degrade gracefully** — continue anyway with a safe
  fallback behavior, used when the failure is recoverable and stopping
  entirely would just be unhelpful without actually protecting anyone
  (e.g., a reordering step failing just means results stay in their
  original, still-valid order).

### False premise
A question that contains an incorrect assumption baked into how it's
asked — for example, "why did X score 0.45" when X actually scored 0.87.
A good system should recognize the assumption is wrong and correct it,
rather than politely answering around the mistake or, worse, inventing an
explanation for a number that was never real.

### Trace / traced query
A complete, detailed record of everything that happened while answering
one specific question — every stage it went through, every check that
ran, how long each step took, and why. Used both for debugging and for
letting a person inspect exactly how a given answer was produced.

### Ingestion / ingestion pipeline
The process of pulling in new source documents (fetching a paper from
arXiv, extracting its text, splitting it into chunks, and adding it to
the search index) so the system has something to search over in the
first place.

### Corpus
The full collection of documents the system can search over and answer
questions about. This project's corpus is a set of real, publicly
available computer-science research papers pulled from arXiv.
