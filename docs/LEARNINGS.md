# SentioBot: Engineering Learnings Log

Purpose: a living, in-depth teaching companion to the production-grade
workstream. STATUS-prod.md is the terse ledger (what changed, which commit,
which gate). THIS file is the narrative: the architecture, the concepts, and
the lesson behind every decision, written to be learned from.

Seeded by the planner across Increments 0 to 2. From here, every increment:
the BUILDER appends what it built, the architecture and concepts it touched,
and the lesson; the REVIEWER appends what it found, the deeper principle, and
how to avoid the mistake next time. Keep it in-depth and plain. No em dashes.

How to use it: read top to bottom once for the story, then use the Glossary at
the end as a quick reference. Convert to PDF anytime with the pdf skill or
`pandoc LEARNINGS.md -o LEARNINGS.pdf`.

---

## Part 1 - What SentioBot actually is

SentioBot is an AI customer-support assistant for a fictional electronics
brand (Nexora). It is not one program; it is three cooperating pieces:

- A FRONTEND (Next.js 14 + React + TypeScript + Tailwind): the chat UI in the
  browser. It streams the answer token by token, shows "tool call" badges, and
  keeps a conversation sidebar.
- A BACKEND (FastAPI, Python): the brain. It authenticates the user, decides
  how to answer, talks to the language model, runs retrieval, and saves state.
- A DATABASE (Supabase / Postgres): the memory. Users, products, orders,
  conversations, messages, analytics, and support tickets live here.

Two more components sit inside the backend:

- RETRIEVAL (ChromaDB + BM25): the "library." The product manuals and policies
  are chunked, embedded, and stored so the bot can look up facts instead of
  making them up. This is the RAG in "RAG chatbot."
- THE LLM (Google Gemini originally, now Groq Llama 3.3 70B): the "writer." It
  turns retrieved facts into a fluent, cited answer, and it decides when to
  call a tool (check an order, check a warranty, open a ticket).

Why this shape matters: separating the writer (LLM) from the library
(retrieval) from the memory (database) is what lets you make the answer
trustworthy. The LLM is told to answer ONLY from retrieved context, so it
cites sources and cannot invent policy.

## Part 2 - The request lifecycle (one question, end to end)

Follow "How do I install the LumiGlow bulb?" through the system:

1. The browser POSTs to `/chat/stream` with a JWT (proves who you are).
2. The backend checks the CACHE first (three tiers): an exact-match memory
   (L1), a semantic "did someone ask something very similar" match (L2), and
   an optional shared Redis (L3). A hit returns instantly with no LLM call.
3. On a miss, the backend ROUTES the question. A keyword check ("order",
   "warranty", "ticket"...) decides: a plain documentation question takes the
   RAG path; anything needing an action takes the agent/tool path.
4. RAG path: the retriever finds the most relevant manual sections, the LLM
   streams a cited answer from them, token by token.
5. Tool path: a small state machine (LangGraph) lets the LLM call tools
   (look up docs, check warranty, open a ticket), then finalizes an answer.
6. The backend PERSISTS the exchange (your message, the answer) to Supabase,
   logs analytics, and warms the cache. A final "metrics" event records how
   long everything took and how many calls it made.

The whole answer streams to your screen as it is generated, which is why it
feels fast even when the full answer takes several seconds.

## Part 3 - Lessons by increment

### Increment 0 - Recon: ground truth is the running code
The README described a simple Streamlit app with ChromaDB. The actual code was
a completely different, far more advanced full-stack app, and the entire
rewrite was sitting UNCOMMITTED in one folder, backed up nowhere, while GitHub
still held the old version. Lesson: never trust documentation as ground truth;
read the code, and check git provenance (what is committed, what is pushed,
which branch) before you plan anything.

### Increment 1 - Foundation: make it real before you make it fancy
- Boot integrity: the "production" app could not actually start. Imports
  contradicted each other and the Docker start command pointed at the wrong
  module. Lesson: a system needs ONE canonical way to run that works on a
  fresh clone and inside a container, not "it works on my machine."
- Cache privacy: the cache ignored who was asking but stored personalized
  answers (names, serial numbers, warranty state). One user could be served
  another user's answer. Lesson: any cache that can hold personal data MUST be
  scoped to the identity that data belongs to.
- Config hygiene: no secret should have a working insecure default; secrets
  come from the environment, never from code.
- Concepts introduced: SSE (server-sent events) streaming, JWT auth, and
  centralized settings.

### Increment 1.5 - Provider migration: vendors vanish, so abstract them
Gemini's free tier dropped to zero mid-project and blocked everything,
including measurement. We swapped to Groq. Lesson: put any external provider
behind an abstraction (a config flag plus a factory function) so switching is
a config change, not surgery across the codebase. Vendor risk is a real
engineering constraint, not a footnote.
- F-1 lesson: never forward a raw provider error to the user. The raw 429 body
  leaked internal URLs and quota details. Sanitize errors at the boundary and
  show the user a clean, generic message; log the detail server-side.

### Increment 1.6 - Agent reliability: guarantee behavior, do not hope for it
The tool path was flaky: the model kept re-calling the same tool, looping
until it crashed. The gate had passed once by luck (a 1-in-3 pass), and the
separate reviewer caught it. Lessons:
- Do not make correctness depend on the model behaving well. Guarantee it
  STRUCTURALLY: a hard cap on tool rounds plus a "finalize" node that forces
  the model to answer without any tools available. Even a model that loops
  forever now terminates.
- Never trust an LLM to supply identity or security data. The support-ticket
  tool took the user_id from the LLM's arguments, which it got wrong, so the
  database insert silently failed. Fix: inject the REAL user id from the
  authenticated request, server-side, using a ContextVar (per-request-safe).
- A swallowed error is a lie. `except: pass` turned a failed insert into a
  fake "ticket created" message. Lesson: fail honestly; a hidden failure
  becomes a user-facing falsehood, which is worse than an error.

### Increment 2 - The baseline: measure before you optimize
This is the increment that proves the whole discipline. We changed nothing; we
only instrumented and measured. The payoff:
- The baseline PROVED the multi-query retriever spends one of the two LLM calls
  on every documentation answer, purely to rephrase your question three ways
  before answering. "I think we should cut it" became "here is exactly how much
  it costs." That is the difference between a guess and a decision.
- Instrument at the right seam: LLM and token counts are captured with a
  LangChain callback passed through the call config, so it propagates into
  NESTED calls (the hidden multi-query LLM the caller never sees directly).
  Counting at the obvious spot would have under-reported by half. Lesson:
  measure where the work truly happens, not where you assume it does.
- Fast paths hide work: the visible "done" event fires BEFORE the save and
  analytics writes, so naive counting missed two database calls. A trailing
  metrics event, emitted after all work, gets the honest number. (Same reason
  a cache hit is not instant at ~1.4s: it still embeds the query and writes to
  the database.)
- Cost per request is a budget metric: Groq's free tier is ~100K tokens per
  DAY, and each answer costs ~2.4K tokens, so the free app supports ~40 answers
  per day per account. Lesson: on any metered resource, tokens (or dollars)
  per request is a first-class metric next to milliseconds.

The numbers (Groq Llama 3.3 70B, temp 0, localhost, free tier; INTERNAL
baseline for measuring deltas, not a published claim):
- RAG doc answer, cache miss: ~8.6 to 11.3s to first token, 2 LLM calls, 7
  local embeddings, 5 database round trips, ~2.5K tokens.
- Tool answer, cache miss: ~11s to first token, 2 to 3 LLM calls, ~2.26K tokens.
- Cache hit: ~1.4s, 0 LLM calls, ~0 tokens.
- Plus a one-time ~22s cold start on the first request (lazy-loaded models).

## Part 4 - Cross-cutting principles (the transferable lessons)

1. Ground truth is the running code, not the README.
2. You cannot optimize what you have not measured; a number you can regenerate
   beats an intuition you cannot.
3. Measure DELTAS, not just absolutes. Deltas (before vs after on the same
   machine) cancel out environmental noise, which is why a noisy free-tier
   localhost is still a valid place to prove an optimization worked.
4. Baseline before optimization, and freeze the eval set BEFORE results exist,
   so you cannot (even accidentally) shop for a flattering metric.
5. Guarantee behavior structurally; do not hope the model or the network
   cooperates.
6. Never trust an LLM with identity or security data; inject it server-side.
7. Sanitize errors at the boundary; never leak provider internals; never
   swallow an error into a lie.
8. Abstract external providers; free tiers disappear without warning.
9. Scope every cache to the identity whose data it holds.
10. Keep a terse ledger (STATUS) and a teaching narrative (this file) separate;
    each does its job better alone.

## Part 5 - How to judge an optimization (the four questions)

Before building any optimization, answer four things. This is the exact frame
the planner uses to rank candidates, and it is worth internalizing:

- Expected win: what specifically gets better, and roughly how much?
- Proof metric: what measurement, run before and after, would PROVE it (or
  disprove it)?
- Cost: engineering effort, plus any recurring money or latency it adds.
- Risk: what could it break, and how would you catch that?

Worked example, the top candidate (cut or gate the multi-query retriever):
- Win: removes one of two LLM calls per doc answer, so lower time-to-first-token
  and roughly half the tokens per answer (which also doubles daily capacity on
  the free tier).
- Proof: the Increment 2 latency and call-count baseline, re-run after the cut.
- Cost: engineering only; it REDUCES spend.
- Risk: multi-query may be lifting recall, so answers could get slightly worse.
  That risk is exactly why we build the QUALITY baseline first, so the cut can
  be judged on faithfulness and hit-rate, not just speed.

## Glossary

- LLM: large language model (the "writer"); here Groq Llama 3.3 70B.
- RAG: retrieval-augmented generation; look up real documents, then answer from
  them, so the model cites facts instead of inventing them.
- Embedding: a text turned into a vector of numbers so similar meanings sit
  close together; used for semantic search. Model here: MiniLM, run locally.
- Vector store / ChromaDB: the database of embeddings that answers "which
  chunks are most similar to this query."
- BM25: a classic keyword-search ranking; strong on exact terms and codes.
- Hybrid / ensemble retrieval: combining keyword (BM25) and semantic (vector)
  search to catch both exact matches and meaning-based ones.
- Multi-query retriever: asks the LLM to rephrase the question several ways
  before searching, to improve recall; costs an extra LLM call.
- Parent-document retrieval: search small chunks, but return the larger parent
  section they belong to, so the answer has full context.
- TTFT: time to first token; how long until the first word appears. The key
  latency number for a streaming chat.
- p50 / p95: the median and the 95th-percentile of a set of measurements; p95
  is the "slow but not rare" experience. (Only meaningful with enough samples.)
- SSE: server-sent events; a one-way stream from server to browser, used to
  push tokens as they are generated.
- LangGraph: a small state machine for LLM agents; each step (call model, run
  tool, finalize) is an explicit node, which makes control and recovery clear.
- ContextVar: Python per-task variable; safe way to carry request-scoped data
  (like the real user id) without it leaking between concurrent requests.
- JWT: JSON web token; a signed proof of identity the browser sends on each
  request.
- Cache tiers L1/L2/L3: in-process exact match, semantic-similarity match, and
  a shared cross-process cache (Redis).
- Cold start: the one-time slowness while lazy-loaded models initialize on the
  first request.
- RAGAS / faithfulness / hit-rate@k: an eval toolkit and its metrics;
  faithfulness = is the answer supported by the retrieved context; hit-rate@k =
  did the right source appear in the top k results.
