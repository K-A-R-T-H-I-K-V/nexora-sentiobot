# SentioBot: AI Features Plan (the intelligence roadmap)

Status: the production foundation (P0 to P6 + fail-closed RLS) is complete and
verified. This document plans the AI features that turn a correct RAG support
bot into a genuinely great product. Written by the planner. Each feature becomes
its own planner -> builder -> reviewer increment, and none is allowed to regress
the frozen baselines (retrieval hit@5 0.913) or the security gates (injection
containment, fail-closed authorization).

---

## The vision (what "great" means here)

Today SentioBot answers questions from documents and can call a few tools. That
is a good assistant. A GREAT one does four things a person does:

1. UNDERSTANDS what you actually want, even when you phrase it badly.
2. FEELS the conversation: it notices frustration and adapts, and it never lets
   an angry customer wait. (The product is literally named "I feel." Right now it
   feels nothing. That is the single biggest missed opportunity in the app.)
3. REMEMBERS you across visits and is proactive with what it knows.
4. Is TRUSTWORTHY: every answer is visibly grounded in a real source, and it says
   "I do not know" instead of inventing.

The plan below is organized so that each tier makes the app noticeably better on
its own, and the tiers stack into that vision.

---

## Governing constraints (the rules every feature obeys)

- TOKEN BUDGET is the hard limit. Groq free tier is ~100K tokens/day. Every
  feature states its token cost. Strongly prefer features that run on LOCAL
  embeddings (zero tokens) or the cheap fast model (llama-3.1-8b-instant) over
  the 70B model. Cache aggressively (the 3-tier cache already exists).
- MEASURE OR IT DID NOT HAPPEN. Any feature touching answer quality is evaluated
  against the frozen 50-question golden set before it ships; report the delta.
- DO NO HARM. No feature may regress hit@5 0.913, the injection red-team, or the
  10/10 cross-user denial suite. These are CI-gated, so a regression fails the
  build.
- SECURITY TRAVELS WITH FEATURES. Anything that ingests user content (uploads,
  memory) reopens the indirect-injection surface we deferred; it carries its own
  defense.

## Our building blocks (what we already have to build on)

- Agent: a LangGraph StateGraph (nodes: agent, tools, finalize). New behavior is
  a new NODE or a new TOOL, not a rewrite.
- Models on Groq: llama-3.3-70b-versatile (reasoning/answers), llama-3.1-8b-instant
  (cheap/fast classification), whisper-large-v3 (speech to text, cheap and fast).
- Retrieval: Chroma + BM25 ensemble, ONNX MiniLM embeddings (LOCAL, zero-token).
- Data: Supabase Postgres with fail-closed RLS (new user-owned tables inherit the
  same per-row security pattern).
- Frontend: Next.js streaming chat with SSE event types already wired
  (token / tool_start / tool_end / done / error) - new event types plug straight
  into the existing switch.

---

## TIER 1 - The Smarter Core (fix the weak spots, earn trust)

### F1. Intent-aware routing (retire the keyword router)
- What: replace the brittle keyword match (`if "warranty" in message`) that
  decides RAG-path vs tool-path with a real intent classifier.
- Why it is great: this is a KNOWN weakness flagged since day-one recon. "What
  does the warranty policy say about water damage" is a documentation question,
  but the word "warranty" wrongly routes it to the tool path. Fixing routing
  makes a large class of answers better at once.
- Architecture: a front-of-graph classifier node. Two options, ranked:
  (a) EMBEDDING-based (recommended, zero-token): embed the query with the local
      ONNX model, compare to a small set of labeled intent prototype vectors
      (doc_lookup, order, warranty, ticket, chitchat, escalation), take the
      nearest. Deterministic, free, fast.
  (b) llama-3.1-8b-instant classification if the embedding approach is not
      accurate enough (costs a tiny call).
- How to add: add `classify_intent(query) -> intent` before the router; store
  the intent in AgentState; route on it. Keep the keyword router as a fallback
  behind a flag for one increment so you can A/B them.
- Token cost: ~0 (embedding path).
- Risk: misclassification. Low, and measurable.
- Measurement/gate: build a small labeled routing set (reuse the golden set's
  categories), measure routing accuracy old vs new; ship only if it beats the
  keyword router. Must not drop hit@5.
- Sequence: FIRST. Highest value-to-cost in the whole plan, and it unblocks
  cleaner behavior for everything downstream.

### F2. Groundedness guard + inline source highlighting
- What: (1) after generating, verify the answer is supported by the retrieved
  context and show a "Grounded" badge (or soften/withhold if it is not); (2)
  highlight the exact source sentence(s) behind each claim in the sources panel.
- Why it is great: this is the trust feature. Recruiters and users both ask "how
  do I know it is not making this up?" A visible grounded-check and a click-to-
  see-the-exact-sentence answer that question. It also directly showcases the
  RAGAS faithfulness metric you already built.
- Architecture:
  - Inline citation: prompt the model (already does [Source N]) to also emit the
    quoted span it used; the frontend renders the quote under the answer.
  - Groundedness check: a lightweight verifier. Cheapest first: a local NLI /
    string-overlap heuristic between answer claims and context. If that is too
    weak, a single llama-3.1-8b-instant "is this answer supported by this
    context? yes/no + which sentences" call.
- How to add: a `verify_groundedness` step after generation; add a `grounded`
  field + `citations` to the `done` SSE event; the frontend shows the badge and
  the highlighted spans (a new render branch, no new event plumbing).
- Token cost: ~0 (heuristic) to small (8B verifier). Budget the 8B path.
- Risk: a wrong "grounded" badge is worse than none. Tune the threshold on the
  golden set; when unsure, show "partially grounded," never a false green.
- Measurement/gate: correlate the guard's verdict with RAGAS faithfulness on the
  frozen set; the badge must not claim grounded where faithfulness is low.
- Sequence: SECOND. Trust is foundational and this is highly visible.

### F3. Reranking for retrieval precision (measured, optional)
- What: after the ensemble returns top-k, re-order with a cross-encoder reranker
  so the best passages lead.
- Why it is great: better context in, better answer out; can lift faithfulness
  and answer quality, especially as the corpus grows.
- Architecture: a small local cross-encoder (e.g. a MiniLM reranker via ONNX,
  zero-token) as a post-retrieval node; take top-k from the ensemble (say 10),
  rerank, keep top 3-5 for the prompt.
- How to add: a `rerank` node between retrieve and generate; behind a config
  flag so it is A/B-able against the frozen baseline.
- Token cost: ~0 (local reranker).
- Risk: MAY NOT beat 0.913 on this small corpus (few distractors to reorder).
  This is a MEASURE-FIRST feature: only ship if it moves hit-rate or faithfulness
  on the frozen set without adding meaningful latency.
- Sequence: THIRD, and honestly optional. Do it only if F1/F2 leave retrieval
  quality as the next real bottleneck. Great resume line IF it measures a win.

---

## TIER 2 - Emotional Intelligence (the SentioBot differentiator)

This tier is what makes the app memorable and on-brand. It is also mostly cheap
(local or 8B classifiers), so it fits the token budget.

### F4. Sentiment and frustration awareness -> adaptive tone + fast escalation
- What: detect the user's emotional state (calm / confused / frustrated / angry)
  each turn; adapt the answer's tone; and when frustration crosses a threshold,
  proactively offer a human (create_support_ticket) instead of making them ask.
- Why it is great: it delivers on the product's NAME. It is the feature nobody
  expects from a support bot and everybody remembers. "The bot noticed I was
  annoyed and got me to a human" is a story. It is also genuinely good support
  practice.
- Architecture: a `detect_sentiment` step (local classifier or llama-3.1-8b-
  instant returning a label + score). Store a running frustration signal in
  AgentState / the conversation. Feed the label into the system prompt ("the user
  seems frustrated; be concise, acknowledge the issue, offer a human"). A
  threshold rule triggers proactive escalation.
- How to add: sentiment node before generation; a `sentiment` field on the
  message metadata (so the UI can show a subtle mood indicator if you want); an
  escalation branch in the graph when the signal is high.
- Token cost: ~0 (local) to small (8B). Prefer local.
- Risk: misreading tone (calling a calm user "angry" is jarring). Keep the
  adaptation subtle; never announce "you seem angry"; just adjust behavior.
- Measurement/gate: a small labeled set of frustrated vs calm messages; measure
  detection accuracy; verify escalation fires on the frustrated cases and NOT on
  calm ones (a false escalation is annoying). Add these as red-team-style cases.
- Sequence: the FLAGSHIP of this tier. Do it early in Tier 2.

### F5. Clarify-before-answering (ask instead of guessing)
- What: when a query is ambiguous or missing a needed detail ("my device is
  broken" - which device? "is it under warranty" - which serial?), the agent
  asks ONE targeted clarifying question instead of guessing or dumping options.
- Why it is great: it is the difference between a form and a conversation. It
  also reduces wrong answers and wasted tool calls.
- Architecture: a decision in the agent node: if the query lacks a required slot
  for the detected intent (e.g., warranty needs a serial, and none is in the
  message or the user profile), emit a clarifying question rather than proceeding.
  The user profile (owned products/serials) already lets it resolve many cases
  WITHOUT asking (proactive), so it only asks when it truly cannot.
- How to add: slot-check logic keyed off the F1 intent; a clarify branch that
  returns a question and pauses.
- Token cost: ~0 extra (it is a branch in the existing call).
- Risk: over-asking (annoying). Bias toward resolving from profile/context first;
  ask at most one question.
- Sequence: pairs naturally with F1 (intent) and F4.

### F6. Long-term user memory (it remembers you)
- What: remember durable facts across sessions (products owned, past issues and
  resolutions, preferences) and use them proactively: "Last time your Thermostat
  Pro had a pairing issue - is that resolved, or should we continue?"
- Why it is great: personalization is the biggest jump in perceived
  intelligence. A bot that remembers feels like a concierge, not a kiosk.
- Architecture: a `user_memory` table in Supabase (user_id, fact, source,
  created_at) under the SAME fail-closed RLS pattern (a user's memory is theirs
  only). After each conversation, a summarization step extracts durable facts and
  upserts them (dedup against existing). On a new session, load the memory into
  the system prompt (bounded to N most-relevant facts to control tokens).
- How to add: the memory table + RLS policy; a post-conversation summarize-and-
  store step (budget: one 8B call per ended conversation, not per turn); a
  load-memory step at conversation start.
- Token cost: SMALL but recurring (one summarization per conversation). Budget it;
  summarize on conversation end, not every message; cap stored facts per user.
- Risk: (1) privacy - covered by RLS (memory is user-scoped, fail-closed); (2)
  memory hallucination - only store facts traceable to something the user said or
  a tool returned, never model guesses; (3) staleness - let users view/clear their
  memory (a "what SentioBot remembers about me" panel is itself a trust feature).
- Measurement/gate: correctness of extracted facts on sample conversations; RLS
  denial test extended to user_memory; token-per-conversation stays in budget.
- Sequence: the richest Tier 2 feature; do it after F4/F5 since it is the most
  moving parts.

---

## TIER 3 - Multimodal and Voice (the wow factor)

### F7. Voice mode (talk to SentioBot)
- What: press a mic, speak your question, get a streamed spoken/te xt answer.
- Why it is great: high cool-factor, real accessibility win, and it fits the
  stack almost for free because Groq hosts Whisper.
- Architecture: browser captures audio -> send to a new backend endpoint ->
  Groq whisper-large-v3 transcribes (fast, cheap) -> the transcript flows into
  the EXISTING chat pipeline unchanged -> answer streams back -> optional TTS via
  the browser's built-in SpeechSynthesis (zero cost) or a TTS API.
- How to add: a `/voice/transcribe` endpoint (Groq Whisper); a mic button + Web
  Audio capture on the frontend; reuse the whole existing chat path for the
  answer; browser TTS for readback.
- Token cost: Whisper is billed in audio-seconds, cheap; the answer is a normal
  chat turn. Browser TTS is free.
- Risk: audio handling edge cases (permissions, formats); keep it an ADD-ON, text
  chat stays the default so nothing regresses.
- Measurement/gate: transcription sanity on sample clips; the text path is
  unchanged so the frozen eval is unaffected.
- Sequence: first of Tier 3, because it is cheap and demos incredibly well.

### F8. Multimodal RAG (see the manual, and your device)
- What: (1) ingest diagrams/images from the manuals so the bot can answer "show
  me the wiring diagram for the Thermostat Pro"; (2) let a user upload a PHOTO of
  their device/error and have the bot reason about it ("that blinking red LED
  means...").
- Why it is great: technical support is deeply visual; this unlocks questions the
  text-only bot simply cannot answer. It is the most impressive single feature.
- Architecture: a vision-capable model (a Groq vision model, availability to be
  confirmed at build time) for image understanding; multimodal or caption-based
  indexing of manual images so they are retrievable; an upload path for user
  images.
- How to add: extend ingestion to extract + caption/embed images; add an image-
  upload input; route image-bearing turns to the vision model.
- Token/cost: HIGHER (vision tokens); budget carefully, and gate uploads by
  size/rate. This is the heaviest feature in the plan.
- Risk: (1) COST and the token budget; (2) INDIRECT INJECTION - user-uploaded
  images/text reopen the second-order injection surface we deferred; malicious
  instructions in an uploaded image caption must be treated as DATA, not
  instructions (carry the injection defense into this path); (3) vision accuracy.
- Measurement/gate: a small visual-QA set; the indirect-injection defense must be
  demonstrated on uploaded content before it ships.
- Sequence: later. High payoff, highest lift and cost. Do it once Tier 1/2 have
  made the core excellent.

---

## TIER 4 - Agentic Expansion (it does more)

### F9. More real actions (return/RMA, callback scheduling, recommendations)
- What: new tools so the agent can DO more: start a return/RMA, schedule a
  callback, recommend a compatible product for what the user owns.
- Architecture: each is a new LangGraph tool + a Supabase table (returns,
  callbacks) under RLS, using the ContextVar user_id pattern from Increment 1.6
  (never trust the LLM for identity). Recommendations can use the existing
  embeddings over a product catalog.
- Token cost: minimal (tool calls). Risk: each new tool is new authorization
  surface - it inherits the fail-closed pattern and the denial suite gets a case.
- Sequence: incremental; add tools as the product needs them.

### F10. Self-improving loop (learn from thumbs-down)
- What: mine the analytics (thumbs-down + low-groundedness answers) to surface
  KNOWLEDGE GAPS - the questions SentioBot answers badly - and turn them into a
  prioritized list for corpus expansion or golden-set additions.
- Why it is great: it closes the loop between the eval harness you built and the
  product. It is also a fantastic admin/portfolio artifact ("the system tells you
  where it is weak").
- Architecture: an analytics query + a clustering/summarization pass over
  negative-feedback queries; surface in the admin dashboard.
- Token cost: batchable and occasional; cheap.
- Sequence: after feedback volume exists; pairs with the admin dashboard.

---

## Recommended sequence (and why)

1. F1 Intent routing - fixes a known flaw, near-zero cost, makes everything
   downstream cleaner.
2. F2 Groundedness + citations - trust, highly visible, showcases your eval work.
3. F4 Sentiment/escalation - the on-brand differentiator, cheap, memorable.
4. F5 Clarify-before-answer - natural pair with F1/F4, cheap.
5. F7 Voice - cheap via Groq Whisper, demos brilliantly.
6. F6 Long-term memory - richest personalization; more moving parts.
7. F3 Rerank - only if retrieval quality is the measured bottleneck.
8. F9 More tools / F10 self-improving loop - as the product grows.
9. F8 Multimodal - the big finale; highest lift and cost, do it last and
   deliberately with its injection defense.

This order front-loads cheap, high-impact, on-brand wins and defers the
expensive/heavy ones, which fits both the token budget and the "cool factor per
day of work" curve.

## How each feature ships (the discipline stays)

Every feature is a normal increment: planner ratifies the design + gate; builder
inspects, builds behind a config flag where sensible, and measures against the
frozen baseline; reviewer independently re-runs the gate and tries to break it.
A feature ships only if it improves what it targets AND regresses nothing
(hit@5 0.913, injection containment, 10/10 authz). The learnings log gets a new
entry per feature so the "why" is captured, not just the "what".

## The north star

Ship Tier 1 and Tier 2 and SentioBot stops being "a RAG chatbot" and becomes
"the support agent that understands what you mean, feels how you feel, remembers
who you are, and proves what it says." That is a product people remember, and a
portfolio piece that tells a story no tutorial project can: not just built, but
measured, hardened, deployed, and then made genuinely intelligent, one verified
increment at a time.
