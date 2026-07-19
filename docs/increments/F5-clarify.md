# Feature F5 - Clarify-before-answering

Status: PHASE A (inspect + propose). Builder does Phase A ONLY, then STOPS.
Standing Conventions + FEATURE CADENCE in CLAUDE.md apply.

## Goal
When a query is ambiguous or missing a needed detail, ask ONE targeted clarifying
question instead of guessing - but resolve from the user profile / conversation
FIRST (proactive), so it only asks when the answer truly is not knowable. At most
one question. The difference between a form and a conversation.

## PHASE A - INSPECT + PROPOSE (do this now, then STOP; write no feature code)
Grep/read the real code and write a PROPOSAL covering:
- The F1 intent + confidence output; the user profile (owned_products/serials);
  chat_history availability; how the agent handles missing info TODAY (does it
  guess, ask, or dump options?); the graph structure and where a clarify branch
  would sit.
- A SLOT model per intent: which intents have a required slot (warranty -> serial;
  order_status -> order_id) and which are answerable without one (doc_lookup).
- The RESOLVE-BEFORE-ASK order: try to fill the missing slot from (a) the message,
  (b) the user PROFILE, (c) chat_history, and only ask if still unknown. Reuse the
  F4/F1 machinery where sensible.
- The clarify decision + where it fires in the graph; the clarifying question is
  just an assistant message (no new SSE plumbing). At most ONE question; bias hard
  toward resolving from context.
- Interaction rules: a FRUSTRATED user (F4) should not be met with a bureaucratic
  question - keep any clarification concise/empathetic or skip to action; do not
  clarify when the profile already answers it.
- Proposed GATE: a labeled set of genuinely-ambiguous vs well-specified /
  profile-resolvable queries. The safety property here (Convention 10) is NO
  OVER-ASK: it must ask on truly-ambiguous cases and must NOT ask when the answer
  is resolvable from profile/history - and that must be tested on the HARD cases
  (queries that LOOK ambiguous but are resolvable), not just the obvious ones. No
  regression (hit@5 0.913, injection, authz); zero/low token.
- Risks: over-asking (annoying); asking for info already in the profile/history;
  clashing with F4 tone.
STOP after the proposal. Await the planner's ratified Phase B build spec below.

## PHASE A PROPOSAL (builder, 2026-07-20) - awaiting ratification

### 1. Code reality: intent/confidence, profile, history, missing-info handling, graph

F1 intent + confidence. `route_message(message, embedding=None)` (backend/agent/
intent_router.py) returns `RouteDecision(route, intent, confidence, router, scores)`.
Intents: doc_lookup, order_status, warranty, ticket_or_escalation, chitchat,
out_of_scope; INTENT_ROUTE (intent_router.py:58) maps doc_lookup/out_of_scope -> rag,
the rest -> tool. In stream_agent_response the decision is already computed
(agent.py:585 `decision = route_message(...)`), so `decision.intent` is available to
drive a per-intent slot check with no extra work.

User profile. `user_profile = {"name", "owned_products"}` (backend/api/main.py:172),
where owned_products = `[{"product_name", "serial_number"}]` from the verified user row
(auth.get_current_user -> db.get_user_by_id). It reaches stream_agent_response
(agent.py:558) and the graph state. IMPORTANT: serials are kept SERVER-SIDE - the
system prompt lists products by NAME only (agent.py:383) and check_warranty_status
resolves the serial from `current_user_products` (tools.py:84), never from the prompt.
A clarify resolver may read owned_products server-side (to decide ask/don't-ask and to
pass a product NAME hint), without putting serials in the prompt.

chat_history. Available in stream_agent_response as `chat_history` (last 12 messages,
agent.py:566-578) and already sliced for user turns by F4. So "resolve from history" is
free here (unlike stateless routing, F1-R6).

Slot sources per tool (the load-bearing asymmetry):
- check_order_status(order_id) (tools.py:25): the ONLY order lookup is
  db.get_order_by_id(order_id) (database.py:78). There is NO get_orders_for_user and no
  owner column on orders (schema.sql:51). So an order_id CANNOT be resolved from the
  profile; it must come from the message or history. Format is `NX-YYYY-NNN`
  (schema.sql:59, golden set NX-2025-301..303).
- check_warranty_status(product_name OR serial) (tools.py:69): RESOLVES from the
  profile. Pass a product name the user owns -> it matches against owned_products and
  resolves the serial server-side. So warranty is often answerable WITHOUT asking, from
  the profile alone.
- create_support_ticket(conversation_summary) and lookup_documentation(query): no
  user-supplied slot; the summary is model-generated and the doc query is the message.

How missing info is handled TODAY: by the model, non-deterministically. Behaviour Rule 2
(agent.py:413) already says "call check_warranty_status with the PRODUCT NAME ... Do NOT
ask for the serial" - a partial resolve-before-ask for warranty. For order_status there
is NO rule, so a missing order_id leads the model to either ask, guess (call the tool
with a bad id -> "No order found"), or dump. There is no deterministic, gateable
clarify. That is the gap F5 fills.

Graph + where clarify sits. The clarify decision fits BEST as a deterministic pre-check
in stream_agent_response, AFTER F1 routing (agent.py:585) and F4 sentiment
(agent.py:590ish), BEFORE dispatching to the RAG/tool path. It only engages for
slot-bearing intents (order_status, warranty). No new graph node is needed; the
clarifying question is a normal assistant message (token + done events), so no new SSE
plumbing (same as F2/F4). This mirrors F4's deterministic offer backstop.

### 2. Slot model per intent

| intent | required slot | resolvable from |
| --- | --- | --- |
| doc_lookup, out_of_scope, chitchat | none | n/a (never clarify) |
| ticket_or_escalation | none (summary is generated) | n/a (never clarify) |
| order_status | order_id (`NX-YYYY-NNN`) | message, history (NOT profile) |
| warranty | which product | message, PROFILE (owned_products), history |

Only order_status and warranty can trigger a clarify, and warranty rarely does (the
profile usually answers it).

### 3. Resolve-before-ask order (message -> profile -> history), then ask

For order_status:
- (a) message: regex `\bNX-\d{4}-\d{3}\b` (permissive; case-insensitive). If found ->
  resolved, proceed.
- (c) history: same regex over recent user/assistant turns (a just-discussed order).
- else -> ASK: "Which order number should I check? (for example NX-2025-301)".

For warranty:
- (a) message names a serial (SN-...) or a product the user owns (substring match both
  ways, aligned with tools.py:87-95) -> resolved to that product.
- (b) profile: if the user owns exactly ONE product -> that product (a generic "is my
  device under warranty?" is answerable). If the message names one of several owned
  products -> that one.
- (c) history: a product from owned_products discussed in recent turns.
- else -> ASK: 0 products -> "I do not see a registered product on your account - could
  you share the serial number printed on the device?"; multiple products + a generic
  reference -> "Which product should I check: <A>, <B>, or <C>?" (names only, no
  serials).
When the slot resolves from profile/history (not the message), inject a one-line HINT
into the system prompt / tool path ("The user is asking about their <ProductName>; use
check_warranty_status with that name") so the model uses it and does not re-ask.

### 4. Clarify decision + where it fires

A new backend/core/clarify.py exposes `decide(intent, message, profile, history,
sentiment) -> ClarifyDecision(ask: bool, question: str, resolved_hint: str)`. Pure
function, ZERO tokens. In stream_agent_response, for order_status/warranty: if
`ask` -> stream the canned question as an assistant message (token + done, no LLM call)
and return; else attach `resolved_hint` to the tone/hint passed into
_build_system_message and proceed normally. At most ONE question by construction (one
decision per turn). Re-ask guard: if the LAST assistant message was itself a clarify
question for this intent (detected from chat_history) and the slot is STILL unresolved,
do NOT ask again - proceed best-effort (let the tool return its own "not found" message
or offer a human), so the bot never loops on a question.

Design recommendation: make the DECISION deterministic (gateable per Convention 10) and
the QUESTION a canned per-intent template (zero-token), lightly tone-adapted for F4
(prepend a warm lead for a frustrated user). Rejected alternative: inject a "you may ask
a clarifying question" instruction and let the LLM decide - more fluid, but the ask/
don't-ask decision becomes non-deterministic and the no-over-ask property cannot be
tested for certain. A small Behaviour Rule ("resolve from profile/history first; ask at
most one question only if still unknown") is added as a BELT for the LLM on the paths the
pre-check does not short-circuit.

### 5. Interaction rules (F4 + profile)

- Never clarify when the profile/history already answers it - the resolver's whole job,
  and the load-bearing gate case.
- A FRUSTRATED user (F4 label frustrated/angry or escalate) is NOT met with a
  bureaucratic question: keep the clarify to ONE short warm line (prepend a brief lead),
  and never stack it with the human-offer in a wall of text. If the missing slot is
  genuinely unavoidable (no order_id can be invented), still ask, but warmly.
- Clarify runs AFTER F4 so the sentiment label is available to shape tone; it runs after
  the injection guard too, so a clarify path never weakens the guard.

### 6. Proposed GATE

- Labeled clarify set results/clarify_set_v1.json: each item = {intent, message, mock
  profile (owned_products), optional history, expect_ask}. Four buckets: truly_ambiguous
  (expect_ask=true), well_specified (false), PROFILE_RESOLVABLE (false - the HARD cases:
  "is my thermostat under warranty" with a thermostat owned; "is my product under
  warranty" with exactly one owned), HISTORY_RESOLVABLE (false).
- Safety property NO OVER-ASK (Convention 10, the F2/F4 analogue): the over-ask rate on
  the resolvable buckets must be 0 - it must NOT ask when resolvable from profile/history
  - and this is tested on the HARD cases (look ambiguous, are resolvable), not just the
  obvious ones. Also report under-ask (failing to ask on truly_ambiguous). Over-ask is
  the load-bearing number.
- NO REGRESSION: hit@5 0.913, injection red-team, 10/10 authz all green; ZERO token on
  the deterministic path (a short-circuited clarify makes no LLM call at all).
- A test_clarify.py CI gate + a clarify_eval.py harness, deterministic (mock profiles).

### 7. Risks

- OVER-ASKING (annoying, and asking for info already known): mitigated by exhausting
  message -> profile -> history before asking, and gated by no-over-ask on the HARD
  resolvable cases.
- order_id regex brittleness: if a valid id is phrased oddly the regex could miss it and
  we would over-ask. Mitigation: bias toward PROCEEDING when an order reference is
  plausibly present, and keep the regex permissive; better a tool "not found" than a
  needless question.
- Warranty fuzzy match: keep the resolver's matching ALIGNED with tools.py's own
  substring logic so the pre-check and the tool agree (no "resolved here, not-found
  there" mismatch).
- Re-ask loops: the last-turn guard prevents asking the same clarify twice.
- Clash with F4 tone: clarify runs after F4 and stays to one warm line.
- Scope creep into INTENT ambiguity: F5 is SLOT-filling only; low intent confidence stays
  F1's fallback domain (we do NOT add "did you mean order or warranty?"). Flagged, not
  built.

Files Phase B would touch: backend/core/clarify.py (new: slot model + resolver + decide),
agent.py (pre-check + short-circuit + resolved-hint), config.py (clarify_enabled flag),
metrics.py (a clarify field), a system-prompt Behaviour Rule belt, results/
clarify_set_v1.json + backend/scripts/clarify_eval.py + backend/tests/test_clarify.py.
Frontend: none (clarify is a normal assistant message). Token cost: 0 on the shipped
deterministic path.

STOPPING HERE for planner ratification. No feature code written. Open questions for the
planner: (a) deterministic decision + canned/templated question (recommended, gateable,
zero-token) vs an LLM-phrased clarify; (b) confirm SLOT-filling scope only (not intent
ambiguity); (c) on an unavoidable missing slot for a frustrated user, ask warmly
(recommended) or skip straight to the human offer.

## PHASE B - RATIFIED BUILD SPEC
(planner fills after ratifying the Phase A proposal)

## BUILD LOG
(builder fills during Phase B)

## REVIEW
(fresh reviewer fills after)

---

## PHASE B - RATIFIED BUILD SPEC (planner, 2026-07-17)
Proposal APPROVED. The message->profile->history resolve-before-ask order, the
order-vs-warranty asymmetry, and the deterministic/gateable stance are all correct.

Q1 (deterministic vs LLM-phrased): CONFIRM DETERMINISTIC decision + TEMPLATED
question. The no-over-ask property must be testable for certain (Convention 10), so
the DECISION to ask is deterministic and zero-token. The wording is a short,
context-aware TEMPLATE (name the product/context so it is not robotic; warm if F4
flags frustration) - but the decision is never the model's.

Q2 (slot-filling only): CONFIRM. F5 handles MISSING SLOTS (order_status needs an ID;
warranty needs a product/serial when the profile has more than one, or none). It does
NOT do intent disambiguation ("did you mean order or warranty") - that stays F1's job
/ routing v2. No scope creep.

Q3 (frustrated + missing slot): ASK WARMLY by default (one short empathetic line +
the templated ask). BUT if F4's escalation offer is ALREADY firing (sustained
frustration crossed threshold), DEFER - do not stack a form question on an active
human offer; let the escalation stand. Clarify yields to an active F4 escalation.

BUILD:
- clarify.py: deterministic, zero-token pre-check in stream_agent_response, placed
  AFTER the injection guard, F1 routing, and F4 sentiment, BEFORE the tool/answer,
  firing only for order_status and warranty intents.
  * Resolve the slot: order_status -> message, then history (NOT profile - no
    per-user order list exists). warranty -> message, then profile (owned_products),
    then history.
  * Resolved -> inject a hint so the model uses it and does not re-ask; NEVER echo a
    serial the user did not themselves provide (read the profile server-side; do not
    leak it into the question).
  * Unresolved -> stream ONE templated clarifying question (assistant message, no LLM
    call, no new SSE plumbing) and return. Warm line first if F4 flags frustration;
    DEFER entirely if F4 escalation is already firing.
  * RE-ASK GUARD: never ask the same clarify twice in a loop; if the follow-up still
    lacks the slot, give guidance / offer a human rather than re-ask.
- Keep warranty's prompt-level Rule 2 as a soft backup, but the deterministic
  pre-check is authoritative; ensure no double-ask conflict.

GATE (Convention 10 - test the HARD cases):
- NO OVER-ASK: 0 on the PROFILE_RESOLVABLE bucket ("is my thermostat under warranty"
  with a thermostat owned; "is my product under warranty" with exactly one owned) AND
  a HISTORY case (order ID given last turn -> status this turn -> no re-ask). Tested
  on these hard cases, not just obvious ones.
- CORRECT ASK: asks when the slot is genuinely missing/unresolvable (order status
  with no ID anywhere; warranty with multiple owned products, none named). Report
  under-ask too.
- FRUSTRATION INTERACTION: warm-ask on mild frustration; DEFER when F4 escalation is
  active (no stacked form question).
- NO REGRESSION: hit@5 0.913, injection (a jailbreak is refused, never clarified),
  10/10 authz; zero-token; latency near-zero.
Append LEARNINGS: resolve-before-ask (proactivity beats interrogation); decision
deterministic/gateable while only the wording is templated; the order-vs-warranty
slot asymmetry; clarify defers to escalation.

Builder: proceed to PHASE B build, verify the gate, report.

## BUILD LOG (builder, 2026-07-20) - GATE MET

Built to the ratified Phase B spec. Deterministic decision + templated question,
zero-token, slot-filling only, defers to an active F4 escalation.

WHAT SHIPPED:
- backend/core/clarify.py (new): a pure, zero-token `decide(intent, message, profile,
  history, sentiment, settings) -> ClarifyDecision(ask, question, hint, slot, reason)`.
  Fires only for order_status and warranty. Resolve order: message id (regex
  `NX-\d{4}-\d{3}`) then USER-turn history (NOT profile, and NOT the assistant's own
  message, whose example id would false-resolve to a stranger's order). Resolve
  warranty: serial in message, then an OWNED product named in message/history, then a
  non-owned product noun (slot filled, let the tool answer "not registered"), then the
  profile (owns exactly one -> that one), else ask. Ask templates name the product/
  order and never echo a serial. DEFER when sentiment.escalate is set; RE-ASK guard so
  it never loops; a warm lead prepended for a (non-escalating) frustrated user.
- backend/agent/agent.py: the pre-check runs in stream_agent_response AFTER the
  injection guard, F1 routing, and F4 sentiment, BEFORE the RAG/tool dispatch. ask ->
  stream ONE templated assistant message (token + done, no LLM call, no new SSE
  plumbing) and return; resolved-with-hint -> fold the hint into the per-turn guidance
  (below the confidentiality block) so the model uses it and does not re-ask. New
  Behaviour Rule "RESOLVE BEFORE ASKING" as the soft belt; the pre-check is
  authoritative. _build_system_message's per-turn section broadened from "Tone" to
  "Guidance" to carry the tone and/or the clarify hint.
- config.py (clarify_enabled), metrics.py (clarify reason field). results/
  clarify_set_v1.json (labeled set), backend/scripts/clarify_eval.py, backend/tests/
  test_clarify.py (7-test CI gate).

GATE (met):
- NO OVER-ASK (load-bearing, Convention 10): **0.000** over-ask rate; decision accuracy
  **1.000** on 21 items. The HARD resolvable buckets are perfect: profile_resolvable
  5/5 ("is my thermostat under warranty" with a thermostat owned; "is my product under
  warranty" with one owned; a non-owned "my camera"), history_resolvable 3/3 (order id
  from a user turn; product from history; user answering the clarify). CORRECT ASK on
  truly_ambiguous 6/6; under-ask 0.
- DEFER + RE-ASK guard + warm lead: all verified (tests).
- PRIVACY: a clarify question lists product NAMES only, never a serial (test + live: the
  "which product" question shows "Nexora Thermostat Pro, LumiGlow Smart Light").
- JAILBREAK refused, never clarified: the injection guard runs BEFORE the clarify
  pre-check (asserted by source-order test); a jailbreak is refused and never reaches
  clarify.
- NO REGRESSION: hit@5 0.913, injection filter, output guard, 10/10 authz all green;
  full suite 49 passed. ZERO tokens end-to-end: the clarify short-circuit streams the
  question with NO LLM call (verified with GROQ empty); the decision is pure regex,
  ~microseconds.

INCONVENIENT / SCOPE (honest):
- The clarify DECISION uses F1's SEMANTIC intent, so it is active under ROUTER=embedding
  (the default). Under ROUTER=keyword (legacy fallback) decision.intent is "keyword" and
  clarify is a no-op - acceptable, keyword is not the shipped router.
- Slot-filling only (ratified): F5 does NOT do intent disambiguation ("did you mean
  order or warranty"); that stays F1's fallback domain.
- Warranty resolution reuses the tool's own substring matching philosophy but is a
  heuristic (product-noun list + owned-name tokens); unusual product phrasings could
  slip. Bias is toward PROCEEDING (no over-ask) over asking, so a miss degrades to the
  tool's own "not found", never a needless question.
- LIVE end-to-end through the full HTTP endpoint (clarify message persisted, answered
  next turn over Groq+Supabase) is component-verified (module + eval + a direct
  stream_agent_response short-circuit) but not driven through the real endpoint this
  session; recommended for the reviewer.

## REVIEW (fresh reviewer, 2026-07-20)

Scope: F5 as committed (536539e code, a325af6 docs). Re-derived from code and re-ran
the gates; did not trust the write-up. This is a slot-filling feature that reads the
user profile (with serials) and conversation history and can call the order/warranty
tools, so I focused on the cross-user and privacy surface.

VERDICT: CLEAN. The no-over-ask safety property holds under my own probes, the
self-inflicted cross-user exposure bug is genuinely fixed and I verified it directly,
the injection ordering is correct, the clarify path never bypasses the tools' own
authorization, and nothing regressed. One P3 privacy-hygiene note. No P0/P1/P2.

### Independently confirmed (re-ran / re-derived)
- NO OVER-ASK (load-bearing): clarify_eval reports over-ask 0.000, decision accuracy
  1.000 across 21 items, every bucket clean (profile_resolvable 5/5, history_resolvable
  3/3). My own stress probes outside the set agree: "is my thermostat under warranty"
  (owns 2) resolves, "is my Nexora under warranty" (owns 1) resolves via profile, a
  generic ref with 2 owned correctly ASKS, a non-owned product noun does NOT ask
  (named_not_owned -> the tool says "not registered"), and an order query with no id
  asks. It errs toward proceeding, not interrogating.
- THE EXPOSURE FIX IS REAL - verified directly. An assistant turn containing the seeded
  id ("Order NX-2025-301 is processing") plus a vague "has it shipped yet?" does NOT
  resolve (ask=True, empty hint): _history_user_text filters to role=="user", so the
  bot's own example id can never be machine-resolved into a stranger's order lookup. The
  clarify-question case degrades to reask_guard. Both order and warranty history
  resolution search user turns only.
- DEFENSE IN DEPTH: even a user-typed foreign id (I probed "NX-2025-999") only becomes a
  prompt HINT; the tools (check_order_status / check_warranty_status) enforce ownership
  server-side from the ContextVar user_id (Increment 7, in the 10/10 authz suite), so
  clarify can never bypass authorization - the worst case is a hint the tool then
  denies. History itself is the current user's RLS-scoped conversation, so it carries no
  other user's turns.
- INJECTION UNCHANGED: the layer-1 guard runs BEFORE clarify (verified in the source and
  by test_jailbreak_is_refused_before_clarify_not_clarified's source-order assertion);
  a jailbreak is refused, never clarified. The clarify hint sits BELOW the
  confidentiality block, folded into the same guidance slot as the F4 tone.
- PRIVACY: the "which product" question lists product NAMES only ("Nexora Thermostat
  Pro, LumiGlow Smart Light"), never a serial, though the resolver reads serials
  server-side. The serial-ask template asks the user to provide one; it echoes none.
- ZERO-TOKEN: decide() is pure regex/list work (no embedding, even lighter than F1/F4);
  the ask short-circuit streams the templated question and returns with no LLM call
  (verified GROQ empty). No new SSE plumbing.
- NO REGRESSION: full suite 49 passed (hit@5 0.913, injection, output guard, 10/10
  authz, routing, groundedness, sentiment, clarify); ruff clean.

Credit: the builder found and fixed the search-whose-turns exposure themselves and
wrote it up as the sharpest lesson - exactly the self-adversarial instinct this
workstream wants. I confirmed the fix rather than the story.

### Finding

F5-R1 [P3, CONFIRMED] The clarify template's hardcoded example order id is a REAL
  seeded order, shown to every user. _ORDER_ASK (clarify.py:50) reads "...for example,
  NX-2025-301", and NX-2025-301 is Bob's real order (supabase/schema.sql:59, :146:
  "-> Bob"). It is LOW sensitivity - an order-id string with no PII, and the tool denies
  any cross-user lookup - and it is NOT the exposure bug (that was machine resolution,
  fixed). But it is a hygiene inconsistency: the feature that just closed a cross-user
  path THROUGH this example id still DISPLAYS a real customer's order id as its public
  example. Replace it with an obviously-synthetic placeholder (e.g. "NX-XXXX-XXX" or
  "NX-0000-000"). Trivial, and it closes the loop honestly. (Test fixtures and the F1
  routing set also use NX-2025-301, which is fine - the finding is only the user-facing
  template string.)

BOTTOM LINE: F5 is well-built and safe to ship. The safety property (no over-ask) is
real and I could not break it, the cross-user exposure the builder caught is genuinely
closed and I proved it, clarify never bypasses the tools' authorization, injection is
untouched, and nothing regressed. The one P3 is a one-line placeholder swap. The
pending live end-to-end through the HTTP endpoint (builder-disclosed) is the only thing
I did not exercise this session.

---

## F5 CLOSED (2026-07-17) - CLEAN
Reviewer confirmed the load-bearing NO-OVER-ASK property under adversarial probing
(0.000 over-ask / 1.000 decision accuracy, 21 items incl. profile-resolvable +
history hard cases); the exposure fix is real (verified directly); the injection
guard runs before clarify; the clarify question shows product names only, never
serials; zero tokens; no regression (49 tests, hit@5 0.913, 10/10 authz).
Standout: the builder self-caught a cross-user exposure - the order resolver scanned
ASSISTANT turns, and the clarify question itself contains an example order id, so a
vague follow-up could false-resolve to a stranger's order. Fixed to search USER turns
only (LEARNINGS Part 21: "watch whose words you search" - the bot's own output is
attacker-influenceable content). Defense-in-depth held regardless (tools enforce
ownership server-side; history is RLS-scoped). Commits 536539e, a325af6.
Trivial ride-along required: F5-R1 [P3] - the clarify template hardcodes a REAL
seeded order id (NX-2025-301 = Bob's) as its example; swap to a synthetic placeholder
(NX-XXXX-XXX). Include it in the release-prep commit. F5 done.
