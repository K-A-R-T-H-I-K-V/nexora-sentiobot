# Feature F2 - Groundedness guard + inline source highlighting

Status: PHASE A (inspect + propose). Builder does Phase A ONLY, then STOPS for
planner ratification. Standing Conventions + FEATURE CADENCE in CLAUDE.md apply.

## Goal
Make every answer visibly grounded, the trust feature. Two parts:
(1) a GROUNDEDNESS CHECK on the RAG answer (is it actually supported by the
    retrieved context) surfaced as a badge, and a soft-withhold / "partially
    grounded" when it is not (never a false green);
(2) INLINE SOURCE HIGHLIGHTING: show the exact source sentence(s) behind the
    answer's claims in the sources panel.

## PHASE A - INSPECT + PROPOSE (do this now, then STOP; write no feature code)
Grep/read the real code and write a PROPOSAL here covering:
- The current RAG answer path: how the sources list is built and what the `done`
  SSE payload contains; how the frontend renders sources today (the sources panel
  in frontend/app/chat/page.tsx) and how the SSE event types are handled.
- GROUNDEDNESS-CHECK design options + recommendation:
  * Cheapest: a LOCAL heuristic/NLI (embedding or n-gram overlap between the
    answer's claims and the retrieved context) - zero-token, preferred if it is
    accurate enough.
  * Fallback: one llama-3.1-8b-instant "is this answer supported by this context?
    which sentences?" call - small token cost.
  Recommend one; state the token cost and where it runs in the graph.
- INLINE-CITATION design options + recommendation: generation-time (prompt the
  model to emit the quoted supporting span per [Source N]) vs post-hoc (match
  answer claims to source spans locally). Recommend one.
- CONTRACT changes: what you add to the `done` event (e.g. `grounded` score +
  `citations` spans) and the frontend render branch (badge + highlighted spans).
  Confirm no new SSE plumbing is needed (extend the existing `done`).
- Proposed GATE: how you will validate the guard against RAGAS faithfulness on the
  FROZEN golden set (it must not claim grounded where faithfulness is low), the
  token-cost target, and no regression (hit@5 0.913, injection + 10/10 authz).
- Risks: a FALSE-GREEN badge is worse than none; token budget of the 8B path;
  latency; citation spans that do not actually appear in the source.
STOP after the proposal. Await the planner's ratified Phase B build spec below.

## PHASE A PROPOSAL (builder, 2026-07-19) - awaiting ratification

### 1. Current RAG answer + sources path (code reality)

Sources are metadata-only; the retrieved TEXT never reaches the client.
- `_format_sources` (backend/agent/agent.py:236) maps each retrieved Document to
  `{"source": <filename>, "section": <section_title>}`. No page_content, no span.
- RAG path: `_retrieve_context` (agent.py:246) returns `(context_str, sources)`.
  `context_str` (agent.py:261, tagged `[Source N] (file | section)\n<text>`) is fed
  to the LLM and then DISCARDED; only `sources` (metadata) survives to `done`.
- Tool path: `lookup_documentation` (agent.py:269) returns JSON
  `{"context": ..., "sources": ...}`; `_extract_tool_sources` (agent.py:286) keeps
  only `sources` and throws away `context`.
- The `done` SSE payload is `{"answer", "sources"}` on both paths (agent.py:589 RAG,
  agent.py:670 tool). The injection-refusal, empty-context, and output-guard-blocked
  exits send `sources: []` (agent.py:525/553/586/667).
- main.py `/chat/stream` generate() (backend/api/main.py:242) parses the SSE, and on
  `done` it injects `interaction_id` and RE-DUMPS the whole payload
  (main.py:263-266). Consequence that matters: ANY extra key added to the `done`
  data dict in agent.py passes through to the client unchanged. No new SSE event
  type or plumbing is needed. The cache-hit path (main.py:206) sends its own `done`
  with `sources: [], cached: true` and no context.
- Persistence: only `sources` is stored, in the assistant message metadata
  `{"sources": sources}` (main.py:275) and in analytics `retrieved_docs` (main.py:283).
- The answer text ALREADY contains inline `[Source N]` markers: the system prompt
  rule 1 (agent.py:365) and the lookup_documentation docstring (agent.py:274)
  instruct "answer ONLY from context, cite inline as [Source N]", and the context is
  built with `[Source N]` tags. So markers exist in the prose, but there is NO
  mapping from `[Source N]` to a supporting sentence, and the source text is not sent.

Frontend rendering.
- `Source = {source, section}` (frontend/lib/api.ts:30); the `done` event type is
  `{answer, sources, cached?, interaction_id?}` (api.ts:45). `streamChat`
  (api.ts:130) yields each parsed SSE event generically, so new `done` fields flow
  through once the TS type + render branch are extended.
- Chat page: `UIMessage.sources?` (frontend/app/chat/page.tsx:52); the `done`
  handler sets `content/sources/interactionId/isStreaming` (page.tsx:178). Sources
  render in a collapsed `<details>` panel, one row per source as
  `<source> - <section>` (page.tsx:341-356), no text. History reload maps
  `m.metadata?.sources` (page.tsx:102). `metrics` events are ignored (no case).

RAGAS faithfulness harness (for the gate): backend/scripts/quality_baseline_llm.py,
judge = `llama-3.1-8b-instant` (line 30, distinct from the 70B generator),
faithfulness judged against the RETRIEVED context (line 198), over a doc/policy
subset of the FROZEN golden set, 2 runs. Named "indicative, weak 8B judge" per
Increment 3. The local zero-token embedder `get_embeddings()` (ONNX MiniLM,
backend/core/onnx_embeddings.py) is reusable here exactly as F1 reused it.

### 2. Groundedness-check design (options + recommendation)

Core idea shared by both options: split the answer into claim sentences and the
retrieved context into source sentences; a claim is "supported" if some source
sentence backs it. Groundedness score = supported_claims / total_claims.

- Option A (LOCAL, zero-token, RECOMMENDED as primary): cosine between each answer
  claim and each source sentence using the in-memory ONNX MiniLM. Reuses F1's
  embedder, adds NO LLM call and ~0 tokens. Runs post-answer (see section 5).
- Option B (8B NLI, small token cost): one `llama-3.1-8b-instant` call per doc
  answer: "which answer sentences are supported by this context?" Cost roughly
  context+answer in / short out, order ~0.8-1.5K tokens per RAG answer, i.e. a
  ~30-60% increase over the current ~2.4K/answer and a matching cut to daily
  free-tier capacity. Better at entailment than overlap.

Recommendation: ship Option A as the default, because it is zero-token and on-brand,
BUT make the badge conservative and 3-state so its known weakness (below) cannot
produce a false green, and keep Option B behind a config flag
(`GROUNDEDNESS_NLI=on`, default off) as a ratified escalation if the golden-set
validation shows Option A is not accurate enough. The honest weakness of Option A,
stated up front: embedding overlap measures TOPICAL SUPPORT, not ENTAILMENT, so it
cannot reliably catch a contradiction ("warranty does NOT cover water" vs a claim
that it does) or a wrong number (source "2 years", answer "3 years") - both can be
embedding-similar. This is exactly why the badge must be conservative and why the
gate's job is to prove no-false-green, not to chase a high absolute score.

Badge semantics (conservative, never false-green):
- `grounded` (green): EVERY claim has a supporting source sentence above threshold T.
- `partial` (amber): some but not all claims supported (this is the soft-withhold
  state the kickoff asks for: show the answer, but flag it as not fully verified).
- `unverified` (grey): no retrieved context or near-zero support (e.g. the
  "could not find documentation" reply).
T and the grounded/partial cutoffs are CALIBRATED on the golden set (section 5) and
committed, never hand-tuned to a single example.

Scope (important, avoids a misleading badge): the check applies to DOC-GROUNDED
answers only - the RAG path, and tool-path answers that actually called
lookup_documentation. Pure tool-action answers (order/warranty status via
check_order/warranty_status, which have no retrieved context) get NO groundedness
badge rather than a false one; their trust signal is the tool-call badge that
already exists. This must be explicit so an order-status answer is not painted
grey/amber as if it were an unsupported doc answer.

### 3. Inline-citation design (options + recommendation)

- Option A (post-hoc LOCAL extraction, RECOMMENDED): the SAME claim-to-source-
  sentence matching produces, per cited `[Source N]`, the exact source sentence(s)
  that back the answer. Spans are EXTRACTED from the retrieved text, so they
  provably appear in the source (the gate asserts substring). Zero-token, and it
  reuses the groundedness computation (one pass yields both the score AND the spans).
- Option B (generation-time): prompt the model to emit a quoted supporting span per
  `[Source N]`. Costs output tokens and can HALLUCINATE a quote that is not in the
  source (the precise risk the kickoff names). Rejected as primary for that reason.

Recommendation: Option A. It is the natural byproduct of the groundedness pass and
carries a hard guarantee (spans are real source text). The existing `[Source N]`
markers in the prose stay; the panel gains the highlighted supporting sentence(s).

### 4. Contract changes (extend `done`, no new plumbing)

Add two keys to the existing `done` data dict (they pass through main.py unchanged):
- `grounded`: `{"score": float 0..1, "label": "grounded"|"partial"|"unverified",
  "supported": int, "total": int}` (omitted / label unset for pure tool answers).
- `citations`: `[{"n": <source index>, "source", "section", "spans": [<exact source
  sentence>, ...]}]`.
Frontend: extend `Source`/`done` types + `UIMessage` (api.ts, page.tsx); render a
groundedness BADGE by the answer (green/amber/grey with the score and a tooltip) and
the highlighted `spans` inside the existing sources `<details>` panel. Persist
`grounded` + `citations` in the assistant message metadata (main.py:275) and map
them on history reload (page.tsx:102) so the badge survives a refresh. Cache-hit
badge: the cache stores only the answer text, so a cache hit has no context to
recompute from; propose either persisting/replaying the stored grounded+citations or
DEFERRING the cache-hit badge (show none) - planner to rule; leaning defer to keep
scope tight.

### 5. Where it runs + token cost

A post-answer, pre-`done` step inside stream_agent_response, after `full_answer` is
assembled and while the retrieved text is in scope. It does NOT affect
time-to-first-token (the answer has already streamed); it adds a small CPU-only
delay (embedding a handful of answer + source sentences, tens of ms) before the
`done`/badge appears. Requires threading the per-source retrieved TEXT to the check:
the RAG path already has the docs in `_retrieve_context` (today it drops
page_content in `_format_sources`); the tool path needs `context` captured from the
lookup_documentation JSON (extend `_extract_tool_sources`). Both are small, contained
changes. Token cost of the shipped default: 0 (local ONNX). If `GROUNDEDNESS_NLI` is
ratified on, state its measured per-answer token cost in the results recipe.

### 6. Proposed GATE

Zero-token deterministic core (CI-able, the headline):
- Citation spans are SUBSTRINGS of the retrieved source text (no hallucinated
  quotes) - a hard assertion.
- No-false-green, synthetic: take a known-good grounded answer and inject one
  unsupported claim; the guard must drop it out of `grounded` (to partial). A fully
  supported known answer must be labeled `grounded`. This is the core safety property.
- No regression: hit@5 0.913, injection red-team, 10/10 authz all still green (this
  feature is post-answer and touches neither retrieval, routing, nor authz).
- A test_groundedness.py mirroring test_routing.py.

Correlation vs RAGAS faithfulness (local/paid, NOT in CI, indicative):
- On the golden doc/policy answerable items, compute the local groundedness label
  AND the existing 8B RAGAS faithfulness; the gate PASSES only if there are ZERO
  false greens (no item the guard calls `grounded` that the judge scores low
  faithfulness). Report agreement + the calibrated thresholds, stamped
  (commit/model/date) like every results file. Lead with the deterministic no-false-
  green property; treat RAGAS as indicative (weak 8B judge, per Increment 3), and if
  the two disagree, read the actual answers before trusting the judge (the
  Increment 4 lesson).
- Token target: 0 for the shipped path; the RAGAS validation spends the same 8B
  judge tokens the Increment 3 harness already budgets, run locally across days.

### 7. Risks

- FALSE-GREEN is worse than no badge (the cardinal risk). Mitigated by the
  conservative 3-state label (green requires ALL claims supported), the no-false-
  green gate, and extraction-based (not generated) citation spans.
- Embedding overlap is not entailment: it can miss a contradiction, a negation, or a
  wrong number that stays topically similar. Named honestly; the 8B NLI escalation
  (Option B, config-flagged) is the mitigation IF the golden-set validation shows the
  local path is not safe enough. The planner should rule on the zero-token vs
  accuracy tradeoff after seeing the measured no-false-green numbers.
- 8B token budget: if the NLI path is enabled, ~0.8-1.5K tokens/answer, cutting free-
  tier daily capacity meaningfully; that is why it is off by default.
- Latency: post-stream CPU embed only; does not affect TTFT; bounded to a few
  sentences.
- Scope confusion: a badge on a pure tool-action answer would mislead; the design
  suppresses it there. Cache-hit answers have no recomputable context (defer or
  persist-and-replay).

STOPPING HERE for planner ratification. No feature code written. The two open
questions for the planner: (a) ship local-only zero-token and rely on the
conservative label + no-false-green gate, or ratify the 8B NLI escalation for the
groundedness LABEL despite its token cost; (b) cache-hit badge - defer, or persist
and replay the stored grounded/citations.

## PHASE B - RATIFIED BUILD SPEC
(planner fills after ratifying the Phase A proposal)

## BUILD LOG (builder, 2026-07-19) - GATE MET

Built to the ratified Phase B spec. Local-only, zero-token; 8B NLI kept behind a
default-off flag.

WHAT SHIPPED:
- backend/core/groundedness.py (new): the local pass. Splits the answer into factual
  claims (filters headers, colon lead-ins, citation-apparatus/filename references,
  pleasantries) and the retrieved source into sentences, cosine-matches with the
  in-memory ONNX MiniLM (0 tokens), and returns a 3-state label
  (grounded/partial/unverified: grounded requires EVERY claim matched) + citations
  (the literal source sentence per matched claim). Caps: max_citations=6,
  span_max=240, max_source_sentences=120. GROUNDEDNESS_NLI (off) adds an 8B overlay
  that can only DOWNGRADE.
- backend/agent/agent.py: _retrieve_context now also returns per-source text;
  lookup_documentation carries source_texts; a post-answer _groundedness_payload
  runs via asyncio.to_thread and emits `grounded` + `citations` on the done event,
  SCOPED to doc-grounded answers (empty source_texts -> no badge, so pure
  order/warranty tool answers get none). Fails safe: any error -> no badge, never a
  false one, never blocks the answer.
- backend/api/main.py + backend/services/cache.py: PERSIST-AND-REPLAY. cache stores a
  structured entry {answer, grounded, citations, sources} (new get_cached_entry; the
  string get_cached_response kept for compat); the endpoint persists grounded +
  citations in message metadata and replays them on a cache hit. Cross-user cache
  privacy preserved (check_cache_privacy PASS).
- backend/core/config.py (flags) + metrics.py (groundedness_ms, measurement only).
- frontend: 3-state badge (honest wording: "Grounded/Partially grounded/Unverified",
  never "verified/correct") + inline highlighted citation spans in the sources panel.
  Spans render as TEXT ({c.span}, React-escaped) - XSS-safe, no dangerouslySetInnerHTML.
  grounded+citations flow through the done event, persist in metadata, and render on
  reload.
- results/routing... no. results/groundedness_validation.{json,md} (stamped);
  backend/scripts/groundedness_validation.py; backend/tests/test_groundedness.py.

GATE (met):
- NO FALSE GREEN (deterministic): test_groundedness poison test + the validation
  harness's synthetic poison over 5 real answers - an injected unsupported claim
  never stays "grounded". PASS.
- Citation spans are literal SUBSTRINGS of a source (anti-hallucination): asserted in
  test_groundedness. PASS.
- Scope: no badge without retrieved sources (pure tool answers). PASS.
- VALIDATION vs 8B RAGAS on the frozen golden set (indicative, 0 tokens, reuses the
  Increment 4 human-verified answers + committed faithfulness): local greened doc-03
  (RAGAS 1.0) and doc-09 (0.8); greened doc-12 which RAGAS scored 0.0 - READ the
  answer: it is genuinely grounded (weather-resistant/not-waterproof matches the
  manual), so the weak judge erred (Increment 4 failure mode), NOT a false green;
  pol-02/pol-03 conservatively "partial" (the safe direction). Threshold 0.5 EARNED by
  this validation, not guessed.
- NO REGRESSION: hit@5 0.913 (test_retrieval_eval), injection filter + output guard +
  10/10 authz all green; full suite 31 passed. Default path 0 tokens (no LLM call).
- Latency: local pass ~0.8-1.0s (measured), POST-STREAM + threaded, so no TTFT impact;
  bounded by max_source_sentences. Memoizing the fixed corpus's source embeddings is a
  logged forward optimization.
- XSS-safe citation rendering (text nodes only). ruff + frontend build green.

INCONVENIENT / DEFERRED (honest):
- The ~0.8-1.0s post-stream latency is real; the badge resolves about a second after
  the answer finishes. Forward optimization: cache source-sentence embeddings.
- Topical overlap is not entailment (can miss a wrong number / negation). The 8B NLI
  flag is the documented mitigation (default off); the human backstop is the shown
  source sentence. Wording avoids "verified/correct" for exactly this reason.
- pol-02/pol-03 read "partial" though human-faithful (a list-item fragment did not
  match) - the SAFE direction; documented, not gold-plated.
- LIVE end-to-end (badge in the browser, cache-hit + reload replay against real
  Groq/Supabase) is component-verified (module + cache round-trip + gate) but not run
  against live infra this session; recommended for the reviewer.

## REVIEW (fresh reviewer, 2026-07-19)

Scope: F2 as committed (a1cf089 code, eb0f415 docs), with F1-R5 and F4 layered on
top since. I re-derived from the code and re-ran the gates; I did not trust the build
log.

VERDICT: STRONG engineering, ship-worthy, but the headline "NO FALSE GREEN
(deterministic) PASS" is narrower than stated. The design is genuinely good - extracted
(not generated) citations make hallucinated quotes structurally impossible, the badge
wording is honest, and the feature fails safe and is correctly scoped. One P2: a class
of false green the committed gate does not test. One P3 confirming the disclosed
entailment weakness. No P0/P1, no regression, no security issue.

### Independently confirmed (re-ran / re-derived)
- Zero-token default: groundedness_nli defaults False, threshold 0.5; the whole suite
  (incl. test_groundedness -> analyze()) runs green with GROQ_API_KEY empty, so the
  default path makes no LLM call. The 8B NLI overlay is flag-gated and can only
  DOWNGRADE (conservative AND), so it cannot introduce a false green either.
- Anti-hallucination citations: spans are raw slices of the source text (split ->
  strip -> word-boundary truncate, all substring-preserving); test asserts every span
  is a literal source substring. Logic sound, test real.
- Fail-safe + scope: _groundedness_payload (agent.py:285) wraps analyze in try/except
  -> returns {} (no badge) on any error, and returns {} for empty source_texts, so a
  pure order/warranty tool answer gets NO badge (verified in code + test). A failed
  check never blocks the answer and never shows a false badge.
- Cache privacy preserved: the F2 refactor changed only the stored VALUE (now a JSON
  entry {answer,grounded,citations,sources}); keying still goes through
  _cache_key(user_id,query) and the user-scoped semantic tier. check_cache_privacy.py
  PASS (no cross-user leak, same-user hit works). get_cached_entry is user-scoped like
  get_cached_response; legacy bare-string entries are handled.
- XSS-safe + honest wording: citation spans render as escaped React text
  (&ldquo;{c.span}&rdquo;), never dangerouslySetInnerHTML; badge labels are
  Grounded/Partially grounded/Unverified (never "verified"/"correct"), and the tooltip
  says "checks source support, not factual correctness". The mitigation the whole
  design rests on is genuinely present.
- No regression: full suite 38 passed (hit@5 0.913, injection filter, output guard,
  10/10 authz, routing, groundedness, sentiment). 
- Validation is real + stamped; the one disagreement I re-read myself: doc-12 (local
  grounded vs RAGAS 0.0) is a genuine weak-8B-judge error - the answer ("weather-
  resistant, IP65 rated, not fully waterproof", cited to the manual's Safety section)
  is properly grounded, NOT a false green. Local was right there.

### Findings

F2-R1 [P2, CONFIRMED] The "no false green" gate is narrower than the claim: SHORT or
  HEDGED unsupported factual claims evade the claim filter and leave the answer labeled
  "grounded". Distinct from the disclosed entailment weakness (this is claim SELECTION,
  not topical overlap). Reproduced against the test's own source (a 2-year warranty):
  - "...covers defects for two years [Source 1]. Ships worldwide free."
    -> grounded 1/1. The unsupported "Ships worldwide free." is a real factual claim
    but 21 chars < the 25-char floor in _is_factual_claim, so it is dropped from the
    denominator and cannot downgrade the label.
  - "...[Source 1]. Of course it also includes a free smart speaker."
    -> grounded 1/1. "of course" trips the _NONFACTUAL regex, so the fabricated
    smart-speaker claim is dropped.
  The committed test_poisoned_answer_is_never_green (and the validation's synthetic
  poison) inject a LONG off-topic fabrication, which IS caught -> partial, so CI is
  green while the property fails for ordinary short/hedged phrasings a real LLM emits.
  Fix: add a short-claim and a hedged-claim poison case to test_groundedness (they fail
  today, exposing the gap); then either stop excluding short/hedged FACTUAL claims from
  the denominator, or narrow the gate's wording to "no false green for on-topic-
  dissimilar claims above the length floor" and document this evasion class alongside
  the entailment one. It is a claim-scope/eval-coverage gap (the same shape as F1-R5),
  not a security hole: the badge already disclaims correctness and shows the source.

F2-R2 [P3, CONFIRMED - disclosed tradeoff] Wrong-number and negation answers get a
  GREEN "Grounded" badge. Verified: source "two years" / answer "five years [Source 1]"
  -> grounded 1.0; source "water damage ... are not covered" / answer "Water damage is
  covered [Source 1]" -> grounded 1.0, and the citation SHOWN literally reads "are not
  covered", contradicting the answer. This is the ratified, disclosed
  topical-overlap-is-not-entailment tradeoff, and the mitigations are present (honest
  wording + the contradicting source sentence is displayed for the user to catch it).
  Noting it because green (not amber) on a self-contradicting answer is a sharp UX
  signal that rests entirely on the user reading the source; consider downgrading to
  "partial" when a citation contains a negation the claim lacks, or a number mismatch.
  Non-blocking; the design ruled this acceptable.

BOTTOM LINE: F2 is well-built and safe to ship - honest wording, real anti-hallucination
citations, correct scoping, fail-safe, privacy-preserving, zero-token, no regression.
Close F2-R1 (add the short/hedged poison tests and either count those claims or narrow
the "no false green" wording) so the gate's headline matches reality; F2-R2 is the
accepted, disclosed tradeoff, optionally hardened with a negation/number-mismatch
amber downgrade.

---

## PHASE B - RATIFIED BUILD SPEC (planner, 2026-07-17)
Proposal APPROVED. The extraction-based, substring-guaranteed citation design is
excellent: it makes hallucinated quotes structurally impossible, and the shown
source sentence is the REAL trust mechanism. The groundedness score is a
supporting signal, not a correctness guarantee. Two decisions ruled:

DECISION 1 (local vs 8B NLI): SHIP LOCAL-ONLY (zero-token) as default; keep the 8B
NLI path behind a config flag (default OFF) as the documented mitigation. Guard
rails:
- DO NOT OVERCLAIM. The flagged tension is correct: cosine overlap = TOPICAL
  support, not entailment; it cannot catch a topically-similar wrong number or a
  negation. So the badge must NOT say "verified"/"correct"/"entailed". It claims
  only that each claim MATCHES a retrieved source passage, and it SHOWS that
  passage so the USER makes the final check. The highlighted source sentence is
  the backstop to the badge's weakness (a wrong "5 years" next to a source saying
  "2 years" is caught by the human precisely because we show the source).
- The final badge WORDING and the green-bar threshold are DECIDED BY VALIDATION,
  not pre-committed: validate the local groundedness score against 8B RAGAS
  faithfulness on the FROZEN golden set. If local reliably flags the unfaithful
  answers, ship local with honest wording. If local greens answers RAGAS calls
  unfaithful (false greens), tighten the wording/bar OR enable the 8B path. Report
  the comparison; ship no label the validation did not earn.

DECISION 2 (cache-hit badge): PERSIST-AND-REPLAY. Store grounded + citations in
the message metadata (alongside sources, already persisted) and with the cache
entry; replay on a cache hit and on conversation reload. Same answer must show the
same badge everywhere.

BUILD:
- Backend: a local groundedness+citation pass after the RAG answer. RETAIN the
  retrieved source sentence text (currently discarded at agent.py:246 and the tool
  path at :286). Split answer into claims, context into sentences, cosine-match
  with the in-memory ONNX MiniLM (F1's embedder, zero tokens). Emit on `done`:
  `grounded` (3-state grounded/partial/unverified + score) and `citations` (per
  matched claim: the literal source sentence + source/section). CAP citation count
  and span length to bound payload. Scope to DOC-grounded answers only (no badge
  on pure order/warranty tool answers).
- Frontend: 3-state badge (honest wording) + inline highlighted source spans.
  Render citation text ESCAPED / as text (never dangerouslySetInnerHTML on source
  content) to avoid XSS if source text ever contains markup. Persist
  grounded+citations in message metadata; render on reload.
- Config: GROUNDEDNESS_NLI flag (default off) for the 8B escalation path.

GATE:
- DETERMINISTIC / CI-able: (a) NO FALSE GREEN (green only if every claim matches a
  source above the ratified threshold); (b) every citation span is a literal
  SUBSTRING of a retrieved source (assert in a test) - structural anti-
  hallucination; (c) scoped to doc answers only.
- VALIDATION (indicative): local groundedness vs 8B RAGAS faithfulness on the
  frozen golden set; report disagreements; this decides final wording + whether 8B
  is needed. Budget the RAGAS run; log tokens.
- NO REGRESSION: hit@5 0.913, injection red-team, 10/10 authz green; default path
  ~0 tokens (no LLM call); local-pass latency measured + bounded.
- PERSIST-AND-REPLAY consistency verified (cache hit + reload show the same badge).
- XSS-safe citation rendering.
Append docs/LEARNINGS.md: extraction-vs-generation citations (extracted = no
hallucinated quotes), topical-overlap-is-not-entailment (why the badge must not
overclaim + the human-in-the-loop backstop), validate-the-label-before-you-ship.

Builder: proceed to PHASE B build to this spec, then verify the gate and report.

---

## F2 CLOSED (2026-07-17) - engineering CLEAN
All gates met: no-false-green (synthetic poison PASS); citation spans are literal
source substrings (asserted); the 0.5 threshold was EARNED by validation vs 8B
RAGAS on the frozen golden set - and the builder correctly caught doc-12 as a
WEAK-JUDGE error (RAGAS 0.0 but the answer is genuinely grounded; local was right,
same failure mode as Increment 4), which is local being correct, not a false green.
No regression (hit@5 0.913, injection, output guard, 10/10 authz; 31 tests). Zero
tokens on the default path; 8B NLI behind GROUNDEDNESS_NLI (off). Persist-and-replay
+ XSS-safe rendering done. Honest residual: ~0.8-1.0s local-pass latency
(post-stream, threaded, no TTFT impact; badge resolves ~1s after the answer);
forward opt logged (memoize corpus sentence embeddings).
DEFERRED VERIFICATION (explicit, not a blocker): the live badge-in-browser +
cache-hit/reload replay against real Groq/Supabase was component-verified but NOT
run on live infra this session. Fold it into the convention-9 LIVE SMOKE at the
next main release PR (F2 is on v2-fullstack, not yet on main/prod).
Commits: a1cf089 feat(groundedness); eb0f415 docs(f2). F2 done.

---

## POST-CLOSE CORRECTION - F2-R1 [P2] (planner, 2026-07-17)
A later reviewer found the "no false green" gate is narrower than the headline:
short or hedged unsupported factual claims EVADE the claim filter and still get a
"grounded" badge (a <25-char unsupported clause is dropped; an "of course ..."
clause trips the non-factual filter). The committed poison test only injects a LONG
off-topic fabrication - the easy case - so CI is green while the property fails for
ordinary phrasings. Same shape as F1-R5. This must close before "no false green"
becomes a resume line.
F2-R1 FIX:
- WIDEN the poison set: add short and hedged unsupported-claim cases (they fail
  today). These become the regression guard.
- FIX the property so it actually holds: do not drop short factual clauses from
  claim extraction, and do not treat hedged openers ("of course ...") as
  non-factual. If full coverage is impractical, NARROW the badge wording so
  "grounded" does not overclaim for those - but prefer fixing the property.
- Re-run; report honestly. Per new STANDING CONVENTION 10, the gate must hold on
  the HARD cases, not just the obvious one.
F2-R2 [P3, disclosed] wrong-number/negation greens (the ratified topical-overlap
tradeoff): OPTIONAL amber-downgrade on a detected number/negation mismatch; not
required, but a cheap, high-trust win if easy.

## F2-R1 FIX DONE (builder, 2026-07-19)
Fixed the PROPERTY (convention 10: hold on the hard cases), not the wording.
- Claim extraction no longer drops short or hedged FACTUAL claims: the length floor
  dropped 25 -> 12 chars and the word floor 4 -> 3, so "Ships worldwide free." (3
  words) counts; hedged openers ("of course", "sure", "certainly", ...) are STRIPPED
  as a prefix (leaving the claim behind them) instead of being treated as non-factual,
  and "of course"/"sure thing" were removed from the pleasantry filter.
- WIDENED the poison set with two named regression tests that FAILED before the fix:
  test_short_unsupported_claim_is_not_green ("Ships worldwide free.") and
  test_hedged_unsupported_claim_is_not_green ("Of course it also includes ..."). Both
  now downgrade to partial. The old long-off-topic poison test still passes.
- NO REGRESSION: the increment4 real-answer validation is byte-identical to the F2
  build (doc-03/09/12 grounded, pol-02/03 partial, synthetic poison PASS) - the lower
  floors were not binding on real full-sentence claims. Full suite 42 passed.
- F2-R2 (wrong-number/negation greens) remains the DISCLOSED entailment tradeoff, not
  touched here (the badge wording already disclaims correctness and shows the source).
