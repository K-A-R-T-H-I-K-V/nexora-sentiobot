# Feature F1 - Intent-aware routing

Status: PHASE A (inspect + propose). Builder does Phase A ONLY, then STOPS for
planner ratification. Standing Conventions in CLAUDE.md apply (do not restate).

## Goal
Replace the brittle keyword router in stream_agent_response (the day-one flaw:
"what does the warranty policy cover" misroutes to the tool path because it
contains the word "warranty") with a real, zero-token, LOCAL intent classifier.

## PHASE A - INSPECT + PROPOSE (do this now, then STOP; write no feature code)
Grep/read the real routing code and write a PROPOSAL here covering:
- Where routing happens today and exactly what it decides (RAG path vs LangGraph
  tool path). Quote the current keyword check.
- The intent classes you propose (starter set: doc_lookup, order_status, warranty,
  ticket_or_escalation, chitchat, out_of_scope) and how each maps to a route.
- Classifier design options with a recommendation. Planner's steer: embedding
  based (local ONNX MiniLM, zero-token, cosine-match to labeled prototype phrases,
  confidence threshold + fallback). Say so if inspection suggests otherwise.
- Files you would change; the config flag to keep the old router A/B-able; the
  proposed LABELED ROUTING SET (size, sources, where it lives - SEPARATE from the
  frozen quality golden set); the token cost (target ~0); the proposed GATE.
- Risks (misclassification, latency, any hidden LLM call).
STOP after writing this. Await the planner's ratified Phase B build spec below.

## PHASE B - RATIFIED BUILD SPEC
(planner fills this after ratifying the Phase A proposal)

## BUILD LOG
F1 was built in a SINGLE pass off the earlier bundled kickoff in STATUS (which
predated this two-phase feature file), so PHASE A/B above stayed as placeholders;
the planner retroactively ratified that timing artifact (see RATIFICATION). The
full build record lives in STATUS-prod.md under "Feature F1 - INTENT-AWARE ROUTING";
this is the short pointer the reviewer asked for.

What changed: backend/agent/intent_router.py (new; local ONNX MiniLM intent
classifier, prototypes embedded once, max-cosine match, keyword fallback below a
0.35 threshold), agent.py (keyword block -> route_message), config.py (ROUTER flag
+ intent_confidence_threshold), backend/core/metrics.py (intent field),
results/routing_set_v1.json (28-query labeled set), backend/scripts/routing_eval.py
(A/B harness + leakage guard), backend/tests/test_routing.py (7-test CI gate),
results/routing_eval.{json,md}.

Verified vs the gate: embedding 0.929 vs keyword 0.714 (delta +0.214) on 28 queries;
6/8 keyword misroutes fixed (2 mixed-intent residuals disclosed); zero-token (no LLM
call, GROQ key empty); hit@5 0.913 + injection + 10/10 authz unregressed; ruff +
full pytest green.

F1-R1 (eval leakage) fixed the CLEAN way before commit: the reviewer named 2 leaked
eval items; the full diff (normalized string match OR cosine >= 0.90) found 5, all
replaced with independent phrasings; the concrete-serial prototype generalized; the
false "not copied" comment corrected; and "no leakage" made an ENFORCED invariant in
routing_eval.py + test_routing.py. Honest post-fix number is UNCHANGED at +0.214
(replaced, not dropped; leaked items were both-router-correct filler, so the delta
was never leak-dependent). F1-R2 (non-English fallback) and F1-R3 (threshold noise
band) recorded as documented limitations. F1-R4 (result stamp) closes on the
post-commit re-stamp.

## REVIEW (fresh reviewer, 2026-07-18)

Scope reviewed: the uncommitted working-tree F1 change on v2-fullstack:
backend/agent/intent_router.py (new), the agent.py routing swap, config.py
(ROUTER flag + threshold), metrics.py (intent field), backend/scripts/routing_eval.py,
backend/tests/test_routing.py, results/routing_set_v1.json, results/routing_eval.*,
LEARNINGS Part 17. I re-derived from the code and re-ran the gates myself; I did not
trust the STATUS numbers.

VERDICT: STRONG, no P0/P1 blockers. The feature does what it claims: it is genuinely
zero-token and deterministic, it reproduces bit-identically, it beats the keyword
router on the real misroute cases, and it does not regress any frozen gate. There is
one real eval-integrity finding (a false "no overlap" claim in the code plus a small
leaked item) and three low-severity notes. None of them changes the headline
conclusion. Fix F1-R1 before promotion; the rest are dev's call.

### What I independently reproduced (re-ran, did not read)
- Routing A/B, twice, into a scratch results dir: keyword 0.7143 / embedding 0.9286,
  delta +0.2143, 6/8 keyword-misroutes fixed, 1 low-confidence fallback. My two runs
  are byte-identical to each other AND to the builder's committed results/routing_eval.json
  rows: every per-row confidence matches to < 1e-6. Deterministic, stable, honest.
- Zero-token: both the eval and route_message() run with GROQ_API_KEY empty and make
  no LLM call (a provider call would raise on the empty key). Routing is one local
  ONNX MiniLM embed of the message against a cached prototype matrix. Confirmed.
- No frozen-gate regression:
  * hit@5 == 0.913 with misses [doc-01, doc-13] (test_retrieval_eval hard-asserts the
    exact value and miss set; passed). F1 does not touch retrieval.
  * results/golden_set_v1.json (the FROZEN quality set) is untouched by this change.
  * Injection defense is NOT weakened: the layer-1 refusal (_is_injection) runs at
    agent.py:523 and returns route="refused" BEFORE route_message() is reached at
    agent.py:541, so an injection attempt never reaches the classifier. Verified in code.
  * Authorization denial suite: test_authz.py 10/10 passed offline.
  * Full deterministic suite (routing + retrieval + injection filter + output guard +
    authz): green.
- Adversarial input battery (my own, none in the 28-item set): empty, whitespace,
  100K-char message, prompt-injection string, gibberish, SQL string, emoji-only,
  non-English. NO crashes; every input returns a clean RouteDecision, and the
  low-confidence fallback correctly catches gibberish/emoji/SQL/injection/non-English
  and defers rather than confidently misrouting. Graceful degradation confirmed.

### Findings

F1-R1 [P2, CONFIRMED] Eval leakage + a false eval-integrity claim in the code.
  file: backend/agent/intent_router.py:67-72 (the prototype-block comment) and
  results/routing_set_v1.json (r-war-02, r-oos-01).
  The code comment states the prototypes are "GENERIC exemplars ... deliberately NOT
  copied from the labeled routing eval set ... so scoring the classifier on that set
  is not grading it against its own prototype text." That claim is false:
  - r-war-02 "Check the warranty status for serial number SN-NTS-PRO-XYZ987." is a
    character-identical copy (modulo case + trailing period) of the prototype at
    intent_router.py:101 "check the warranty status for serial number SN-NTS-PRO-XYZ987"
    (normalized exact match; embedding cosine 0.992).
  - r-oos-01 "What is the best brand of refrigerator to buy this year?" is a trivial
    word-reorder of the prototype "what is the best refrigerator brand to buy this year"
    (cosine 0.978).
  How I reproduced: normalized string set-membership + ONNX cosine of every eval
  question against every prototype (max 0.992). Scored items are literally graded
  against their own text, so the absolute 0.929 is inflated on those two rows.
  Impact is bounded and honest to state: BOTH leaked items are keyword-correct, so
  they are NOT part of the "6/8 misroutes fixed" story, and removing them widens the
  delta over keyword (embedding 0.923 vs keyword 0.692, delta +0.231). The comparative
  headline is fully robust; only the absolute number and the code comment are tainted.
  Fair-to-the-builder note: the LEARNINGS entry makes only the NARROWER, true claim
  (they did not add a prototype for the failing items r-tik-03/04 - real
  teaching-to-test discipline, which I confirmed). The overreach is solely the
  categorical "NOT copied from the labeled routing eval set" in the code comment.
  Fix (either): (a) replace the r-war-02-shaped prototype with a generic form that
  drops the concrete seeded serial and reword r-oos-01's twin, so there is genuinely
  no overlap; or (b) correct the comment to disclose the overlap honestly. Given the
  repo's stated honesty bar, the claim must not ship as written.

F1-R2 [P3, CONFIRMED] Non-English queries silently misroute (to a safe fallback).
  "cual es el estado de mi pedido" (Spanish for an order-status question) scores below
  threshold on the English-only prototypes -> low_confidence_fallback -> keyword router
  -> rag (wrong path; the order tool is never called). No crash, degrades safely. The
  corpus and product are English-only so this is a real but low-impact limitation;
  worth one line in LEARNINGS/limitations so it is not a surprise later.

F1-R3 [P3, CONFIRMED] The 0.35 confidence threshold sits in the noise band.
  Empty/whitespace input scores 0.381 (chitchat, ABOVE threshold -> tool), while a
  genuine indirect escalation (r-tik-04) scores 0.302 (BELOW -> deferred, and missed).
  So the threshold admits some junk and rejects some real intent. r-tik-04 is already
  disclosed; the empty-string -> tool case is harmless (the endpoint would not pass an
  empty message). Consider a min-length/empty short-circuit, or revisit 0.35 if the
  routing set grows. Low priority.

F1-R4 [P3, provenance] The results stamp names a commit that does not contain the code.
  results/routing_eval.json is stamped commit ddfeaa1, but ALL F1 code is uncommitted
  working tree, so checking out ddfeaa1 cannot reproduce it. Resolves automatically
  when F1 is committed and the eval re-run/re-stamped; flagged so the re-stamp is not
  forgotten.

### Process note (not a code finding; for the dev/planner to rule)
This feature file shows PHASE B (ratified build spec) EMPTY and BUILD LOG EMPTY, and
states "Builder does Phase A ONLY, then STOPS for planner ratification." Yet F1 is
fully built and logged "GATE MET" in the STATUS spine (line ~1367). The ratified
two-phase feature cadence (PLANNER NOTE: "No inspect-and-build in one pass for
features") and the "log the build in the feature file, keep the spine lean"
convention both appear to have been skipped. This has NO bearing on the code's
correctness (which I verified independently), but the planner should either ratify
the one-pass build retroactively or note the deviation, and the build log belongs in
this file's BUILD LOG section.

BOTTOM LINE: the intent router is real, local, zero-token, deterministic, and
independently reproduces the reported A/B numbers; it fixes the day-one keyword
misroutes and regresses nothing frozen (hit@5 0.913, injection ordering, 10/10 authz).
Close F1-R1 (the false "no overlap" claim + the one leaked prototype) before promoting;
F1-R2..R4 and the process note are the dev's call.

---

## PLANNER RATIFICATION (2026-07-17, retroactive)

Cadence note: F1 was built in a single pass (inspect -> build -> verify) off the
earlier BUNDLED kickoff in STATUS, because that kickoff predated the two-phase
feature cadence and this feature file (both landed AFTER F1 was already in
flight). This is a timing artifact of the process change, not builder misconduct.
Design reviewed now and RETROACTIVELY RATIFIED. From F2 onward the two-phase gate
(Phase A propose -> planner ratify -> Phase B build) is enforced; the empty
PHASE A/B placeholders above reflect that F1 skipped the split, not a gap to
backfill with fiction.

Design ratified:
- Local ONNX embedding intent classifier reusing the in-memory MiniLM; old keyword
  router behind ROUTER=keyword for A/B and rollback; zero added tokens. Correct.
- DESIGN REFINEMENT ratified: split "warranty" by KIND (policy -> doc_lookup/RAG;
  status -> tool). Required by the kickoff goal (fix the "what does the warranty
  policy cover" misroute); separates cleanly (policy 0.77-0.88, status 0.94);
  reversible. Approved.

REQUIRED before commit/promotion - F1-R1 [P2] eval integrity. Planner enforces the
CLEAN fix, not the cosmetic one:
- The routing eval set must NOT contain copies of the classifier's own prototypes.
  r-war-02 is character-identical to a prototype (cosine 0.992); r-oos-01 is a
  trivial reorder (0.978). DE-DUPLICATE: replace those two eval items with
  genuinely independent phrasings (or drop them), so the eval measures
  generalization, not memorization. THEN fix the false comment at
  intent_router.py:67-72. Re-run the A/B and report the HONEST numbers (removing
  the leak WIDENS the delta to ~+0.231, so the story gets stronger). Same standard
  as the frozen golden set: no leakage, honest deltas, no inflated absolute number.

Accepted residuals (P3, non-blocking; add as limitations lines, do NOT gold-plate):
- F1-R2 non-English queries fall back safely (no crash). Note as a limitation.
- F1-R3 the 0.35 threshold sits in a noise band (empty 0.381 passes; a real
  escalation 0.302 defers). Nudge only if cheap; otherwise document. Do not overfit
  the threshold to the eval.
- F1-R4 provenance resolves on commit + re-stamp (already planned).

Commit plan (APPROVED, on v2-fullstack, AFTER the F1-R1 fix):
- Commit 1 feat(routing): F1 code + results ONLY (intent_router.py, agent.py,
  config.py, metrics.py, routing_set_v1.json [de-leaked], routing_eval.py,
  test_routing.py, results/routing_eval.{json,md}).
- Commit 2 docs(process+plan): the planner-authored files, committed SEPARATELY,
  not folded into the feature (CLAUDE.md, agents/*.md, docs/AI-FEATURES-PLAN.md,
  docs/ai-features-roadmap.html, docs/increments/, STATUS-prod.md,
  docs/LEARNINGS.md). Legitimate repo content; do not leave uncommitted.
- Never git add .; name files explicitly. Re-run routing_eval after committing so
  results stamp the real commit hash (closes F1-R4).

STATUS: F1 built + reviewed STRONG; ACCEPTED pending the F1-R1 de-leak fix, then
commit + promote.
