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
[CORRECTED by F1-R5, 2026-07-19: these 28-item numbers were OVER-STATED. The honest
widened figures are embedding 0.839 / keyword 0.645 / +0.194, doc_lookup 12/15,
6-of-11 misroutes fixed on a 31-item set that includes over-trigger residuals the
router does NOT fix. See "F1-R5 CORRECTION DONE" at the end of this file.]

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

### RE-REVIEW (fresh reviewer, 2026-07-18) - F1-R1 verified fixed, VERDICT now CLEAN

I re-derived the F1-R1 fix from the changed files and re-ran every gate. It is fixed
the CLEAN way, not cosmetically, and I confirmed it independently rather than trusting
the eval's own leakage function:

- LEAKAGE GONE (my own recompute, not their leakage_report): I re-embedded every eval
  question against every prototype myself. Closest surviving pair is r-pol-01 at cosine
  0.880; nothing is >= 0.90 and there are zero normalized string copies. The five
  near-copies I would have flagged (r-war-02 0.99, r-oos-01 0.98, r-doc-03 0.92,
  r-tik-01 0.91, r-oos-02 0.91) are all replaced with independent phrasings, and the
  concrete-serial prototype is generalized to "...for my product's serial number".
- THE GUARD ACTUALLY BITES: I fed a verbatim prototype ("is water damage covered under
  the warranty policy") into leakage_report as a synthetic eval item; it flagged it
  (cosine 1.0, string_copy true). The invariant is wired into both the eval `passed`
  flag (routing_eval.py:166-167) and test_no_prototype_leakage, so a reintroduced leak
  fails CI. Not a no-op.
- THE FALSE CLAIM IS CORRECTED: intent_router.py:67-72 no longer asserts a bare "not
  copied"; it states the prototypes are kept DISJOINT and points at the enforced check.
- HONEST NUMBERS, UNCHANGED: re-ran the A/B twice - keyword 0.714 / embedding 0.929 /
  delta +0.214, 6/8 misroutes fixed, 1 fallback, byte-identical to the committed
  results/routing_eval.json (which the builder correctly re-ran after the fix: it shows
  the new r-war-02 text and leakage_guard PASS, max cosine 0.8798). The leaked items
  were both-router-correct filler, so replacing them left the headline untouched, as
  claimed; the delta was never leak-dependent.
- NO REGRESSION: full deterministic suite 26 passed / 0 failed (routing now 7 tests
  incl. the leakage guard, hit@5 == 0.913 misses [doc-01, doc-13], injection filter,
  output guard, authz 10/10). My hostile battery (empty, 100K-char, injection, SQL,
  gibberish, emoji, non-English) still degrades gracefully with no crash.
- P3s ADDRESSED: F1-R2 (non-English) and F1-R3 (threshold noise band) are now written
  up as limitations in routing_set_v1.json and surfaced in the eval report. F1-R4
  (results stamp names a commit that lacks the code) still resolves on the pending
  commit + re-stamp; that is the only open thread and it is mechanical.
- PROCESS NOTE RESOLVED: the one-pass cadence was a genuine timing artifact (the
  two-phase feature file post-dated F1) and is retroactively ratified by the planner
  in this file, with two-phase enforced from F2 on. The builder's BUILD-LOG claim of
  that ratification is truthful (I checked the ratification exists; it is not
  fabricated), and the empty PHASE A/B were honestly left empty rather than backfilled.

RE-REVIEW VERDICT: CLEAN. The one P2 is genuinely closed and independently verified,
nothing frozen regressed, and the only remaining item (F1-R4) is a stamp that closes
on commit. F1 is ready to commit and promote.

### STRONG RE-REVIEW (fresh reviewer, round 3, 2026-07-19) - committed as e1dfec7, F2 layered on

This pass went past the eval numbers into the deployed integration and the feature's
own promise under inputs the 28-item set never touches. F1 is now committed
(e1dfec7, single commit; the approved plan said two, immaterial) with F2 groundedness
committed on top (a1cf089).

Still-holding confirmations (re-run on the CURRENT HEAD, not the state I first saw):
- F2 did NOT regress F1: routing A/B on HEAD is still keyword 0.714 / embedding 0.929
  / +0.214, leakage guard PASS (max cosine 0.880), 7/7 routing tests pass, ruff clean.
- Integration trace (agent.py stream_agent_response, the real deployed path): the
  layer-1 injection refusal (line 566) still precedes route_message (line 585);
  routing sets route + intent metrics; both branches (RAG line 591, LangGraph tool
  line 642) are intact and unaffected by F1 beyond the router swap.
- The event-loop-blocking worry I had (route_message embeds synchronously, no await)
  is bounded: main.py:158 refuses any message over max_input_chars (8000) BEFORE
  routing, and MiniLM truncates to 256 tokens, so the synchronous embed is sub-ms.
  Not a finding.

NEW findings this round (neither prior review nor the builder surfaced these):

F1-R5 [P2, CONFIRMED] The "fixes the keyword over-trigger" headline is over-stated;
  realistic documentation/compatibility questions that contain a tool keyword STILL
  misroute to the tool path, by two mechanisms the 28-item eval never exercises.
  (a) The low-confidence fallback IS the keyword router (intent_router.py:233 calls
      keyword_route on conf < 0.35), so it re-introduces the exact over-trigger F1
      exists to remove. Reproduced:
      - "do you support HomeKit?"        conf 0.332 -> fallback -> keyword sees
        "support" -> TOOL (a pure compatibility/doc question).
      - "can I order replacement parts?" conf 0.309 -> fallback -> keyword sees
        "order" -> TOOL (a how-to/doc question).
  (b) The embedder itself over-triggers at moderate/high confidence on some
      warranty-POLICY phrasings (the very class F1 promises to send to RAG):
      - "is a cracked screen a warranty thing"       0.564 -> warranty -> TOOL
        (a coverage/policy question; the warranty-STATUS tool cannot answer it).
      - "what human languages does the app support?"  0.392 -> ticket -> TOOL.
  None of these four are in routing_set_v1.json, so the 0.929 (and the "12/12
  doc_lookup") does not reflect them. Reproduced with route_message(default router)
  under GROQ empty. This is NOT a regression - F1 is never worse than keyword here
  (the fallback equals keyword; the embedder-confident cases keyword also sent to
  tool), and both paths still answer - but the eval overstates the real-world
  over-trigger fix, and the docs frame the day-one flaw as solved. Per the repo's
  honesty bar, this needs disclosure: add these adversarial phrasings to
  routing_set_v1.json and report the honest (lower) doc_lookup number, and/or document
  that low-confidence keyword-bearing doc questions fall back to the legacy behaviour.
  Fixing the fallback itself is a genuine tension (defaulting low-confidence to RAG
  would break the indirect-escalation under-trigger fixes), so I recommend disclose +
  widen-the-eval over changing the fallback.

F1-R6 [P3, CONFIRMED] Routing is stateless - it ignores chat_history and routes on the
  bare current turn, so multi-turn STATUS follow-ups misroute. Reproduced:
  "is mine covered?" (a warranty-status follow-up about the user's own product) ->
  rag 0.391, so it gets a generic policy answer from RAG instead of the warranty
  tool. "how long does that last?" -> rag 0.360. Degraded, not broken (RAG still
  answers), and all eval items are single-turn so this is unmeasured. Document as a
  limitation, or feed the last user turn into the classifier input.

STRONG-REVIEW VERDICT: the feature is real, correct on its measured scope, zero-token,
deterministic, security-neutral (injection ordering preserved), and F2 did not regress
it - everything my first two rounds certified still holds on the committed HEAD. The
new finding is a claim-scope / disclosure gap (F1-R5, P2): the over-trigger is only
partially fixed and easy to break with ordinary phrasings outside the small curated
set, so the "fixed the day-one flaw" framing and the 0.929 are more generous than the
real behaviour. It does not block the feature (never worse than baseline, both paths
answer), but it should be disclosed and the eval widened before this becomes a resume
line. F1-R6 (stateless multi-turn) is a P3 limitation to document.

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

---

## F1 CLOSED (2026-07-17) - CLEAN
F1-R1 fixed the clean way and independently re-reviewed CLEAN. Builder swept wider
than the finding (5 leaked items, not 2), replaced all with independent phrasings,
generalized the seeded serial, corrected the false comment, and added a MEASURED
leakage guard (max eval->prototype cosine must stay < 0.90; wired into the eval
pass flag + test_no_prototype_leakage in CI). Reviewer recomputed cosines
independently (max 0.880), confirmed the guard bites, confirmed no regression
(suite 26 passed, hit@5 0.913, injection + 10/10 authz green).
Honest number: keyword 0.714 / embedding 0.929, delta +0.214 (NOT +0.231; the
planner's commit-message figure assumed dropping items, the builder replaced all 5
and kept 28-question coverage, so the score held - which is itself the proof of
genuine generalization, not memorization). Zero added tokens (local ONNX).
Commits on v2-fullstack: e1dfec7 feat(routing); 84396d3 docs(process) with the
re-stamped results. P3s (non-English fallback, threshold noise band) documented as
limitations, not gold-plated. F1-R4 provenance resolved by the re-stamp.
F1 is done. This is the first AI feature shipped through the full loop.

---

## POST-CLOSE CORRECTION - F1-R5 [P2] + F1-R6 [P3] (planner, 2026-07-17)
While reviewing F2, a later reviewer found F1's headline is OVER-STATED: the 28-item
routing eval omits the residual over-trigger cases, so "+0.214 / fixed the day-one
flaw" is rosier than reality. This is a claim-scope / honesty gap, NOT a regression
(F1 is never worse than keyword on these; both paths still answer). Ruling:
DISCLOSE-AND-WIDEN. Do NOT re-architect the fallback - defaulting low-confidence to
RAG would break the indirect-escalation fixes; that tension is logged for a future
routing v2.

F1-R5 FIX (small correction increment):
- Add the residual over-trigger phrasings to results/routing_set_v1.json as new
  labeled cases: the low-confidence fallback re-triggering the keyword bug
  ("do you support HomeKit?" -> doc_lookup; "can I order replacement parts?" ->
  doc_lookup) and the embedder over-trigger ("is a cracked screen a warranty thing"
  -> coverage = doc_lookup, not the status tool). Keep the leakage guard passing.
- Re-run routing_eval and REPORT THE HONEST, lower accuracy (it will drop below
  0.929 - that is the point). Update EVERY place the +0.214 / "fixed the flaw" claim
  appears (STATUS, LEARNINGS, results, and any future README/resume line) to the
  honest number + its scope.
- DOCUMENT the fallback behavior (low-confidence -> keyword router, which can
  over-trigger) and the residual as a known limitation.
F1-R6 [P3]: routing is STATELESS (ignores chat_history), so multi-turn status
follow-ups ("is mine covered?") misroute. DOCUMENT as a limitation now; a
history-aware classifier is a queued routing-v2 refinement, not this increment.
LEARNINGS meta-lesson to append: your eval's COVERAGE is your claim's SCOPE - an
accuracy number only speaks for the distribution you tested; a reviewer probing
BEYOND the eval is how you find the number's blind spots (second time now; see
Part 9). Widen the eval, restate the honest number.
Housekeeping: also commit the two files left uncommitted (CLAUDE.md and this F1
file); never git add .

---

## F1-R5 CORRECTION DONE (builder, 2026-07-19)
Disclose-and-widen, per the ruling above. No re-architecture of the fallback.

- Added the residual over-trigger phrasings to results/routing_set_v1.json as
  labeled doc_lookup/rag cases, all keyword_misroute + known_hard (F1 does NOT fix
  them), all leakage-clean (max prototype cosine 0.564 < 0.90):
  - r-res-01 "do you support HomeKit?"      -> embedding conf 0.332 -> fallback ->
    keyword sees "support" -> TOOL (wrong; ideal rag).
  - r-res-02 "can I order replacement parts?" -> conf 0.309 -> fallback -> "order"
    -> TOOL (wrong).
  - r-res-03 "is a cracked screen a warranty thing" -> embedder over-triggers to
    warranty (TOOL) at 0.564 (wrong; a coverage/policy question).
- Re-ran routing_eval. HONEST widened numbers (was 0.929 / 0.714 / +0.214 / 6-of-8
  on the 28-item set): embedding 0.839 vs keyword 0.645, delta +0.194, doc_lookup
  12/15, 6/11 keyword misroutes fixed. Leakage guard still PASS (max 0.880 < 0.90).
  Eval still PASS (embedding > keyword; gate items route correctly; no stale labels).
- test_routing.py accuracy floor lowered 0.85 -> 0.78 to sit below the honest 0.839
  with headroom (still a real regression tripwire). test suite green.
- Documented F1-R5 (fallback inherits the keyword over-trigger; over-trigger REDUCED
  not removed) and F1-R6 (stateless routing; multi-turn status follow-ups misroute)
  as limitations in routing_set_v1.json + LEARNINGS Part 17.
- Corrected the +0.214 / "fixed the flaw" claim to the honest number + scope in
  STATUS (FEATURE INDEX), LEARNINGS Part 17 (headline + new correction subsection +
  meta-lesson "your eval's coverage is your claim's scope"), and the results files.
- Housekeeping: committed CLAUDE.md and this F1 file (were left uncommitted after F1).
- NOT changed: the low-confidence -> keyword fallback (defaulting to RAG would break
  the indirect-escalation under-trigger fixes) and stateless routing; both are queued
  as routing v2. This is a claim-SCOPE correction, not a behaviour change: F1 is never
  worse than the keyword router on these cases, and both paths still answer.

### F1-R5 CORRECTION VERIFIED (fresh reviewer, 2026-07-19)
Re-derived and re-ran, did not trust the write-up. The correction is honest and clean:
- Numbers reproduced exactly on committed HEAD: embedding 0.8387 / keyword 0.6452 /
  delta 0.1935, doc_lookup 12/15, misroutes 6/11, fallbacks 3/31. Matches the claimed
  0.839 / 0.645 / +0.194.
- The three added items are HONEST MISSES, not gamed: r-res-01 (0.332 -> fallback ->
  tool), r-res-02 (0.309 -> fallback -> tool), r-res-03 (0.564 -> warranty -> tool) all
  route wrong and all carry gate_must_route_correct=false, so they legitimately pull the
  score DOWN. Their intents confirm the exact two mechanisms of F1-R5 (fallback-inherits-
  keyword, and direct embedder over-trigger).
- No teaching-to-test: git confirms INTENT_PROTOTYPES is byte-unchanged since F2; only
  routing_set_v1.json (+3 items) and test_routing.py (floor) changed.
- No new leakage: my own recompute over all 31 items has max prototype cosine 0.880
  (r-pol-01), nothing >= 0.90, zero string copies.
- Floor test honestly lowered 0.85 -> 0.78 (sits below 0.839 with ~0.06 headroom, still
  a real tripwire, not set just under the number). Full deterministic suite 26 passed,
  hit@5 0.913 intact.
F1-R5 CLOSED. The over-trigger is now honestly framed as REDUCED not removed, the eval
covers the residuals, and the headline number reflects real behaviour. F1-R6 (stateless
multi-turn) remains a documented P3 for routing v2. Nothing outstanding on F1.
