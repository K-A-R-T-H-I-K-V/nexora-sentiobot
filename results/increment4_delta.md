# Increment 4 (P3 Rank 1): cut the multi-query retriever - measured delta

Controlled A/B on the RAG answer path (same LLM + prompt; only the retriever differs), on 5 questions from the frozen RAGAS subset. Retrieval hit@k before/after is the frozen Increment 3 deterministic baseline.

## Provenance

- Generator `llama-3.3-70b-versatile` temp 0; judge `llama-3.1-8b-instant` (weak 8b, indicative). Commit `b778bd144cad`. As-of 2026-07-18.
- Hardware: Windows 11 (build 10.0.26200) | AMD64 | py3.11.9 (stamp fixed, R3-2). Golden items SHA256 `d81f1e317437baff...` asserted at start (R3-1).
- Tokens spent: 37127 (gen 19707, judge 17420).

## Delta (before = multi-query ON, after = base ensemble)

Headline = grade-1 metrics only: deterministic, reproduce bit-for-bit on re-run.
These are the claims the data supports.

| Metric | Before | After | Change |
|---|---|---|---|
| LLM calls / doc query | 2.0 | **1.0** | **2 -> 1** (verified all 5) |
| Tokens / answer | 2690.4 | **1251.0** | **-53.5%** (verified all 5) |
| Retrieval time (ms) | 819.5 | 45.76 | **-94.4%** (the eliminated multi-query LLM round-trip was most of it) |
| Retrieval hit@5 (frozen Inc3) | 0.913 | 0.913 | **identical** (reproduced byte-for-byte) |

Consequence of -53.5% tokens: roughly ~2x the daily free-tier answer capacity.

## Not headlined, and why (R4-1: report only what the data supports)

Two numbers from this run are NOT trustworthy claims and are deliberately kept
out of the headline. Stating them there would imply a bigger win than exists.

- Generation time (answer LLM call): mean 5777 -> 1868 ms. DROPPED as a claim,
  not attributable to the cut. `gen_ms` times the same answer call in both arms;
  the retriever change does not touch it except via prompt size. Per row the
  sign even flips (pol-02 964 -> 5768 ms, pol-03 683 -> 1798 ms got SLOWER
  after), and the -68% "mean" is manufactured by two free-tier outliers in the
  before arm (doc-09 13658 ms, doc-12 12590 ms). That is Groq server-load
  variance, not a speedup from cutting multi-query. ("TTFT proxy" was also a
  misnomer: this is full non-streaming generation time, not time-to-first-token.)
  The genuine, defensible latency win is retrieval -94% above, which already
  contains the removed expansion round-trip. The raw per-row gen_ms stays in
  results/increment4_delta.json for the record; it is just not a claim.
- RAGAS-style faithfulness (8b judge): 0.81 -> 0.55. Kept in the raw JSON,
  NOT headlined. This is a weak-judge number dominated by noise (two spurious
  0.0 scores on demonstrably-correct answers this run). Read as a headline it
  invites a "quality dropped" misread that is false; see the ANSWER check below,
  which is how quality was actually confirmed.

## Quality: the ANSWER check (not just retrieval membership)

The gate requires checking the written answer, because identical top-5 membership does not prove identical answers. Reading all 5 before/after pairs (in results/increment4_delta.json `rows`):
- doc-03 (LumiGlow specs): both give 9W + 25,000 hours, cited. After judged 1.0 (>= before 0.8).
- doc-09 (Thermostat 24V): both give 24V AC; after adds the R/C terminals. Both 0.8.
- doc-12 (camera IP65): both give 'IP65, not waterproof, do not submerge', cited. After judged 0.0 which is WRONG (the answer is faithful).
- pol-02 (warranty water damage): both give 'flood/water excluded', cited. After judged 0.0 which is WRONG.
- pol-03 (30-day return): both give 30-day full-refund conditions. Both 0.95.

VERDICT: quality HELD. The faithfulness mean dropping 0.81 -> 0.55 is driven ENTIRELY by two spurious 0.0 scores from the weak 8b judge on demonstrably-correct after-answers (doc-12, pol-02) - the same judge failure mode disclosed in Increment 3 (doc-14). Every after-answer carries the same key facts and cites sources (after_cites_source 5/5). The automated fact-substring check (2/4) also mis-matched (it looked for 'LED', '(IP65)'); it is not evidence of a real drop. No MATERIAL faithfulness regression on inspection.

## Decision

Ship the cut as DEFAULT (use_multiquery=False). Multi-query is retained behind USE_MULTIQUERY (reversible, A/B-able), so if the reviewer's independent read disagrees, gating it on low-confidence first-pass retrieval is one flag away - no code lost. The before/after answers are committed in the results JSON for independent verification of the judge-artifact claim.

## Honest caveats

- The faithfulness judge is a weak 8b model; treat its number as indicative and read the answers. This run is a live case where the metric flagged a false regression.
- Timing is small-N, free-tier, single-machine. The trustworthy latency claim is retrieval time (-94%): it reproduces because it is dominated by the eliminated LLM round-trip, not by server load. Generation-time is NOT trustworthy even as a delta on the same box, because per-request Groq server load swings it more than the cut does (see "Not headlined" above); that is why it is dropped, not merely caveated.
- Small frozen corpus (85 sections): multi-query's value tends to grow with corpus size/messiness, so 'no gain here' may not generalize; re-measure if the corpus grows (logged).