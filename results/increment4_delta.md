# Increment 4 (P3 Rank 1): cut the multi-query retriever - measured delta

Controlled A/B on the RAG answer path (same LLM + prompt; only the retriever differs), on 5 questions from the frozen RAGAS subset. Retrieval hit@k before/after is the frozen Increment 3 deterministic baseline.

## Provenance

- Generator `llama-3.3-70b-versatile` temp 0; judge `llama-3.1-8b-instant` (weak 8b, indicative). Commit `b778bd144cad`. As-of 2026-07-18.
- Hardware: Windows 11 (build 10.0.26200) | AMD64 | py3.11.9 (stamp fixed, R3-2). Golden items SHA256 `d81f1e317437baff...` asserted at start (R3-1).
- Tokens spent: 37127 (gen 19707, judge 17420).

## Delta (before = multi-query ON, after = base ensemble)

| Metric | Before | After | Change |
|---|---|---|---|
| LLM calls / doc query | 2.0 | **1.0** | **2 -> 1** |
| Tokens / answer | 2690.4 | **1251.0** | **-53.5%** |
| Generation time (ms) | 5777.0 | 1867.62 | -67.7% (TTFT proxy; noisy free tier) |
| Retrieval time (ms) | 819.5 | 45.76 | -94.4% (the multi-query LLM call was most of it) |
| Retrieval hit@5 (frozen Inc3) | 0.913 | 0.913 | **identical** |
| RAGAS-style faithfulness (8b, indicative) | 0.81 | 0.55 | see caveat below |

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
- Timing is a small-N, free-tier, single-machine measurement; the DELTA (before/after on the same box) is the trustworthy part, not the absolute ms.
- Small frozen corpus (85 sections): multi-query's value tends to grow with corpus size/messiness, so 'no gain here' may not generalize; re-measure if the corpus grows (logged).