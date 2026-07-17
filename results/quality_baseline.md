# SentioBot quality baseline (Increment 3, P2)

Frozen golden set: `results/golden_set_v1.json` (v1, 50 questions, as-of 2026-07-17).

## Recipe (stamped)

- Generator: `groq` / `llama-3.3-70b-versatile` (temperature 0, k=5). Judge: `llama-3.1-8b-instant` (weak 8b judge, distinct from the 70b generator).
- Dataset version: v1 (frozen). Commit: `b34a8a91dc07`. As-of date: 2026-07-17.
- Hardware: Windows 10 | AMD64 | py3.11.9.
- Tokens spent (LLM pass, this run): generation 53156, judge 7058, total 60214 (multi-query reference retrieval LLM calls are extra and not in this counter).
- SMALL-CORPUS CAVEAT: the corpus is 85 sections (3 manuals + policies + 50 FAQ rows). A small, clean corpus makes retrieval EASIER, so hit-rate here is optimistic versus a larger, messier corpus. Do not oversell it.

## PRIMARY: retrieval on the BASE ENSEMBLE (free, local, zero Groq tokens, bit-stable)

Scored on the 23 single-turn retrieval items (15 doc + 8 policy). Reproduced twice, IDENTICAL (bit-stable).

| Metric | Value |
|---|---|
| hit-rate@1 | 0.5217 |
| hit-rate@3 | 0.7826 |
| hit-rate@5 | **0.913** |
| MRR | 0.6659 |
| context precision@5 | 0.2261 (low by construction: ~1-2 relevant of 5 retrieved) |

Per-category hit@5: {'doc_lookup': '13/15', 'policy_lookup': '8/8'}. Misses@5: ['doc-01', 'doc-13'] (doc-01 Wi-Fi band, doc-13 QR-scan; both facts also appear in sibling product sections, diluting rank).

Reproducibility: run1 == run2 exactly (0.913 hit@5, same misses).

## Multi-query retriever reference (one pass; spends tokens)

Full MultiQueryRetriever hit-rate@5 = **0.913** on the same 23 items. This EQUALS the base-ensemble hit@5 (0.913): on this frozen set, the extra multi-query LLM call per query buys **no** hit-rate improvement. That is the central quality input to Increment 4 (cut/gate multi-query): the latency + token win (Increment 2) may come at no measured retrieval cost here. To be confirmed on faithfulness once RAGAS runs.

## Tool-call correctness (sampled, through the real router)

Sample of 8 agent-route items (order valid + unknown, warranty active/expired/unknown, escalation): **8/8 correct tool**, **8/8 correct args**. The keyword router + agent selected the right tool and arguments on every sampled item.

## SECONDARY (indicative): RAGAS-style faithfulness/relevancy + adversarial refusal %

STATUS: BUDGET-DEFERRED. The Groq free tier is ~100K tokens/DAY (F1.5-4); the deterministic baseline is free but the LLM pass (multi-query reference + tool-call + RAGAS generation) exhausted the day's budget mid-RAGAS (429 TPD). Per the ratified recipe, RAGAS is SECONDARY/indicative and may be spread across days. The harness `backend/scripts/quality_baseline_llm.py` resumes these two stages on a fresh window (8 doc/policy Q, judge 2x with the 8b judge; 5 adversarial items, mechanical injection checks + judged refuse). Deterministic PRIMARY numbers do NOT depend on them.

## Interpretation

- The retrieval baseline is solid (hit@5 0.91) and reproduces exactly. It is the anchor Increment 4 measures against.
- Multi-query adds cost with no hit@5 gain here, strengthening the case to cut it, PENDING the faithfulness read (multi-query could still change answer quality even if top-5 membership is unchanged).
- Tool routing is reliable on the sample.
- Numbers are on a small, frozen corpus; treat as internal baselines for deltas, not published absolutes.
