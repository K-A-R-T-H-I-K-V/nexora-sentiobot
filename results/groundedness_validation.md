# Feature F2 - Groundedness validation (local vs RAGAS + no-false-green)

- Commit: `ff70bd5e4abd3331e9dfc4922f48e8d65773fcfe`  |  As-of: 2026-07-19
- Embedding: all-MiniLM-L6-v2 (ONNX Runtime via fastembed) (local, 0 tokens)  |  Threshold: 0.5
- Source: increment4_delta.json (human-verified real answers + committed 8B RAGAS faithfulness)

No-false-green (synthetic poison): **PASS**

| id | local label | supported | RAGAS faith | poisoned label | disagree |
| --- | --- | --- | --- | --- | --- |
| doc-03 | grounded | 1/1 | 1.0 | partial |  |
| doc-09 | grounded | 1/1 | 0.8 | partial |  |
| doc-12 | grounded | 3/3 | 0.0 | partial | yes |
| pol-02 | partial | 3/4 | 0.0 | partial |  |
| pol-03 | partial | 3/4 | 0.95 | partial |  |

Disagreements are local=grounded where the WEAK 8B judge scored low faithfulness. Per Increment 4, the judge gave spurious 0.0s to human-verified-correct answers; these are judge noise, not false greens. The build gate is the synthetic no-false-green above.
