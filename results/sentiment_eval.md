# Feature F4 - Sentiment detection + no-false-escalation

- Commit: `d2d691990bf4f3a83ae8659d71c9b86a17340965`  |  As-of: 2026-07-19
- Embedding: all-MiniLM-L6-v2 (ONNX Runtime via fastembed) (local, 0 tokens)
- Escalation threshold 0.52, EMA alpha 0.5, window 3

## Headline
- Detection accuracy: **0.783** (23 messages)
- FALSE-ESCALATION rate on no-escalate controls: **0.000** (0/13) [load-bearing; must be 0]
- Abuse-override misses: 0
- Escalation-scenario failures: 0
- Leakage guard: PASS (max cosine 0.569 < 0.9)

## Accuracy by label
- calm: 10/10
- confused: 1/3
- frustrated: 3/6
- angry: 4/4

## Escalation scenarios

| id | turns | ema | expect | got | ok |
| --- | --- | --- | --- | --- | --- |
| esc-sustained-01 | 3 | 0.61 | True | True | ok |
| esc-sustained-02 | 3 | 0.71 | True | True | ok |
| esc-calm-01 | 2 | 0.00 | False | False | ok |
| esc-single-frustrated | 1 | 0.35 | False | False | ok |
| esc-resolved | 3 | 0.26 | False | False | ok |
| esc-abuse-single | 1 | 0.50 | True | True | ok |
