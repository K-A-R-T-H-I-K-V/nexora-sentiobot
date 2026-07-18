# Feature F1 - Intent routing A/B (keyword vs embedding classifier)

- Commit: `e1dfec713fa42148544f52c5c6dee1122c9ea9b0`  |  As-of: 2026-07-18
- Embedding model: all-MiniLM-L6-v2 (ONNX Runtime via fastembed) (local, 0 LLM calls, 0 tokens)
- Hardware: Windows 11 (build 10.0.26200) | AMD64 | py3.11.9
- Confidence threshold: 0.35

## Headline
- Keyword router accuracy:   **0.714** (28 queries)
- Embedding router accuracy: **0.929**  (delta +0.214)
- Keyword-misroute cases fixed by embedding: **6/8**
- Low-confidence fallbacks to keyword: 1/28

## Leakage guard (F1-R1: measures generalization, not memorization)
- No eval question copies or near-copies a prototype: **PASS** (max cosine to any prototype over the set = 0.880, threshold 0.9)

## Embedding accuracy by intent
- doc_lookup: 12/12
- order_status: 3/3
- warranty: 4/4
- ticket_or_escalation: 2/4
- chitchat: 2/2
- out_of_scope: 3/3

## Known limitations
- F1-R2 (non-English): prototypes are English-only, so a non-English query (e.g. 'cual es el estado de mi pedido') scores below threshold and falls back to the keyword router (safe, no crash, but may take the wrong path). The corpus and product are English-only, so this is low-impact; multilingual routing would need multilingual prototypes or a language-aware step.
- F1-R3 (threshold noise band): the 0.35 confidence threshold sits in a noise band. Empty/whitespace input scores ~0.381 (above -> tool) while a genuine indirect escalation (r-tik-04) scores 0.302 (below -> deferred and missed). The endpoint rejects empty messages upstream, so the empty-string case is harmless; the threshold is documented rather than overfit to this small set.

## Per-query

| id | ideal | keyword | emb intent | emb route | ok | conf |
| --- | --- | --- | --- | --- | --- | --- |
| r-doc-01 | rag | rag | doc_lookup | rag | ok | 0.805 |
| r-doc-02 | rag | rag | doc_lookup | rag | ok | 0.579 |
| r-doc-03 | rag | rag | doc_lookup | rag | ok | 0.505 |
| r-doc-04 | rag | rag | doc_lookup | rag | ok | 0.413 |
| r-doc-05 | rag | rag | doc_lookup | rag | ok | 0.425 |
| r-doc-06 | rag | rag | doc_lookup | rag | ok | 0.667 |
| r-doc-07 | rag | rag | doc_lookup | rag | ok | 0.610 |
| r-pol-01 | rag | tool (wrong) | doc_lookup | rag | ok | 0.880 |
| r-pol-02 | rag | tool (wrong) | doc_lookup | rag | ok | 0.771 |
| r-pol-03 | rag | tool (wrong) | doc_lookup | rag | ok | 0.847 |
| r-pol-04 | rag | rag | doc_lookup | rag | ok | 0.574 |
| r-pol-05 | rag | rag | doc_lookup | rag | ok | 0.528 |
| r-ord-01 | tool | tool | order_status | tool | ok | 0.701 |
| r-ord-02 | tool | tool | order_status | tool | ok | 0.690 |
| r-ord-03 | tool | rag (wrong) | order_status | tool | ok | 0.642 |
| r-war-01 | tool | tool | warranty | tool | ok | 0.812 |
| r-war-02 | tool | tool | warranty | tool | ok | 0.846 |
| r-war-03 | tool | tool | warranty | tool | ok | 0.860 |
| r-war-04 | tool | tool | warranty | tool | ok | 0.638 |
| r-tik-01 | tool | tool | ticket_or_escalation | tool | ok | 0.718 |
| r-tik-02 | tool | tool | ticket_or_escalation | tool | ok | 0.744 |
| r-tik-03 | tool | rag (wrong) | doc_lookup | rag | MISS | 0.520 |
| r-tik-04 | tool | rag (wrong) | low_confidence_fallback | rag | MISS | 0.302 |
| r-cht-01 | tool | rag (wrong) | chitchat | tool | ok | 0.513 |
| r-cht-02 | tool | rag (wrong) | chitchat | tool | ok | 0.594 |
| r-oos-01 | rag | rag | out_of_scope | rag | ok | 0.393 |
| r-oos-02 | rag | rag | out_of_scope | rag | ok | 0.471 |
| r-oos-03 | rag | rag | out_of_scope | rag | ok | 0.732 |
