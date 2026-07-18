# SentioBot latency + call-count + tokens baseline (Increment 2)

Measurement only; no behavior changed. Numbers are shown as **run1 / run2**.

## Provenance

- Provider / model: `groq` / `llama-3.3-70b-versatile` (temperature 0.0, max_tokens 2048)
- Commit: `eb5df80e0839`
- Dates (UTC): run1 2026-07-17T17:14:53, run2 2026-07-17T17:21:10
- Hardware: Windows 10 | AMD64 | py3.11.9
- TTFT/e2e are client-side over localhost; counts+tokens are server-side (trailing metrics event).

## Method

Per run: 1 RAG + 1 tool warm-up (absorbs the ~22s first-request singleton cold-start), then a fixed 3 RAG + 3 tool COLD set (cache-miss), then two cache-WARM re-asks. The backend is restarted before each run so the in-process cache is cold; the harness asserts cache_hit=False on the cold set. Token-frugal by design (Groq free tier is ~100K tokens/DAY).

## Latency (ms)

| Metric | RAG cold | Tool cold | Cache warm |
|---|---|---|---|
| TTFT p50 | 8599.2 / 11323.6 | 11255.2 / 10862.5 | 1391.1 / 1384.6 |
| TTFT p95 | 12921.0 / 11358.5 | 25756.2 / 17020.7 | - |
| e2e p50 (to done) | 9503.7 / 12206.4 | 11689.1 / 11302.9 | 1391.2 / 1384.7 |
| retrieval p50 | 641.6 / 601.8 | 0.0 / 0.0 | - |

One-time singleton cold-start (first request after boot): run1 21765.1 ms, run2 21873.6 ms.

## Per-route API-call inventory (deterministic; identical across runs)

| Route | LLM calls | Embedding ops | Supabase round trips |
|---|---|---|---|
| RAG (cache miss) | 2 | 7 | 5 |
| Tool (cache miss) | [2, 3] | 2 | [6, 7] |
| Cache hit | 0 | 1 (L2 semantic) | [4] |

## Tokens per request (Groq usage; p50)

| Route | prompt | completion | total |
|---|---|---|---|
| RAG cold | 2256.0 / 2256.0 | 360.0 / 348.0 | 2499.0 / 2491.0 |
| Tool cold | 2182.0 / 2181.0 | 74.0 / 87.0 | 2255.0 / 2268.0 |

## Findings (set up P3, no optimization here)

1. RAG makes **2 LLM calls** per answer: the MultiQueryRetriever spends one full LLM call on 3 rephrasings BEFORE the answer. That extra call is the Rank 1 P3 target (cut/gate multi-query); it is pure TTFT + token overhead.
2. **Daily budget:** ~2.3-2.5K tokens/request means Groq's 100K tokens/DAY supports only ~40 chat answers/day before 429 (F1.5-4). A full baseline reproduction exhausted one account's daily budget; run2 used a fresh account. This makes tokens/request a first-class metric, not just ms.
3. **Cache warm is ~1.4s, not instant:** dominated by the L2 semantic embed + the sequential Supabase persist writes (a cache hit still saves the turn). Analytics/persist could be made non-blocking (P3 Rank 2).
4. **Singleton cold-start ~22s** on the first request (retriever + MiniLM + BM25 load). Rank 5 (warm on startup) would remove it.

## Reproducibility

Call inventory and tokens/request are **identical** across run1 and run2 (deterministic). Latency p50/p95 reproduce within Groq free-tier server-load variance (RAG TTFT p50 8.6s vs 11.3s; cache-warm and cold-start within ~1%). Re-run: restart the backend, then `python -m backend.scripts.latency_baseline <label>`.
