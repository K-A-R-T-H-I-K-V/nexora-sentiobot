# STATUS: index

Workstreams:
- STATUS-prod.md: production-grade upgrade of nexora-sentiobot
  (boot integrity, latency + quality baselines, measured optimization,
  hardening, container, CI, deploy, honest README, then features).
  ACTIVE.

Cross-cutting principles (promote here when something applies beyond
one workstream):
- Make it run and honest before you measure it; you cannot baseline a
  system that does not boot.
- Metrics before optimization; baselines frozen and committed. Latency
  (TTFT, retrieval, e2e p50/p95) is a first-class baseline, not an
  afterthought, for any streaming system.
- No secrets in code, images, or git history; scan before push.
- Cold-start proof for anything packaging/deploy related (fresh clone,
  container from scratch, public URL answers).
- Personalized or user-scoped data must never be shared across users by
  a cache or shared store; verify scope before shipping any cache.
- README claims must match code reality at every commit.

Cross-cutting decisions:
- 2026-07-17: Architecture reality is a multi-service app (FastAPI +
  LangGraph + Next.js + Supabase, ChromaDB retrieval), not the single
  Streamlit app the seeded plans assumed. Plans are revised to match
  code, ordering principle preserved. See STATUS-prod.md Ground Truth.
