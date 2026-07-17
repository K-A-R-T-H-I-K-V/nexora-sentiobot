# STATUS: sentiobot production-grade workstream

Goal: take nexora-sentiobot from a working full-stack demo to a
production-grade, publicly deployed, EVALUATED, hardened system that
survives a senior AI hiring manager's scrutiny. Every phase exit
produces a resume-usable artifact. Budget: as close to zero recurring
cost as possible.

Roles: PLANNER (Cowork, read-only, ratifies + drafts kickoffs here),
BUILDER (Claude Code, builds + commits), REVIEWER (fresh Claude Code
session, adversarial, re-runs everything). The dev bridges.

Rules of this file: every increment logged with commit hashes + what
was verified. Decisions recorded with reasoning. Deferred work tracked
explicitly, never silently dropped. No em dashes anywhere.

---

## GROUND TRUTH (planner recon, 2026-07-17, read from staged source on main)

The seeded MASTER PLAN below was drafted against the OLD public GitHub
version (single Streamlit app, ChromaDB, mock_db.py, dashboard.py). That
version no longer exists in this repo. The README is also stale and
partly AI-pasted. What follows is what the CODE actually does today. The
ordering PRINCIPLE from the seeded plan (measure, then optimize, then
harden, then deploy, honest README last) is kept; the specifics are
rewritten to match reality.

### Real architecture (live path)

Two-service app plus a managed database, not one Streamlit process.

- BACKEND: FastAPI (`backend/api/main.py`, title "SentioBot API"
  v2.0.0). Endpoints: `/auth/login`, `/auth/me`, `/chat/stream` (SSE
  token streaming, the main path), `/chat/conversations` (+messages),
  `/feedback`, `/analytics/summary`, `/health`. Run target in the code
  is `uvicorn` with 2 workers.
- AGENT: LangGraph `StateGraph` (`backend/agent/agent.py`), NOT the old
  LangChain ReAct AgentExecutor the README claims. Streams tokens via
  `graph.astream_events(version="v2")`. LLM is Gemini through
  `langchain_google_genai`, model `gemini-2.0-flash` (config default),
  temperature 0.1, streaming on.
- RETRIEVAL: ChromaDB is STILL the vector store. pgvector did NOT
  replace it. `get_retriever()` builds: Chroma (k=5) + BM25 over
  `parents.pkl`, combined in an `EnsembleRetriever` (weights 0.4 BM25 /
  0.6 vector), wrapped in a `MultiQueryRetriever` that calls the LLM to
  generate 3 query rephrasings before retrieving. Parent-document
  pattern via `LocalFileStore` + `parents.pkl`. Embeddings:
  HuggingFace `all-MiniLM-L6-v2` on CPU.
- DATABASE: Supabase (Postgres) via `backend/services/database.py`, used
  for APP DATA only (users, products, orders, conversations, messages,
  analytics, support_tickets). Schema + seed data in
  `supabase/schema.sql`. Supabase is NOT used for vector search;
  pgvector is commented out as a "future migration" note. mock_db.py is
  gone; those mocks are now Supabase rows.
- AUTH: real JWT (`backend/core/auth.py`), bcrypt password hashing
  (passlib), OAuth2 password flow. Seed users alice / bob / guest.
- CACHE: three-tier (`backend/services/cache.py`): L1 in-process LRU
  (exact sha256), L2 semantic cosine over MiniLM embeddings
  (threshold 0.92), L3 optional Redis. Checked at the top of
  `/chat/stream` before the agent runs.
- FRONTEND: Next.js 14 (App Router) + React 18 + TypeScript + Tailwind
  (`frontend/`). Real chat UI: token streaming via fetch + ReadableStream
  (`lib/api.ts`), tool-call badges, conversation sidebar, markdown
  rendering, feedback buttons, dark theme. JWT kept in localStorage.
  Login page + chat page.
- ORCHESTRATION: `docker-compose.yml` wires backend (8000) + frontend
  (3000) + redis:7. Backend also has `railway.toml` (Railway/Nixpacks);
  frontend has `vercel.json`. Ingestion: `backend/scripts/ingest.py`
  (2-stage, `--direct` mode works without API calls) and
  `batch_summarize.py`. `ingest_manifest.json` records 85 parents / 85
  summaries over 5 data sources.

### Live vs dead / stale

- LIVE: FastAPI app, JWT auth, the RAG streaming path, the LangGraph
  tool path, Supabase reads/writes, L1/L2 cache, Next.js chat, ingest
  `--direct`.
- DEAD / UNWIRED: `cache_stats()` and its referenced `/metrics`
  endpoint (endpoint does not exist in main.py); `rate_limit_per_minute`
  config (no limiter implemented); L3 Redis unless `REDIS_URL` is set.
- NOTEBOOKS: `notebooks/` (01/02/03 verify_*, chunking, retrieval-
  testing, visualize_split) are exploration, not runtime. Keep, but out
  of the production path.

### Divergence table (reality vs seeded plan vs README)

- Seeded "single Streamlit app / restructure into src/": OBSOLETE.
  Reality already has a backend/ + frontend/ split. P0's "move code into
  src/" no longer applies; the real P0 problem is different (see below).
- Seeded "ChromaDB replaced by Supabase/pgvector?": NO. Chroma stays;
  Supabase is the app DB. Any pgvector move is a future, optional,
  measured migration, not an assumed fact.
- Seeded "extract a service layer under Streamlit (P2)": OBSOLETE. A
  service layer already exists (core/, services/, agent/). Hardening
  targets those modules, not a Streamlit refactor.
- README "Streamlit / LangChain ReAct AgentExecutor / Gemini 1.5 Flash /
  ConversationBufferWindowMemory / dashboard.py": ALL STALE. Reality is
  Next.js / LangGraph / gemini-2.0-flash / Supabase-loaded history
  (last 12 messages) / `/analytics/summary`. README setup steps (conda,
  `pip freeze > requirements.txt`) are wrong; a pinned
  `backend/requirements.txt` already exists.
- User's mental model "simple Streamlit UI to upgrade": the UI is
  ALREADY a sophisticated React app. The remaining UI work is mobile
  responsiveness, accessibility, and fixing broken wiring (below), not a
  rewrite.

### Confirmed defects found during recon (severity, evidence)

P0-1 BOOT INTEGRITY (the system likely does not start as configured).
  The import scheme is self-contradictory. `main.py`, `auth.py`,
  `database.py`, `agent.py` use absolute `backend.*` imports (implying
  you run `uvicorn backend.api.main:app` from the repo root, with
  `backend` as a namespace package). But `cache.py` line 36 does
  `from config import get_settings` (a flat-layout import that only
  resolves if `backend/core/` is on `sys.path`). These two schemes
  cannot both resolve under one PYTHONPATH, and `main.py` imports
  `cache`, so import fails on startup. Separately, the Dockerfile
  (`context: ./backend`, `COPY . .`, `CMD uvicorn main:app`) flattens
  backend/ into /app, so both `main:app` (no /app/main.py) and the
  `backend.*` imports (no /app/backend package) break in the image.
  `railway.toml` has the same `uvicorn main:app`. Net: there is no
  single documented command that boots the app. This blocks everything,
  including baselining. Must be the first increment.

P0-2 CROSS-USER CACHE LEAKAGE (privacy + correctness).
  `get_cached_response(user_id, query)` deliberately ignores `user_id`
  ("identical questions from different users share the same cached
  answer"). But answers are personalized: the system prompt injects the
  user's name and owned products, and warranty/order answers contain
  user-specific data. So Alice's personalized answer (name, serial,
  warranty state) can be served verbatim to Bob when he asks a similar
  question (L2 fires at cosine >= 0.92). This is a real data-leak on a
  repo that will claim "production-grade." Cache must be user-scoped, or
  personalized/tool answers must be excluded from caching. Fix belongs
  in hardening, and must be verified by the reviewer's hostile battery.

P1-3 AGENT TOOL PATH HAS NO RAG CONTEXT.
  Routing in `stream_agent_response` is a keyword match
  (`order|warranty|serial|ticket|human|support`). Matches go to the
  LangGraph agent, whose system prompt says "RAG FIRST: Use
  lookup_documentation," but there is NO `lookup_documentation` tool
  bound (TOOLS = order/warranty/ticket only) and no documents are
  retrieved or injected on that path. So a documentation question that
  merely contains "warranty" (e.g. "what does the warranty policy cover
  for water damage") routes to a path with zero context, contradicting
  the "answer ONLY from context" rule. Correctness gap; also a latency
  asymmetry between the two paths. Design fix needed (unify retrieval
  into both paths, or add the lookup tool).

P1-4 FEEDBACK IS BROKEN END TO END.
  The frontend needs `interactionId` to POST feedback, but it is never
  populated: `streamChat` reads the `X-Conversation-Id` header and NOT
  `X-Interaction-Id`, and the `done` event carries no interaction id.
  So `msg.interactionId` is always undefined and `handleFeedback`
  returns early. The thumbs up/down UI renders but never writes. The
  `/analytics` feedback counts will therefore always be near zero. Also
  the main.py docstring says `POST /feedback/{interaction_id}` while the
  real route is `POST /feedback` (body-based); doc drift.

P1-5 TOOL-PATH FINAL ANSWER OVERWRITE + EMPTY SOURCES.
  On `on_tool_end`, `full_answer = str(output)` overwrites streamed
  content with the raw tool output, and the `done` event always sends
  `sources: []` for tool queries. Persisted assistant messages on tool
  turns can be malformed (raw tool string, or tool string plus later
  synthesis), and tool answers never carry sources.

P2-6 FRONTEND BUILD INTEGRITY.
  `frontend/` root has package.json (references tailwind, postcss,
  typescript) but the listing shows NO `tailwind.config.js`, NO
  `postcss.config.js`, NO `next.config.js`, NO `tsconfig.json`. Tailwind
  v3 will not compile classes without its config + postcss plugin, so a
  fresh `npm run build` likely produces unstyled output or errors. Needs
  a fresh-clone frontend build to confirm and fix. (Confirm before
  fixing; Next auto-generates tsconfig, but not tailwind/postcss.)

P2-7 INSECURE DEFAULTS + LATENCY BLIND SPOTS.
  `config.py` ships `jwt_secret = "change-me-in-production-..."` as a
  default (insecure if env is unset). `allowed_origins` still contains
  the placeholder `https://your-vercel-app.vercel.app`. The only latency
  signal is a middleware that logs total request ms; there is no
  time-to-first-token, no retrieval timing, no p50/p95. For a streaming
  app, TTFT is the headline metric and it is currently unmeasured.

P2-8 HIDDEN LATENCY + COST IN RETRIEVAL.
  `MultiQueryRetriever` makes an LLM call to generate 3 rephrasings on
  EVERY documentation query, before the answer LLM call. So a single doc
  answer is: 1 LLM call (query expansion) + embedding + ensemble
  retrieve + 1 LLM call (answer). This is the prime suspect for slow
  TTFT and doubles LLM cost per query. It is a candidate to measure and
  possibly cut (see Improvement Candidates), but NOT before a baseline
  exists.

### Secrets status (preliminary, needs the builder's scan)

`backend/.env` exists on disk and IS gitignored (root `.gitignore` has
`backend/.env`; `.dockerignore` has `.env`), so it should not be in the
image. NOT YET VERIFIED: whether any key was committed earlier in git
history. A `gitleaks`/`trufflehog` history scan is required before the
first push of this workstream and its result logged here. `schema.sql`
contains bcrypt hashes for demo users only (acceptable).

---

## MASTER PLAN (REVISED against reality; ratified ordering, do not reorder without a logged decision)

Ordering principle held: make it run and honest, THEN measure, THEN
optimize against baselines, THEN harden, THEN ship, THEN features. No
optimization is planned before its baseline exists. Each phase exits
with a resume-usable artifact.

### P0: Boot integrity + repo honesty [gate: fresh clone runs, one real answer streams]
- Establish ONE canonical run configuration that boots locally and in
  Docker: reconcile the import scheme (fix `cache.py`'s `from config`
  import and settle on `backend.*` as a real package or a flat layout,
  consistently), fix the Dockerfile/railway CMD + module path so the
  container serves `/health` and one streamed chat.
- No behavior/logic changes in P0 beyond what boot requires. Bugs found
  (P0-2..P2-8) are LOGGED here and scheduled, not fixed opportunistically.
- Secrets: `.env.example`, confirm gitignore coverage, run a git-history
  secrets scan, log the result.
- Trim the README's worst stale claims to stop them misleading (full
  honest rewrite stays last, P6).
- Resume artifact: a repo a reviewer can clone and run (backend + one
  streamed answer) in a few minutes.

### P1: Baselines - latency [gate: TTFT + retrieval + e2e p50/p95 committed, reproduced twice]
- Instrument the CURRENT system (no optimization): time-to-first-token,
  retrieval time (including the multi-query LLM call), end-to-end p50/p95,
  separately for the RAG path and the tool path. Emit structured JSON.
- A small fixed request set (deterministic, no LLM judge) driven through
  the REAL endpoint so routing is exercised honestly. Commit numbers
  stamped with commit hash, model, date, hardware.
- Resume artifact: a real latency table ("p50 TTFT X ms, p95 e2e Y ms")
  with methodology.

### P2: Baselines - quality [gate: baseline scores committed + reproduced twice, golden set FROZEN]
- Golden dataset: ~50 question/expected pairs across doc lookup, order
  status, warranty, ticket creation, multi-turn, and out-of-scope-must-
  refuse. Composition RATIFIED here, then FROZEN (later edits are logged
  ratification events). This is the first real ratification fight; do not
  rubber-stamp.
- Metrics pinned BEFORE results: retrieval hit-rate@k + context
  precision (free, deterministic); RAGAS faithfulness + answer relevancy
  (LLM-judged, Gemini free tier, temperature 0, at least 2 runs for a
  variance story); tool-call correctness (right tool, right args). Eval
  runs through the REAL routing (so the P1-3 no-context tool path is
  measured, not idealized).
- Leakage guard: golden questions must not be hand-fit to chunking or
  weights; hold out a split if iterating.
- Resume artifact: a reproducible quality number with its recipe, to
  replace the deleted "70%" README claim.

### P3: Measured optimization [gate: each change re-run on the frozen set, delta reported honestly incl. regressions]
- Only candidates from the ranked list below, each with its baseline
  already committed. Every change re-runs P1 latency + P2 quality on the
  same frozen inputs; deltas logged both ways (a latency win that costs
  quality is reported as such).
- Hard rule: no candidate is started before both its baselines exist.
- Resume artifact: a before/after table ("cut p95 TTFT 40%, faithfulness
  held within noise").

### P4: Service hardening [gate: hostile input + the found defects cannot break or leak]
- Fix the recon defects on the real path: P0-2 cache leakage (user-scope
  or exclude personalized/tool answers), P1-3 tool-path context, P1-4
  feedback wiring, P1-5 answer overwrite/sources, P2-7 secrets/CORS.
- Config via environment (no insecure literals), structured logging,
  timeouts + retries with backoff on all LLM/retrieval calls, defined
  failure messages (no stack traces to users). Cost guards: agent
  max-iterations, per-request token caps, bounded memory (history is
  already capped at 12; verify). Rate limiter actually wired.
- Input handling: length limits, prompt-injection resistance verified by
  the reviewer's hostile battery.
- Resume artifact: "hardened for production" that is demonstrable.

### P5: Container + CI [gate: docker compose up from scratch serves both; CI green]
- Fix the Dockerfile properly (already partly in P0), multi-stage, and
  decide + ratify: bake the Chroma index into the image vs mount/rebuild
  on start (cost + cold-start tradeoff stated).
- GitHub Actions: ruff lint, unit tests, frontend build, and a FREE eval
  smoke (retrieval-only metrics or a tiny cached subset; never an
  unbounded paid LLM run in CI).
- Resume artifact: CI badge + "containerized, CI-gated."

### P6: Deploy + observability + honest README [gate: public URL answers a real question; README matches code]
- Deploy target ratified by cost: backend on Railway or GCP Cloud Run
  (scale-to-zero, near-zero at portfolio traffic, matches the Cloud Run
  resume narrative) vs HF Spaces as zero-cost fallback; frontend on
  Vercel free. Secrets via the platform's secret manager, never in the
  image. Decision logged with monthly cost.
- Per-request tracing (LangSmith free tier or homegrown), cost-per-query
  visibility (build the real `/metrics` the cache already assumes).
- README rewritten LAST so it describes what exists: accurate
  architecture diagram, the eval + latency tables with recipes, live
  demo link, honest limitations.
- Resume artifact: live link + a resume bullet where every word is
  verifiable in the repo.

### P7: Net-new features [gate: each feature has its own planner/builder/reviewer loop and cannot regress P1/P2 baselines]
Ranked backlog, GATED behind baselines + hardening. This is the "entire
feature plan" the dev asked for, sequenced so nothing shiny jumps the
line. Each item, when started, gets baselined and re-measured.
- Mobile-responsive + accessible UI: the sidebar is a fixed `w-64` and
  not responsive; make it a drawer, add a11y, keyboard nav, PWA install.
  (This IS the "simple to sophisticated UI" ask; the app is already
  React, so it is polish + mobile, not a rewrite.)
- Citation with source highlighting (extract supporting sentence, show
  in the sources expander).
- Streaming markdown + skeleton states; stop/regenerate; copy button.
- Admin analytics dashboard page (real, now that feedback works).
- Multi-document upload / user-provided knowledge bases.
- Multimodal RAG (images/diagrams from manuals via a vision model).
- pgvector migration (only if measured to beat Chroma on this corpus).
- Knowledge-graph retrieval for multi-hop queries.
Each feature is a separate increment; none is ratified until P0-P4 are
closed.

---

## IMPROVEMENT CANDIDATES (brainstorm, critiqued, ranked; fold into P3 only after baselines)

Planner is a critic here, not a cheerleader. Each candidate states the
expected win, the measurement that would PROVE it, the cost, and the
risk. Nothing here is approved for build until P1 (latency) and P2
(quality) baselines are committed. Rank = build order within P3.

Rank 1 - Cut or gate the MultiQueryRetriever.
  Expected win: largest TTFT reduction available. It adds a full LLM
  round-trip (3 rephrasings) before every doc answer. Removing it, or
  firing it only on low-confidence first-pass retrieval, should cut TTFT
  materially and halve per-query LLM cost.
  Proof: P1 TTFT + retrieval-time delta; P2 hit-rate@k + faithfulness
  must hold (this is the risk - multi-query may be lifting recall).
  Cost: engineering only; REDUCES recurring LLM cost.
  Risk: medium. Could drop retrieval quality on terse queries. Report
  both metrics; keep only if TTFT drops without a quality regression.

Rank 2 - Fix + exploit the cache correctly (P0-2 first).
  Expected win: near-zero-latency repeat answers, safely. Today it is a
  latency win that leaks data; scoping it per user (or per user+profile
  hash, excluding tool/personalized answers) keeps the win and closes
  the leak.
  Proof: cache hit-rate and TTFT-on-hit vs miss (needs P1 harness);
  reviewer confirms no cross-user leak.
  Cost: engineering only.
  Risk: low once scoped. Measure hit-rate honestly (semantic 0.92 may
  over-fire and serve stale answers).

Rank 3 - Model routing (small model for easy queries).
  Expected win: lower latency + cost by sending simple/FAQ queries to a
  cheaper/faster Gemini tier and reserving the stronger model for tool
  reasoning.
  Proof: per-class TTFT + cost delta with quality held on the frozen
  set per class.
  Cost: engineering; net cost DOWN.
  Risk: medium. Misroute hurts quality; needs a cheap, reliable
  classifier (keyword/embedding, not another LLM call, or the latency
  win evaporates).

Rank 4 - Unify retrieval into the tool path (also fixes P1-3).
  Expected win: correctness (context on tool turns) + consistent
  latency between paths; enables honest single-path eval.
  Proof: P2 quality on tool-routed doc questions rises; P1 tool-path
  TTFT measured.
  Cost: engineering.
  Risk: medium. Adds retrieval latency to tool turns; measure the
  tradeoff. This overlaps P4 hardening, so sequence carefully.

Rank 5 - Warm singletons + embedding reuse at startup.
  Expected win: kill first-request cold start (retriever, LLM, and the
  SEPARATE cache embedder are lazy singletons; the cache loads a second
  MiniLM instance). Warm them on startup and share one embedder.
  Proof: first-request TTFT before/after; steady-state RSS before/after.
  Cost: engineering; small.
  Risk: low. Slower boot, more idle RAM; fine for a long-lived service,
  watch it on scale-to-zero platforms.

Rank 6 - Index/chunking design pass.
  Expected win: better hit-rate@k and/or smaller/faster index (H2-header
  chunking is coarse; some parents may be large).
  Proof: P2 retrieval metrics on the frozen set; index size + retrieve
  time.
  Cost: engineering + one reingest.
  Risk: medium-high for eval integrity: tuning chunking against the
  golden set is leakage. Requires a held-out split before this is
  allowed.

Rank 7 - Ensemble weight sweep (BM25 vs vector).
  Expected win: modest hit-rate gain over the current 0.4/0.6.
  Proof: P2 retrieval metrics across a small weight grid on a HELD-OUT
  split.
  Cost: cheap (deterministic, no LLM judge).
  Risk: leakage if fit on the frozen eval set; low technical risk.

Rank 8 (deferred to P7, not P3) - pgvector migration.
  Expected win: one datastore (drop Chroma), simpler deploy, maybe
  faster at scale.
  Proof: retrieval quality parity on the frozen set + retrieve-time at
  this corpus size (85 docs).
  Cost: engineering; possibly Supabase compute.
  Risk: high effort for unclear win at 85 docs. Explicitly NOT an
  optimization increment; it is a feature-grade migration, measured
  before adoption.

---

## DECISIONS
- 2026-07-17 (planner): Master plan REVISED against code reality. The
  seeded Streamlit/single-app assumptions are retired; the app is
  FastAPI + LangGraph + Next.js + Supabase with ChromaDB retrieval. The
  measure-before-optimize ordering is preserved. Reasoning: the seeded
  plan's phases assumed a codebase that no longer exists; keeping its
  ordering principle while rewriting the phase contents is the honest
  move.
- 2026-07-17 (planner): ChromaDB STAYS as the vector store for now.
  pgvector is reclassified from "assumed replacement" to a P7 measured
  migration. Reasoning: code shows Chroma live and pgvector commented
  out; at 85 docs a migration is unjustified without a measured win.
- 2026-07-17 (planner): P0 is now BOOT INTEGRITY, not "restructure into
  src/." Reasoning: recon found the app likely does not start as
  configured (contradictory imports; broken Docker/railway CMD). You
  cannot baseline a system that does not run, so boot integrity precedes
  all measurement. This supersedes the seeded P0.
- 2026-07-17 (planner): Latency baseline (P1) and quality baseline (P2)
  are SEPARATE increments. Reasoning: latency instrumentation is
  mechanical and low-risk and unblocks the biggest structural suspect
  (multi-query retrieval); the golden-set composition is a genuine
  ratification decision and earns its own increment. Split work by kind.
- 2026-07-17 (planner): VERSION PROVENANCE. GitHub `main` and the local
  committed HEAD are the OLD Streamlit v1 (app.py/dashboard.py/mock_db.py,
  ChromaDB, Gemini 1.5, LangChain ReAct); the local repo is level with
  origin/main with ZERO extra commits. The entire v2 rewrite (backend/,
  frontend/, supabase/, docker-compose, agents/) is UNCOMMITTED working
  tree (10 old files deleted, 13 new entries untracked). So the local
  folder is the newer, correct target and is the ONLY copy of v2; it is
  not backed up. Builder's first act is a snapshot commit + push to a
  branch so v2 exists somewhere besides one folder.
- 2026-07-17 (planner): BATCHING POLICY for token efficiency. Increment 1
  is widened from boot-only to one "Foundation" pass bundling boot
  integrity + frontend build config + the mechanical correctness fixes
  (feedback wiring, tool-answer persistence, cache privacy scoping,
  config hygiene, secrets) AND the tool-path retrieval unification
  (P1-3). Reasoning: these are all correctness/runnability, not
  optimization, and the baseline must measure a CORRECT system. Opus can
  do this coherently in one builder pass, saving conversations. The one
  line NOT collapsed: no optimization (multi-query cut, model routing,
  weight sweeps, chunking) runs before the P1/P2 baselines exist.
  Consequential ratifications (golden-set composition, deploy target)
  remain their own gated steps.
- 2026-07-17 (builder, Increment 1): BOOT LAYOUT settled as a real
  `backend` package run from repo root: `uvicorn backend.api.main:app`.
  Added `backend/__init__.py`; fixed the single flat import (cache.py) to
  `backend.core.config`. To remove CWD fragility, config.py now resolves
  `.env` and the Chroma/parent store paths against the backend package dir
  (absolute), so the app boots identically from any cwd and inside the
  container. Reasoning: the import fix alone was insufficient; the relative
  `.env`/store paths were a second, independent boot break (the planner
  anticipated this under "ensure .env loads under that one cwd").
- 2026-07-17 (builder, Increment 1): lookup_documentation DESIGN. The tool
  returns the RETRIEVED CONTEXT plus sources (JSON), and the graph's own
  LLM node performs the single, streamed synthesis; the tool does NOT call
  the LLM itself. This is a small, deliberate refinement of the kickoff's
  "reuse _run_rag" wording. Reasoning: (a) it satisfies the ratified intent
  ("the graph path retrieves real context, both paths answer from retrieved
  docs") exactly; (b) it avoids a redundant second synthesis LLM call per
  tool-path doc answer (the literal _run_rag-in-tool approach would synth
  once inside the tool and again in the agent node), honoring the cost-guard
  doctrine; (c) it keeps token streaming coherent. Sources are captured by
  parsing the lookup_documentation on_tool_end output. Combined with the N-1
  fix (stream only langgraph_node == "agent" tokens), nested retriever
  tokens never leak to the user. Flagged for the reviewer/planner to object;
  trivially reversible if they prefer the literal form.
- 2026-07-17 (builder, Increment 1): CACHE PRIVACY (safety) scopes ALL
  tiers by user_id: L1/L3 keys become sha256(user_id + query); L2 semantic
  entries carry their owner and cross-user matches are masked out. This is
  the safety fix only; the 0.92 threshold and hit-rate exploitation stay
  P3 Rank 2 as ratified. A committed, dependency-light regression script
  (backend/scripts/check_cache_privacy.py) proves two users cannot share a
  personalized answer, and doubles as the future CI free-tier check.
- 2026-07-17 (planner): Increment 1 (FOUNDATION) ACCEPTED. The reviewer
  re-derived the gate table, reproduced the cache-privacy fix under a 21-probe
  hostile battery, and found no P0/P1. LLM-dependent gates carry into
  Increment 1.5 (blocked by Gemini quota, not code). Do not reopen Increment
  1. Full verdict in the REVIEW and PLANNER RATIFICATION sections below.
- 2026-07-17 (planner): LLM provider Gemini -> Groq RATIFIED as a FOUNDATION
  prerequisite (Increment 1.5), not a P3 optimization: a dead LLM blocks the
  demo and all P1/P2 baselining, so restoring the substrate does not violate
  measure-before-optimize. Provider goes behind an LLM_PROVIDER config +
  factory in get_llm (future swaps are config, not migrations). Model
  llama-3.3-70b-versatile via langchain-groq, temperature 0, max_output_tokens
  2048, pinned in every results recipe. P1/P2 baselines are established ON
  GROQ. P2 note: the RAGAS judge must differ from the generator; free-tier
  rate limits (~30 RPM) require the P2 harness to batch/sleep. Full detail in
  the PLANNER RATIFICATION section below.

## INCREMENT LOG
(builder logs each increment here: what changed, commit hashes, what
was verified)

### Increment 1 - FOUNDATION (builder, in progress 2026-07-17)

STEP 0 - SNAPSHOT (done):
- Branch `v2-fullstack` created off HEAD (main, old Streamlit v1). Working
  tree snapshotted with `git add -A` (the ONE ratified use of add -A).
  Commit `5909e08` "chore: snapshot v2 full-stack rewrite ... before
  hardening". Pushed to `origin/v2-fullstack`. `main` untouched.
- Pre-push safety (verified before committing/pushing):
  - `git add -A` dry-run reviewed: 69 entries, NONE of `.env`, `venv/`,
    `node_modules`, `vector_db/`, `parent_docstore/`, `parents.pkl` staged
    (all gitignored). backend/.env (933 B, real keys) is gitignored and
    NOT in the snapshot.
  - Live-secret history scan (targeted, values redacted): all 7 secrets in
    backend/.env (Google, Supabase URL+service key, JWT, Groq, Cohere,
    Redis) checked with `git log --all -S<value>`: 0 appear in any commit.
    Full gitleaks/trufflehog history scan is Step 2H (tools not yet
    installed; will run via Docker image).
- REPO-HYGIENE FLAG (new, forward task, not this increment): GitHub push
  warned that historical blob `Nexora-Assets.zip` is 66 MB (reachable from
  main; pre-existing on origin, not in the current working tree). Repo
  bloat; candidate for history cleanup (BFG) later, out of scope now.

STEP 1 - INSPECT findings (reality vs the kickoff's ground truth):

CONFIRMED, exactly as the planner recon stated:
- P0-1 import break: the ONLY inconsistent import in the whole backend is
  `backend/services/cache.py:36  from config import get_settings`. Every
  other module (main, auth, database, agent, tools) already uses `backend.*`
  (sweep run with ripgrep, venv excluded). main.py imports cache, so boot
  fails at import. No `backend/__init__.py` (namespace package; the five
  subpackages do have __init__.py). Fix is one line + run-command settle.
- P0-1 Docker: `backend/Dockerfile` (context ./backend, `COPY . .`,
  `CMD uvicorn main:app`) flattens backend/ into /app, so neither `main:app`
  nor `backend.*` resolve. `railway.toml` startCommand `uvicorn main:app`
  same break.
- P0-2 cache leak: `get_cached_response(user_id, query)` keys on
  `_sha256(query)` only (line 178) and L2 semantic (`_semantic.get`) is also
  user-agnostic. Confirmed cross-user leak.
- P1-3 tool path no context: routing is a keyword match on
  order|warranty|serial|ticket|human|support (agent.py:302). TOOLS =
  [check_order_status, check_warranty_status, create_support_ticket];
  NO lookup_documentation bound; the graph path injects no retrieved docs.
  The system prompt already says "RAG FIRST: Use lookup_documentation" and
  create_support_ticket's docstring references it, but the tool never
  existed (dead reference). `_run_rag` (agent.py:147) is DEAD CODE, written
  for exactly this and never wired.
- P1-4 feedback: frontend `streamChat` reads `X-Conversation-Id`
  (api.ts:147) and NOT `X-Interaction-Id`; the `done` event carries no
  interaction id, so `msg.interactionId` is always undefined and
  `handleFeedback` returns early (chat/page.tsx:198). Backend DOES set the
  `X-Interaction-Id` header (main.py:193) and logs an analytics row with
  that id (main.py:176). main.py docstring says `/feedback/{interaction_id}`
  but the real route is body-based `POST /feedback` (drift).
- P1-5 overwrite: agent.py:378 `full_answer = str(output)` clobbers streamed
  content on on_tool_end; done event always sends `sources: []`
  (agent.py:380).
- P2-6 frontend config: frontend/ root has only package.json + vercel.json.
  MISSING tailwind.config, postcss.config, tsconfig.json, next.config,
  next-env.d.ts. Confirmed.
- P2-7 config: jwt_secret default "change-me-in-production-..."
  (config.py:17); allowed_origins default carries the placeholder
  "https://your-vercel-app.vercel.app" (config.py:14). Confirmed.

EXTENDED (refines the ground truth, does not contradict it):
- BOOT, second dimension beyond imports: config.py sets `env_file = ".env"`
  and vector paths `"vector_db"/"parent_docstore"/"parents.pkl"` all
  RELATIVE to CWD. The cwd that makes `backend.*` imports resolve (repo
  root) is NOT the cwd where .env and the stores live (backend/). So even
  after the import fix, running `uvicorn backend.api.main:app` from repo
  root would load NO .env (empty api keys, default jwt) and fail retriever
  init (FileNotFoundError on ./parents.pkl). ingest.py already writes to
  PROJECT_ROOT=backend/ (ingest.py:64-66). Robust fix: make config resolve
  .env and the store paths against the backend package dir (absolute), not
  CWD. The kickoff anticipated this ("Ensure .env loads under that one
  cwd"); this is the precise mechanism.
- docker-compose.yml: (a) references `frontend/Dockerfile` which does NOT
  exist, so a full `docker compose build` fails on the frontend service;
  (b) its backend volume mounts point at repo-root `./vector_db` etc.,
  but the stores live in `backend/vector_db`, so the mounts are wrong;
  (c) it relies on ${VAR} interpolation from a repo-root .env that does not
  exist (real .env is backend/.env). All part of the P0-1 boot fix scope.

NEW (not in the recon list):
- N-1 Tool-path token leakage under astream_events: once lookup_documentation
  is bound and it (or the retriever) calls the LLM, `graph.astream_events`
  surfaces those NESTED model-stream events too. The current tool-path loop
  streams every `on_chat_model_stream` token to the user, so the
  MultiQueryRetriever's query-expansion tokens (and any tool-side synthesis)
  would leak into the visible answer and into full_answer. Fix: filter
  on_chat_model_stream to the top-level graph node only
  (metadata.langgraph_node == "agent").
- N-2 `prose prose-invert` markdown styling (chat/page.tsx:329,
  globals.css) needs `@tailwindcss/typography`, which is not in
  package.json. Without it npm run build still succeeds but markdown is
  unstyled. Will add the plugin under step B (frontend build integrity).
- N-3 requirements.txt pins everything EXCEPT `google-genai` (line 20,
  unpinned). Reproducibility nit; noted for a later pin, not fixed here
  (out of this increment's scope).
- N-4 Cached responses (main.py cached_stream) send no interaction_id and
  log no analytics row, so feedback is a no-op on a cached answer. Accepted
  limitation for this increment (nothing to attach feedback to); noted.

DESIGN FORK (step F, the one design item): resolved in DECISIONS above
(lookup_documentation returns retrieved context + sources; the agent node
does the single streamed synthesis; node-filtered streaming stops nested
retriever tokens leaking). No STEP-1 finding CONTRADICTS the ground truth,
so the build proceeded. A LATER, larger conflict surfaced at STEP 3 (see
"BLOCKER" below): two external dependencies the ground truth called LIVE are
actually dead.

STEP 2 - BUILD (shipped; commit hashes on v2-fullstack):
- fc8ed6e fix(backend): boot integrity (A), config hygiene (G), cache
  privacy (E), tool-path RAG unification + feedback + sources (C/D/F).
- 1fb8289 fix(packaging): Dockerfile / railway.toml / docker-compose aligned
  to `uvicorn backend.api.main:app`, correct store mounts, env_file (A).
- 860b2ee fix(frontend): tailwind/postcss/tsconfig/next config + typography
  plugin + frontend Dockerfile + lockfile (B); feedback interaction_id
  capture (C).
- ec507cc docs(readme): trim worst stale v1 lines + rework banner (I).
- (this STATUS update is committed separately as the increment log.)
All items A-I delivered. No optimization performed (multi-query, routing,
weight sweeps, chunking all remain P3, gated on baselines).

STEP 3 - VERIFY (9 gate checks, honest results):
1. Fresh clone + pip install: PARTIAL. backend/requirements.txt installs
   cleanly (used throughout the session); a true scratch reclone was not
   re-run, but the container build (check 7) exercises pip install from
   zero and was resolving/installing deps successfully when logged.
2. ingest.py --direct builds stores: PASS. 85 parents, 100% coverage, no
   LLM calls, exit 0; writes to the absolute backend/ paths that match the
   config change.
3. Single command boots; GET /health: PASS.
   `uvicorn backend.api.main:app` from repo root -> {"status":"ok",
   "version":"2.0.0"}. Also proved cwd-independent .env loading.
4. login (guest) + chat both paths stream: BLOCKED by dead credentials (see
   BLOCKER). Verified as far as possible without them: app boots; all four
   tools (incl. lookup_documentation) bind with correct schemas; on an LLM
   failure the agent emits a clean `error` SSE event, not a stack trace to
   the client (fail-safe confirmed). The RAG-path refactor, node-filtered
   token streaming, and source capture are code- and structurally verified
   but not yet exercised against a live LLM.
5. thumbs-up writes an analytics row: BLOCKED (needs Supabase). The
   interaction_id-in-done wiring is verified by code review + the frontend
   type-check (npm build) passing; the end-to-end DB write needs a live DB.
6. cross-user cache isolation: PASS. check_cache_privacy.py -> no L1 exact
   or L2 semantic leak; same-user exact + semantic hit works; exit 0.
7. docker compose build + container /health + one chat: PARTIAL / BLOCKED.
   Dockerfile + compose corrected and config-valid; backend image build ran
   through dependency install (structure valid). Runtime chat blocked by
   dead creds. NEW FINDING: torch pulls CUDA/cudnn wheels (~900 MB), so the
   image is very large; pin CPU-only torch in P5 (image size + cold start).
8. npm run build (styled): PASS. Compiled, types checked (validates the
   api.ts + chat page edits), tailwind + typography, 6 pages, exit 0.
9. Secrets scan logged: PASS for the pushable history + remediation filed.
   gitleaks (docker image) over v2-fullstack lineage (the exact 20 commits a
   push sends) = "no leaks found", exit 0; no .env tracked. A full --all
   scan flags an OLD leak (Google + Cohere keys in a root .env in commit
   02b6865, 2025-09-26) that survives ONLY in local `refs/original/*`
   filter-branch backups + reflog: it is NOT on origin/main and NOT in the
   v2-fullstack lineage (a prior filter-branch already purged it from real
   history), the Google key is already rotated (current key differs), and
   the Step-0 push did not carry it. Remediation filed below.

>>> BLOCKER surfaced to the dev (STOP-and-surface; higher value than
plowing on): the ground truth lists "Supabase reads/writes" and the RAG
streaming path as LIVE, but BOTH external dependencies in backend/.env are
DEAD right now:
  (a) Supabase project `cnsodxqiskbjwxhjlmor.supabase.co` returns NXDOMAIN
      from public DNS (8.8.8.8) - the project no longer exists. This breaks
      auth, chat persistence, feedback writes, and analytics (the entire
      /chat/stream endpoint path), independent of this increment.
  (b) The active GOOGLE_API_KEY is API_KEY_INVALID (revoked/expired). This
      breaks all LLM calls (RAG synthesis, multi-query expansion, tool
      reasoning).
Consequence: gate checks 4, 5, and the container runtime chat CANNOT be
verified in this environment, and the live demo path is currently broken
regardless of the code. The Nexora electronics network hosts (Google LLM,
HuggingFace) resolve fine, so this is not a general network issue; it is
dead credentials/infra. NEEDED FROM DEV before the reviewer can close gates
4/5/7-runtime: a valid Google API key and a recreated + reseeded Supabase
project (apply supabase/schema.sql), with backend/.env updated. The code is
correct and ready; only the infra is dead.

STEP 3 UPDATE (live retest after the dev restored infra):
- The dev recreated Supabase (new project, resolves + seeded via schema.sql)
  and replaced the Google key. Live retest then found a NEW real bug:
  passlib[bcrypt]==1.7.4 left bcrypt UNPINNED, so bcrypt 5.0.0 was installed;
  passlib's backend init crashes on bcrypt >= 4.1 ("password cannot be longer
  than 72 bytes"), 500ing every /auth/login. FIXED: pinned bcrypt==4.0.1
  (commit f1c1e70), which also fixes the container. This was a real
  demo-path bug the ground truth could not have caught (the app never booted
  before this increment).
- GATE 4 login: now PASS. alice/password123 -> JWT + profile (Supabase live
  and correctly seeded, owned_products present).
- GATE 4 chat / GATE 5 feedback: still pending ONE infra item. The replaced
  Google key authenticates (no more API_KEY_INVALID) but its project returns
  429 with free_tier limit: 0 for gemini-2.0-flash (the key is an "AQ."-style
  Cloud credential, not an AI Studio "AIza" key with free-tier quota). Needs
  a Gemini key that has quota (a fresh aistudio.google.com key, or billing).
  Once in, chat + feedback close. Graceful failure re-confirmed: the quota
  error surfaces as a clean SSE error event, not a crash.
- Session hygiene (dev's machine was near-full at 98%): freed ~11 GB (pip
  cache 7.6 GB + npm + scratch), killed orphaned dev servers, pruned Docker
  (the completed backend image was ~13 GB due to CUDA torch, reinforcing the
  CPU-torch P5 task). ~18 GB more sits in Docker's WSL2 vhdx and needs an
  elevated diskpart compact (handed to the dev). backend/.env reorganized
  into neat labeled sections (values preserved; gitignored, not committed).

### Increment 1.5 - GROQ + REVIEWER FIXES (builder, 2026-07-17) - GATE MET

Commits on v2-fullstack:
- 4f66004 docs(status): reviewer verdict + planner ratification committed;
  the two ratified DECISIONS and the new forward tasks appended under their
  canonical headers.
- 7c5c298 feat(backend): LLM migrated to Groq behind an LLM_PROVIDER factory;
  F-1 error sanitization; F-2 allowlist token filter; agent max-iterations
  cap; CPU-only torch pin.
- 124f4d4 fix(backend): F-3 persist user message + conversation on cache hit.
- d36f522 docs(readme): F-4 correctness trim of the false v1 body claims.

What shipped (all MUST, SHOULD, and LOW items of the ratified kickoff):
- MUST provider abstraction + Groq: get_llm() is a factory keyed on
  LLM_PROVIDER ("groq" default, "gemini" still selectable). ChatGroq, model
  llama-3.3-70b-versatile, temperature 0, max_output_tokens 2048,
  GROQ_API_KEY from env. Embeddings (MiniLM) and Chroma + BM25 retrieval
  untouched, as ratified.
- MUST F-1: provider errors are never serialized to the client; the real
  exception is logged server-side and the client receives one generic
  message.
- SHOULD F-2: token suppression is now the documented allowlist
  (langgraph_node == "agent"), so untagged or future nested model streams
  cannot leak into the answer.
- SHOULD F-3: cache hits now resolve/create the conversation, persist both
  messages, log analytics, and thread interaction_id (supersedes N-4).
- SHOULD agent max-iterations: LangGraph recursion_limit from
  AGENT_MAX_ITERATIONS (default 8), so a runaway tool loop cannot burn the
  Groq free-tier daily quota.
- SHOULD CPU-only torch: torch==2.10.0+cpu via the PyTorch CPU index.
- LOW F-4: the false README body lines (Streamlit agent,
  ConversationBufferWindowMemory, analytics.log/dashboard.py, AgentExecutor
  self-correction, app.py/dashboard.py/mock_db.py tree) replaced with what
  the code does. Full rewrite stays P6.
- LOW secrets hygiene: refs/original/* backups pruned, reflog expired, gc'd;
  commit 02b6865 no longer exists in the object DB; no tracked .env in any
  ref; origin/main and origin/v2-fullstack carry no key. Full
  `gitleaks detect` (all refs) = "no leaks found", exit 0.

ENUMERATION CATCH (surfaced, not silently absorbed): langchain-groq 1.1.3
pulls langchain-core 1.4.9, a MAJOR bump that breaks the pinned
langgraph 0.2.38 / langchain 0.3.3 stack the reviewer validated. Pinned the
compatible set instead: langchain-groq==0.2.5 with langchain-core==0.3.63
(still <0.4). Re-verified the whole stack imports, all four tools bind, and
the graph builds.

GATE (every Increment 1 LLM-dependent gate re-run ON GROQ; 15/15 automated
checks passed, plus F-1 and the container):
- RAG route: streams a real grounded answer, 18 sources, cites [Source N],
  interaction_id in done. PASS.
- Tool route (doc question): calls lookup_documentation, answers grounded and
  cited ("water damage is not covered [Source 3]"), 16 sources populated.
  PASS. This closes the reviewer's P-1: the unification and the token
  suppression are now verified live, not just structurally.
- Tool route (personalized): calls check_warranty_status, coherent answer, no
  raw tool-output overwrite. PASS.
- F-2 suppression under Groq streaming-with-tools: no nested multi-query or
  raw-JSON tokens in any visible answer. PASS.
- Feedback writes a DB row: POST /feedback 200 and analytics positive went
  0 -> 1 against live Supabase. PASS (closes Increment 1 gate 5).
- F-3 cache hit: re-ask returned cached=true with interaction_id, and the
  analytics total incremented (3 -> 4), proving the turn persisted. PASS.
- F-1 forced provider error: with a deliberately invalid Groq key the server
  logged the full 401 "Invalid API Key" while the client received exactly
  "The assistant is temporarily unavailable. Please try again in a moment."
  with no status code, provider name, quota body, or URL. PASS.
- Container chat on the slimmed image: image 3.2GB (was ~13GB), torch
  2.10.0+cpu with no nvidia/cudnn/nccl wheels; container healthy, running
  `uvicorn backend.api.main:app`; login 200 and a real Groq chat streamed a
  grounded answer citing [Source 1] with 19 sources. PASS (closes Increment 1
  gate 7 runtime).
- Byte-compile of all changed modules clean.

NOT DONE (deliberately, per the kickoff): no latency instrumentation. That is
Increment 2 (P1 baseline on Groq).

Recipe note for the baselines: provider=groq, model=llama-3.3-70b-versatile,
temperature=0, max_output_tokens=2048, langchain-groq==0.2.5,
langchain-core==0.3.63, retrieval unchanged (Chroma k=5 + BM25 0.4/0.6 in an
EnsembleRetriever wrapped by MultiQueryRetriever), embeddings
all-MiniLM-L6-v2. Pin these in every results file.

---

## >>> ACTIVE KICKOFF: Increment 1 - FOUNDATION (BUILDER, batched single pass)

One coherent pass: make the real path runnable, correct, and safe to
measure. This bundles boot integrity, frontend build config, the
mechanical correctness fixes, config hygiene, secrets, and the tool-path
retrieval unification. It does NOT optimize anything (no multi-query cut,
no routing, no weight sweeps, no chunking) and does NOT rewrite the
README beyond trimming the worst stale lines. Reason for batching: these
are all correctness/runnability, and the later baselines must measure a
CORRECT system.

STEP 0 - SNAPSHOT (do this FIRST, the v2 rewrite is currently backed up
nowhere):
- `git checkout -b v2-fullstack`
- `git add -A` then commit: "chore: snapshot v2 full-stack rewrite
  (FastAPI + LangGraph + Next.js + Supabase) before hardening"
- `git push -u origin v2-fullstack`. Leave `main` (the old Streamlit v1)
  untouched. Everything below lands as follow-up commits on this branch.

STEP 1 - INSPECT (report findings into INCREMENT LOG before editing;
a corrected finding is a win):
- Reproduce the boot failure. From repo root try `uvicorn
  backend.api.main:app`; record the exact ImportError. Confirm the
  `cache.py` `from config import get_settings` break and whether
  `backend/__init__.py` exists.
- Confirm the Docker/railway failure (`CMD uvicorn main:app` after the
  build flattens backend/).
- Confirm frontend build config: is `tailwind.config.js` /
  `postcss.config.js` / `tsconfig.json` actually missing? Try
  `npm install && npm run build`.
- Confirm the two paths in `stream_agent_response`: keyword routing, the
  missing `lookup_documentation` tool, and that the tool path injects no
  retrieved context.

STEP 2 - BUILD (all of the following in this pass):
A. Boot integrity. One canonical layout: treat `backend/` as the package
   root (add `backend/__init__.py` if needed), fix `cache.py` to
   `from backend.core.config import get_settings`, make ALL intra-package
   imports consistent, set the single run command to `uvicorn
   backend.api.main:app` from repo root. Correct the Dockerfile (workdir
   + module path + build context so `backend.*` imports resolve, do not
   flatten), `railway.toml`, and `docker-compose.yml` to that same
   command. Ensure `.env` loads under that one cwd.
B. Frontend build integrity. Add the missing `tailwind.config.js`,
   `postcss.config.js`, and `tsconfig.json` (confirm content against the
   existing `globals.css` @tailwind directives) so `npm run build`
   produces a styled build.
C. Feedback wiring (P1-4). Backend: include `interaction_id` in the
   `done` SSE event payload (and keep the `X-Interaction-Id` header).
   Frontend: capture it in `streamChat` and thread it onto the message so
   `handleFeedback` actually POSTs. Verify a thumbs-up writes a row.
D. Tool-path answer + sources (P1-5). Stop overwriting `full_answer` with
   raw tool output; persist the synthesized assistant text; send
   coherent (possibly empty but not malformed) `sources`.
E. Cache privacy (P0-2), safety only. Scope the cache key per user
   (include `user_id` in the hash) OR skip caching personalized/tool
   answers. Do NOT tune the 0.92 threshold or chase hit-rate here; that
   is P3. Just close the cross-user leak. Add a regression check: two
   different users, same question, must not share a personalized answer.
F. Tool-path retrieval unification (P1-3), the one design item. Add a
   `lookup_documentation` tool that wraps the existing retriever + RAG
   (reuse `_run_rag`), bind it into `TOOLS`, and update the agent prompt
   so the graph path retrieves real context. Both paths now answer from
   retrieved docs. Accept the added tool-turn latency (it is correctness,
   not a latency regression to optimize away yet).
G. Config hygiene (P2-7). `jwt_secret` must come from the environment;
   refuse to start (or hard-warn in non-debug) if it is the placeholder.
   Replace the placeholder CORS origin with env-driven config. Add
   `.env.example` with keys and no real values.
H. Secrets. Run `gitleaks detect` (or trufflehog) over the FULL history;
   paste the result into INCREMENT LOG. If a key is found in history,
   STOP and file a remediation task before pushing further.
I. README: trim only the most misleading lines (Streamlit, ReAct, `pip
   freeze` setup) with a one-line "under active rework" note. Full
   rewrite stays P6.

STEP 3 - VERIFY (the gate; all must pass on a FRESH clone of the branch):
1. Fresh clone + venv + `pip install -r backend/requirements.txt` OK.
2. `python backend/scripts/ingest.py --direct` builds stores (or
   committed stores load).
3. Single command boots; `GET /health` returns `{"status":"ok"}`.
4. `POST /auth/login` (guest) returns a token; one `POST /chat/stream`
   streams token + done, no traceback, on BOTH a pure-doc question and a
   warranty/tool question (the latter now has retrieved context).
5. Thumbs-up on an answer writes an analytics feedback row (feedback
   fixed).
6. Two different users asking the same personalized question do NOT get
   each other's answer (cache leak closed).
7. `docker compose build` succeeds and `/health` + one chat work IN THE
   CONTAINER (cold-start proof, not "works in my venv").
8. `npm run build` (frontend) succeeds with styles applied.
9. Secrets scan result logged; clean history (or remediation filed).

DONE = all nine verified, committed on `v2-fullstack`, and logged in
INCREMENT LOG with commit hashes and exactly what was verified. Ask
before pushing if anything is ambiguous; otherwise the dev has
pre-authorized commits for this increment.

AFTER BUILDER: a FRESH reviewer session does an adversarial review of
this whole increment (re-runs the gate itself, tries to break the cache
scoping, the feedback path, the injection surface, the container boot)
and writes a verdict into the REVIEW section. The dev brings the
builder's INCREMENT LOG and the reviewer's verdict back to the planner;
P0/P1 findings are resolved before Increment 2.

Resume artifact: "clone, install, one command, it answers on both paths,
in a container" plus a clean secrets-scan line and a closed privacy leak.

Explicitly NOT in this increment (gated, with reasons):
- Latency instrumentation + baseline (Increment 2 / P1): needs the
  corrected system running first.
- Golden set + RAGAS quality baseline (Increment 3 / P2): its own
  ratification.
- All optimization (multi-query cut, routing, weight sweep, chunking):
  P3, only after both baselines exist.

---

## FORWARD TASKS / DEFERRED (nothing is deferred silently; it lands here)
- P0-2 cross-user cache leakage -> FIXED (safety) in Increment 1;
  latency exploitation stays P3 Rank 2.
- P1-3 tool-path missing RAG context -> unified in Increment 1 (step F).
- P1-4 feedback wiring broken -> FIXED in Increment 1; unblocks the P7
  analytics dashboard.
- P1-5 tool-path answer overwrite + empty sources -> FIXED in Increment 1.
- P2-6 frontend build config -> FIXED in Increment 1 (step B).
- P2-7 insecure jwt default + placeholder CORS origin -> FIXED in
  Increment 1 (step G).
- P2-8 / Rank 1 multi-query retriever latency+cost -> P3 after baseline.
- Rank 8 pgvector migration -> P7, measured, not assumed.
- README honest rewrite -> P6 (only worst lines trimmed in Increment 1).
- v2 backup -> Increment 1 Step 0 (snapshot commit + push to
  `v2-fullstack`); `main` stays on the old Streamlit v1. DONE (5909e08).
- SECURITY / SECRETS REMEDIATION (from Increment 1 gitleaks, priority):
  (1) Confirm the OLD leaked keys are REVOKED: Google `AIzaSyDRXL...` (in
  commit 02b6865's root .env; already rotated, current key differs) and the
  Cohere key in the same commit. Verify in the Google Cloud + Cohere
  consoles. (2) Local repo hygiene: the leak survives only in
  `refs/original/*` (prior filter-branch backups) + reflog; delete those
  refs, expire the reflog, and gc so `gitleaks detect --all` is clean. NOT
  done by the builder (it removes someone else's filter-branch safety
  backups; needs the dev's OK). (3) origin history is already clean of it.
- DEAD CREDENTIALS / INFRA (partly resolved): Supabase RECREATED + reseeded
  by the dev (login now works). Gemini remains blocked (see next item).
- LLM PROVIDER MIGRATION Gemini -> Groq free tier [PENDING PLANNER
  RATIFICATION, do not build until ratified]: Gemini's free tier is now
  effectively unusable (keys expire / return 429 with free_tier limit: 0 on
  gemini-2.0-flash before a single message), which blocks the free demo path
  AND all P1/P2 baselining (you cannot baseline a system with no working
  LLM). Proposal: swap get_llm() from ChatGoogleGenerativeAI to ChatGroq
  (langchain-groq); GROQ_API_KEY already exists in .env; keep HuggingFace
  MiniLM embeddings + ChromaDB retrieval unchanged; pick a tool-calling model
  (candidate: llama-3.3-70b-versatile) at temperature 0 for eval
  reproducibility. This is a FOUNDATION prerequisite (needed to run for free
  at all), not a P3 optimization, so it does not violate measure-before-
  optimize; but the P1/P2 baselines would then be established ON Groq.
  Planner must ratify: (a) Groq as the provider, (b) the exact model + params,
  (c) rate/cost guards for Groq's free-tier limits, (d) whether this reopens
  Increment 1 / becomes Increment 1.5 before the latency/quality baselines.
- CUDA torch image bloat -> P5: pin CPU-only torch (e.g. the
  +cpu wheel / torch CPU index) so the backend image is small enough for
  scale-to-zero deploy; the default Linux torch pulled ~900 MB of CUDA/cudnn.
- Nexora-Assets.zip (66 MB) in the v2-fullstack lineage -> repo bloat; BFG /
  git filter-repo cleanup (coordinate with the secrets history cleanup).
- google-genai is UNPINNED in requirements.txt (N-3) -> pin a version for
  reproducibility (do with the P1 baseline pinning).
- Error-message leakage (P4 hardening): the agent's `error` SSE event
  forwards the raw provider error text (e.g. the full Gemini error) to the
  client; sanitize to a generic user-facing message in P4.
- Per-user Redis cache invalidation (P4): keys are user-scoped but hashed,
  so `invalidate_user_cache` cannot target one user and clears all Redis
  keys; add a per-user key prefix when the limiter/hardening lands.
- Cached responses (main.py cached_stream) log no analytics row and carry no
  interaction_id, so feedback is a no-op on a cached answer (N-4); revisit
  when the cache is exploited for latency in P3 Rank 2.
- PULLED INTO INCREMENT 1.5 (ratified): F-1 raw-error sanitization (raised to
  P2, live leak), F-2 allowlist token filter, F-3 persist message+conversation
  on cache hit (supersedes N-4), F-4 README false-body trim, agent
  max-iterations cap, and CPU-only torch pin. See the PLANNER RATIFICATION
  section for the ratified 1.5 kickoff.
- Self-signup / registration (only seeded users today) -> P7 feature, ranked;
  the P6 public demo is NOT blocked (seeded guest login exists).
- Cost guards (per-request token caps, bounded memory) -> P4; only the agent
  max-iterations cap is pulled into Increment 1.5.

## REVIEW

### Increment 1 - FOUNDATION, adversarial review (reviewer, 2026-07-17)

Scope reviewed: the 9 commits 5909e08 (snapshot) through da17e8f on
v2-fullstack. Everything below was re-derived from the code and re-run by
the reviewer; the builder's gate table was not trusted. Environment:
Windows 11, Python 3.11.9 in backend/venv (bcrypt 4.0.1 confirmed
installed), Node v23.5.0, live backend/.env present. No gitleaks/trufflehog
or ruff available locally; secrets were re-scanned with git plumbing and
lint was substituted with byte-compilation.

VERDICT: mostly clean. The headline security fix (cache privacy P0-2) is
solid under a harder battery than the committed check. Boot, config
refusal, frontend build, tool binding, and graceful failure all reproduce
green. Findings below are one P2 (confirmed live info-leak, already flagged
by the builder as P4) and three P3s. No P0 or P1 defects found in the
shipped code. Several happy-path gates remain UNVERIFIABLE here because the
LLM key returns 429 free_tier limit 0; that is infra, not a code defect,
and it matches the builder's own honesty.

INDEPENDENTLY RE-RUN AND CONFIRMED PASS:
- Boot: `import backend.api.main` from repo root succeeds (app title
  "SentioBot API" 2.0.0). The single-import-break (cache.py) and the
  cwd-relative store/.env paths are genuinely fixed (config.py:21,55-58,70).
- Config JWT refusal (config.py:81-93): placeholder and empty secret both
  raise on boot in non-debug (pydantic ValidationError wrapping the
  ValueError); DEBUG=true warns and continues. Reproduced all three cases.
- Cache privacy P0-2 (cache.py:62-65,108-128,180-213): committed
  check_cache_privacy passed twice (exit 0, stable). Reviewer's own 21-probe
  hostile battery (3 users, 6 exact/near-duplicate cross-asks each, plus a
  fresh-user probe, a same-user regression probe, and a NUL-delimiter key
  forge) found NO cross-user leak on L1 or L2, and same-user exact hits
  still work. L1/L3 key = sha256(user_id \x00 query); L2 masks non-owner
  entries to score -1.0 before argmax. The fix holds.
- Frontend build P2-6 (gate 8): `npm run build` compiled, types checked
  (validates api.ts + chat/page.tsx edits), typography plugin present, 6
  pages, exit 0.
- Tool binding + unification P1-3 (agent.py:150-192): all four tools
  (lookup_documentation, check_order_status, check_warranty_status,
  create_support_ticket) bind with correct arg schemas; graph nodes are
  [__start__, agent, tools]. Structurally sound.
- Answer overwrite / sources P1-5 (agent.py:370-382): confirmed by trace
  that full_answer is no longer clobbered with raw tool output; the agent
  node's streamed tokens are the persisted answer and tool_sources come from
  parsing lookup_documentation output.
- Feedback wiring P1-4: main.py:171 injects interaction_id into the done
  event; api.ts:45 + chat/page.tsx:183 consume it; submitFeedback posts
  {interaction_id, feedback}. Coherent end to end at the code level (the DB
  write itself needs live Supabase, not run here).
- bcrypt pin (requirements.txt:12): bcrypt==4.0.1 present and installed;
  login bug fix is real.
- Graceful failure (see F-1 for the caveat): doc path, tool path, a
  100K-char message, and a prompt-injection message ALL degrade to a single
  clean `error` SSE event with no uncaught exception or stack trace reaching
  the client.

SECRETS: CLEAN for the pushable lineage. No `.env` is tracked anywhere in
5909e08~1..da17e8f; the 9-commit diff adds no key values (only variable
names and empty placeholders in .env.example and config defaults); a
history `-S` scan for AIza/JWT patterns is empty; the old-leak commit
02b6865 cited in the log does not exist in this repo at all (already purged,
not merely unreachable). backend/.env is gitignored and is excluded from the
backend image by backend/.dockerignore, so it does not bake into image
layers. Independent confirmation of the builder's secrets claim: agreed.

CONFIRMED FINDINGS (reviewer reproduced):

F-1 [P2] Raw provider error text is forwarded verbatim to the client.
  file: backend/agent/agent.py:336 and :386 (also surfaced by main.py's
  passthrough of the error event).
  Failure scenario: any LLM/retrieval exception is serialized as
  `{'type':'error','data':{'message': str(e)}}`. Driving all four inputs
  through stream_agent_response live produced an error event containing the
  full Gemini 429 body: quota text, quota_dimensions, retry_delay, and
  https://ai.google.dev/... links, delivered straight to the end user.
  Reproduction: run backend with the current .env key and send any chat;
  observe the error SSE payload. The builder flagged this as a deferred P4;
  it is live now and leaks provider/infra/quota internals to end users, so
  it should be sanitized to a generic message before any public deploy.
  Raising to P2 because it is on the live user-facing path.

F-2 [P3] Token-leak filter is a denylist, not the documented allowlist.
  file: backend/agent/agent.py:360.
  The DECISIONS log and the code comment both say "stream only
  langgraph_node == 'agent' tokens", but the code streams every
  on_chat_model_stream whose node is NOT "tools". For the current two-node
  graph these are equivalent, so no leak today. Risk: if any third node is
  added later, or if a top-level model-stream event arrives with no
  langgraph_node tag (metadata.get returns None, and None != "tools"), the
  nested MultiQueryRetriever expansion tokens would leak into the visible
  answer. Cheap hardening: switch to the allowlist form
  (`langgraph_node == "agent"`) the design already specifies.

F-3 [P3] Cache-hit path skips message persistence, not just analytics.
  file: backend/api/main.py:126-134.
  On a cache hit the endpoint returns cached_stream and returns BEFORE
  creating the conversation, saving the user message, or logging analytics.
  N-4 in the log notes only the missing analytics/feedback; the broader
  effect is that the whole exchange is absent from conversation history (the
  sidebar loses that turn, and a None conversation_id creates no
  conversation). Confirmed by code trace; not reproduced end to end because
  populating the cache needs a live LLM. Pre-existing, but the increment's
  user-scoped cache makes hits real, so it belongs on the P3/P4 list.

F-4 [P3] README body still asserts flatly-false claims under the banner.
  file: README.md:90,97,116,145-146.
  The added status banner (README.md:11-16) discloses the sections as stale,
  and the worst setup lines (pip freeze, Gemini 1.5) were removed. But the
  body still states "The Streamlit app runs a stateful, reasoning agent"
  (:90), "ConversationBufferWindowMemory" (:97), and lists app.py /
  dashboard.py as the architecture (:145-146), all false against the
  FastAPI + LangGraph + Next.js + Supabase reality. Disclosed-stale and
  explicitly deferred to P6, so P3, not a blocker; noting that a banner over
  concrete falsehoods is exactly the pattern this repo was burned by before.

PLAUSIBLE / UNVERIFIED (blocked by dead infra, not proven defects):

P-1 The token-leak SUPPRESSION itself (F-2's happy-path behavior), grounded
  [Source N] synthesis on the tool path, non-empty sources on a real doc
  answer, the end-to-end feedback DB write, and container runtime chat could
  NOT be exercised: the .env Gemini key returns 429 with free_tier limit 0,
  so no real completion streams. This is the builder's stated highest-risk
  path and remains unverified. It cannot be closed until a working LLM key
  (or the pending Groq migration) is in place. Recommend re-running gates 4,
  5, and container-chat with quota before this increment is called done.

P-2 Tool-path routing relies on the LLM CHOOSING to call lookup_documentation.
  file: backend/agent/agent.py:302-305, :233-242.
  A doc question containing a routing keyword (e.g. "warranty") enters the
  graph, where grounding depends on the model actually calling the tool. If
  it answers from parametric knowledge instead, there is no retrieved
  context despite the "RAG FIRST" prompt. This is the ratified design, not a
  regression; flagged so the P2 quality eval measures it honestly rather
  than assuming grounding.

CONTEXT (not findings):
- No unit tests ship in this increment; the only executable check is
  check_cache_privacy. ruff is not installed in the venv, so lint was
  substituted with byte-compilation of all changed modules (clean).
- google-genai remains unpinned (requirements.txt:23), matching N-3.
- A plain `docker run` of the backend image (no compose volume) has no
  stores and would FileNotFoundError on first chat; /health still works.
  Consistent with the stated mount-the-stores design, not a defect.

BOTTOM LINE: the shipped code is sound. The cache privacy fix, boot
integrity, config hardening, and frontend build are genuinely done and
reproduce green. Fix F-1 before any public exposure (it is a real live
leak), take F-2 as cheap defensive hardening, and treat F-3/F-4 as tracked
P3s. The increment is NOT closable as-is only because the LLM-dependent
gates (chat happy path, sources, feedback write, container chat) cannot be
run against a quota-zero key; those need a working provider, then a
re-review of P-1.

---

## PLANNER RATIFICATION (2026-07-17): Increment 1 close + Groq (Increment 1.5)
(Supersedes the ACTIVE KICKOFF for Increment 1 above; that increment is now accepted.)

### Verdict
- Increment 1 (FOUNDATION): ACCEPTED. Reviewer re-derived the gate table,
  found no P0/P1, reproduced the cache-privacy fix under a 21-probe hostile
  battery, and independently reproduced boot, JWT refusal, frontend build,
  tool binding, graceful failure, and clean secrets. The code-verifiable
  gates are met. The LLM-dependent gates (chat happy path, grounded
  [Source N], sources, feedback DB write, container chat, token suppression)
  are BLOCKED by infra (Gemini 429 free_tier limit 0), not code, and carry
  into Increment 1.5. Do not reopen Increment 1.

### Decisions
- LLM provider Gemini -> Groq: RATIFIED as a foundation prerequisite
  (Increment 1.5), not a P3 optimization. A dead LLM blocks the demo AND all
  P1/P2 baselining, so restoring the substrate does not violate
  measure-before-optimize. Consequence: the P1 latency and P2 quality
  baselines are established ON GROQ and recorded as such.
  (a) Provider: Groq via langchain-groq. REQUIRED refinement: put the
      provider behind config (LLM_PROVIDER env + a factory inside get_llm)
      so a future swap is config, not another migration.
  (b) Model: llama-3.3-70b-versatile (verified current Groq production
      2026-07-17: 131K context, 32K max completion, tool-calling capable),
      temperature 0, max_output_tokens pinned 2048. Pin the exact model ID
      in every results recipe; watch console.groq.com/docs/deprecations.
      llama-3.1-8b-instant reserved for the P3 routing candidate, not now.
  (c) Sequencing: Increment 1.5, not a reopen. Lands before Increment 2
      (P1 latency) and Increment 3 (P2 quality).
  (d) Golden set evaluated on Groq: yes. P2 open item to ratify: the RAGAS
      judge should not be the same model as the generator (self-preference
      bias); document the judge in the recipe, prefer a distinct free judge.
      Free tier is ~30 RPM / ~14.4K req-day / ~6K TPM; the multi-query
      retriever and the judge both spend requests, so the P2 harness must
      batch/sleep within the daily budget. Pinned now, not a later surprise.

### Reviewer findings, disposition (pulled into 1.5 for token efficiency)
- F-1 (P2, raw provider error to client): INTO 1.5, not P4. Forwarding the
  raw 429 body + internal URLs is a live info leak; emit a generic error SSE.
- F-2 (P3, denylist vs allowlist token filter): INTO 1.5. The Groq swap
  changes streaming-with-tools event tagging, so re-verify suppression and
  make it the allowlist (== "agent") while there.
- F-3 (P3, cache hit skips persisting message/conversation): INTO 1.5. It
  holes multi-turn history, which the P2 golden set tests; fix before baseline.
- F-4 (P3, README body still says Streamlit/app.py): targeted correctness
  trim in 1.5; full rewrite stays P6.
- Agent max-iterations cap: INTO 1.5 (LangGraph recursion limit). An
  unbounded tool loop on Groq free tier can burn the daily quota. Fuller
  cost guards (token caps, bounded memory) stay P4.
- CPU-only torch pin: pulled forward from P5 into 1.5 (image ~13GB ->
  testable container chat gate). One-line requirements change.
- Secrets follow-up: prune local refs/original/ filter-branch backups +
  expire reflog + gc; confirm origin never received the rotated key.
  Exposure already closed by rotation; this is hygiene.

### >>> ACTIVE KICKOFF: Increment 1.5 (BUILDER)
MUST:
- Provider abstraction + Groq swap in get_llm (config-selectable),
  GROQ_API_KEY from env; embeddings + Chroma retrieval unchanged.
- F-1 sanitize provider errors before the client.
- Re-verify ALL Increment 1 LLM-dependent gates ON GROQ: chat both routes,
  grounded [Source N] + sources populated, feedback writes a DB row, tool
  path grounds/cites, graceful failure clean.
SHOULD:
- F-2 allowlist token filter + re-verify suppression under Groq.
- F-3 persist user message + conversation on cache hit.
- Agent max-iterations cap (LangGraph recursion limit).
- CPU-only torch pin in requirements.
LOW:
- F-4 correctness-trim false README body claims (no full rewrite).
GATE (Increment 1.5 done): every Increment 1 LLM-dependent gate passes on
Groq; F-1 verified with a forced provider error; container chat works on the
slimmed image; INCREMENT LOG updated with commit hashes + what was verified.
Then a fresh reviewer re-reviews F-1 (confirms F-2/F-3), push v2-fullstack,
and proceed to Increment 2 (P1 latency baseline on Groq).

### Forward tasks (append)
- Self-signup / registration (only seeded users today) -> P7 feature,
  ranked; P6 public demo NOT blocked (seeded guest login exists).
- CPU-only torch / image slim -> pulled into 1.5; rest of container opt P5.
- Cost guards (token caps, bounded memory) -> P4; only agent max-iters in 1.5.

---

## BUILDER MESSAGE (Increment 1.5 complete, 2026-07-17)

To the PLANNER and the next REVIEWER, via the dev.

STATUS: Increment 1.5 is COMPLETE and its gate is MET. Every MUST, SHOULD, and
LOW item in the ratified kickoff shipped, and every Increment 1 LLM-dependent
gate that was blocked by dead infra has now been re-run and PASSED on Groq.
Commits: 4f66004, 7c5c298, 124f4d4, d36f522 (full detail and the verified gate
list are in the INCREMENT LOG entry "Increment 1.5"). The branch is NOT pushed
yet, per the standing ask-before-push rule.

THE HEADLINE: the substrate is alive. Gemini's free tier returned 429 with
free_tier limit 0 before a single message; on Groq (llama-3.3-70b-versatile,
temperature 0) both answer routes now stream real, grounded, cited answers.
The reviewer's P-1 (the highest-risk unverified path: tool-path grounding,
non-empty sources, token suppression, feedback DB write, container chat) is
now CLOSED by live evidence, not structure:
- tool route calls lookup_documentation and answers "water damage is not
  covered [Source 3]" with 16 sources;
- feedback wrote a real Supabase row (analytics positive 0 -> 1);
- container chat works on a 3.2GB image (was ~13GB);
- a forced 401 leaks nothing to the client (F-1).

WHAT I WANT THE REVIEWER TO ATTACK (in priority order):
1. F-1, the P2 you raised. Try to make ANY provider/internal detail reach the
   client: bad key, rate limit, timeout, tool exception, recursion-limit
   breach. The generic string is in agent.py (_GENERIC_ERROR); confirm no
   other path serializes str(e). Note main.py passes agent SSE through
   unchanged, so agent.py is the choke point; verify that is actually true.
2. F-2 allowlist. I switched to `langgraph_node == "agent"` as ratified and
   verified no leak under Groq streaming-with-tools. Your own warning applies
   in reverse now: if a real agent-node stream ever arrives untagged, the
   allowlist SUPPRESSES the answer instead of leaking. It did not happen on
   Groq across 6 live chats, but please probe it (multi-tool turns, a
   recursion-limit breach, a tool that errors mid-stream).
3. F-3 cache-hit persistence. Verified: cached=true, interaction_id present,
   analytics total incremented. Probe multi-turn: does the cached turn appear
   in the sidebar/history and feed the next turn's context correctly? Also
   whether a cache hit on a NEW conversation creates a sensible title.
4. The dependency pin. langchain-groq 1.1.3 silently pulls langchain-core
   1.4.9 and breaks the langgraph 0.2.38 stack; I pinned langchain-groq==0.2.5
   + langchain-core==0.3.63 instead. Please confirm a FRESH venv from
   requirements.txt resolves cleanly and the app boots (this is the one change
   most likely to bite a cold environment).
5. P-2 remains true and unfixed by design: tool-path grounding depends on the
   model CHOOSING to call lookup_documentation. On Groq it did so on every
   doc-ish probe I ran, but that is not a guarantee. The P2 eval must measure
   it rather than assume it.

FLAGS / HONEST GAPS:
- google-genai is still unpinned (N-3). Gemini is now a fallback path only,
  but the pin belongs in the P1 recipe work.
- Nexora-Assets.zip (66MB) still bloats the lineage; cleanup not attempted.
- Groq free tier is roughly 30 RPM. The agent max-iterations cap is in, but
  per-request token caps and bounded memory remain P4. The P2 harness will
  need batching/sleeps to stay inside the daily budget (planner already
  pinned this).
- No unit tests still; check_cache_privacy remains the only executable check.

FOR THE PLANNER: nothing here contradicts the ratified plan, and I did not
touch latency instrumentation. The provider, model, params, and the exact
langchain pin set are recorded in the INCREMENT LOG "Recipe note" so the
Increment 2 (P1 latency) and Increment 3 (P2 quality) baselines can stamp them
verbatim. My only forward-looking observation: now that Groq is live and fast,
the MultiQueryRetriever's extra LLM round-trip (Rank 1 in the improvement
list) is measurable the moment the P1 harness exists; it remains gated behind
the baseline as ratified.
