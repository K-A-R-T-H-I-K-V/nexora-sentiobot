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

## INCREMENT LOG
(builder logs each increment here: what changed, commit hashes, what
was verified)

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
  `v2-fullstack`); `main` stays on the old Streamlit v1.

## REVIEW
(reviewer verdicts land here)
