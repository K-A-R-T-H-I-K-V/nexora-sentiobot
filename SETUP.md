# Setup: the three-agent workflow for sentiobot-prod

## 1. Install into your local repo
Copy into the root of your nexora-sentiobot clone:

    agents/planner.md
    agents/builder.md
    agents/reviewer.md
    CLAUDE.md
    STATUS.md
    STATUS-prod.md

Commit them (they contain no secrets). CLAUDE.md makes every default
Claude Code session in this repo boot as the BUILDER automatically.

## 2. The three agents

PLANNER (Claude Cowork):
- Open Cowork on the repo folder. First message:
  "Read agents/planner.md and act as it. Then read STATUS-prod.md and
   the repo, and ratify/draft the kickoff for the next increment."
- The planner reads but never edits code. It writes specs and
  ratifications into STATUS-prod.md (via you, or directly if you let
  Cowork edit only that file).

BUILDER (Claude Code, your main session):
- `claude` in the repo root. CLAUDE.md boots it as the builder.
- Say: "Start the next increment per STATUS-prod.md." It will inspect
  first and report findings for ratification before building.

REVIEWER (Claude Code, a FRESH session, after an increment ships):
- Open a NEW `claude` session (no shared context with the builder;
  that is the point). First message:
  "Ignore CLAUDE.md's builder role. Read agents/reviewer.md and act as
   it. Review the latest increment: <commit hash or 'uncommitted diff'>."
- It re-runs everything itself and writes a verdict into
  STATUS-prod.md.

## 3. The relay loop (same as your SkyCRM cadence)
1. Builder inspects -> findings into STATUS-prod.md.
2. You carry findings to the Planner -> it ratifies + writes the
   kickoff into STATUS-prod.md.
3. Builder builds the increment -> verifies against the gate -> you
   approve the commit.
4. Reviewer (fresh session) tears it apart -> verdict into
   STATUS-prod.md. P0/P1 findings go back to the Builder before the
   next increment starts.
5. Repeat. Phase order lives in STATUS-prod.md; the Planner holds it.

## 4. First session, concretely
Start with the Planner. Give it planner.md + STATUS-prod.md and say:
"P0 is next. Ratify the P0 scope and draft the Builder kickoff."
Then take that kickoff to the Builder. P0 needs no design debate, so
expect it to move fast; the first real ratification fight should be
P1's metric definitions and golden-set composition. Let the Planner
push back there; that is what it is for.

## 5. Cost guardrails (you are on a budget)
- Gemini free tier covers dev + eval runs at this scale if batched.
- CI must never make unbounded paid API calls (builder + reviewer both
  enforce this).
- Deployment: HF Spaces / Streamlit Community Cloud are zero cost;
  Cloud Run is near-zero at portfolio traffic and matches your resume
  narrative. The Planner ratifies the choice with monthly cost stated.
