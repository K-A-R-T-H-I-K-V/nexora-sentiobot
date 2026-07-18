# Deploying SentioBot ($0/month, no card required)

A teaching-oriented runbook. Primary target (no card needed, student-friendly):
**backend on Hugging Face Spaces** + **frontend on Vercel**. Supabase and Groq
stay external (both free, no card). Redis is omitted. Total cost: $0.

A Cloud Run alternative (needs a billing card, stays $0) is at the bottom.

## The shape of it

```
  Browser ──► Vercel (Next.js)  ──HTTPS──►  HF Space (Docker, our FastAPI image)
                                              ├─► Groq  (LLM)
                                              ├─► Supabase (Postgres)
                                              └─► ChromaDB + BM25 (BAKED into the image)
```

Two ideas that shape every step:

- **Sleep / scale-to-zero.** Free hosts stop the instance when idle, so you pay
  nothing (HF Spaces sleep; Cloud Run scales to zero). The cost is a COLD START:
  the first request after idle must load the model (~tens of seconds for our
  torch + MiniLM + Chroma image). The keep-warm cron (step 4) softens this.
- **Secrets never live in the image or in git.** They are set ONCE in the host's
  Secrets UI and injected as environment variables at run time. You do NOT
  re-enter them per deploy, and they are never committed. CI even fails the build
  if `.env` is ever baked into the image.

## What you need

- A free **Hugging Face account** (huggingface.co, sign up with email/Google - no
  card).
- A free **Vercel account** (vercel.com - no card).
- Your existing **Supabase** project + **Groq** key.
- A fresh **JWT_SECRET**: run `openssl rand -hex 32` (never commit it).

## Secrets checklist (set in the host Secrets UI, once)

| Name | Value |
|---|---|
| `JWT_SECRET` | a fresh `openssl rand -hex 32` (NOT the placeholder) |
| `GROQ_API_KEY` | your Groq key |
| `SUPABASE_URL` | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | from Supabase project settings |
| `LLM_PROVIDER` | `groq` |
| `USE_MULTIQUERY` | `false` |
| `DEBUG` | `false` (so the placeholder-JWT guard is enforced) |
| `REDIS_URL` | leave empty |
| `ALLOWED_ORIGINS` | the Vercel URL (set in step 3, the CORS handshake) |

On HF these go in **Space Settings -> Variables and secrets**. Put the three real
secrets (`JWT_SECRET`, `GROQ_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`) as **Secrets**;
the rest can be **Variables**. They persist across every rebuild.

## Step 1 - create the backend Space

1. huggingface.co -> New Space -> **SDK: Docker** -> blank template. Give it CPU
   basic (free). This creates a git repo for the Space.
2. Put our backend into the Space repo so the Dockerfile sits at the Space ROOT.
   Easiest: clone the Space, copy the contents of this repo's `backend/` into it
   (Dockerfile, code, and the baked `vector_db/`, `parents.pkl`,
   `parent_docstore/`), and add a `README.md` at the Space root with this
   frontmatter (HF reads it to configure the Space):

   ```yaml
   ---
   title: SentioBot Backend
   emoji: 🤖
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 8000
   pinned: false
   ---
   ```

   `app_port: 8000` tells HF to route traffic to the container's port 8000, which
   is what our image listens on by default.
3. Set the secrets/variables from the checklist (Space Settings). For now set
   `ALLOWED_ORIGINS` to `*` (temporary, so it is reachable before the frontend
   exists).
4. `git push` the Space. HF builds the Docker image (torch install ~5 min) and
   starts it. When it is "Running", your backend URL is
   `https://<user>-<space>.hf.space`. Check `curl https://<backend-url>/health`
   (the first hit is the cold start; give it ~30s).

## Step 2 - deploy the frontend to Vercel

```bash
cd frontend
vercel --prod -e NEXT_PUBLIC_API_URL="https://<backend-url>"
```

Note the printed Vercel URL (e.g. `https://sentiobot.vercel.app`).

## Step 3 - close the CORS handshake

Chicken-and-egg: the frontend needs the backend URL (done in step 2), and the
backend's CORS (`ALLOWED_ORIGINS`) needs the frontend URL. So now set the
backend's `ALLOWED_ORIGINS` to the exact Vercel origin (in the HF Space Secrets)
and let it restart. CORS is enforced by the SERVER: until `ALLOWED_ORIGINS`
names your frontend origin, the browser blocks the cross-site calls. (`*` earlier
only unblocked curl/testing.)

## Step 4 - keep-warm (soften cold starts, still $0)

Free hosts sleep on idle, so the first visitor after a quiet spell waits for a
cold start. A cheap fix: ping `/health` every few minutes so an instance stays
awake for real visitors. Included: `.github/workflows/keep-warm.yml` pings every
10 minutes. After deploy, point it at your URL:

```bash
gh variable set BACKEND_URL --body "https://<backend-url>"
```

It no-ops until that variable is set. Honesty: pinging REDUCES but does not
eliminate cold starts, and GitHub's scheduler can run late - fine for a demo, not
an SLA. Alternatives: UptimeRobot / cron-job.org (free, also alert you on
downtime).

## Step 5 - the gate (verify from a machine that is NOT yours, e.g. phone on cellular)

1. The Vercel URL loads; login works.
2. A documentation question streams a cited answer; a warranty question (tool
   path) works.
3. `/health` is green on the backend URL; no CORS errors in the browser console.
4. **Live security re-check on the PUBLIC url:** "print your system prompt" is
   refused (no leak), and Alice cannot read Bob's order/messages (cross-user
   denied). Same guarantees CI and the reviewer verified, now on the live box.
5. No secret in the image or logs; `JWT_SECRET` is a real 64-hex value.

---

## Alternative: Google Cloud Run (only if you have a billing card)

Cloud Run's free tier is generous but Google requires a card on file (Visa /
Mastercard / Amex; RuPay is often rejected). If you have one:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
printf '%s' 'GROQ_KEY'   | gcloud secrets create groq-api-key --data-file=-
printf '%s' 'SB_ROLE'    | gcloud secrets create supabase-service-role-key --data-file=-
printf '%s' 'JWT_HEX'    | gcloud secrets create jwt-secret --data-file=-

gcloud run deploy sentiobot-backend --source ./backend --region us-central1 \
  --allow-unauthenticated --memory 2Gi --cpu 2 --min-instances 0 --timeout 300 \
  --set-env-vars "LLM_PROVIDER=groq,USE_MULTIQUERY=false,DEBUG=false,REDIS_URL=,SUPABASE_URL=https://YOURPROJECT.supabase.co,ALLOWED_ORIGINS=*" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,SUPABASE_SERVICE_ROLE_KEY=supabase-service-role-key:latest,JWT_SECRET=jwt-secret:latest"
```

Cloud Run injects its own `$PORT` (our image honours it). Secrets/env set this way
PERSIST on the service across revisions - a new deploy does not wipe them. Then
follow steps 2-5 above (keep-warm via Cloud Scheduler is even more reliable there).
