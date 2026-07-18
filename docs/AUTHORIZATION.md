# Authorization model (SentioBot)

This document states, explicitly, WHO enforces object-level access control and
HOW, so it is a deliberate decision rather than tribal knowledge. Added in
Increment 7 (app-layer sweep); upgraded in Increment 10 to a hybrid model where
the database itself enforces ownership for user-owned data (fail-closed).

## The gatekeeper: hybrid (database RLS + app-layer defense in depth)

There are two clients, used deliberately:

- **User-JWT client (RLS enforces).** For the request-path user-owned tables
  (conversations, messages, analytics), the backend builds a PER-REQUEST Supabase
  client authed with the caller's JWT (`database._user_db`). The auth token is
  signed with the **Supabase JWT secret** and carries `role=authenticated` and
  `sub = public.users.id`, so Postgres **Row-Level Security** filters every query
  to the caller's own rows. This is **fail-closed**: if a future endpoint forgets
  its app-level check, the database still denies cross-user access. Policies live
  in `supabase/migrations/increment10_rls.sql`.
- **Service-role client (bypasses RLS), used only where it must be.** Pre-auth
  and system operations that have no user JWT or are not user-scoped: `users`
  (login lookup and `get_current_user` run before/at auth), `products`/warranty
  (reference data), and the agent tool path (`orders`, `tickets`), which keeps
  the Increment 7 `current_user_id` ContextVar check as a documented, tested
  exception. Rationale: the request-path endpoints get the biggest fail-closed
  win for the least complexity; threading a JWT through the LangGraph tool nodes
  is real complexity for marginal benefit on a path that is already app-checked
  and denial-suite-covered.

The app-layer checks from Increment 7 (`require_conversation_owner`,
`require_analytics_owner`, the tool owner checks) are **kept** as defense in
depth: they give clean 404/403 responses and a second, independent barrier. RLS
is now the load-bearing control for user-owned data; the app checks are the belt
on top.

Why the policies use the claim directly: our users live in `public.users`, NOT
`auth.users` (we do not use Supabase Auth), so `auth.uid()` would never match and
would deny everything. The policies use `user_id = (auth.jwt() ->> 'sub')::uuid`.

Staged rollout: the user-JWT path activates only when both `SUPABASE_JWT_SECRET`
and `SUPABASE_ANON_KEY` are set (and the migration applied). Without them the app
falls back to the Increment 1-9 behaviour (service-role + app-layer checks), so
the change is safe to deploy before the Supabase side is configured.

## Where ownership is enforced

| Resource | Enforcement | Rule |
|---|---|---|
| Conversations / messages | **DB RLS (user-JWT client)** + `require_conversation_owner` (defense in depth) | RLS: `user_id = jwt sub` (messages via EXISTS on the parent conversation). App-check: 404 if missing/not owned |
| Analytics summary | **DB RLS** + scoped query (`get_analytics_for_user`) | RLS returns only the caller's rows; never the whole table |
| Feedback | **DB RLS** + `require_analytics_owner` | RLS: the row's `user_id = jwt sub`; app-check 404 otherwise |
| Orders | `agent/tools.check_order_status` (service-role + app-check) | order.user_id must equal the caller (`current_user_id`); orders-owner migration is live |
| Warranty | `agent/tools.check_warranty_status` (service-role + app-check) | resolved only against the caller's OWN registered products (Increment 6) |
| Tickets | `agent/tools.create_support_ticket` (service-role) | always created for the authenticated `current_user_id`, never an LLM-supplied id (Increment 1.6) |
| Own conversations list / profile | `/chat/conversations`, `/auth/me` | scoped to the authenticated user (RLS for conversations) |

Verified by:
- the cross-user denial suite `backend/scripts/authz_negative_test.py`
  (results/authz_negative_test.json): user A is DENIED user B's object and A's own
  access works, per resource;
- the **fail-closed proof** `backend/scripts/fail_closed_proof.py`
  (results/increment10_fail_closed.json): with the app-level check removed from
  the path, the DB STILL returns none of another user's rows, proving RLS is the
  enforcement and not the app check masking it.

## Required migration before public deploy

Orders had no owner column, so `check_order_status`'s ownership check was inert
(any logged-in user could read any order by id). The fix is written but needs a
one-time DDL migration that the service-role PostgREST client cannot run:

    supabase/migrations/increment7_orders_owner.sql

Run it once in the Supabase SQL editor (or via `backend/scripts/apply_migration.py`
with a direct `SUPABASE_DB_URL`). After it runs, the denial suite's live orders
row goes from SKIP to PASS (Alice refused Bob's NX-2025-301). **P6 deploy is
gated on this.**

## Deferred (logged, not implied)

- **Extend RLS to the tool path (orders / tickets).** These still use the
  service-role client with the Increment 7 app-check (a documented exception),
  because threading the user JWT through the LangGraph tool nodes is real
  complexity for a path that is already app-checked and denial-suite-covered. A
  future increment could move it to the user-JWT client for full fail-closed
  coverage.
- **Org-wide / admin analytics dashboard.** The old `/analytics/summary` behaviour
  (all users' data) is an admin feature, not a normal-user one. It needs a
  `users.is_admin` flag (another migration) and an admin gate; deferred. Normal
  users get only their own summary.
