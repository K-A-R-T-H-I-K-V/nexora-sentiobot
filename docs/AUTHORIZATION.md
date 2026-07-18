# Authorization model (SentioBot)

This document states, explicitly, WHO enforces object-level access control and
HOW, so it is a deliberate decision rather than tribal knowledge. Added in
Increment 7 after a BOLA/IDOR sweep.

## The gatekeeper: application code, not RLS

The database (`supabase/schema.sql`) defines Row-Level Security policies such as
"a user can only select their own row". Those policies are **not** the enforcement
for app traffic, because the backend connects with the **Supabase service-role
key** (`backend/services/database.py`), and the service role **bypasses RLS by
design**. A service-role query sees everything.

Consequence, and the rule for this codebase:

> **Every user-scoped resource MUST have its ownership checked in application
> code, on every endpoint and every tool that takes an object id.** RLS is a
> backstop and documentation of intent; it enforces nothing while we use the
> service role.

Treating the RLS policies as if they protected app queries is exactly the false
sense of safety that caused the Increment 7 defects.

## Where ownership is enforced

| Resource | Enforcement point | Rule |
|---|---|---|
| Conversations / messages | `main.require_conversation_owner`, called by GET `/chat/conversations/{id}/messages` AND POST `/chat/stream` (conversation_id) | conversation.user_id must equal the caller; 404 if missing, 403 if not owned |
| Analytics summary | GET `/analytics/summary` | scoped to the caller's own rows (`get_analytics_for_user`); never the whole table |
| Feedback | POST `/feedback` | the analytics row's user_id must equal the caller; 403 otherwise |
| Orders | `agent/tools.check_order_status` | order.user_id must equal the caller (`current_user_id`); needs the orders-owner migration to be live (below) |
| Warranty | `agent/tools.check_warranty_status` | resolved only against the caller's OWN registered products (Increment 6) |
| Tickets | `agent/tools.create_support_ticket` | always created for the authenticated `current_user_id`, never an LLM-supplied id (Increment 1.6) |
| Own conversations list / profile | `/chat/conversations`, `/auth/me` | already scoped to the authenticated user |

Verified by the cross-user denial suite: `backend/scripts/authz_negative_test.py`
(results/authz_negative_test.json). It asserts, per resource, that user A is
DENIED user B's object AND that A's own access still works.

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

- **RLS-with-user-JWT model.** A cleaner long-term option is to stop using the
  service role for user requests and instead pass the user's JWT to Supabase so
  the RLS policies do the enforcing. That is a larger change (every query path,
  plus policies for every table); deferred. Until then, app-layer checks are the
  single source of truth and must stay consistent.
- **Org-wide / admin analytics dashboard.** The old `/analytics/summary` behaviour
  (all users' data) is an admin feature, not a normal-user one. It needs a
  `users.is_admin` flag (another migration) and an admin gate; deferred. Normal
  users get only their own summary.
