-- =============================================================================
--  Increment 10 migration - RLS-with-JWT (fail-closed authorization)
--  Run this ONCE in the Supabase SQL editor. Idempotent (safe to re-run).
--
--  The backend now signs auth tokens with the Supabase JWT secret
--  (role=authenticated, sub=public.users.id) and routes the user-owned tables
--  (conversations, messages, analytics) through a per-request user-JWT client.
--  These policies then enforce ownership in the DATABASE: if a future app-level
--  check is forgotten, the DB still denies cross-user access (fail-closed).
--  The service-role client (users/auth, products, orders+tickets tool path)
--  bypasses RLS and is unaffected.
--
--  IMPORTANT: users live in public.users, NOT auth.users, so policies use the
--  JWT `sub` claim directly - (auth.jwt() ->> 'sub')::uuid - NOT auth.uid()
--  (which targets auth.users and would never match => break all access).
-- =============================================================================

-- The `authenticated` role (from the JWT) needs table access; RLS then filters
-- rows. Idempotent; Supabase may already grant these.
grant usage on schema public to authenticated;
grant select, insert, update, delete on public.conversations to authenticated;
grant select, insert, update, delete on public.messages to authenticated;
grant select, insert, update, delete on public.analytics to authenticated;

alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.analytics enable row level security;

-- conversations: a user reads/writes only their own rows.
drop policy if exists conv_owner on public.conversations;
create policy conv_owner on public.conversations
  for all
  using (user_id = (auth.jwt() ->> 'sub')::uuid)
  with check (user_id = (auth.jwt() ->> 'sub')::uuid);

-- messages: no user_id column; ownership is via the parent conversation.
drop policy if exists msg_owner on public.messages;
create policy msg_owner on public.messages
  for all
  using (exists (
    select 1 from public.conversations c
    where c.id = messages.conversation_id
      and c.user_id = (auth.jwt() ->> 'sub')::uuid))
  with check (exists (
    select 1 from public.conversations c
    where c.id = messages.conversation_id
      and c.user_id = (auth.jwt() ->> 'sub')::uuid));

-- analytics: a user reads/updates only their own interactions.
drop policy if exists analytics_owner on public.analytics;
create policy analytics_owner on public.analytics
  for all
  using (user_id = (auth.jwt() ->> 'sub')::uuid)
  with check (user_id = (auth.jwt() ->> 'sub')::uuid);

-- Verify: with a user JWT these return only the caller's rows; the service-role
-- key still sees everything (it bypasses RLS by design).
-- select * from public.conversations;   -- as a user: only theirs
