-- =============================================================================
--  Increment 7 migration - give orders an owner (BOLA/IDOR fix A7-ORDERS)
--  Run this ONCE in the Supabase SQL editor (Project -> SQL Editor -> New query)
--  BEFORE public deploy. It is idempotent (safe to re-run).
--
--  Why: the backend connects with the service-role key, which bypasses RLS, so
--  check_order_status enforces ownership in application code. That check is inert
--  until orders carries a user_id. After this runs, Alice is refused Bob's
--  NX-2025-301 and each user sees only their own orders.
-- =============================================================================

alter table public.orders add column if not exists user_id uuid references public.users(id);
create index if not exists orders_user_id_idx on public.orders(user_id);

-- Owner associations (derived from the seed: order 301 = camera -> Bob;
-- 302 = thermostat, 303 = light -> Alice).
update public.orders set user_id = (select id from public.users where username = 'bob')
  where order_id = 'NX-2025-301';
update public.orders set user_id = (select id from public.users where username = 'alice')
  where order_id in ('NX-2025-302', 'NX-2025-303');

-- Verify (should return each order with a non-null owner):
-- select order_id, user_id from public.orders order by order_id;
