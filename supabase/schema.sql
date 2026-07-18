-- =============================================================================
--  SentioBot — Supabase Schema
--  Run this in your Supabase SQL editor (Project → SQL Editor → New Query)
-- =============================================================================

-- Enable pgvector extension (optional — for future migration from ChromaDB)
-- create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- users
-- ---------------------------------------------------------------------------
create table if not exists public.users (
  id              uuid primary key default gen_random_uuid(),
  username        text unique not null,
  name            text not null,
  password_hash   text not null,
  owned_products  jsonb default '[]'::jsonb,
  created_at      timestamptz default now()
);

-- Row-level security (users can only see their own row)
alter table public.users enable row level security;
create policy "users_select_own" on public.users
  for select using (auth.uid() = id);

-- ---------------------------------------------------------------------------
-- products (warranty data)
-- ---------------------------------------------------------------------------
create table if not exists public.products (
  id                uuid primary key default gen_random_uuid(),
  serial_number     text unique not null,
  product_name      text not null,
  purchase_date     date not null,
  warranty_months   int not null default 12,
  created_at        timestamptz default now()
);

-- Seed data matching mock_db.py
insert into public.products (serial_number, product_name, purchase_date, warranty_months) values
  ('SN-NTS-PRO-XYZ987', 'Nexora Thermostat Pro', '2023-08-15', 24),
  ('SN-NTS-PRO-ABC123', 'Nexora Thermostat Pro', '2024-11-01', 24),
  ('SN-NLRGB-LMO456',  'LumiGlow Smart Light',  '2025-05-20', 12),
  ('SN-NCS360-CAM789', 'SecureSphere 360 Camera','2024-02-10', 12)
on conflict do nothing;

-- ---------------------------------------------------------------------------
-- orders
-- ---------------------------------------------------------------------------
create table if not exists public.orders (
  id          uuid primary key default gen_random_uuid(),
  order_id    text unique not null,
  status      text not null,
  shipped_on  date,
  items       jsonb default '[]'::jsonb,
  created_at  timestamptz default now()
);

insert into public.orders (order_id, status, shipped_on, items) values
  ('NX-2025-301', 'Shipped',    '2025-09-28', '["SecureSphere 360 Camera","LumiGlow Smart Light"]'),
  ('NX-2025-302', 'Processing', null,          '["Nexora Thermostat Pro"]'),
  ('NX-2025-303', 'Delivered',  '2025-09-22', '["LumiGlow Smart Light"]')
on conflict do nothing;

-- ---------------------------------------------------------------------------
-- conversations
-- ---------------------------------------------------------------------------
create table if not exists public.conversations (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references public.users(id) on delete cascade,
  title       text default 'New Conversation',
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

create index if not exists conversations_user_id_idx on public.conversations(user_id);

-- ---------------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------------
create table if not exists public.messages (
  id                uuid primary key default gen_random_uuid(),
  conversation_id   uuid references public.conversations(id) on delete cascade,
  role              text check (role in ('user', 'assistant', 'tool')) not null,
  content           text not null,
  metadata          jsonb default '{}'::jsonb,
  created_at        timestamptz default now()
);

create index if not exists messages_conv_id_idx on public.messages(conversation_id);

-- ---------------------------------------------------------------------------
-- analytics
-- ---------------------------------------------------------------------------
create table if not exists public.analytics (
  id                text primary key,  -- interaction_id from backend
  user_id           uuid references public.users(id),
  conversation_id   uuid references public.conversations(id),
  user_query        text,
  bot_response      text,
  retrieved_docs    jsonb default '[]'::jsonb,
  feedback          int default 0 check (feedback in (-1, 0, 1)),
  timestamp         timestamptz default now()
);

-- ---------------------------------------------------------------------------
-- support_tickets
-- ---------------------------------------------------------------------------
create table if not exists public.support_tickets (
  id          uuid primary key default gen_random_uuid(),
  ticket_id   text unique not null,
  user_id     uuid references public.users(id),
  summary     text,
  status      text default 'open' check (status in ('open', 'in_progress', 'closed')),
  created_at  timestamptz default now()
);

-- ---------------------------------------------------------------------------
-- Seed users  (passwords are bcrypt hashes of the originals)
-- bcrypt("password123") and bcrypt("password456")
-- Generate fresh hashes: python -c "from passlib.context import CryptContext; c=CryptContext(schemes=['bcrypt']); print(c.hash('password123'))"
-- ---------------------------------------------------------------------------
insert into public.users (username, name, password_hash, owned_products) values
  ('alice', 'Alice',
   '$2b$12$HhtTGc49f8r67SpJ3r0ht.0JFt0VLAHiY3wuDL6qLMgEL3cmGUoeu',
   '[{"product_name":"Nexora Thermostat Pro","serial_number":"SN-NTS-PRO-ABC123"},{"product_name":"LumiGlow Smart Light","serial_number":"SN-NLRGB-LMO456"}]'),
  ('bob', 'Bob',
   '$2b$12$VEfo9es6mrkSBavu05uWweVc3PQIRjEpN98n9.XP1jO/tY2t4yrqy',
   '[{"product_name":"SecureSphere 360 Camera","serial_number":"SN-NCS360-CAM789"}]'),
  ('guest', 'Guest',
   '$2b$12$Qnm4VlXdXalNruxydGFbzOkYRjf6aFGLUoyX5dVrZKApG0RVsJNs2',
   '[]')
on conflict do nothing;

-- ---------------------------------------------------------------------------
-- Increment 6 (blast-radius containment): give orders an owner so
-- check_order_status can be scoped to the authenticated user. Without this,
-- any logged-in user can read any order by guessing its ID. Runs after users
-- exist (subqueries below resolve usernames -> ids). EXISTING deployments must
-- run this migration before public deploy; the tool treats a missing user_id
-- as "unscoped" and falls back to bearer-token-by-ID behaviour until then.
-- ---------------------------------------------------------------------------
alter table public.orders add column if not exists user_id uuid references public.users(id);
create index if not exists orders_user_id_idx on public.orders(user_id);

update public.orders set user_id = (select id from public.users where username = 'bob')
  where order_id = 'NX-2025-301';        -- SecureSphere 360 Camera -> Bob
update public.orders set user_id = (select id from public.users where username = 'alice')
  where order_id in ('NX-2025-302', 'NX-2025-303');  -- Thermostat, LumiGlow -> Alice
