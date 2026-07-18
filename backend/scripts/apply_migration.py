"""
apply_migration.py - run a .sql migration against the Supabase Postgres.

The Supabase Python client speaks PostgREST, which cannot run DDL (ALTER TABLE),
so the orders-owner migration needs a DIRECT Postgres connection. This script
applies a .sql file when given the connection string, so the DDL step is one
command once the string is available - otherwise run the .sql in the Supabase
SQL editor by hand.

Connection string: Supabase Dashboard -> Project Settings -> Database ->
Connection string (URI). Provide it via the SUPABASE_DB_URL env var; it is a
secret and must never be committed.

Usage:
  SUPABASE_DB_URL='postgresql://...:5432/postgres' \
    python -m backend.scripts.apply_migration supabase/migrations/increment7_orders_owner.sql
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m backend.scripts.apply_migration <path-to.sql>")
        return 2
    sql_path = (REPO / sys.argv[1]) if not os.path.isabs(sys.argv[1]) else Path(sys.argv[1])
    url = os.environ.get("SUPABASE_DB_URL", "").strip()
    if not url:
        print("SUPABASE_DB_URL not set. Either export the direct Postgres URL, or")
        print(f"paste {sql_path} into the Supabase SQL editor and run it there.")
        return 3
    try:
        import psycopg2  # type: ignore
    except Exception:
        print("psycopg2 not installed. Run: pip install psycopg2-binary")
        return 4
    sql = sql_path.read_text(encoding="utf-8")
    conn = psycopg2.connect(url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
        print(f"applied: {sql_path}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
