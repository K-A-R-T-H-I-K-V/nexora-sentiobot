"""
fail_closed_proof.py - Increment 10: prove RLS enforces ownership at the DATABASE,
independent of any application-level check.

This removes the app-check from the equation entirely: it calls the RAW DB
functions with a given user's JWT in context (no endpoint, no
require_conversation_owner) and asserts the database itself returns only that
user's rows. With RLS on + the user-JWT client, Alice reading Bob's
conversation/messages returns NOTHING - the DB denied it, not the app. Under the
old service-role model this same call would leak Bob's data; the empty result is
the proof that RLS is now the load-bearing control.

Requires RLS mode: SUPABASE_JWT_SECRET + SUPABASE_ANON_KEY set AND
supabase/migrations/increment10_rls.sql applied.

Usage:  python -m backend.scripts.fail_closed_proof
Writes: results/increment10_fail_closed.json   (exit 1 on any hard failure)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.core.config import get_settings  # noqa: E402
from backend.core import auth, metrics  # noqa: E402
from backend.core.request_context import current_access_token  # noqa: E402
import backend.services.database as db  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    metrics.start()
    s = get_settings()
    if not s.rls_enabled:
        print("SKIP: RLS not enabled. Set SUPABASE_JWT_SECRET + SUPABASE_ANON_KEY and")
        print("apply supabase/migrations/increment10_rls.sql, then re-run.")
        return 2

    alice = db.get_user_by_username("alice")
    bob = db.get_user_by_username("bob")
    a_tok = auth.create_access_token({"sub": alice["id"]})
    b_tok = auth.create_access_token({"sub": bob["id"]})

    # Setup via SERVICE-ROLE (bypasses RLS): a Bob-owned conversation + message.
    conv = db.get_db().table("conversations").insert(
        {"user_id": bob["id"], "title": "fail-closed-proof"}).execute().data[0]
    db.get_db().table("messages").insert(
        {"conversation_id": conv["id"], "role": "user",
         "content": "BOB-PRIVATE-MSG", "metadata": {}}).execute()

    rows = []
    def check(name, ok):
        rows.append({"test": name, "passed": bool(ok)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    try:
        # FAIL-CLOSED: Alice's JWT, RAW db functions, NO app-level check.
        current_access_token.set(a_tok)
        alice_msgs = db.get_messages_for_conversation(conv["id"])
        check("Alice (no app-check) gets 0 of Bob's messages - DB denied", len(alice_msgs) == 0)
        check("Alice (no app-check) cannot read Bob's conversation row - DB denied",
              db.get_conversation(conv["id"]) is None)

        # POSITIVE control: Bob's JWT sees his own row + message.
        current_access_token.set(b_tok)
        bob_msgs = db.get_messages_for_conversation(conv["id"])
        check("Bob sees his own message (RLS allows the owner)",
              any("BOB-PRIVATE-MSG" in (m.get("content") or "") for m in bob_msgs))
        check("Bob can read his own conversation row",
              (db.get_conversation(conv["id"]) or {}).get("id") == conv["id"])
    finally:
        # cleanup via service-role
        current_access_token.set("")
        db.get_db().table("messages").delete().eq("conversation_id", conv["id"]).execute()
        db.get_db().table("conversations").delete().eq("id", conv["id"]).execute()

    passed = sum(1 for r in rows if r["passed"])
    out = {
        "test": "Increment 10 fail-closed proof (RLS denies with no app-level check)",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "rls_enabled": True, "n": len(rows), "passed": passed,
        "hard_failures": [r["test"] for r in rows if not r["passed"]],
        "results": rows,
    }
    (REPO / "results/increment10_fail_closed.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{passed}/{len(rows)} passed")
    return 1 if passed != len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
