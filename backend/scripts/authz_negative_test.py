"""
authz_negative_test.py - Increment 7 cross-user denial suite (BOLA/IDOR).

For every user-scoped resource, assert that user A (Alice) is DENIED user B's
(Bob's) object, AND that A's own access still works. An authorization fix without
a cross-user denial test is unverified.

Almost token-free: the /chat/stream ownership check fires BEFORE the LLM, so a
cross-user conversation_id is a 403 at zero cost. Orders are tested at the code
level (the live orders table has no owner column until the migration runs).

Usage:  python -m backend.scripts.authz_negative_test
Writes: results/authz_negative_test.json  (exit 1 if any test fails)
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import backend.services.database as db  # noqa: E402
from backend.core.request_context import current_user_id, current_user_products  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8000"
results = []


def check(name, ok, detail=""):
    results.append({"test": name, "passed": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def login(username, password):
    r = httpx.post(f"{BASE}/auth/login", data={"username": username, "password": password}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["access_token"], j["user"]["id"]


def H(tok):
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


async def order_tool(order_id):
    from backend.agent.tools import check_order_status
    return await check_order_status.ainvoke({"order_id": order_id})


def main() -> int:
    a_tok, a_id = login("alice", "password123")
    b_tok, b_id = login("bob", "password456")

    # ---- setup: a conversation for each user, analytics rows for each ----
    a_conv = httpx.post(f"{BASE}/chat/conversations", headers=H(a_tok), json={"title": "alice-authz"}, timeout=30).json()["id"]
    b_conv = httpx.post(f"{BASE}/chat/conversations", headers=H(b_tok), json={"title": "bob-authz"}, timeout=30).json()["id"]
    BOB_Q = "BOB-SECRET-QUERY-zzz"
    a_iid, b_iid = "authz-test-alice-iid", "authz-test-bob-iid"
    db.get_db().table("analytics").upsert({
        "id": b_iid, "user_id": b_id, "conversation_id": b_conv, "user_query": BOB_Q,
        "bot_response": "bob-only answer", "retrieved_docs": [], "feedback": 0}).execute()
    db.get_db().table("analytics").upsert({
        "id": a_iid, "user_id": a_id, "conversation_id": a_conv, "user_query": "alice-own-query",
        "bot_response": "alice answer", "retrieved_docs": [], "feedback": 0}).execute()

    print("MESSAGES / CONVERSATIONS (A7-MESSAGES):")
    r = httpx.get(f"{BASE}/chat/conversations/{b_conv}/messages", headers=H(a_tok), timeout=30)
    check("Alice DENIED Bob's conversation messages", r.status_code == 403, f"HTTP {r.status_code}")
    r = httpx.get(f"{BASE}/chat/conversations/{a_conv}/messages", headers=H(a_tok), timeout=30)
    check("Alice CAN read her own conversation messages", r.status_code == 200, f"HTTP {r.status_code}")
    # chat_stream with Bob's conversation_id -> 403 before any LLM work (0 tokens)
    r = httpx.post(f"{BASE}/chat/stream", headers=H(a_tok), json={"message": "hi", "conversation_id": b_conv}, timeout=30)
    check("Alice DENIED posting into Bob's conversation (chat/stream)", r.status_code == 403, f"HTTP {r.status_code}")

    print("ANALYTICS (A7-ANALYTICS, P1):")
    r = httpx.get(f"{BASE}/analytics/summary", headers=H(a_tok), timeout=30)
    body = r.text
    check("analytics/summary returns 200 for Alice", r.status_code == 200, f"HTTP {r.status_code}")
    check("analytics/summary does NOT contain Bob's query (no cross-user leak)", BOB_Q not in body,
          "leaked" if BOB_Q in body else "clean")

    print("FEEDBACK (A7-FEEDBACK):")
    r = httpx.post(f"{BASE}/feedback", headers=H(a_tok), json={"interaction_id": b_iid, "feedback": 1}, timeout=30)
    check("Alice DENIED feedback on Bob's interaction", r.status_code == 403, f"HTTP {r.status_code}")
    r = httpx.post(f"{BASE}/feedback", headers=H(a_tok), json={"interaction_id": a_iid, "feedback": 1}, timeout=30)
    check("Alice CAN submit feedback on her own interaction", r.status_code == 200, f"HTTP {r.status_code}")

    print("ORDERS (A7-ORDERS - code enforcement; live pending migration):")
    orders_live_scoped = False
    try:
        db.get_db().table("orders").select("user_id").limit(1).execute()
        orders_live_scoped = True
    except Exception:
        pass

    async def order_tests():
        current_user_id.set(a_id)
        current_user_products.set([])
        real = db.get_order_by_id
        # code path: Bob-owned order refused, Alice-owned shown
        db.get_order_by_id = lambda oid: {"order_id": oid, "user_id": b_id, "status": "Shipped",
                                          "items": ["SecureSphere 360 Camera"], "shipped_on": "2025-09-28"}
        ro = await order_tool("NX-2025-301")
        check("Alice DENIED Bob's order (code owner-check)", "No order found" in ro and "SecureSphere" not in ro)
        db.get_order_by_id = lambda oid: {"order_id": oid, "user_id": a_id, "status": "Delivered",
                                          "items": ["LumiGlow Smart Light"], "shipped_on": "2025-09-22"}
        ra = await order_tool("NX-2025-303")
        check("Alice CAN see her own order (code owner-check)", "Delivered" in ra)
        db.get_order_by_id = real
        # live path, only if the migration has been applied
        if orders_live_scoped:
            ro = await order_tool("NX-2025-301")
            check("LIVE: Alice DENIED Bob's real order NX-2025-301", "No order found" in ro and "SecureSphere" not in ro)
        else:
            check("LIVE orders scoping (migration applied)", False,
                  "SKIPPED - orders.user_id not present; run supabase/migrations/increment7_orders_owner.sql")
    asyncio.run(order_tests())

    passed = sum(1 for r in results if r["passed"])
    # The live-orders row is an expected SKIP until the migration runs; count it separately.
    skips = [r for r in results if not r["passed"] and "SKIPPED" in r["detail"]]
    hard_fail = [r for r in results if not r["passed"] and "SKIPPED" not in r["detail"]]
    out = {
        "test": "Increment 7 cross-user denial suite (BOLA/IDOR)",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "n": len(results), "passed": passed,
        "hard_failures": [r["test"] for r in hard_fail],
        "pending_migration_skips": [r["test"] for r in skips],
        "orders_live_scoped": orders_live_scoped,
        "results": results,
    }
    (REPO / "results/authz_negative_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{passed}/{len(results)} passed; hard failures: {len(hard_fail)}; pending-migration skips: {len(skips)}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
