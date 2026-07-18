"""Cross-user authorization (BOLA/IDOR) denial suite - unit level, FREE.

This is the CI gate the planner mandated: if a future change drops an ownership
check, one of these fails and the build goes red. It mocks the DB owner lookups
(no Supabase) and drives the enforcement code directly (no server, no LLM).

Convention (I7-1): cross-user access returns 404, not 403, so an id oracle is
not exposed; the assertions check for the raised HTTPException, not the code
specifically, except where the code is the point.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import backend.api.main as main
import backend.services.database as db
from backend.agent import tools
from backend.core.request_context import current_user_id, current_user_products

ALICE, BOB = "alice-id", "bob-id"


# --- conversations / messages -------------------------------------------------

def test_conversation_owner_allows_own(monkeypatch):
    monkeypatch.setattr(db, "get_conversation", lambda cid: {"id": cid, "user_id": ALICE})
    assert main.require_conversation_owner("c1", ALICE)["user_id"] == ALICE


def test_conversation_owner_denies_other_user(monkeypatch):
    monkeypatch.setattr(db, "get_conversation", lambda cid: {"id": cid, "user_id": BOB})
    with pytest.raises(HTTPException) as e:
        main.require_conversation_owner("c1", ALICE)
    assert e.value.status_code == 404  # I7-1: no existence oracle


def test_conversation_owner_denies_missing(monkeypatch):
    monkeypatch.setattr(db, "get_conversation", lambda cid: None)
    with pytest.raises(HTTPException):
        main.require_conversation_owner("nope", ALICE)


# --- feedback (analytics ownership) ------------------------------------------

def test_feedback_owner_allows_own(monkeypatch):
    monkeypatch.setattr(db, "get_analytics_by_id", lambda iid: {"id": iid, "user_id": ALICE})
    assert main.require_analytics_owner("i1", ALICE)["user_id"] == ALICE


def test_feedback_owner_denies_other_user(monkeypatch):
    monkeypatch.setattr(db, "get_analytics_by_id", lambda iid: {"id": iid, "user_id": BOB})
    with pytest.raises(HTTPException):
        main.require_analytics_owner("i1", ALICE)


# --- analytics summary is scoped to the caller, never the whole table ---------

def test_analytics_summary_scoped_and_no_raw_table(monkeypatch):
    captured = {}

    def fake_scoped(uid):
        captured["uid"] = uid
        return [{"user_query": "alice q", "feedback": 1}]

    monkeypatch.setattr(db, "get_analytics_for_user", fake_scoped)
    # If the endpoint ever reverts to db.get_db().table("analytics").select("*"),
    # this raises and the test fails.
    def boom(*a, **k):
        raise AssertionError("analytics_summary must NOT touch the raw table")
    monkeypatch.setattr(db, "get_db", boom)

    out = asyncio.run(main.analytics_summary(current_user={"id": ALICE}))
    assert captured["uid"] == ALICE
    assert out["total"] == 1


# --- orders tool: scoped to the caller ---------------------------------------

def test_order_tool_denies_other_users_order(monkeypatch):
    current_user_id.set(ALICE)
    monkeypatch.setattr(db, "get_order_by_id",
                        lambda oid: {"order_id": oid, "user_id": BOB, "status": "Shipped",
                                     "items": ["SecureSphere 360 Camera"], "shipped_on": "2025-09-28"})
    out = asyncio.run(tools.check_order_status.ainvoke({"order_id": "NX-2025-301"}))
    assert "No order found" in out and "SecureSphere" not in out


def test_order_tool_allows_own_order(monkeypatch):
    current_user_id.set(ALICE)
    monkeypatch.setattr(db, "get_order_by_id",
                        lambda oid: {"order_id": oid, "user_id": ALICE, "status": "Delivered",
                                     "items": ["LumiGlow Smart Light"], "shipped_on": "2025-09-22"})
    out = asyncio.run(tools.check_order_status.ainvoke({"order_id": "NX-2025-303"}))
    assert "Delivered" in out


# --- warranty tool: scoped to the caller's OWN products ----------------------

def test_warranty_tool_denies_non_owned_serial():
    current_user_id.set(ALICE)
    current_user_products.set([{"product_name": "Nexora Thermostat Pro", "serial_number": "SN-NTS-PRO-ABC123"}])
    out = asyncio.run(tools.check_warranty_status.ainvoke({"serial_number": "SN-NCS360-CAM789"}))  # Bob's
    assert "registered to your account" in out
    assert "Warranty Expires" not in out


def test_warranty_tool_allows_owned_product(monkeypatch):
    current_user_id.set(ALICE)
    current_user_products.set([{"product_name": "Nexora Thermostat Pro", "serial_number": "SN-NTS-PRO-ABC123"}])
    monkeypatch.setattr(db, "get_product_by_serial",
                        lambda sn: {"serial_number": sn, "product_name": "Nexora Thermostat Pro",
                                    "purchase_date": "2024-11-01", "warranty_months": 24})
    out = asyncio.run(tools.check_warranty_status.ainvoke({"serial_number": "Nexora Thermostat Pro"}))
    assert "Warranty Status" in out and "SN-NTS-PRO-ABC123" in out
