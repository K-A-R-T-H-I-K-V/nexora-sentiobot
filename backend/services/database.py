"""
database.py — Supabase client wrapper.

Tables used (see supabase/schema.sql for full DDL):
  - users          : profile + owned_products (jsonb)
  - orders         : mock order data
  - products       : mock product / warranty data
  - conversations  : chat session metadata
  - messages       : individual messages per conversation
  - analytics      : query logs + feedback
  - support_tickets: escalated tickets
"""

from __future__ import annotations
from supabase import create_client, Client
from backend.core.config import get_settings

_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        s = get_settings()
        _client = create_client(s.supabase_url, s.supabase_service_role_key)
    return _client


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user_by_username(username: str) -> dict | None:
    result = get_db().table("users").select("*").eq("username", username.lower()).single().execute()
    return result.data


def get_user_by_id(user_id: str) -> dict | None:
    result = get_db().table("users").select("*").eq("id", user_id).single().execute()
    return result.data


# ---------------------------------------------------------------------------
# Products / Warranty
# ---------------------------------------------------------------------------

def get_product_by_serial(serial_number: str) -> dict | None:
    result = get_db().table("products").select("*").eq("serial_number", serial_number).single().execute()
    return result.data


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def get_order_by_id(order_id: str) -> dict | None:
    result = get_db().table("orders").select("*").eq("order_id", order_id).single().execute()
    return result.data


# ---------------------------------------------------------------------------
# Conversations & Messages
# ---------------------------------------------------------------------------

def create_conversation(user_id: str, title: str = "New Conversation") -> dict:
    result = get_db().table("conversations").insert({
        "user_id": user_id,
        "title": title,
    }).execute()
    return result.data[0]


def get_conversation(conversation_id: str) -> dict | None:
    """Fetch a single conversation row (incl. user_id) for ownership checks.

    The backend uses the Supabase SERVICE ROLE key, which bypasses RLS, so the
    caller MUST verify conversation["user_id"] against the authenticated user
    before returning or writing anything scoped to it (Increment 7, BOLA).
    """
    try:
        result = get_db().table("conversations").select("*").eq("id", conversation_id).single().execute()
        return result.data
    except Exception:
        return None


def get_conversations_for_user(user_id: str) -> list[dict]:
    result = (
        get_db().table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    return result.data


def save_message(conversation_id: str, role: str, content: str, metadata: dict | None = None) -> dict:
    result = get_db().table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }).execute()
    return result.data[0]


def get_messages_for_conversation(conversation_id: str) -> list[dict]:
    result = (
        get_db().table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )
    return result.data


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def log_analytics(entry: dict) -> None:
    get_db().table("analytics").insert(entry).execute()


def get_analytics_by_id(interaction_id: str) -> dict | None:
    """Fetch a single analytics row (incl. user_id) for ownership checks."""
    try:
        result = get_db().table("analytics").select("id,user_id").eq("id", interaction_id).single().execute()
        return result.data
    except Exception:
        return None


def get_analytics_for_user(user_id: str) -> list[dict]:
    """All analytics rows for ONE user (Increment 7: summary is scoped to the
    caller, never the whole table)."""
    result = get_db().table("analytics").select("*").eq("user_id", user_id).execute()
    return result.data


def update_analytics_feedback(interaction_id: str, feedback: int) -> None:
    get_db().table("analytics").update({"feedback": feedback}).eq("id", interaction_id).execute()


# ---------------------------------------------------------------------------
# Support Tickets
# ---------------------------------------------------------------------------

def create_ticket(user_id: str, summary: str, ticket_id: str) -> dict:
    result = get_db().table("support_tickets").insert({
        "user_id": user_id,
        "ticket_id": ticket_id,
        "summary": summary,
        "status": "open",
    }).execute()
    return result.data[0]


# ---------------------------------------------------------------------------
# Instrumentation: count Supabase round trips per request (measurement only;
# does not change behavior). Each public function does one .execute() round
# trip, so wrapping them counts round trips for the Increment 2 baseline.
# ---------------------------------------------------------------------------

from backend.core import metrics as _metrics  # noqa: E402

for _name in (
    "get_user_by_username", "get_user_by_id", "get_product_by_serial",
    "get_order_by_id", "create_conversation", "get_conversation",
    "get_conversations_for_user", "save_message", "get_messages_for_conversation",
    "log_analytics", "get_analytics_by_id", "get_analytics_for_user",
    "update_analytics_feedback", "create_ticket",
):
    globals()[_name] = _metrics.count_supabase(globals()[_name])
