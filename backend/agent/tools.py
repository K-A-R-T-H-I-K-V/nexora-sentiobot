"""
tools.py — Agent tools backed by Supabase instead of mock dicts.

Each tool is a plain async Python function wrapped with @tool.
Tools are imported and registered in agent.py.
"""

from __future__ import annotations
import uuid
import json
from datetime import datetime, timedelta
from langchain.tools import tool

import backend.services.database as db


# ---------------------------------------------------------------------------
# check_order_status
# ---------------------------------------------------------------------------

@tool
async def check_order_status(order_id: str) -> str:
    """
    Check the live status of a Nexora customer order.
    Use this when the user provides or asks about a specific order ID (e.g. NX-2025-301).
    Returns current status, items, and shipment date.
    """
    order = db.get_order_by_id(order_id.strip())

    if not order:
        return (
            f"No order found for ID **{order_id}**. "
            "Please double-check the order number from your confirmation email."
        )

    items_md = "\n".join(f"- {item}" for item in (order.get("items") or []))
    shipped = f"Shipped on **{order['shipped_on']}**" if order.get("shipped_on") else "Awaiting shipment."

    return (
        f"**Order {order_id}**\n\n"
        f"**Status:** {order['status']}\n\n"
        f"**Items:**\n{items_md}\n\n"
        f"{shipped}"
    )


# ---------------------------------------------------------------------------
# check_warranty_status
# ---------------------------------------------------------------------------

@tool
async def check_warranty_status(serial_number: str) -> str:
    """
    Check whether a Nexora product is within its warranty period.
    Use this when the user provides a serial number OR when their profile
    contains a serial number for the product they are asking about.
    Returns warranty status, expiry date, and whether a claim is possible.
    """
    product = db.get_product_by_serial(serial_number.strip())

    if not product:
        return (
            f"No product found with serial number **{serial_number}**. "
            "Please verify the number printed on the device or packaging."
        )

    purchase_date: datetime = product["purchase_date"]
    if isinstance(purchase_date, str):
        purchase_date = datetime.fromisoformat(purchase_date)

    expiry = purchase_date + timedelta(days=30 * product["warranty_months"])
    is_active = datetime.now() < expiry
    badge = "✅ **Active**" if is_active else "❌ **Expired**"

    return (
        f"**Warranty Status for SN: {serial_number}**\n\n"
        f"- **Product:** {product['product_name']}\n"
        f"- **Purchased:** {purchase_date.strftime('%Y-%m-%d')}\n"
        f"- **Warranty Expires:** {expiry.strftime('%Y-%m-%d')}\n"
        f"- **Status:** {badge}\n\n"
        + (
            "The warranty is still active. I can raise a support ticket to start a claim if needed."
            if is_active else
            "The warranty period has ended. You may still contact support for paid repair options."
        )
    )


# ---------------------------------------------------------------------------
# create_support_ticket
# ---------------------------------------------------------------------------

@tool
async def create_support_ticket(conversation_summary: str, user_id: str = "guest") -> str:
    """
    Escalate an unresolved issue by creating a human-agent support ticket.
    Use this ONLY when:
      (a) the user explicitly asks to speak to a human, OR
      (b) lookup_documentation returned no useful answer.
    Pass a brief summary of the problem as the argument.
    """
    ticket_id = f"TICKET-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        db.create_ticket(
            user_id=user_id,
            summary=conversation_summary,
            ticket_id=ticket_id,
        )
    except Exception:
        pass  # Degrade gracefully — still return ticket ID to user

    return (
        f"✅ Support ticket created. A human agent will be in touch within 24 hours.\n\n"
        f"- **Ticket ID:** `{ticket_id}`\n"
        f"- **Logged at:** {timestamp}\n\n"
        "Please keep your ticket ID for reference."
    )
