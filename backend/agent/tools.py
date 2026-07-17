"""
tools.py — Agent tools backed by Supabase instead of mock dicts.

Each tool is a plain async Python function wrapped with @tool.
Tools are imported and registered in agent.py.
"""

from __future__ import annotations
import logging
import uuid
import json
from datetime import datetime, timedelta
from langchain.tools import tool

import backend.services.database as db
from backend.core.request_context import current_user_id, current_user_products

logger = logging.getLogger(__name__)


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

    # Increment 6 blast-radius: if the orders table carries an owner (user_id),
    # only reveal the order to that user, so a jailbroken persona cannot read a
    # stranger's order by guessing an ID. On the current demo schema orders have
    # NO owner column, so this is a no-op there (documented residual risk;
    # supabase/schema.sql adds the column, and a live migration is required
    # before public deploy). Never confirms another user's order even exists.
    owner = order.get("user_id")
    if owner and owner != current_user_id.get():
        logger.info("Blocked cross-user order access to %s", order_id)
        return (
            f"No order found for ID **{order_id}** on your account. "
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
    Pass the PRODUCT NAME of a product the user owns (e.g. "Nexora Thermostat
    Pro"); the system resolves their registered serial. You may also pass an
    explicit serial number if the user provides one.
    Returns warranty status, expiry date, and whether a claim is possible.
    """
    # Increment 6 blast-radius + prompt minimization: resolve the argument (a
    # product name OR a serial) against the AUTHENTICATED user's own registered
    # products, bound server-side in request context. A jailbroken model cannot
    # check a serial the user does not own, and serials no longer live in the
    # (leakable) system prompt. inj-02 / war-04 style cross-user serial probes
    # are refused here rather than answered.
    arg = (serial_number or "").strip()
    owned = current_user_products.get() or []
    match = None
    for p in owned:
        if p.get("serial_number", "").lower() == arg.lower():
            match = p
            break
    if match is None:
        for p in owned:
            pname = p.get("product_name", "").lower()
            if pname and arg and (arg.lower() in pname or pname in arg.lower()):
                match = p
                break
    if match is None:
        return (
            "I can only check warranty for a product registered to your account, "
            "and I do not see that serial or product on your profile. If you "
            "recently purchased it, please register it first, or share the serial "
            "printed on the device so support can verify it."
        )

    serial_number = match["serial_number"]
    product = db.get_product_by_serial(serial_number)

    if not product:
        return (
            f"No warranty record found for **{match.get('product_name', serial_number)}**. "
            "Please contact support so we can look into it."
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
async def create_support_ticket(conversation_summary: str) -> str:
    """
    Escalate an unresolved issue by creating a human-agent support ticket.
    Use this ONLY when:
      (a) the user explicitly asks to speak to a human, OR
      (b) lookup_documentation returned no useful answer.
    Pass a brief summary of the problem as the argument.
    """
    # The real authenticated user_id comes from request context (F1.5-3), never
    # from the model. The model does not know the user's UUID; a made-up value
    # would fail the users foreign key.
    user_id = current_user_id.get()
    if not user_id:
        logger.error("create_support_ticket called with no authenticated user in context")
        return (
            "I could not create a support ticket because your session could not "
            "be verified. Please sign in again or contact support directly."
        )

    ticket_id = f"TICKET-{uuid.uuid4().hex[:6].upper()}"

    try:
        db.create_ticket(
            user_id=user_id,
            summary=conversation_summary,
            ticket_id=ticket_id,
        )
    except Exception:
        # Do NOT claim success on a failed write (F1.5-3): that is a visible lie
        # and leaves nothing for a human to action. Report the failure honestly.
        logger.exception("create_ticket failed for user %s", user_id)
        return (
            "I was unable to create a support ticket right now. Please try again "
            "in a moment, or contact support directly if it keeps failing."
        )

    return (
        f"✅ Support ticket created. A human agent will follow up within 24 hours.\n\n"
        f"- **Ticket ID:** `{ticket_id}`\n\n"
        "Please keep your ticket ID for reference."
    )
