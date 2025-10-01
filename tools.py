# tools.py - UPGRADED WITH INFORMATIVE USER OUTPUT

import uuid
import json
from datetime import datetime, timedelta
from langchain.tools import tool
from mock_db import ORDERS, PRODUCTS

# The log_action helper function remains the same
def log_action(tool_name: str, input_data: dict, result: str):
    """A helper function to print structured logs."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "="*60)
    print(f"[{timestamp}] Executing Tool: {tool_name}")
    print("-"*60)
    print("INPUT:")
    print(json.dumps(input_data, indent=2))
    print("\nRESULT:")
    print(result)
    print("="*60 + "\n")


@tool
def check_order_status(order_id: str) -> str:
    """
    Use this tool to check the status of a specific customer order by its ID.
    It takes an order_id string and returns the order's current status.
    """
    order = ORDERS.get(order_id)
    log_action(
        tool_name="check_order_status",
        input_data={"order_id": order_id},
        result=f"Found Order Details: {json.dumps(order, indent=2) if order else 'None'}"
    )
    
    if not order:
        return f"I couldn't find any order with the ID '{order_id}'. Please double-check the order number."
    else:
        # NEW INFORMATIVE OUTPUT for the user
        shipped_date = f"Shipped on: {order['shipped_on']}" if order['shipped_on'] else "Awaiting shipment."
        items_list = "\n".join([f"- {item}" for item in order['items']])
        return (
            f"I've found the details for order **{order_id}**:\n\n"
            f"**Status:** {order['status']}\n"
            f"**Items:**\n{items_list}\n"
            f"*{shipped_date}*"
        )

@tool
def check_warranty_status(serial_number: str) -> str:
    """
    Use this tool to check the warranty status of a Nexora product using its serial number.
    It takes a serial_number string and returns if the product is under warranty.
    """
    product = PRODUCTS.get(serial_number)
    
    if not product:
        log_action("check_warranty_status", {"serial_number": serial_number}, "Product not found in database.")
        return f"I couldn't find a product with the serial number '{serial_number}'. Please verify the number on your device."

    warranty_end_date = product["purchase_date"] + timedelta(days=30 * product["warranty_months"])
    is_active = datetime.now() < warranty_end_date
    status_text = "✅ Active" if is_active else "❌ Expired"

    log_details = {
        "product_details": {k: (v.strftime('%Y-%m-%d') if isinstance(v, datetime) else v) for k, v in product.items()},
        "warranty_expires_on": warranty_end_date.strftime('%Y-%m-%d'),
        "is_active": is_active
    }
    log_action(
        tool_name="check_warranty_status",
        input_data={"serial_number": serial_number},
        result=json.dumps(log_details, indent=2)
    )

    # NEW INFORMATIVE OUTPUT for the user
    return (
        f"Here is the warranty status for serial number **{serial_number}**:\n\n"
        f"- **Product:** {product['product_name']}\n"
        f"- **Purchase Date:** {product['purchase_date'].strftime('%Y-%m-%d')}\n"
        f"- **Warranty Expires:** {warranty_end_date.strftime('%Y-%m-%d')}\n"
        f"- **Status:** {status_text}"
    )

@tool
def lookup_documentation(query: str) -> str:
    """
    Use this tool FIRST to answer general questions about Nexora products,
    troubleshooting, and company policies (like returns or defects).
    This should be your default tool unless the user asks for a specific
    order status, warranty status, or explicitly wants to talk to a human.
    """
    # NOTE: The implementation of this tool is now in app.py
    # This is just a placeholder for the agent's reference.
    pass

@tool
def create_support_ticket(conversation_summary: str) -> str:
    """
    Use this tool ONLY as a last resort if the user wants to speak to a human
    OR if you have already tried the `lookup_documentation` tool and could not
    find a satisfactory answer. Do not use this tool for simple questions.
    """
    ticket_id = f"TICKET-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    ticket_details = {"ticket_id": ticket_id, "timestamp": timestamp, "summary": conversation_summary}
    with open("support_tickets.log", "a", encoding="utf-8") as f:
        f.write(f"--- TICKET CREATED ---\n{json.dumps(ticket_details, indent=2)}\n\n")
    
    log_action(
        tool_name="create_support_ticket",
        input_data={"conversation_summary": conversation_summary},
        result=f"Successfully created ticket. Details logged. Ticket ID: {ticket_id}"
    )

    # NEW INFORMATIVE OUTPUT for the user
    return (
        f"I've created a support ticket for you. A human support agent will be in touch shortly.\n\n"
        f"**Your Reference Details:**\n"
        f"- **Ticket ID:** `{ticket_id}`\n"
        f"- **Time Logged:** {timestamp}"
    )
