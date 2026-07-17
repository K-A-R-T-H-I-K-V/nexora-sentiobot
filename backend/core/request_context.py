"""Per-request context shared across the agent graph and its tools.

Carries the authenticated user_id so server-side tools (create_support_ticket)
use the REAL user from the request, never an LLM-supplied value. Lives in its
own module so agent.py and tools.py can both import it without a cycle.
"""

from contextvars import ContextVar

# Empty string means "no authenticated user in context".
current_user_id: ContextVar[str] = ContextVar("sentiobot_current_user_id", default="")

# The authenticated user's registered products: list of
# {"product_name": str, "serial_number": str}. Set per request from the verified
# profile (never from the LLM). check_warranty_status resolves and scopes against
# this, so a jailbroken model cannot check a serial the user does not own, and
# serial numbers no longer need to sit in the (leakable) system prompt.
current_user_products: ContextVar[list] = ContextVar(
    "sentiobot_current_user_products", default=[]
)
