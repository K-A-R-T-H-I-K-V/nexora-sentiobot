"""Per-request context shared across the agent graph and its tools.

Carries the authenticated user_id so server-side tools (create_support_ticket)
use the REAL user from the request, never an LLM-supplied value. Lives in its
own module so agent.py and tools.py can both import it without a cycle.
"""

from contextvars import ContextVar

# Empty string means "no authenticated user in context".
current_user_id: ContextVar[str] = ContextVar("sentiobot_current_user_id", default="")
