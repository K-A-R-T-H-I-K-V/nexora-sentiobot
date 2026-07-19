"""
clarify.py - deterministic, zero-token clarify-before-answering (Feature F5).

When a slot-bearing intent (order_status needs an order id; warranty needs to know
WHICH product) is missing its detail, ask ONE targeted question instead of guessing.
But RESOLVE FIRST, proactively, from the message, then the user PROFILE, then the
conversation history, and only ask if it is still not knowable. Proactivity beats
interrogation: the difference between a form and a conversation.

Two design commitments (ratified):
- The DECISION to ask is DETERMINISTIC and zero-token, so the no-over-ask safety
  property is testable for certain (Convention 10). Only the wording is a short,
  context-aware TEMPLATE (it names the product/context; it is warmed if F4 flags
  frustration). The model never decides whether to ask.
- Only order_status and warranty can clarify. doc_lookup / chitchat / out_of_scope /
  ticket_or_escalation never do. Intent DISAMBIGUATION stays F1's job, not F5's.

Privacy: the resolver reads owned_products (with serials) SERVER-SIDE, but a clarify
question NEVER echoes a serial the user did not themselves provide (it names products
only). Order ids are the user's own and may be carried forward from their history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ORDER_ID = re.compile(r"\bNX-\d{4}-\d{3}\b", re.I)
_SERIAL = re.compile(r"\bSN-[A-Z0-9][A-Z0-9-]{3,}\b", re.I)

# A generic self-reference that does NOT name a specific product.
_GENERIC_REF = re.compile(
    r"\b(my (device|product|item|unit|gadget|thing|one|purchase|warranty|order)|"
    r"(is|are) (it|this|that|mine)|still (under|covered|valid|in warranty))\b", re.I)

# Concrete product-category nouns (used to tell "my camera" (a named thing the user may
# not own) from "my device" (truly generic)).
_PRODUCT_NOUN = re.compile(
    r"\b(thermostat|camera|cam|light|bulb|lamp|speaker|sensor|plug|lock|doorbell|bell|"
    r"switch|monitor|hub)\b", re.I)

# Marketing/filler tokens dropped when matching a product name to free text.
_STOP = {"nexora", "pro", "the", "smart", "plus", "max", "mini", "series", "gen", "and"}

# Signature of a clarify question we already asked (for the re-ask guard).
_CLARIFY_SIG = re.compile(
    r"which order number should i check|share the serial number|"
    r"which product should i check", re.I)

# Templated questions (zero-token; named/contextual, not robotic).
# F5-R1: the example uses a SYNTHETIC placeholder, not a real seeded order id
# (NX-2025-301 is a real order). It is also not `NX-\d{4}-\d{3}`-shaped, so it can never
# be mistaken for an id by any resolver, in addition to the user-turns-only scan.
_ORDER_ASK = ("Which order number should I check? You can find it on your confirmation "
              "email (for example, NX-XXXX-XXX).")
_WARRANTY_ASK_SERIAL = ("I do not see a product registered to your account yet. Could "
                        "you share the serial number printed on the device so I can "
                        "check its warranty?")
_WARM_LEAD = "Happy to sort this out. "


def _sig_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (name or "").lower())
            if t not in _STOP and len(t) > 2]


def _match_owned(text: str, owned: list[dict]) -> str | None:
    """The owned product whose significant name tokens appear in text, or None."""
    low = (text or "").lower()
    for p in owned:
        toks = _sig_tokens(p.get("product_name", ""))
        if toks and any(t in low for t in toks):
            return p.get("product_name")
    return None


def _history_user_text(history: list[dict]) -> str:
    return " ".join(m.get("content", "") for m in (history or []) if m.get("role") == "user")


def _last_assistant(history: list[dict]) -> str:
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


@dataclass
class ClarifyDecision:
    ask: bool = False
    question: str = ""
    hint: str = ""       # injected into the prompt when resolved from profile/history
    slot: str = ""       # "order_id" | "warranty_product" (observability)
    reason: str = ""     # why (observability): resolved_message / resolved_profile / ...


def _resolve_order(message: str, history: list[dict]) -> ClarifyDecision:
    if _ORDER_ID.search(message):
        return ClarifyDecision(ask=False, slot="order_id", reason="resolved_message")
    # Search USER turns only. The assistant's own clarify question contains an EXAMPLE
    # id ("NX-2025-301"); matching that would false-resolve to a stranger's order. An id
    # the bot legitimately knows always originated from a user turn anyway.
    m = _ORDER_ID.search(_history_user_text(history))
    if m:
        oid = m.group(0).upper()
        return ClarifyDecision(ask=False, slot="order_id", reason="resolved_history",
                               hint=f"The user is referring to order {oid} from earlier "
                                    f"in the conversation; call check_order_status with it.")
    return ClarifyDecision(ask=True, question=_ORDER_ASK, slot="order_id", reason="missing")


def _resolve_warranty(message: str, owned: list[dict], history: list[dict]) -> ClarifyDecision:
    owned = owned or []
    if _SERIAL.search(message):
        return ClarifyDecision(ask=False, slot="warranty_product", reason="resolved_message")

    named = _match_owned(message, owned)
    if named:
        return ClarifyDecision(ask=False, slot="warranty_product", reason="resolved_message",
                               hint=f"The user is asking about their {named}; call "
                                    f"check_warranty_status with that product name.")

    # A concrete but NON-owned product noun ("my camera" when they own a thermostat):
    # the slot is filled, let the tool answer "not registered" rather than over-ask.
    if _PRODUCT_NOUN.search(message) and not _GENERIC_REF.search(message):
        return ClarifyDecision(ask=False, slot="warranty_product", reason="named_not_owned")

    # Generic reference from here on. Try the profile, then history.
    if len(owned) == 1:
        name = owned[0].get("product_name")
        return ClarifyDecision(ask=False, slot="warranty_product", reason="resolved_profile",
                               hint=f"The user is asking about their {name}; call "
                                    f"check_warranty_status with that product name.")

    named_hist = _match_owned(_history_user_text(history), owned)
    if named_hist:
        return ClarifyDecision(ask=False, slot="warranty_product", reason="resolved_history",
                               hint=f"The user is asking about their {named_hist}; call "
                                    f"check_warranty_status with that product name.")

    if len(owned) == 0:
        return ClarifyDecision(ask=True, question=_WARRANTY_ASK_SERIAL,
                               slot="warranty_product", reason="missing_no_products")

    names = [p.get("product_name", "") for p in owned if p.get("product_name")]
    q = f"Which product should I check the warranty for: {', '.join(names)}?"
    return ClarifyDecision(ask=True, question=q, slot="warranty_product", reason="missing_multiple")


def decide(intent: str, message: str, profile: dict, history: list[dict],
           sentiment: dict | None, settings) -> ClarifyDecision:
    """Deterministic clarify decision. Zero-token. Returns ask=False (with an optional
    hint) when the slot resolves or the intent does not clarify; ask=True (with a
    templated question) only when a needed slot is genuinely unresolvable."""
    if not getattr(settings, "clarify_enabled", True):
        return ClarifyDecision(ask=False)
    if intent not in ("order_status", "warranty"):
        return ClarifyDecision(ask=False)

    # Yield to an ACTIVE F4 escalation: do not stack a form question on a live human
    # offer for a sustainedly-frustrated user (the escalation stands).
    if sentiment and sentiment.get("escalate"):
        return ClarifyDecision(ask=False, reason="defer_escalation")

    if intent == "order_status":
        d = _resolve_order(message, history)
    else:
        d = _resolve_warranty(message, (profile or {}).get("owned_products", []), history)

    if not d.ask:
        return d

    # Re-ask guard: if we already asked this exact clarify last turn and the slot is
    # STILL missing, do NOT loop - proceed and let the tool/model guide or offer a human.
    if _CLARIFY_SIG.search(_last_assistant(history)):
        return ClarifyDecision(ask=False, slot=d.slot, reason="reask_guard")

    # Warm the wording for a (mildly) frustrated user; escalating users already deferred.
    if sentiment and sentiment.get("label") in ("frustrated", "angry"):
        d.question = _WARM_LEAD + d.question
    return d
