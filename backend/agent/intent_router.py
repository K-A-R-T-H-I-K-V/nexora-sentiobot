"""
intent_router.py - local, zero-token intent classification for request routing.

Feature F1. The day-one router in stream_agent_response was a bare keyword match:

    any(kw in message.lower() for kw in
        ["order", "warranty", "serial", "ticket", "human", "support"])

It is brittle in BOTH directions:
  - It OVER-triggers: "what does the warranty policy cover for water damage" is a
    general policy/documentation question, but the bare word "warranty" sends it
    down the heavier agent tool path (slower, more tokens).
  - It UNDER-triggers: "please escalate this to a person" needs a ticket, but it
    contains none of the trigger words, so it wrongly takes the RAG path and
    never opens one.

This replaces it with an embedding intent classifier. We embed a small set of
labeled prototype phrases per intent ONCE (lazily, on first use, then cached)
using the SAME local ONNX MiniLM the retriever and cache already load, so there
is no extra model, no LLM call, and ~0 tokens on the hot path. At query time we
embed the message (one local embed) and cosine-match it to the nearest prototype;
the winning intent maps to a route.

Intent taxonomy (ratified kickoff): doc_lookup, order_status, warranty,
ticket_or_escalation, chitchat, out_of_scope. NOTE the load-bearing distinction
that makes the feature work: "warranty" here means warranty STATUS for the user's
OWN product or serial (needs the check_warranty_status tool); warranty POLICY /
coverage / period questions are doc_lookup and answer from the docs on the RAG
path. Separating those two is the whole point of F1.

Route map:
  doc_lookup, out_of_scope                         -> "rag"   (retrieve + answer)
  order_status, warranty, ticket_or_escalation,
  chitchat                                         -> "tool"  (LangGraph agent)

Low-confidence fallback: below intent_confidence_threshold the classifier defers
to the legacy keyword router (a safe, known-behaviour default) instead of
guessing on a weak match. Both routers are selectable via the ROUTER config flag
so the change is reversible and A/B-able.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.core.config import get_settings
from backend.core.onnx_embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Legacy keyword-router signals (unchanged from the original inline check). Kept
# for the ROUTER=keyword path and as the low-confidence fallback.
_KEYWORD_TOOL_SIGNALS = ["order", "warranty", "serial", "ticket", "human", "support"]

# Which route each intent takes. doc_lookup / out_of_scope -> RAG; the rest -> tool.
INTENT_ROUTE: dict[str, str] = {
    "doc_lookup": "rag",
    "out_of_scope": "rag",
    "order_status": "tool",
    "warranty": "tool",
    "ticket_or_escalation": "tool",
    "chitchat": "tool",
}

# Labeled prototype phrases per intent, exemplars written from domain knowledge.
# They are kept DISJOINT from the labeled routing eval set
# (results/routing_set_v1.json): routing_eval.py enforces that no eval question is
# a normalized string copy of a prototype and that every eval question stays below
# 0.90 cosine to its nearest prototype, so the eval measures GENERALIZATION, not
# memorization (F1-R1). That check fails the eval and CI if it ever regresses.
# The doc_lookup set intentionally carries warranty/return POLICY phrasings so a
# "what does the warranty cover" question lands on doc_lookup, not the
# warranty-status tool intent (the load-bearing split for this feature).
INTENT_PROTOTYPES: dict[str, list[str]] = {
    "doc_lookup": [
        "how do I set up the smart bulb",
        "how do I factory reset my device",
        "what wifi band does the camera need for setup",
        "my thermostat screen is blank, how do I fix it",
        "troubleshooting steps for a flickering light",
        "what does privacy mode do on the camera",
        "what video resolution does the camera record at",
        # warranty / return POLICY questions are documentation, not status checks:
        "what does the warranty cover",
        "is water damage covered under the warranty policy",
        "how long is the warranty period for the thermostat",
        "what is the return and refund policy",
        "how long do refunds take to process",
        "what personal data does the company collect about me",
        "what are the customer support hours",
    ],
    "order_status": [
        "what is the status of my order",
        "where is my order and has it shipped",
        "has my order been delivered yet",
        "track my package for order number NX-2025-301",
        "what items are in my recent order",
    ],
    "warranty": [
        "is my thermostat still under warranty",
        "is my smart light still covered under warranty",
        "check the warranty status for my product's serial number",
        "what is the current warranty status of my device",
        "is the product with this serial number under warranty",
    ],
    "ticket_or_escalation": [
        "I want to speak to a human agent",
        "please escalate this to a person",
        "open a support ticket for my broken camera",
        "connect me to a real support representative",
        "raise a ticket for someone to follow up with me",
    ],
    "chitchat": [
        "hello there",
        "hi, how are you today",
        "thanks so much for your help",
        "good morning",
        "what can you help me with",
    ],
    "out_of_scope": [
        "what is the best refrigerator brand to buy this year",
        "can you recommend a good samsung television",
        "what medication should I take for a headache",
        "write me a poem about the ocean",
        "who won the football game last night",
    ],
}


@dataclass
class RouteDecision:
    route: str                                    # "rag" | "tool"
    intent: str                                   # intent, or "keyword"/fallback label
    confidence: float                             # top cosine (0.0 for keyword router)
    router: str                                   # "embedding" | "keyword"
    scores: dict[str, float] = field(default_factory=dict)  # per-intent top cosine (debug)


def keyword_route(message: str) -> str:
    """The legacy day-one router: 'tool' if any keyword signal is present, else 'rag'."""
    low = message.lower()
    return "tool" if any(kw in low for kw in _KEYWORD_TOOL_SIGNALS) else "rag"


class _IntentClassifier:
    """Embeds the prototype phrases once, then cosine-matches messages to intents."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None    # (num_prototypes, dim), L2-normalized
        self._proto_intent: list[str] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        emb = get_embeddings()
        phrases: list[str] = []
        proto_intent: list[str] = []
        for intent, protos in INTENT_PROTOTYPES.items():
            for p in protos:
                phrases.append(p)
                proto_intent.append(intent)
        vecs = np.array(emb.embed_documents(phrases), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._matrix = np.where(norms > 0, vecs / norms, 0.0)
        self._proto_intent = proto_intent
        self._built = True
        logger.info(
            "Intent classifier ready: %d prototypes across %d intents",
            len(phrases), len(INTENT_PROTOTYPES),
        )

    def scores(self, message: str, embedding=None) -> dict[str, float]:
        """Top (max) cosine of the message against each intent's prototypes. Accepts
        an optional precomputed query embedding so the caller can embed the message
        ONCE and share it (F4 shares this with the sentiment pass)."""
        self.build()
        q = np.array(embedding if embedding is not None
                     else get_embeddings().embed_query(message), dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        sims = self._matrix @ qn                              # (num_prototypes,)
        best: dict[str, float] = {}
        for intent, s in zip(self._proto_intent, sims):
            fs = float(s)
            if intent not in best or fs > best[intent]:
                best[intent] = fs
        return best

    def classify(self, message: str, embedding=None) -> tuple[str, float, dict[str, float]]:
        s = self.scores(message, embedding)
        intent = max(s, key=s.get)
        return intent, s[intent], s


_classifier: _IntentClassifier | None = None


def _get_classifier() -> _IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = _IntentClassifier()
    return _classifier


def warm() -> None:
    """Build the prototype matrix ahead of the first request (optional). The
    classifier is otherwise built lazily on first route_message, matching the
    app's other lazy singletons (retriever, LLM, cache embedder)."""
    _get_classifier().build()


def classify_intent(message: str, embedding=None) -> tuple[str, float, dict[str, float]]:
    """Raw embedding classification, ignoring the confidence fallback. Returns
    (intent, confidence, per_intent_scores). Used by the routing eval."""
    return _get_classifier().classify(message, embedding)


def route_message(message: str, router: str | None = None, embedding=None) -> RouteDecision:
    """Top-level routing decision. router defaults to the ROUTER config flag
    ('embedding' or 'keyword'). Makes no LLM call and spends no tokens. An optional
    precomputed embedding is reused (shared with the F4 sentiment pass) to embed once."""
    settings = get_settings()
    router = router or settings.router

    if router == "keyword":
        return RouteDecision(
            route=keyword_route(message), intent="keyword", confidence=0.0,
            router="keyword",
        )

    intent, conf, scores = _get_classifier().classify(message, embedding)
    if conf < settings.intent_confidence_threshold:
        # Weak match: defer to the legacy keyword router rather than guess.
        return RouteDecision(
            route=keyword_route(message), intent="low_confidence_fallback",
            confidence=conf, router="embedding", scores=scores,
        )
    return RouteDecision(
        route=INTENT_ROUTE[intent], intent=intent, confidence=conf,
        router="embedding", scores=scores,
    )
