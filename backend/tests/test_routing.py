"""Feature F1 routing gate (free, deterministic, 0 LLM tokens).

Turns the routing A/B into a CI tripwire: the embedding intent classifier must
keep beating the legacy keyword router on the labeled routing set, the specific
keyword-misroute cases the feature exists to fix must stay fixed, and routing
must never touch a provider (no LLM call). A regression in prototypes, the
threshold, or the route map fails the build.

Loads the local ONNX MiniLM model once (a free, public download, same as the
retrieval eval); no Supabase, no secrets, no paid tokens.
"""
from __future__ import annotations

import pytest

from backend.scripts.routing_eval import evaluate
from backend.agent.intent_router import route_message


@pytest.fixture(scope="module")
def report():
    return evaluate()


def test_embedding_beats_keyword(report):
    assert report["embedding_accuracy"] >= report["keyword_accuracy"], (
        f"embedding {report['embedding_accuracy']} did not beat/match "
        f"keyword {report['keyword_accuracy']}"
    )
    # It should genuinely BEAT it, not merely tie, on this set.
    assert report["embedding_accuracy"] > report["keyword_accuracy"]


def test_embedding_accuracy_floor(report):
    # Well below the observed 0.929, leaving headroom, but catches a real
    # regression (a broken prototype set or route map).
    assert report["embedding_accuracy"] >= 0.85, report["embedding_accuracy"]


def test_gate_items_route_correctly(report):
    # The unambiguous over/under-trigger fixes (the feature's core promise) must
    # route to the ideal path under the embedding router.
    assert report["gate_failures"] == [], report["gate_failures"]


def test_core_over_trigger_misroutes_fixed(report):
    # The exact day-one flaw: policy/coverage questions containing a tool keyword
    # must go to the RAG path, not the tool path.
    by_id = {r["id"]: r for r in report["rows"]}
    for rid in ("r-pol-01", "r-pol-02", "r-pol-03"):
        assert by_id[rid]["embedding_route"] == "rag", (rid, by_id[rid])


def test_dataset_labels_consistent(report):
    # Every keyword_misroute claim in the dataset is actually misrouted by keyword.
    assert report["stale_labels"] == [], report["stale_labels"]


def test_no_prototype_leakage(report):
    # F1-R1: the eval must measure generalization, not memorization. No question
    # may copy or near-copy (>= 0.90 cosine) a classifier prototype.
    assert report["leakage_guard"]["leaks"] == [], report["leakage_guard"]["leaks"]


def test_routing_is_zero_token():
    # Routing must not depend on a provider. In CI GROQ_API_KEY is empty, so if
    # route_message tried an LLM call it would raise; a clean decision proves the
    # classifier is purely local (ONNX embedding), spending no tokens.
    dec = route_message("Does the warranty policy cover water damage?", router="embedding")
    assert dec.route in ("rag", "tool")
    assert dec.router == "embedding"
