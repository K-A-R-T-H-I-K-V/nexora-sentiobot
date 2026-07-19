"""Feature F2 groundedness gate (free, deterministic, 0 LLM tokens).

CI tripwire for the trust feature's core safety properties:
  - NO FALSE GREEN: an answer with an unsupported claim is never labeled grounded.
  - Citation spans are literal SUBSTRINGS of the retrieved source (extraction, not
    generation, so a hallucinated quote is structurally impossible).
  - SCOPE: an answer with no retrieved source (a pure tool answer) gets no badge.

Uses controlled source fixtures so it is self-contained and deterministic; the
validation harness (groundedness_validation.py) exercises real corpus answers.
"""
from __future__ import annotations

from backend.core.config import get_settings
from backend.core.groundedness import analyze

SETTINGS = get_settings()

SRC = [{
    "source": "policies.md",
    "section": "Warranty and Returns",
    "text": (
        "The standard warranty covers manufacturing defects for two years from the "
        "date of purchase. Water damage and accidental damage are not covered under "
        "the standard warranty. Refunds are issued within fourteen days of return."
    ),
}]


def test_fully_supported_answer_is_grounded():
    answer = ("The standard warranty covers manufacturing defects for two years "
              "[Source 1]. Water damage is not covered [Source 1].")
    r = analyze(answer, SRC, SETTINGS)
    assert r["grounded"]["label"] == "grounded", r["grounded"]
    assert r["citations"], "expected citations for a grounded answer"


def test_poisoned_answer_is_never_green():
    # One true claim plus one fabricated claim absent from the source.
    answer = ("The warranty covers manufacturing defects for two years [Source 1]. "
              "It also includes free lifetime international shipping and a "
              "complimentary smart speaker in every box.")
    r = analyze(answer, SRC, SETTINGS)
    assert r["grounded"]["label"] != "grounded", r["grounded"]


def test_short_unsupported_claim_is_not_green():
    # F2-R1: a SHORT fabricated factual clause ("Ships worldwide free." = 3 words)
    # must still count and downgrade the label. It evaded the old 25-char floor.
    answer = ("The warranty covers manufacturing defects for two years [Source 1]. "
              "Ships worldwide free.")
    r = analyze(answer, SRC, SETTINGS)
    assert r["grounded"]["label"] != "grounded", r["grounded"]


def test_hedged_unsupported_claim_is_not_green():
    # F2-R1: a HEDGED fabrication ("Of course it also ...") must count. The hedge
    # opener is stripped and the claim behind it is checked; it used to be dropped
    # wholesale by the non-factual filter.
    answer = ("The warranty covers manufacturing defects for two years [Source 1]. "
              "Of course it also includes a complimentary smart speaker.")
    r = analyze(answer, SRC, SETTINGS)
    assert r["grounded"]["label"] != "grounded", r["grounded"]


def test_citation_spans_are_substrings_of_source():
    answer = ("The warranty covers manufacturing defects for two years [Source 1]. "
              "Refunds are issued within fourteen days [Source 1].")
    r = analyze(answer, SRC, SETTINGS)
    assert r["citations"], "expected citations"
    joined = " ".join(st["text"] for st in SRC)
    for c in r["citations"]:
        assert c["span"] in joined, f"citation span is not a literal source substring: {c['span']!r}"


def test_no_retrieved_source_is_unverified_no_badge():
    # A pure tool answer (order/warranty status) has no retrieved doc context.
    r = analyze("Your order NX-2025-301 shipped yesterday and arrives Friday.", [], SETTINGS)
    assert r["grounded"]["label"] == "unverified"
    assert r["citations"] == []


def test_partial_when_some_claims_unsupported():
    answer = ("Refunds are issued within fourteen days [Source 1]. "
              "The device includes a built-in holographic projector.")
    r = analyze(answer, SRC, SETTINGS)
    assert r["grounded"]["label"] == "partial", r["grounded"]
    assert 0.0 < r["grounded"]["score"] < 1.0
