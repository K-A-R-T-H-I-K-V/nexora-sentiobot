"""Feature F4 sentiment gate (free, deterministic, 0 LLM tokens).

CI tripwire for the trust-critical properties:
  - NO FALSE ESCALATION: the proactive human offer never fires on calm / confused /
    emphatic-calm / positive controls (the F2 no-false-green applied to emotion).
  - Escalation FIRES on sustained frustration (multi-turn) and the profanity override.
  - NEVER ANNOUNCE: injected tone instructions always forbid naming the user's mood,
    and the escalation-offer line contains no emotion words.
  - Leakage-guarded eval (no message near-copies a prototype).
"""
from __future__ import annotations

import re

import pytest

from backend.core import sentiment as S
from backend.core.config import get_settings
from backend.scripts.sentiment_eval import evaluate

SETTINGS = get_settings()
_EMOTION_WORDS = re.compile(r"\b(you (seem|sound|are|look)|angry|frustrated|upset|annoyed|mad)\b", re.I)


@pytest.fixture(scope="module")
def report():
    return evaluate()


def test_no_false_escalation(report):
    # The load-bearing property: zero false escalations on the no-escalate controls.
    assert report["false_escalations"] == [], report["false_escalations"]
    assert report["false_escalation_rate"] == 0.0


def test_escalation_scenarios_pass(report):
    # Fires on sustained frustration + abuse override, not on calm/resolved.
    assert report["escalation_scenarios"]["failures"] == [], report["escalation_scenarios"]["failures"]
    assert report["abuse_override_missed"] == [], report["abuse_override_missed"]


def test_no_prototype_leakage(report):
    assert report["leakage_guard"]["leaks"] == [], report["leakage_guard"]["leaks"]


def test_tone_instructions_never_announce():
    # Every non-empty tone instruction must forbid naming the user's mood.
    for label in ("confused", "frustrated", "angry"):
        for escalate in (False, True):
            tone = S._tone_instruction(label, escalate)
            assert tone, (label, escalate)
            low = tone.lower()
            assert "mood or feelings" in low and "do not comment on" in low, tone
    # calm injects nothing (default voice).
    assert S._tone_instruction("calm", False) == ""


def test_escalation_offer_has_no_emotion_words():
    offer = S.escalation_offer(True, "Here are the steps to reset it.")
    assert offer, "expected an offer when escalate=True and none present"
    assert not _EMOTION_WORDS.search(offer), offer
    # No offer when the model already offered, or when not escalating.
    assert S.escalation_offer(True, "I can connect you with a human agent.") == ""
    assert S.escalation_offer(False, "Here are the steps.") == ""


def test_positive_profanity_is_calm_not_escalated():
    # "this is fucking amazing" is positive: negative-only lexicon + positive guard
    # keep it calm, so profanity alone never forces an escalation.
    r = S.analyze("this is fucking amazing, works great now!!!", [], SETTINGS)
    assert r.label == "calm"
    assert r.escalate is False


def test_sentiment_is_zero_token():
    # Runs with GROQ_API_KEY empty (CI); a provider call would raise. A clean result
    # proves the pass is purely local ONNX.
    r = S.analyze("this still will not connect and it is really frustrating", [], SETTINGS)
    assert r.label in S.LABELS
