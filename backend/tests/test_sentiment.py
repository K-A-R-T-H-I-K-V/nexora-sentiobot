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


def test_single_spike_stays_below_threshold():
    # F4-R1: PIN the single-spike-safe margin. A single MAXIMUM-frustration turn (no
    # history) must land BELOW the escalation threshold, so one angry message never
    # escalates on its own (only sustained frustration, or explicit profanity, does).
    # The seed-at-0 EMA of one turn is exactly alpha * score, so the invariant is
    # alpha * 1.0 < escalation_threshold. This test fails loudly if a future retune of
    # alpha or the threshold silently starts escalating single spikes.
    alpha = SETTINGS.sentiment_ema_alpha
    thr = SETTINGS.sentiment_escalation_threshold
    assert alpha * 1.0 < thr, (
        f"single-spike safety broken: alpha*1.0={alpha} >= threshold={thr}; "
        "a lone maximum-anger turn would now escalate"
    )
    # And prove it end-to-end on a maxed non-profane single turn (no history).
    # (Profanity has its own high-precision override, tested separately.)
    r = S.analyze("this is absolute garbage and a total ripoff, useless", [], SETTINGS)
    assert r.score >= 0.9, r.score          # a maximum-frustration read
    assert r.escalate is False, (r.ema, r.escalate)  # but a single spike does not escalate


def test_sustained_frustration_does_escalate():
    # The paired invariant: two frustrated turns cross the threshold (sustained).
    r = S.analyze("still broken and it is really frustrating",
                  ["this still will not connect", "i tried that already and nothing helped"],
                  SETTINGS)
    assert r.escalate is True, (r.ema, r.escalate)
