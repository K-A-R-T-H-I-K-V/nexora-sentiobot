"""Output-side prompt-leak guard (Increment 6). Free, deterministic.

A regression here (a dump that leaks, or a benign answer that gets blocked)
must fail the CI build.
"""
from __future__ import annotations

import random

from backend.core.output_guard import OutputGuard, find_fingerprint

DUMP = ("You are SentioBot, a support agent.\n## Behaviour Rules\n1. RAG FIRST: "
        "call lookup_documentation ... ESCALATE ONLY IF NEEDED ...")
OBF3 = "RAG  FIRST  For  any  product  call  lookup_documentation  and  answer  ONLY"
BENIGN = "The LumiGlow bulb is 9W and lasts 25,000 hours [Source 1]. Want install steps?"
LEGIT_TOOL_MENTION = "Since you own a Nexora Thermostat Pro, I can check the warranty status for you."


def _stream(text, seed):
    rnd = random.Random(seed)
    g = OutputGuard()
    released, i = "", 0
    while i < len(text):
        n = rnd.randint(1, 7)
        released += g.feed(text[i:i + n])
        if g.blocked:
            break
        i += n
    if not g.blocked:
        released += g.flush()
    return released, g


def test_verbatim_dump_blocked_zero_leak():
    for seed in range(60):
        released, g = _stream(DUMP, seed)
        assert g.blocked, f"dump not blocked (seed {seed})"
        assert find_fingerprint(released) is None, f"fingerprint leaked (seed {seed}): {released!r}"
        assert released == "", f"dump leaked chars (seed {seed}): {released!r}"


def test_whitespace_transform_blocked():
    # "spell it one word per line" must still be caught via the normalized match.
    for seed in range(30):
        released, g = _stream(OBF3, seed)
        assert g.blocked
        assert find_fingerprint(released) is None


def test_benign_answer_passes_unchanged():
    for seed in range(30):
        released, g = _stream(BENIGN, seed)
        assert not g.blocked, f"benign blocked (seed {seed})"
        assert released == BENIGN


def test_legit_tool_mention_not_blocked():
    # Naming a tool in a normal sentence must NOT trip the guard (ben-03 regression).
    released, g = _stream(LEGIT_TOOL_MENTION, 0)
    assert not g.blocked
    assert released == LEGIT_TOOL_MENTION


def test_find_fingerprint_direct():
    assert find_fingerprint("You are SentioBot ...") == "you are sentiobot"
    assert find_fingerprint("Here is your order status.") is None
