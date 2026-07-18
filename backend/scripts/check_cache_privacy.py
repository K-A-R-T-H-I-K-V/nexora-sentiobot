"""
check_cache_privacy.py — Regression guard for the cross-user cache leak
(recon defect P0-2). Two different users asking the same (or semantically
similar) personalized question must NOT receive each other's cached answer.

Run:   python -m backend.scripts.check_cache_privacy
  (or) python backend/scripts/check_cache_privacy.py
Exit:  0 = isolated (pass), 1 = leak detected (fail).

Dependency-light and deterministic: forces the in-process L1/L2 tiers only
(REDIS_URL empty) and DEBUG on, so it needs no Redis and no LLM. It does load
the MiniLM embedder for the L2 semantic tier (downloaded once from HuggingFace).
Suitable as the free-tier cache check in CI later.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make `backend` importable when run as a plain script, and force a
# deterministic, dependency-light config BEFORE importing settings/cache.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DEBUG", "true")
os.environ["REDIS_URL"] = ""
os.environ.setdefault("JWT_SECRET", "regression-test-secret")

from backend.services import cache  # noqa: E402

ALICE = "alice"
BOB = "bob"
QUESTION = "how long is my warranty on my thermostat"
SIMILAR = "what is the remaining warranty period for my thermostat"
ALICE_ANSWER = "Alice PRIVATE: your thermostat SN-A1 warranty expires 2027-01-01."


async def main() -> int:
    failures: list[str] = []

    # Alice caches a personalized answer.
    await cache.set_cached_response(ALICE, QUESTION, ALICE_ANSWER)

    # 1. L1 exact: Bob asks the identical question -> must be a miss.
    bob_exact = await cache.get_cached_response(BOB, QUESTION)
    if bob_exact is not None:
        failures.append(f"L1 LEAK: Bob received Alice's answer for the exact question: {bob_exact!r}")

    # 2. L2 semantic: Bob asks a semantically similar question -> must be a miss.
    bob_similar = await cache.get_cached_response(BOB, SIMILAR)
    if bob_similar is not None:
        failures.append(f"L2 LEAK: Bob received Alice's answer for a similar question: {bob_similar!r}")

    # 3. Same-user exact still works (the cache must not be broken outright).
    alice_exact = await cache.get_cached_response(ALICE, QUESTION)
    if alice_exact != ALICE_ANSWER:
        failures.append(f"REGRESSION: Alice did not get her own exact answer back (got {alice_exact!r})")

    # 4. Same-user semantic hit (informational; depends on the 0.92 threshold).
    alice_similar = await cache.get_cached_response(ALICE, SIMILAR)
    print(f"[info] same-user semantic hit for Alice: {'yes' if alice_similar == ALICE_ANSWER else 'no'}")

    if failures:
        print("CACHE PRIVACY CHECK: FAIL")
        for f in failures:
            print("  -", f)
        return 1

    print("CACHE PRIVACY CHECK: PASS (no cross-user leak; same-user exact hit works)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
