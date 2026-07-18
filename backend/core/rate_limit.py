"""
rate_limit.py — in-process per-user request limiter (Increment 5 cost guard).

Wired onto the expensive /chat/stream endpoint and keyed by the authenticated
user id, using a fixed 60-second window. This closes the last unused config
field (rate_limit_per_minute) so a single account cannot hammer the LLM and
drain the daily token budget.

Honest limitation: the counter lives in this process, so with multiple uvicorn
workers the effective limit is per-worker, not global. A globally-correct limit
needs a shared store (Redis) keyed the same way, the exact L1/L2 vs L3 tradeoff
we already make for the cache. Documented, not hidden.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import HTTPException, status

from backend.core.config import get_settings

_lock = asyncio.Lock()
# user_id -> (window_index, count_in_window)
_windows: dict[str, tuple[int, int]] = {}
# Opportunistic cap on the dict so idle users cannot grow it without bound.
_MAX_TRACKED = 10_000


async def enforce_rate_limit(user_id: str) -> None:
    """Raise HTTP 429 if user_id has exceeded rate_limit_per_minute this window."""
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:  # disabled
        return

    window = int(time.time()) // 60
    async with _lock:
        start, count = _windows.get(user_id, (window, 0))
        if start != window:  # new minute: reset
            start, count = window, 0
        count += 1
        _windows[user_id] = (start, count)

        # Prune entries from older windows if the map grows large (idle users).
        if len(_windows) > _MAX_TRACKED:
            for uid in [u for u, (w, _) in _windows.items() if w != window]:
                _windows.pop(uid, None)

        over_limit = count > limit

    if over_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending more messages.",
            headers={"Retry-After": "60"},
        )
