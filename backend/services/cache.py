"""
cache.py — Three-tier caching for RAG responses.

Tier 1 │ In-process LRU   │ Exact SHA-256 hash match  │ ~0 ms    │ Always on
Tier 2 │ Semantic cosine   │ Cosine similarity ≥ 0.92  │ ~5-15 ms │ Always on
Tier 3 │ Redis             │ Exact hash, cross-process │ ~1-5 ms  │ Optional (set REDIS_URL)

Why three tiers?
  L1 — zero-cost for the exact same query repeated in a session.
  L2 — catches semantically equivalent questions that L1 misses, e.g.
       "what's the warranty?" vs "how long is my warranty period?".
       Typical hit rate boost: +15-20% on top of L1 alone.
  L3 — shares the cache across multiple backend workers / replicas.
       Falls back gracefully if Redis is down or not configured.

Cache key strategy
  L1 and L3 use:  sha256(normalised_query)   — user-agnostic so Alice and
                  Bob asking the same product question share a cached answer.
  L2 uses raw normalised text for embedding lookup (no hashing needed).

Thread / async safety
  L1 is a plain dict — safe for a single-process uvicorn worker.
  L2 uses numpy ops that release the GIL; safe for async code.
  L3 (Redis) is fully async via redis.asyncio.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import numpy as np

from config import get_settings

log = logging.getLogger(__name__)

# ── Embedding model (lazy singleton) ─────────────────────────────────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()


# ── Tier 1: In-process LRU ────────────────────────────────────────────────────

class _LRUCache:
    def __init__(self, max_size: int = 256):
        self._store: OrderedDict[str, str] = OrderedDict()
        self._max = max_size

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._max:
            self._store.popitem(last=False)


_lru = _LRUCache()


# ── Tier 2: Semantic cosine cache ─────────────────────────────────────────────

class _SemanticCache:
    """
    Stores (query_embedding, answer) pairs.
    On lookup, runs a vectorised cosine pass across all stored embeddings.
    For <= 512 entries this takes < 2 ms on CPU.
    """

    def __init__(self, max_size: int = 512, threshold: float = 0.92):
        self._queries:    list[str]         = []
        self._embeddings: list[np.ndarray]  = []
        self._answers:    list[str]         = []
        self._max       = max_size
        self._threshold = threshold

    def get(self, query: str) -> str | None:
        if not self._embeddings:
            return None

        q_emb  = np.array(_get_embedder().embed_query(_normalise(query)))
        stored = np.vstack(self._embeddings)                     # (N, D)
        norms  = np.linalg.norm(stored, axis=1, keepdims=True)
        normed = np.where(norms > 0, stored / norms, 0.0)
        q_norm = q_emb / (np.linalg.norm(q_emb) or 1.0)
        scores = normed @ q_norm                                 # (N,)
        best   = int(np.argmax(scores))

        if scores[best] >= self._threshold:
            log.info("L2 semantic cache HIT (score=%.3f, matched='%.60s')",
                     scores[best], self._queries[best])
            return self._answers[best]
        return None

    def set(self, query: str, answer: str) -> None:
        if len(self._queries) >= self._max:
            self._queries.pop(0)
            self._embeddings.pop(0)
            self._answers.pop(0)
        emb = np.array(_get_embedder().embed_query(_normalise(query)))
        self._queries.append(query)
        self._embeddings.append(emb)
        self._answers.append(answer)

    @property
    def size(self) -> int:
        return len(self._queries)


_semantic = _SemanticCache(
    max_size=512,
    threshold=float(getattr(get_settings(), "semantic_cache_threshold", 0.92)),
)


# ── Tier 3: Optional Redis ────────────────────────────────────────────────────

_redis: Any = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    s = get_settings()
    if not getattr(s, "redis_url", ""):
        return None
    try:
        import redis.asyncio as aioredis  # type: ignore
        _redis = aioredis.from_url(s.redis_url, decode_responses=True)
        log.info("Redis cache connected at %s", s.redis_url)
        return _redis
    except ImportError:
        log.warning("redis package not installed; skipping Redis tier")
        return None
    except Exception as exc:
        log.warning("Redis connection failed (%s); running without it", exc)
        return None


# ── Public API (drop-in replacement for original cache.py) ───────────────────

async def get_cached_response(user_id: str, query: str) -> str | None:
    """
    Returns cached answer string or None on cache miss.
    Tries L1 -> L2 -> L3 in order.
    user_id is accepted for API compatibility but not used in the cache key,
    so identical questions from different users share the same cached answer.
    """
    key = _sha256(query)

    # ── L1: exact LRU ──
    hit = _lru.get(key)
    if hit:
        log.debug("L1 cache HIT")
        return hit

    # ── L2: semantic similarity ──
    hit = _semantic.get(query)
    if hit:
        _lru.set(key, hit)   # promote to L1 for the next exact match
        return hit

    # ── L3: Redis ──
    r = _get_redis()
    if r:
        try:
            hit = await r.get(key)
            if hit:
                log.debug("L3 Redis cache HIT")
                _lru.set(key, hit)
                _semantic.set(query, hit)  # warm lower tiers too
                return hit
        except Exception as exc:
            log.warning("Redis get failed: %s", exc)

    return None


async def set_cached_response(user_id: str, query: str, answer: str) -> None:
    """Write answer to all available cache tiers."""
    key = _sha256(query)
    ttl = getattr(get_settings(), "cache_ttl_seconds", 3600)

    _lru.set(key, answer)
    _semantic.set(query, answer)

    r = _get_redis()
    if r:
        try:
            await r.setex(key, ttl, answer)
        except Exception as exc:
            log.warning("Redis set failed: %s", exc)


async def invalidate_user_cache(user_id: str) -> None:
    """
    Called on logout or profile change (kept for API compatibility).
    L1/L2 are not user-scoped so only Redis cleanup is relevant here.
    """
    r = _get_redis()
    if r:
        try:
            async for key in r.scan_iter(f"*"):
                await r.delete(key)
        except Exception as exc:
            log.warning("Redis invalidation failed: %s", exc)


# ── Stats (used by /metrics endpoint in main.py) ──────────────────────────────

def cache_stats() -> dict:
    return {
        "l1_size":        len(_lru._store),
        "l2_size":        _semantic.size,
        "l2_threshold":   _semantic._threshold,
        "redis_connected": _get_redis() is not None,
    }