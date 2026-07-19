"""
metrics.py — Per-request measurement for the latency + call-count + tokens
baseline (Increment 2). Measurement ONLY: nothing here changes request
behavior or output. A RequestMetrics object lives in a ContextVar, so it is
per-request and concurrency-safe (each request/task sees its own counters).

Captured per request:
  - route            : "rag" | "tool" | "cache"
  - intent           : F1 classified intent (doc_lookup, order_status, ...)
  - cache_hit        : bool
  - llm_calls        : LLM API calls (via a LangChain callback)
  - embedding_ops    : MiniLM embed_query / embed_documents ops
  - supabase_calls   : Supabase round trips (one per db.* function call)
  - prompt_tokens    : summed from Groq usage metadata
  - completion_tokens: summed from Groq usage metadata
  - retrieval_ms     : wall time of the retriever call (incl. multi-query LLM)

Time-to-first-token and end-to-end are measured client-side by the baseline
harness (what the user actually experiences); the fields above are the
server-internal counts that a client cannot see.
"""

from __future__ import annotations

import functools
import logging
from contextvars import ContextVar
from dataclasses import dataclass, asdict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    route: str = ""
    intent: str = ""  # F1: classified intent (or "keyword"/fallback label)
    sentiment: str = ""  # F4: detected emotion label (calm/confused/frustrated/angry)
    cache_hit: bool = False
    llm_calls: int = 0
    embedding_ops: int = 0
    supabase_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_ms: float = 0.0
    groundedness_ms: float = 0.0  # F2: local groundedness+citation pass (0 tokens)
    sentiment_ms: float = 0.0     # F4: local sentiment pass (0 tokens)

    def as_dict(self) -> dict:
        return asdict(self)


_current: ContextVar[RequestMetrics | None] = ContextVar("sentiobot_request_metrics", default=None)


def start() -> RequestMetrics:
    """Begin a fresh measurement for the current request/context."""
    m = RequestMetrics()
    _current.set(m)
    return m


def get() -> RequestMetrics | None:
    return _current.get()


def inc(field_name: str, n: int = 1) -> None:
    m = _current.get()
    if m is not None:
        setattr(m, field_name, getattr(m, field_name) + n)


def add_tokens(prompt: int, completion: int) -> None:
    m = _current.get()
    if m is not None:
        m.prompt_tokens += int(prompt or 0)
        m.completion_tokens += int(completion or 0)


def set_field(field_name: str, value) -> None:
    m = _current.get()
    if m is not None:
        setattr(m, field_name, value)


# ---------------------------------------------------------------------------
# LLM call + token counting via a LangChain callback (observation only)
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallbackHandler):
    """Counts LLM calls and sums Groq token usage. Attached via config so it
    propagates to nested runnables (the retriever's multi-query LLM, the agent
    and finalize nodes)."""

    def on_llm_start(self, *args, **kwargs) -> None:  # non-chat models
        inc("llm_calls")

    def on_chat_model_start(self, *args, **kwargs) -> None:  # chat models (Groq)
        inc("llm_calls")

    def on_llm_end(self, response, **kwargs) -> None:
        prompt_toks, completion_toks = self._extract_usage(response)
        if prompt_toks or completion_toks:
            add_tokens(prompt_toks, completion_toks)

    @staticmethod
    def _extract_usage(response) -> tuple[int, int]:
        # 1) llm_output.token_usage (OpenAI/Groq style)
        try:
            out = getattr(response, "llm_output", None) or {}
            tu = out.get("token_usage") or out.get("usage") or {}
            p = tu.get("prompt_tokens") or tu.get("input_tokens")
            c = tu.get("completion_tokens") or tu.get("output_tokens")
            if p is not None or c is not None:
                return int(p or 0), int(c or 0)
        except Exception:
            pass
        # 2) generation message usage_metadata (streaming path)
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    um = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if um:
                        return int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0))
        except Exception:
            pass
        return 0, 0


def callback_config(extra: dict | None = None) -> dict:
    """A RunnableConfig dict that attaches the metrics callback (plus any extra
    config). Passing this to astream/ainvoke/astream_events is observation only."""
    cfg = {"callbacks": [MetricsCallback()]}
    if extra:
        cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Embedding op counting (wraps an Embeddings; same vectors, just counted)
# ---------------------------------------------------------------------------

class CountingEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings):
        self._inner = inner

    def embed_documents(self, texts):
        inc("embedding_ops", len(texts))
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        inc("embedding_ops")
        return self._inner.embed_query(text)

    async def aembed_documents(self, texts):
        inc("embedding_ops", len(texts))
        return await self._inner.aembed_documents(texts)

    async def aembed_query(self, text):
        inc("embedding_ops")
        return await self._inner.aembed_query(text)


# ---------------------------------------------------------------------------
# Supabase round-trip counting (decorator for db.* functions)
# ---------------------------------------------------------------------------

def count_supabase(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        inc("supabase_calls")
        return fn(*args, **kwargs)
    return wrapper
