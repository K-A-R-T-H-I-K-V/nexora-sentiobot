"""
main.py — FastAPI application entry point.

Endpoints
---------
POST   /auth/login                       → JWT token
GET    /auth/me                          → user profile

POST   /chat/stream                      → SSE streaming chat (main endpoint)
GET    /chat/conversations               → list user's conversations
POST   /chat/conversations               → create new conversation
GET    /chat/conversations/{id}/messages → load message history

POST   /feedback                         → thumbs up/down (body: interaction_id, feedback)

GET    /analytics/summary                → dashboard metrics (admin-only in prod)

Deployment (run from the repository root so the `backend` package resolves)
----------
  Local:      uvicorn backend.api.main:app --reload --port 8000
  Production: uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT --workers 2
"""

from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import backend.core.auth as auth
import backend.services.database as db
import backend.services.cache as cache
from backend.agent.agent import stream_agent_response
from backend.core.config import get_settings
from backend.core import metrics
from backend.core.rate_limit import enforce_rate_limit
from backend.core.request_context import current_access_token

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s = get_settings()

app = FastAPI(
    title="SentioBot API",
    version="2.0.0",
    description="Production-grade RAG + LangGraph agent for Nexora Electronics support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=s.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None  # None → create new conversation


class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback: int  # 1 = thumbs up, -1 = thumbs down


class NewConversationRequest(BaseModel):
    title: str = "New Conversation"


# ---------------------------------------------------------------------------
# Authorization helpers (Increment 7: BOLA/IDOR)
# ---------------------------------------------------------------------------
# The backend connects with the Supabase SERVICE ROLE key, which BYPASSES the
# RLS policies in schema.sql. Authorization is therefore enforced HERE, in
# application code, on every user-scoped resource. Each object-taking route must
# confirm the object belongs to the authenticated caller before touching it.

def require_conversation_owner(conversation_id: str, user_id: str) -> dict:
    """Return the conversation iff it belongs to user_id; else 404.

    I7-1: 404 for BOTH a non-existent id and someone else's, so the response
    never reveals whether another user's conversation exists (no id oracle).
    """
    conv = db.get_conversation(conversation_id)
    if conv is None or conv.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conv


def require_analytics_owner(interaction_id: str, user_id: str) -> dict:
    """Return the analytics row iff it belongs to user_id; else 404 (see I7-1)."""
    row = db.get_analytics_by_id(interaction_id)
    if row is None or row.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interaction not found.")
    return row


# ---------------------------------------------------------------------------
# Middleware: latency logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms:.0f}ms)")
    return response


# ---------------------------------------------------------------------------
# Chat Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: Annotated[dict, Depends(auth.get_current_user)],
    token: Annotated[str, Depends(auth.oauth2_scheme)],
):
    """
    Main streaming chat endpoint.  Returns Server-Sent Events.

    Each SSE event carries a JSON payload:
      data: {"type": "token", "data": "Hello "}
      data: {"type": "tool_start", "data": {"name": "...", "input": "..."}}
      data: {"type": "tool_end",   "data": {"name": "...", "output": "..."}}
      data: {"type": "done",       "data": {"answer": "...", "sources": [...]}}
      data: {"type": "error",      "data": {"message": "..."}}
    """
    # ---- Request guards (Increment 5): reject before any work ----
    # Length cap: a giant paste is refused up front, so it cannot drive retrieval
    # or the LLM and burn tokens. Clean, defined message; no stack trace.
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Message cannot be empty.")
    if len(message) > s.max_input_chars:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Message too long (limit {s.max_input_chars} characters). Please shorten it.")
    # Per-user rate limit on this expensive endpoint (drains the daily budget).
    await enforce_rate_limit(current_user["id"])

    # BOLA (Increment 7): if the caller names an existing conversation, it must be
    # THEIRS. Otherwise a user could read history from and append messages to
    # another user's conversation by passing its id. Checked before any work.
    if req.conversation_id:
        require_conversation_owner(req.conversation_id, current_user["id"])

    metrics.start()  # per-request measurement (measurement only; see core/metrics.py)
    user_id = current_user["id"]
    user_profile = {
        "name": current_user.get("name", "Guest"),
        "owned_products": current_user.get("owned_products", []),
    }

    # ---- Cache check (skip the agent, but still persist the turn) ----
    cached = await cache.get_cached_response(user_id, message)
    if cached:
        metrics.set_field("route", "cache")
        metrics.set_field("cache_hit", True)
        # F-3: persist the exchange so cached turns are not missing from
        # conversation history. Resolve/create the conversation, save both
        # messages, and log an analytics row (so feedback works on cached
        # answers too), threading the interaction_id like the live path.
        conv_id = req.conversation_id
        if not conv_id:
            conv = db.create_conversation(user_id, title=message[:60])
            conv_id = conv["id"]
        db.save_message(conv_id, "user", message)
        db.save_message(conv_id, "assistant", cached, {"sources": [], "cached": True})

        interaction_id = f"{user_id}-{int(datetime.utcnow().timestamp()*1000)}"
        db.log_analytics({
            "id": interaction_id,
            "user_id": user_id,
            "conversation_id": conv_id,
            "user_query": message,
            "bot_response": cached,
            "retrieved_docs": [],
            "feedback": 0,
            "timestamp": datetime.utcnow().isoformat(),
        })

        m = metrics.get()
        async def cached_stream():
            payload = json.dumps({"type": "token", "data": cached})
            yield f"data: {payload}\n\n"
            done_data = {"answer": cached, "sources": [], "cached": True, "interaction_id": interaction_id}
            yield f"data: {json.dumps({'type': 'done', 'data': done_data})}\n\n"
            if m is not None:
                yield f"data: {json.dumps({'type': 'metrics', 'data': m.as_dict()})}\n\n"

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Interaction-Id": interaction_id,
                "X-Conversation-Id": conv_id,
            },
        )

    # ---- Resolve / create conversation ----
    conv_id = req.conversation_id
    if not conv_id:
        conv = db.create_conversation(user_id, title=message[:60])
        conv_id = conv["id"]

    # ---- Load history ----
    history = db.get_messages_for_conversation(conv_id)

    # ---- Save incoming user message ----
    db.save_message(conv_id, "user", message)

    # ---- Stream agent response ----
    interaction_id = f"{user_id}-{int(datetime.utcnow().timestamp()*1000)}"
    final_answer_parts: list[str] = []
    sources: list[dict] = []

    async def generate():
        nonlocal final_answer_parts, sources
        # Re-bind the access token inside the streaming generator so the post-answer
        # DB writes (save_message, log_analytics) carry the user's JWT for RLS,
        # even if the request-context ContextVar does not propagate into the
        # StreamingResponse task (Increment 10).
        current_access_token.set(token)

        async for sse_data in stream_agent_response(message, history, user_profile, user_id):
            out = sse_data

            # Parse SSE to capture final answer for persistence and to thread
            # the interaction_id into the done event so the client can submit
            # feedback against it.
            try:
                raw = sse_data.strip()
                if raw.startswith("data: "):
                    payload = json.loads(raw[6:])
                    ptype = payload.get("type")
                    if ptype == "token":
                        final_answer_parts.append(payload["data"])
                    elif ptype == "done":
                        sources = payload["data"].get("sources", [])
                        payload["data"]["interaction_id"] = interaction_id
                        out = f"data: {json.dumps(payload)}\n\n"
            except Exception:
                out = sse_data

            yield out

        # Persist assistant message after stream completes
        full_answer = "".join(final_answer_parts)
        if full_answer:
            db.save_message(conv_id, "assistant", full_answer, {"sources": sources})
            await cache.set_cached_response(user_id, message, full_answer)
            db.log_analytics({
                "id": interaction_id,
                "user_id": user_id,
                "conversation_id": conv_id,
                "user_query": message,
                "bot_response": full_answer,
                "retrieved_docs": sources,
                "feedback": 0,
                "timestamp": datetime.utcnow().isoformat(),
            })

        # Trailing measurement event with the COMPLETE per-request metrics
        # (including the post-answer Supabase writes). The client ignores
        # unknown event types; this is measurement only.
        _m = metrics.get()
        if _m is not None:
            yield f"data: {json.dumps({'type': 'metrics', 'data': _m.as_dict()})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Interaction-Id": interaction_id,
            "X-Conversation-Id": conv_id,
        },
    )


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------

@app.get("/chat/conversations")
async def list_conversations(current_user: Annotated[dict, Depends(auth.get_current_user)]):
    return db.get_conversations_for_user(current_user["id"])


@app.post("/chat/conversations", status_code=status.HTTP_201_CREATED)
async def new_conversation(
    body: NewConversationRequest,
    current_user: Annotated[dict, Depends(auth.get_current_user)],
):
    return db.create_conversation(current_user["id"], title=body.title)


@app.get("/chat/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    current_user: Annotated[dict, Depends(auth.get_current_user)],
):
    # BOLA (Increment 7): the conversation must belong to the caller before we
    # return any of its messages.
    require_conversation_owner(conversation_id, current_user["id"])
    messages = db.get_messages_for_conversation(conversation_id)
    return messages


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    current_user: Annotated[dict, Depends(auth.get_current_user)],
):
    # BOLA (Increment 7): a user may only rate their OWN interaction.
    require_analytics_owner(req.interaction_id, current_user["id"])
    db.update_analytics_feedback(req.interaction_id, req.feedback)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Analytics (basic public summary — restrict to admin in prod)
# ---------------------------------------------------------------------------

@app.get("/analytics/summary")
async def analytics_summary(current_user: Annotated[dict, Depends(auth.get_current_user)]):
    """Aggregated stats for the CALLER's own interactions.

    BOLA fix (Increment 7, A7-ANALYTICS): this previously did select("*") over the
    whole analytics table and returned EVERY user's queries and bot answers to any
    authenticated user - a bulk cross-user data exposure. It is now scoped to the
    caller's own rows. A cross-user / org-wide admin dashboard is a separate,
    admin-gated feature (deferred: needs a users.is_admin flag + migration; see
    docs/AUTHORIZATION.md), NOT something a normal user may receive.
    """
    import pandas as pd

    rows = db.get_analytics_for_user(current_user["id"])
    if not rows:
        return {"total": 0, "positive": 0, "negative": 0, "top_queries": []}

    df = pd.DataFrame(rows)
    top = df["user_query"].value_counts().head(10).reset_index()
    top.columns = ["query", "count"]

    return {
        "total": len(df),
        "positive": int((df["feedback"] == 1).sum()),
        "negative": int((df["feedback"] == -1).sum()),
        "top_queries": top.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
