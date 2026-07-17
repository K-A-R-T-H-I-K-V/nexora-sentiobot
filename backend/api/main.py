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
    user_id = current_user["id"]
    user_profile = {
        "name": current_user.get("name", "Guest"),
        "owned_products": current_user.get("owned_products", []),
    }

    # ---- Cache check (skip the agent, but still persist the turn) ----
    cached = await cache.get_cached_response(user_id, req.message)
    if cached:
        # F-3: persist the exchange so cached turns are not missing from
        # conversation history. Resolve/create the conversation, save both
        # messages, and log an analytics row (so feedback works on cached
        # answers too), threading the interaction_id like the live path.
        conv_id = req.conversation_id
        if not conv_id:
            conv = db.create_conversation(user_id, title=req.message[:60])
            conv_id = conv["id"]
        db.save_message(conv_id, "user", req.message)
        db.save_message(conv_id, "assistant", cached, {"sources": [], "cached": True})

        interaction_id = f"{user_id}-{int(datetime.utcnow().timestamp()*1000)}"
        db.log_analytics({
            "id": interaction_id,
            "user_id": user_id,
            "conversation_id": conv_id,
            "user_query": req.message,
            "bot_response": cached,
            "retrieved_docs": [],
            "feedback": 0,
            "timestamp": datetime.utcnow().isoformat(),
        })

        async def cached_stream():
            payload = json.dumps({"type": "token", "data": cached})
            yield f"data: {payload}\n\n"
            payload = json.dumps({"type": "done", "data": {"answer": cached, "sources": [], "cached": True, "interaction_id": interaction_id}})
            yield f"data: {payload}\n\n"

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
        conv = db.create_conversation(user_id, title=req.message[:60])
        conv_id = conv["id"]

    # ---- Load history ----
    history = db.get_messages_for_conversation(conv_id)

    # ---- Save incoming user message ----
    db.save_message(conv_id, "user", req.message)

    # ---- Stream agent response ----
    interaction_id = f"{user_id}-{int(datetime.utcnow().timestamp()*1000)}"
    final_answer_parts: list[str] = []
    sources: list[dict] = []

    async def generate():
        nonlocal final_answer_parts, sources

        async for sse_data in stream_agent_response(req.message, history, user_profile):
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
            await cache.set_cached_response(user_id, req.message, full_answer)
            db.log_analytics({
                "id": interaction_id,
                "user_id": user_id,
                "conversation_id": conv_id,
                "user_query": req.message,
                "bot_response": full_answer,
                "retrieved_docs": sources,
                "feedback": 0,
                "timestamp": datetime.utcnow().isoformat(),
            })

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
    db.update_analytics_feedback(req.interaction_id, req.feedback)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Analytics (basic public summary — restrict to admin in prod)
# ---------------------------------------------------------------------------

@app.get("/analytics/summary")
async def analytics_summary(current_user: Annotated[dict, Depends(auth.get_current_user)]):
    """Returns aggregated stats for the dashboard."""
    import pandas as pd

    rows = db.get_db().table("analytics").select("*").execute().data
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
