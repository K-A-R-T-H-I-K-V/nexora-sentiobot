"""
agent.py — Production LangGraph agent with token-level streaming.

Architecture
------------
We replace the old LangChain ReAct AgentExecutor with a LangGraph
StateGraph.  This gives us:

  1. Proper async streaming  — yields tokens as they are generated.
  2. Explicit node control   — each step (RAG retrieval, tool call,
                               LLM generation) is a discrete graph node.
  3. Better error recovery   — each node can independently catch and
                               re-route on failure.
  4. Observable intermediate — callers receive typed event objects so
                               the frontend can show "tool called" badges
                               in real-time.

Streaming protocol (SSE events sent to frontend)
-------------------------------------------------
  {"type": "token",      "data": "Hello "}
  {"type": "token",      "data": "there!"}
  {"type": "tool_start", "data": {"name": "check_warranty_status", "input": "SN-..."}}
  {"type": "tool_end",   "data": {"name": "check_warranty_status", "output": "..."}}
  {"type": "done",       "data": {"answer": "...", "sources": [...]}}
  {"type": "error",      "data": {"message": "..."}}
"""

from __future__ import annotations
import json
import pickle
import os
import logging
from typing import AsyncIterator, TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.request_context import current_user_id
from backend.agent.tools import check_order_status, check_warranty_status, create_support_ticket

logger = logging.getLogger(__name__)

# Generic, user-safe error text. The real exception (which may contain provider
# quota bodies, internal URLs, or stack detail) is logged server-side only and
# never serialized to the client (F-1).
_GENERIC_ERROR = "The assistant is temporarily unavailable. Please try again in a moment."

# ---------------------------------------------------------------------------
# Singletons (initialised once at startup)
# ---------------------------------------------------------------------------

_retriever = None
_llm = None
_graph = None


def get_llm():
    """Return the chat LLM for the configured provider (Groq by default).

    The provider is config-selectable (LLM_PROVIDER) so a future swap is a
    config change, not another migration. Embeddings and Chroma retrieval are
    unaffected by this choice.
    """
    global _llm
    if _llm is None:
        s = get_settings()
        provider = (s.llm_provider or "groq").lower()
        if provider == "groq":
            _llm = ChatGroq(
                model=s.groq_model,
                api_key=s.groq_api_key,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
                streaming=True,  # Critical for token streaming
            )
        elif provider == "gemini":
            _llm = ChatGoogleGenerativeAI(
                model=s.gemini_model,
                google_api_key=s.google_api_key,
                temperature=s.llm_temperature,
                max_output_tokens=s.llm_max_tokens,
                streaming=True,
            )
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER {provider!r}; use 'groq' or 'gemini'."
            )
    return _llm


def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    s = get_settings()
    embedding_model = HuggingFaceEmbeddings(model_name=s.embedding_model)

    vectorstore = Chroma(
        persist_directory=s.vector_db_path,
        embedding_function=embedding_model,
    )
    byte_store = LocalFileStore(s.parent_store_path)
    store = create_kv_docstore(byte_store)

    with open(s.parent_list_path, "rb") as f:
        all_parent_docs = pickle.load(f)

    bm25 = BM25Retriever.from_documents(all_parent_docs)
    chroma_ret = vectorstore.as_retriever(search_kwargs={"k": 5})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, chroma_ret],
        weights=[0.4, 0.6],  # Slightly favour semantic for general queries
    )

    mq_prompt = PromptTemplate.from_template(
        "Generate 3 alternative phrasings of this question for vector search. "
        "Return them separated by newlines.\nQuestion: {question}"
    )
    _retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble,
        llm=get_llm(),
        prompt=mq_prompt,
    )
    return _retriever


# ---------------------------------------------------------------------------
# Shared retrieval
# ---------------------------------------------------------------------------
# Both answer paths ground on the SAME retrieved context: the direct RAG
# stream (below) and the lookup_documentation tool used by the graph path.
# Neither synthesises here; the caller's LLM writes the single streamed answer.


def _format_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "source": d.metadata.get("source", "N/A"),
            "section": d.metadata.get("section_title", "N/A"),
        }
        for d in docs
    ]


async def _retrieve_context(query: str) -> tuple[str, list[dict]]:
    """Retrieve docs and return (formatted_context_str, sources_list)."""
    retriever = get_retriever()
    docs: list[Document] = await retriever.ainvoke(query)
    if not docs:
        return "", []

    context_str = "\n\n".join(
        f"[Source {i+1}] ({d.metadata.get('source','?')} | {d.metadata.get('section_title','?')})\n"
        f"{d.page_content}"
        for i, d in enumerate(docs)
    )
    return context_str, _format_sources(docs)


@tool
async def lookup_documentation(query: str) -> str:
    """Search Nexora product manuals and policy documentation to answer a
    question. Use this FIRST for any product, feature, setup, troubleshooting,
    policy, or warranty-policy question. Returns retrieved documentation
    context tagged with [Source N]; answer using ONLY that context and cite
    the sources inline as [Source N]."""
    context_str, sources = await _retrieve_context(query)
    return json.dumps({"context": context_str, "sources": sources})


def _tool_output_text(output) -> str:
    """Best-effort extraction of a tool's text output (ToolMessage or raw)."""
    content = getattr(output, "content", None)
    return content if isinstance(content, str) else str(output)


def _extract_tool_sources(output) -> list[dict]:
    """Pull the sources list out of lookup_documentation's JSON output."""
    try:
        data = json.loads(_tool_output_text(output))
        srcs = data.get("sources", [])
        return srcs if isinstance(srcs, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# LangGraph Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: dict
    user_id: str
    sources: list[dict]
    final_answer: str
    called_tools: list[str]      # signatures of (tool_name, args) already executed
    tool_rounds: int             # number of tools-node visits so far
    max_tool_rounds: int         # after this many rounds, force finalize


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

TOOLS = [lookup_documentation, check_order_status, check_warranty_status, create_support_ticket]
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_DEDUPE_NUDGE = (
    "You already called this tool with these exact arguments and its result is "
    "above. Do NOT call it again. Answer the user's question now using the "
    "information already gathered."
)


def _tool_signature(name: str, args: dict) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
    except Exception:
        return f"{name}:{args!r}"


def _build_system_message(user_profile: dict) -> str:
    name = user_profile.get("name", "Guest")
    products = user_profile.get("owned_products", [])

    profile_section = f"User name: {name}\n"
    if products:
        lines = "\n".join(
            f"  - {p['product_name']} (Serial: {p['serial_number']})"
            for p in products
        )
        profile_section += f"Owned products:\n{lines}\n"
    else:
        profile_section += "No registered products on file.\n"

    return f"""\
You are SentioBot, a helpful and precise AI support agent for Nexora Electronics.

## User Profile (use this proactively)
{profile_section}

## Behaviour Rules
1. RAG FIRST: For any product, feature, setup, troubleshooting, policy, or \
   warranty-policy question, call lookup_documentation and answer ONLY from \
   the context it returns, citing sources inline as [Source N]. If the context \
   is empty, say so honestly and offer to raise a support ticket.
2. PROACTIVE: If the user asks about a product and you have its serial number \
   above, call check_warranty_status immediately; do NOT ask for it.
3. MEMORY: Do not ask for info the user already gave in this conversation.
4. ESCALATE ONLY IF NEEDED: Use create_support_ticket only when documentation \
   does not resolve the issue or the user explicitly asks for a human.
5. FORMAT: Use Markdown. Be concise. Offer the logical next action at the end.

Available tools: lookup_documentation, check_order_status, \
check_warranty_status, create_support_ticket.
"""


async def call_model(state: AgentState) -> AgentState:
    """Main LLM node: generates next response or tool call."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    sys_msg = SystemMessage(content=_build_system_message(state["user_profile"]))
    messages = [sys_msg] + state["messages"]

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def dedupe_tool_node(state: AgentState) -> AgentState:
    """Execute the requested tool calls, but SKIP any (tool, args) already run
    this turn (F1.5-1). A repeated call returns a nudge instead of re-executing,
    so the model stops looping and the daily token budget is not burned.

    The authenticated user_id is bound into the request context here, in the
    same coroutine that awaits the tools, so create_support_ticket uses the real
    user rather than an LLM-supplied value (F1.5-3).
    """
    current_user_id.set(state.get("user_id", ""))

    last = state["messages"][-1]
    called = list(state.get("called_tools", []))
    out_messages: list[BaseMessage] = []

    for tc in getattr(last, "tool_calls", []) or []:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        tc_id = tc.get("id", "")
        sig = _tool_signature(name, args)

        if sig in called:
            out_messages.append(ToolMessage(content=_DEDUPE_NUDGE, tool_call_id=tc_id, name=name))
            continue

        called.append(sig)
        tool = _TOOLS_BY_NAME.get(name)
        if tool is None:
            out_messages.append(ToolMessage(content=f"Unknown tool: {name}.", tool_call_id=tc_id, name=name))
            continue

        try:
            result = await tool.ainvoke(args)
        except Exception:
            logger.exception("Tool %s raised", name)
            result = f"The {name} tool could not complete right now."
        content = result if isinstance(result, str) else str(result)
        out_messages.append(ToolMessage(content=content, tool_call_id=tc_id, name=name))

    return {
        "messages": out_messages,
        "called_tools": called,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }


async def finalize_node(state: AgentState) -> AgentState:
    """Forced graceful finish: answer WITHOUT tools once the round cap is hit,
    guaranteeing convergence even if the model keeps trying to call tools."""
    llm = get_llm()  # no bind_tools -> the model must answer, not call tools
    sys_msg = SystemMessage(content=(
        _build_system_message(state["user_profile"])
        + "\n\nYou have gathered enough information from the tools above. Answer "
          "the user's question now, concisely, using that information. Do NOT "
          "call any tools."
    ))
    response = await llm.ainvoke([sys_msg] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: tool calls -> tools, unless the round cap is hit -> finalize;
    no tool calls -> END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        if state.get("tool_rounds", 0) >= state.get("max_tool_rounds", 4):
            return "finalize"
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def get_graph():
    global _graph
    if _graph is not None:
        return _graph

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", dedupe_tool_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "finalize": "finalize", END: END},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)

    _graph = workflow.compile()
    return _graph


# ---------------------------------------------------------------------------
# Streaming entry point (used by FastAPI)
# ---------------------------------------------------------------------------

async def stream_agent_response(
    user_message: str,
    chat_history: list[dict],
    user_profile: dict,
    user_id: str = "",
) -> AsyncIterator[str]:
    """
    Yields Server-Sent Event data strings.
    Each yielded string is a JSON object:
      {"type": "token",      "data": "..."}
      {"type": "tool_start", "data": {...}}
      {"type": "tool_end",   "data": {...}}
      {"type": "done",       "data": {"answer": "...", "sources": [...]}}
      {"type": "error",      "data": {"message": "..."}}
    """
    # Bind the authenticated user into request context for server-side tools.
    current_user_id.set(user_id)

    # Convert stored history to LangChain messages
    lc_history: list[BaseMessage] = []
    for msg in chat_history[-12:]:  # Last 6 exchanges
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    # Check if this is a RAG/documentation query (fast path)
    is_tool_query = any(
        kw in user_message.lower()
        for kw in ["order", "warranty", "serial", "ticket", "human", "support"]
    )

    # For pure documentation queries, stream tokens directly via RAG
    if not is_tool_query:
        try:
            context_str, sources = await _retrieve_context(user_message)

            if not context_str:
                yield f"data: {json.dumps({'type': 'token', 'data': 'I could not find relevant documentation for your query. Would you like me to raise a support ticket?'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'data': {'answer': '', 'sources': []}})}\n\n"
                return

            llm = get_llm()
            sys_content = (
                f"{_build_system_message(user_profile)}\n\n"
                f"## Retrieved Documentation\n{context_str}"
            )
            messages = [SystemMessage(content=sys_content)] + lc_history + [HumanMessage(content=user_message)]

            full_answer = ""
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'data': {'answer': full_answer, 'sources': sources}})}\n\n"
            return

        except Exception:
            logger.exception("RAG stream error")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': _GENERIC_ERROR}})}\n\n"
            return

    # For tool-using queries, run the LangGraph agent
    max_rounds = get_settings().agent_max_tool_rounds
    initial_state: AgentState = {
        "messages": lc_history + [HumanMessage(content=user_message)],
        "user_profile": user_profile,
        "user_id": user_id,
        "sources": [],
        "final_answer": "",
        "called_tools": [],
        "tool_rounds": 0,
        "max_tool_rounds": max_rounds,
    }

    graph = get_graph()
    full_answer = ""
    tool_sources: list[dict] = []

    try:
        async for event in graph.astream_events(
            initial_state,
            version="v2",
            # Backstop above the round-based finalize (which triggers first);
            # the force-finalize path, not this limit, is what makes it converge.
            config={"recursion_limit": 2 * max_rounds + 6},
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                # Allowlist: stream ONLY answer-producing node tokens (the agent
                # node and the forced finalize node). Nested LLM calls (the
                # retriever's multi-query expansion inside lookup_documentation,
                # tagged node "tools", or any untagged nested stream) are
                # suppressed so they cannot leak into the visible answer (F-2).
                if event.get("metadata", {}).get("langgraph_node") not in ("agent", "finalize"):
                    continue
                token = event["data"]["chunk"].content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'data': {'name': event['name'], 'input': str(event['data'].get('input', ''))}})}\n\n"

            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                # Capture retrieved sources from the documentation tool for the
                # done event. Do NOT overwrite the streamed answer with raw tool
                # output; the agent node writes the final answer.
                if event.get("name") == "lookup_documentation":
                    tool_sources = _extract_tool_sources(output)
                    display = "Retrieved documentation context."
                else:
                    display = _tool_output_text(output)
                yield f"data: {json.dumps({'type': 'tool_end', 'data': {'name': event['name'], 'output': display}})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'data': {'answer': full_answer, 'sources': tool_sources}})}\n\n"

    except Exception:
        logger.exception("Agent stream error")
        # F1.5-2: if a real answer already streamed, finish gracefully with a
        # done event (carrying whatever sources were captured) instead of
        # flipping a good answer to an error. Only a pre-answer failure errors.
        if full_answer.strip():
            yield f"data: {json.dumps({'type': 'done', 'data': {'answer': full_answer, 'sources': tool_sources}})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': _GENERIC_ERROR}})}\n\n"
