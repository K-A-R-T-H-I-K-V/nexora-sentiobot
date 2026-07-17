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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.agent.tools import check_order_status, check_warranty_status, create_support_ticket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons (initialised once at startup)
# ---------------------------------------------------------------------------

_retriever = None
_llm = None
_graph = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = ChatGoogleGenerativeAI(
            model=s.gemini_model,
            google_api_key=s.google_api_key,
            temperature=s.llm_temperature,
            max_output_tokens=s.llm_max_tokens,
            streaming=True,  # Critical for token streaming
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
# RAG sub-chain (inline tool for lookup_documentation)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are SentioBot, a precise AI support agent for Nexora Electronics.

## Core Rules
1. Answer ONLY from the Provided Context below. Do not use outside knowledge.
2. If the context does not contain the answer, say so honestly.
3. Be concise and use Markdown formatting.
4. Cite sources as [Source N] inline.
5. After answering a product question, proactively offer the next helpful action \
   (e.g. warranty check, support ticket).

## Provided Context
{context}
"""

RAG_CITATION_PROMPT = """\
Answer the question using ONLY the provided context.
Include inline citations like [Source 1], [Source 2].
End with a "**Sources:**" section listing each source used.

Context:
{context}

Question: {question}
"""


async def _run_rag(query: str) -> tuple[str, list[dict]]:
    """Returns (formatted_answer_str, sources_list)."""
    retriever = get_retriever()
    docs: list[Document] = await retriever.ainvoke(query)

    if not docs:
        return "I couldn't find relevant information in the documentation.", []

    context_str = "\n\n".join(
        f"[Source {i+1}] ({doc.metadata.get('source','?')} | {doc.metadata.get('section_title','?')})\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context_str)),
        HumanMessage(content=query),
    ]

    response = await llm.ainvoke(messages)
    sources = [
        {
            "source": d.metadata.get("source", "N/A"),
            "section": d.metadata.get("section_title", "N/A"),
        }
        for d in docs
    ]
    return response.content, sources


# ---------------------------------------------------------------------------
# LangGraph Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: dict
    sources: list[dict]
    final_answer: str


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

TOOLS = [check_order_status, check_warranty_status, create_support_ticket]
TOOL_NODE = ToolNode(TOOLS)


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
1. PROACTIVE: If the user asks about a product and you have its serial number above, \
   call check_warranty_status immediately — do NOT ask for it.
2. MEMORY: Do not ask for info the user already gave in this conversation.
3. RAG FIRST: Use lookup_documentation for all product/policy questions before \
   calling other tools.
4. ESCALATE ONLY IF NEEDED: Use create_support_ticket only when documentation \
   fails or user explicitly asks for a human.
5. FORMAT: Use Markdown. Be concise. Offer the logical next action at the end.

Available tools: check_order_status, check_warranty_status, create_support_ticket.
For documentation questions, respond directly using your knowledge — \
your context already includes the retrieved documents.
"""


async def call_model(state: AgentState) -> AgentState:
    """Main LLM node: generates next response or tool call."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    sys_msg = SystemMessage(content=_build_system_message(state["user_profile"]))
    messages = [sys_msg] + state["messages"]

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: if last message has tool calls → tools node, else → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
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
    workflow.add_node("tools", TOOL_NODE)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    _graph = workflow.compile()
    return _graph


# ---------------------------------------------------------------------------
# Streaming entry point (used by FastAPI)
# ---------------------------------------------------------------------------

async def stream_agent_response(
    user_message: str,
    chat_history: list[dict],
    user_profile: dict,
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
            retriever = get_retriever()
            docs = await retriever.ainvoke(user_message)
            sources = [
                {"source": d.metadata.get("source", "N/A"),
                 "section": d.metadata.get("section_title", "N/A")}
                for d in docs
            ]

            if not docs:
                yield f"data: {json.dumps({'type': 'token', 'data': 'I could not find relevant documentation for your query. Would you like me to raise a support ticket?'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'data': {'answer': '', 'sources': []}})}\n\n"
                return

            context_str = "\n\n".join(
                f"[Source {i+1}] ({d.metadata.get('source','?')} | {d.metadata.get('section_title','?')})\n{d.page_content}"
                for i, d in enumerate(docs)
            )

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

        except Exception as e:
            logger.exception("RAG stream error")
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"
            return

    # For tool-using queries, run the LangGraph agent
    initial_state: AgentState = {
        "messages": lc_history + [HumanMessage(content=user_message)],
        "user_profile": user_profile,
        "sources": [],
        "final_answer": "",
    }

    graph = get_graph()
    full_answer = ""

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'data': {'name': event['name'], 'input': str(event['data'].get('input', ''))}})}\n\n"

            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                yield f"data: {json.dumps({'type': 'tool_end', 'data': {'name': event['name'], 'output': str(output)}})}\n\n"
                # Overwrite full_answer with tool result for done event
                full_answer = str(output)

        yield f"data: {json.dumps({'type': 'done', 'data': {'answer': full_answer, 'sources': []}})}\n\n"

    except Exception as e:
        logger.exception("Agent stream error")
        yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"
