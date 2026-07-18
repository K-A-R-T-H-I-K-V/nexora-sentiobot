"""
check_tool_convergence.py — Regression guard for tool-path non-convergence
(reviewer finding F1.5-1). Proves that when the model keeps requesting the SAME
tool call every turn, the graph still converges to a final answer instead of
looping until GraphRecursionError.

It uses a FAKE looping LLM and a FAKE counting tool, so it spends NO real Groq
tokens and needs no Supabase. Run:

    python -m backend.scripts.check_tool_convergence

Exit 0 = converged with dedupe working; 1 = failed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("JWT_SECRET", "convergence-test-secret")
os.environ.setdefault("GROQ_API_KEY", "gsk_not_used_by_the_fake_llm")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import backend.agent.agent as agent  # noqa: E402

_tool_exec_count = {"n": 0}


@tool
async def fake_tool(query: str = "") -> str:
    """A fake tool that counts how many times it actually executes."""
    _tool_exec_count["n"] += 1
    return "fake tool result"


class _FakeLoopingLLM:
    """When bound with tools, ALWAYS requests the same tool call (a model that
    never converges on its own). When called without tools (the finalize node),
    returns a plain text answer."""

    def __init__(self, bound: bool = False):
        self._bound = bound

    def bind_tools(self, tools):  # noqa: ANN001
        return _FakeLoopingLLM(bound=True)

    async def ainvoke(self, messages, **kwargs):  # noqa: ANN001
        if self._bound:
            return AIMessage(
                content="",
                tool_calls=[{"name": "fake_tool", "args": {"query": "x"}, "id": "call_1"}],
            )
        return AIMessage(content="Here is my final answer from the gathered information.")


async def main() -> int:
    # Inject the fakes.
    agent._llm = _FakeLoopingLLM()
    agent._TOOLS_BY_NAME = {"fake_tool": fake_tool}
    agent._graph = None  # rebuild with the current nodes

    graph = agent.get_graph()
    max_rounds = 2
    state = {
        "messages": [HumanMessage(content="please help")],
        "user_profile": {"name": "Test", "owned_products": []},
        "user_id": "test-user",
        "sources": [],
        "final_answer": "",
        "called_tools": [],
        "tool_rounds": 0,
        "max_tool_rounds": max_rounds,
    }

    failures: list[str] = []
    try:
        final = await graph.ainvoke(state, config={"recursion_limit": 2 * max_rounds + 6})
    except Exception as exc:  # GraphRecursionError would land here
        print(f"CONVERGENCE CHECK: FAIL - graph raised {type(exc).__name__}: {exc}")
        return 1

    last = final["messages"][-1]
    answer = getattr(last, "content", "")
    if _tool_exec_count["n"] != 1:
        failures.append(f"dedupe broken: fake_tool executed {_tool_exec_count['n']} times, expected 1")
    if not (isinstance(last, AIMessage) and answer.strip()):
        failures.append(f"did not finalize to a text answer (last={type(last).__name__}, content={answer!r})")
    if final.get("tool_rounds", 0) > max_rounds:
        failures.append(f"tool_rounds {final.get('tool_rounds')} exceeded max {max_rounds}")

    if failures:
        print("CONVERGENCE CHECK: FAIL")
        for f in failures:
            print("  -", f)
        return 1

    print("CONVERGENCE CHECK: PASS")
    print(f"  fake_tool executed once (dedupe held), rounds={final.get('tool_rounds')}, "
          f"finalized answer len={len(answer)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
