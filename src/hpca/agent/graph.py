"""The checkpointed orchestrator loop (§4.1, §4.2).

Graph shape:

    START → orchestrator ──(pending tool?)──→ execute_tool ─→ orchestrator …
                        └──(direct answer / cap / failure)──→ END

State is plain OpenAI-style message dicts so it round-trips through the
checkpointer and straight into the LLM client without conversion. Tool calls
are stored by name + validated arguments (JSON), never as live objects.
Destructive tools pause at ``interrupt()`` with the exact operation; the TUI
resumes with ``Command(resume={"approved": bool})`` (§5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hpca.agent.middleware import (
    DecisionError,
    DirectResponse,
    decide,
)
from hpca.agent.prompts import orchestrator_system_prompt
from hpca.agent.tools import ToolRegistry
from hpca.llm import Message

MAX_TOOL_ROUNDS = 8


def _append(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Message], _append]
    pending_tool: dict | None
    tool_rounds: int


def build_graph(
    *,
    llm: Any,
    tools: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt_fn: Callable[[], str] | None = None,
    ctx: Any = None,
    max_retries: int = 3,
):
    render_system_prompt = system_prompt_fn or orchestrator_system_prompt

    async def orchestrator(state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        if rounds >= MAX_TOOL_ROUNDS:
            note = (
                f"I stopped after {MAX_TOOL_ROUNDS} tool calls in one turn "
                "(tool budget exhausted). Tell me how to proceed."
            )
            return _final(note)
        system: Message = {"role": "system", "content": render_system_prompt()}
        try:
            decision = await decide(
                llm,
                [system] + list(state.get("messages", [])),
                tools,
                max_retries=max_retries,
            )
        except DecisionError as e:
            return _final(f"I failed to produce a valid action: {e}")
        if isinstance(decision, DirectResponse):
            return _final(decision.text)
        return {
            "pending_tool": {
                "tool": decision.tool.name,
                "arguments": decision.arguments.model_dump(),
            },
            "tool_rounds": rounds + 1,
        }

    async def execute_tool(state: AgentState) -> dict:
        pending = state["pending_tool"]
        assert pending is not None
        tool = tools.get(pending["tool"])
        if tool.destructive:
            verdict = interrupt(
                {
                    "tool": tool.name,
                    "arguments": pending["arguments"],
                    "description": tool.description,
                }
            )
            approved = (
                bool(verdict.get("approved"))
                if isinstance(verdict, dict)
                else bool(verdict)
            )
            if not approved:
                return _tool_message(
                    f"[tool result] {tool.name}: DENIED by the user — "
                    "the operation was not executed."
                )
        arguments = tool.params.model_validate(pending["arguments"])
        try:
            output = await tool.handler(arguments, ctx)
            content = f"[tool result] {tool.name}: {output}"
        except Exception as e:  # surfaced to the model, never crashes the graph
            content = f"[tool error] {tool.name}: {type(e).__name__}: {e}"
        return _tool_message(content)

    def _final(text: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": text}],
            "pending_tool": None,
            "tool_rounds": 0,
        }

    def _tool_message(content: str) -> dict:
        # Tool results use the user role: vLLM/Qwen templates reject
        # mid-conversation system messages, and the native tool role requires
        # the tool-call protocol we deliberately bypass (§4.3).
        return {
            "messages": [{"role": "user", "content": content}],
            "pending_tool": None,
        }

    def route(state: AgentState) -> str:
        return "execute_tool" if state.get("pending_tool") else END

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("execute_tool", execute_tool)
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route)
    builder.add_edge("execute_tool", "orchestrator")
    return builder.compile(checkpointer=checkpointer)


@dataclass
class TurnResult:
    reply: str | None
    interrupt: dict | None
    messages: list[Message] = field(default_factory=list)


async def run_turn(
    graph,
    *,
    session_id: str,
    user_text: str | None = None,
    resume: Command | None = None,
) -> TurnResult:
    """Run one turn (new user message or interrupt resume) on a session thread."""
    config = {"configurable": {"thread_id": session_id}}
    if resume is not None:
        payload: Any = resume
    else:
        payload = {
            "messages": [{"role": "user", "content": user_text}],
            "pending_tool": None,
            "tool_rounds": 0,
        }
    result = await graph.ainvoke(payload, config)
    messages = result.get("messages", [])
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        return TurnResult(reply=None, interrupt=interrupts[0].value, messages=messages)
    reply = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), None
    )
    return TurnResult(reply=reply, interrupt=None, messages=messages)
