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

MAX_TOOL_ROUNDS = 30  # default; overridable per build (llm.max_tool_rounds)


def _append(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Message], _append]
    # The model's reasoning, anchored to the message index it produced. Kept
    # out of `messages` so it is never fed back to the model, only shown and
    # logged (§4.2 context firewall).
    thinking: Annotated[list[dict], _append]
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
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    on_activity: Callable[[str], None] | None = None,
):
    render_system_prompt = system_prompt_fn or orchestrator_system_prompt
    # A turn is silent for seconds or minutes; this is what the TUI's spinner
    # names, so a wait is legible as thinking or as a particular tool running.
    report = on_activity or (lambda activity: None)

    async def orchestrator(state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        if rounds >= max_tool_rounds:
            # Budget spent: don't throw away what the tools found. One last
            # call with NO tools forces the model to answer from the results
            # it already has (it often has the answer and just kept digging).
            return await _summarise_and_stop(state)
        system: Message = {"role": "system", "content": render_system_prompt()}
        report("thinking")
        try:
            decision = await decide(
                llm,
                [system] + list(state.get("messages", [])),
                tools,
                max_retries=max_retries,
            )
        except DecisionError as e:
            return _final(f"I failed to produce a valid action: {e}")
        # This decision produces the next message, whether it is the answer
        # below or the tool result execute_tool appends.
        thinking = _thinking(state, decision.reasoning)
        if isinstance(decision, DirectResponse):
            return _final(decision.text) | thinking
        return {
            "pending_tool": {
                "tool": decision.tool.name,
                "arguments": decision.arguments.model_dump(),
            },
            "tool_rounds": rounds + 1,
            **thinking,
        }

    async def execute_tool(state: AgentState) -> dict:
        pending = state["pending_tool"]
        assert pending is not None
        tool = tools.get(pending["tool"])
        report(f"running {tool.name}")
        arguments = tool.params.model_validate(pending["arguments"])
        context = ctx() if callable(ctx) else ctx  # per-session context provider
        if context is not None:
            # so a tool's own model calls are logged under its name
            context.current_tool = tool.name
        if tool.gates(arguments, context):
            payload = {
                "tool": tool.name,
                "arguments": pending["arguments"],
                "description": tool.description,
            }
            if tool.describe_call is not None:
                payload["details"] = tool.describe_call(arguments, context)
            verdict = interrupt(payload)
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
        try:
            output = await tool.handler(arguments, context)
            content = f"[tool result] {tool.name}: {output}"
        except Exception as e:  # surfaced to the model, never crashes the graph
            content = f"[tool error] {tool.name}: {type(e).__name__}: {e}"
        return _tool_message(content)

    def _thinking(state: AgentState, reasoning: str) -> dict:
        if not reasoning.strip():
            return {}
        return {
            "thinking": [
                {"after": len(state.get("messages", [])), "reasoning": reasoning}
            ]
        }

    async def _summarise_and_stop(state: AgentState) -> dict:
        """Answer from the tool results already gathered, tools withdrawn.

        Uses the empty-registry branch of `decide`, so the model can only
        respond — the same firewalled path, just with nothing left to call.
        """
        budget_note = {
            "role": "user",
            "content": (
                f"[tool budget: you have used all {max_tool_rounds} tool calls "
                "for this turn]\nDo not ask to call more tools. Answer the "
                "user now from what the tool results above already show — state "
                "the finding if you have it, or say plainly what is still "
                "missing and what single next step would get it."
            ),
        }
        system: Message = {"role": "system", "content": render_system_prompt()}
        report("thinking")
        try:
            decision = await decide(
                llm,
                [system] + list(state.get("messages", [])) + [budget_note],
                ToolRegistry(),  # no tools: respond-only
                max_retries=max_retries,
            )
        except DecisionError:
            return _final(
                f"I used all {max_tool_rounds} tool calls this turn without a "
                "clean finish. Tell me how to proceed."
            )
        text = decision.text if isinstance(decision, DirectResponse) else (
            f"I used all {max_tool_rounds} tool calls this turn. Tell me how "
            "to proceed."
        )
        return _final(text) | _thinking(state, decision.reasoning)

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
    thinking: list[dict] = field(default_factory=list)
    # Index of the first message this turn appended: everything from here on
    # is new, which is what the session log needs and history does not.
    first_new: int = 0


async def deliver_event(graph, *, session_id: str, text: str) -> None:
    """Append a system event to a session thread without running the model.

    The cheap half of event delivery: ``aupdate_state`` writes the message
    into the checkpointed thread, so the agent sees it on its next turn at no
    cost. Used when reacting immediately is not wanted — the session is not
    open, or a turn is already in flight. Reacting immediately is just
    ``run_turn(..., user_text=text)`` on the same thread instead; both are the
    same LangGraph primitive, a new input on an existing ``thread_id``.

    The role is "user" because that is how every other machine-generated
    message reaches this model — tool results included — and a 27B model
    follows the shape it has already seen far more reliably than a new one.
    """
    config = {"configurable": {"thread_id": session_id}}
    await graph.aupdate_state(config, {"messages": [{"role": "user", "content": text}]})


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
    before = await graph.aget_state(config)
    first_new = len((before.values or {}).get("messages", []))
    result = await graph.ainvoke(payload, config)
    messages = result.get("messages", [])
    thinking = result.get("thinking", [])
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        return TurnResult(
            reply=None,
            interrupt=interrupts[0].value,
            messages=messages,
            thinking=thinking,
            first_new=first_new,
        )
    reply = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), None
    )
    return TurnResult(
        reply=reply,
        interrupt=None,
        messages=messages,
        thinking=thinking,
        first_new=first_new,
    )
