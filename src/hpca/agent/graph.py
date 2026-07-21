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

from hpca.agent import compact
from hpca.agent.middleware import (
    DecisionError,
    DirectResponse,
    ToolCall,
    decide,
)
from hpca.agent.modes import (
    continue_nudge_for,
    destructive_approval_required,
    mode_prompt_suffix,
    present_plan_reply,
    requires_execution_approval,
    script_preview,
    skipped_message,
    tools_for_mode,
)
from hpca.agent.prompts import orchestrator_system_prompt
from hpca.agent.tools import ToolRegistry
from hpca.llm import Message

# Fallback per-turn tool budget for callers that do not pass one. The app
# passes llm.max_tool_rounds, whose default is -1 (no cap); a non-positive
# value here means the same — the agent works until it is done.
MAX_TOOL_ROUNDS = 30
# A turn never ends on a bare chat reply that only narrates the next step: the
# reply is fed back with a nudge and the model tries again (see
# continue_nudge_for — plan mode nudges any chat reply, other modes only a
# deferred action). Bounded so a model that only ever narrates cannot loop
# forever — after this many nudges the turn ends with whatever it said.
MAX_CONTINUE_NUDGES = 2


# Sentinel for rolling an interrupted turn out of the thread: the messages
# reducer is append-only, so an update cannot otherwise shrink the history.
# ``aupdate_state(config, {"messages": {TRUNCATE_TO: n}})`` keeps the first n.
TRUNCATE_TO = "__truncate_to__"


def _append(left: list, right) -> list:
    if isinstance(right, dict) and TRUNCATE_TO in right:
        return left[: right[TRUNCATE_TO]]
    return left + right


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Message], _append]
    # The model's reasoning, anchored to the message index it produced. Kept
    # out of `messages` so it is never fed back to the model, only shown and
    # logged (§4.2 context firewall).
    thinking: Annotated[list[dict], _append]
    pending_tool: dict | None
    tool_rounds: int
    # The plan-mode checklist (§3.5): list of {"text": str, "done": bool}.
    # Checkpointed with the thread, re-injected into the system prompt every
    # round, replaced wholesale by each update_plan call.
    plan: list[dict] | None
    # Compaction (redesign Phase 6): {"upto": int, "summary": Message}. The
    # stored history is never rewritten — the transcript keeps everything and
    # only the view sent to the model is folded, so compaction can never lose
    # what the user can still scroll back to.
    compacted: dict | None


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
    max_model_len: Callable[[], int | None] | None = None,
    on_evict: Callable[[list[Message]], Any] | None = None,
    on_usage: Callable[[dict], None] | None = None,
    mode_fn: Callable[[], str | None] | None = None,
):
    render_system_prompt = system_prompt_fn or orchestrator_system_prompt

    def client():
        """The LLM client to use right now. ``llm`` may be a value (one client
        for the whole graph) or a callable provider (per-session selection,
        resolved per decision from whichever turn is running) — mirrors ctx."""
        return llm() if callable(llm) else llm

    # The session's interaction mode (§3.5), read per round so a mid-session
    # switch takes effect on the very next decision. None = no mode feature
    # (tests, bare graphs): behaves exactly like before.
    current_mode = mode_fn or (lambda: None)
    # A turn is silent for seconds or minutes; this is what the TUI's spinner
    # names, so a wait is legible as the LLM processing or as a particular
    # tool running.
    report = on_activity or (lambda activity: None)
    window = max_model_len or (lambda: None)
    # The backend's own token count for each decision — what the context
    # meter shows. Reported per round, not per turn, because a tool-heavy
    # turn grows the prompt as it goes and that is exactly what fills a 32k
    # window.
    report_usage = on_usage or (lambda usage: None)

    def _view(state: AgentState) -> list[Message]:
        """The history as the model sees it: folded once compacted."""
        messages = list(state.get("messages", []))
        compacted = state.get("compacted")
        if not compacted:
            return messages
        return [compacted["summary"]] + messages[compacted["upto"] :]

    async def _maybe_compact(state: AgentState) -> dict:
        """Fold the older history into a summary before it overflows.

        Returns the state update (empty when nothing was compacted). The
        messages about to leave the model's view are handed to ``on_evict``
        first: the moment before context is dropped is the last chance to
        extract a durable learning from it.
        """
        view = _view(state)
        if not compact.should_compact(view, max_model_len=window()):
            return {}
        messages = list(state.get("messages", []))
        already = (state.get("compacted") or {}).get("upto", 0)
        older, _ = compact.split(messages[already:])
        if not older:
            return {}
        report("compacting context")
        previous = (state.get("compacted") or {}).get("summary")
        # Fold the previous summary in too, so a second compaction carries
        # the early session forward instead of forgetting it.
        to_summarize = ([previous] if previous else []) + older
        try:
            summary = await compact.summarize(client(), to_summarize)
        except Exception:
            # Best-effort: an oversized prompt is still better than a turn
            # that cannot run at all.
            return {}
        if on_evict is not None:
            try:
                await on_evict(older)
            except Exception:
                pass  # extraction is best-effort too
        return {"compacted": {"upto": already + len(older), "summary": summary}}

    async def orchestrator(state: AgentState) -> dict:
        rounds = state.get("tool_rounds", 0)
        if max_tool_rounds > 0 and rounds >= max_tool_rounds:
            # Budget spent: don't throw away what the tools found. One last
            # call with NO tools forces the model to answer from the results
            # it already has (it often has the answer and just kept digging).
            # A non-positive budget means no cap — this branch never fires.
            return await _summarise_and_stop(state)
        compaction = await _maybe_compact(state)
        if compaction:
            state = {**state, **compaction}
        mode = current_mode()
        # Plan mode withdraws execution tools from the offer itself — a tool
        # the model is never shown is one it cannot call (§3.5).
        active_tools = tools_for_mode(tools, mode)
        system: Message = {"role": "system", "content": _system_text(state, mode)}
        # A bare chat reply that only announces the next step ("Let me dig into
        # the source…") is the model narrating instead of acting, so feed it
        # back with a nudge and let it retry — in every mode (plan mode nudges
        # any chat reply, other modes only a deferred action; see
        # continue_nudge_for). Bounded, and the fumbled narration is never
        # persisted — the same shape as decide()'s validation-retry.
        nudges: list[Message] = []
        for attempt in range(MAX_CONTINUE_NUDGES + 1):
            report("LLM processing")
            try:
                decision = await decide(
                    client(),
                    [system] + _view(state) + nudges,
                    active_tools,
                    max_retries=max_retries,
                )
            except DecisionError as e:
                return _final(f"I failed to produce a valid action: {e}") | compaction
            report_usage(decision.usage)
            # This decision produces the next message, whether it is the answer
            # below or the tool result execute_tool appends.
            thinking = _thinking(state, decision.reasoning)
            if isinstance(decision, DirectResponse):
                nudge = (
                    continue_nudge_for(mode, decision.text)
                    if attempt < MAX_CONTINUE_NUDGES
                    else None
                )
                if nudge is not None:
                    nudges = nudges + [
                        {"role": "assistant", "content": decision.text},
                        {"role": "user", "content": nudge},
                    ]
                    continue
                return _final(decision.text) | thinking | compaction
            if decision.tool.name == "present_plan":
                # The explicit end of a planning turn: record the checklist and
                # answer with the plan; the TUI then offers it for approval.
                return _present_plan(decision.arguments) | thinking | compaction
            return {
                "pending_tool": {
                    "tool": decision.tool.name,
                    "arguments": decision.arguments.model_dump(),
                },
                "tool_rounds": rounds + 1,
                **thinking,
                **compaction,
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
        mode = current_mode()
        # Full-auto is the one mode that waives the destructive gate (§3.5);
        # every other mode keeps §5.3's "always ask".
        destructive = tool.gates(arguments, context) and destructive_approval_required(
            mode
        )
        # Manual and plan modes gate execution tools too (§3.5) — same
        # interrupt/resume machinery, a different question to the user.
        execution = not destructive and requires_execution_approval(mode, tool.name)
        if destructive or execution:
            payload = {
                "tool": tool.name,
                "arguments": pending["arguments"],
                "description": tool.description,
                "kind": "destructive" if destructive else "execution",
            }
            preview = script_preview(tool.name, pending["arguments"], context)
            if preview:
                payload["script"] = preview
            if tool.describe_call is not None:
                payload["details"] = tool.describe_call(arguments, context)
            verdict = interrupt(payload)
            approved = (
                bool(verdict.get("approved"))
                if isinstance(verdict, dict)
                else bool(verdict)
            )
            if not approved:
                if execution:
                    return _tool_message(skipped_message(tool.name))
                return _tool_message(
                    f"[tool result] {tool.name}: DENIED by the user — "
                    "the operation was not executed."
                )
        try:
            output = await tool.handler(arguments, context)
            content = f"[tool result] {tool.name}: {output}"
        except Exception as e:  # surfaced to the model, never crashes the graph
            return _tool_message(f"[tool error] {tool.name}: {type(e).__name__}: {e}")
        update = _tool_message(content)
        if tool.name == "update_plan":
            # The checklist lives in the checkpointed state, not in the tool:
            # that is what makes it survive restarts and prompt re-injection.
            update["plan"] = [step.model_dump() for step in arguments.steps]
        return update

    def _system_text(state: AgentState, mode: str | None) -> str:
        """This round's system prompt: the app's render plus the mode rules
        and the current plan. Appended per round, never only once, so the
        constraint survives compaction and mode switches apply immediately."""
        text = render_system_prompt()
        suffix = mode_prompt_suffix(mode, state.get("plan"))
        return f"{text}\n\n{suffix}" if suffix else text

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
        # This call is the one that rescues the turn's findings, so it must
        # not be the one that overflows: compact first if the accumulated
        # tool results have pushed the history over the line.
        compaction = await _maybe_compact(state)
        if compaction:
            state = {**state, **compaction}
        mode = current_mode()
        system: Message = {"role": "system", "content": _system_text(state, mode)}
        # Plan mode never trails off in chat, not even out of budget: hand the
        # plan over instead. Offer only present_plan so the model finishes the
        # turn the one way it should — the best plan it has now, unknowns as
        # open questions — and the TUI still gets a checklist to approve.
        if mode == "plan" and "present_plan" in tools.names():
            note = {
                "role": "user",
                "content": (
                    f"[tool budget: you have used all {max_tool_rounds} "
                    "look-around steps this turn]\nStop investigating and hand "
                    "the plan over now: call present_plan with the best "
                    "checklist you can from what you have already found, and "
                    "put anything still uncertain in open_questions. Do not ask "
                    "to read more."
                ),
            }
            report("LLM processing")
            try:
                decision = await decide(
                    client(),
                    [system] + _view(state) + [note],
                    tools.subset(["present_plan"]),
                    max_retries=max_retries,
                )
            except DecisionError:
                return _final(
                    f"I used all {max_tool_rounds} look-around steps this turn "
                    "without finishing the plan. Tell me how to proceed."
                ) | compaction
            report_usage(decision.usage)
            thinking = _thinking(state, decision.reasoning)
            if isinstance(decision, ToolCall) and decision.tool.name == "present_plan":
                return _present_plan(decision.arguments) | thinking | compaction
            text = (
                decision.text
                if isinstance(decision, DirectResponse)
                else f"I used all {max_tool_rounds} look-around steps. Tell me "
                "how to proceed."
            )
            return _final(text) | thinking | compaction
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
        report("LLM processing")
        try:
            decision = await decide(
                client(),
                [system] + _view(state) + [budget_note],
                ToolRegistry(),  # no tools: respond-only
                max_retries=max_retries,
            )
        except DecisionError:
            return _final(
                f"I used all {max_tool_rounds} tool calls this turn without a "
                "clean finish. Tell me how to proceed."
            ) | compaction
        report_usage(decision.usage)
        text = decision.text if isinstance(decision, DirectResponse) else (
            f"I used all {max_tool_rounds} tool calls this turn. Tell me how "
            "to proceed."
        )
        return _final(text) | _thinking(state, decision.reasoning) | compaction

    def _final(text: str) -> dict:
        return {
            "messages": [{"role": "assistant", "content": text}],
            "pending_tool": None,
            "tool_rounds": 0,
        }

    def _present_plan(args: Any) -> dict:
        """End a planning turn on an explicit present_plan call: answer with
        the plan and store the checklist so the TUI hands it over (§3.5).

        Empty steps (a plan-blocking question with no checklist yet) leave any
        earlier plan untouched — only a real checklist replaces it.
        """
        update = _final(present_plan_reply(args))
        if args.steps:
            update["plan"] = [step.model_dump() for step in args.steps]
        return update

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
    # The plan checklist as of the end of this turn (None when none exists).
    plan: list[dict] | None = None
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


async def thread_message_count(graph, *, session_id: str) -> int:
    """How many messages the thread holds right now (the point to roll back to
    before a turn appends to it)."""
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    return len((snapshot.values or {}).get("messages", []))


async def rollback_thread(graph, *, session_id: str, keep: int) -> list[Message]:
    """Drop everything a turn appended, keeping the first ``keep`` messages.

    Used when the user aborts a turn mid-flight (§ interrupt): the interrupted
    user message and any partial tool traffic must leave the thread so the
    re-edited prompt starts from a clean history. Returns the surviving
    messages. Relies on the TRUNCATE_TO sentinel the messages reducer honours.
    """
    config = {"configurable": {"thread_id": session_id}}
    await graph.aupdate_state(config, {"messages": {TRUNCATE_TO: keep}})
    snapshot = await graph.aget_state(config)
    return list((snapshot.values or {}).get("messages", []))


async def run_turn(
    graph,
    *,
    session_id: str,
    user_text: str | None = None,
    resume: Command | None = None,
    api_content: str | None = None,
) -> TurnResult:
    """Run one turn (new user message or interrupt resume) on a session thread.

    ``api_content`` is the message as the *model* should see it (recalled
    memory context appended); the transcript stores the clean ``user_text``
    and the LLM client substitutes the sidecar at the wire.
    """
    config = {"configurable": {"thread_id": session_id}}
    if resume is not None:
        payload: Any = resume
    else:
        message: Message = {"role": "user", "content": user_text}
        if api_content is not None:
            message["api_content"] = api_content
        payload = {
            "messages": [message],
            "pending_tool": None,
            "tool_rounds": 0,
        }
    before = await graph.aget_state(config)
    first_new = len((before.values or {}).get("messages", []))
    result = await graph.ainvoke(payload, config)
    messages = result.get("messages", [])
    thinking = result.get("thinking", [])
    interrupts = result.get("__interrupt__") or []
    plan = result.get("plan")
    if interrupts:
        return TurnResult(
            reply=None,
            interrupt=interrupts[0].value,
            messages=messages,
            thinking=thinking,
            plan=plan,
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
        plan=plan,
        first_new=first_new,
    )
