"""The checkpointed orchestrator loop (§4.1, §4.2).

Graph shape:

    START → orchestrator ──(pending tool?)──→ execute_tool ─→ orchestrator …
                        └──(direct answer / cap / failure)──→ END

State is plain OpenAI-style message dicts so it round-trips through the
checkpointer and straight into the LLM client without conversion. Tool calls
are stored by name + validated arguments (JSON), never as live objects.
Destructive tools pause at ``interrupt()`` with the exact operation; the TUI
resumes with ``Command(resume={"approved": bool, "reason": str})`` (§5.3) —
the reason being what the user typed when refusing, empty when they did not.
"""

from __future__ import annotations

import inspect
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
    denied_message,
    destructive_approval_required,
    mode_prompt_suffix,
    requires_execution_approval,
    script_preview,
    skipped_message,
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
# continue_nudge_for). Bounded so a model that only ever narrates cannot loop
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


def _resolve(provider, thread_id):
    """Resolve a per-session dependency for the running ``thread_id``.

    Backward-compatible with the old contract (a plain value, or a zero-arg
    callable resolved per decision) AND the new one (a one-arg provider that
    takes the invoked thread_id). A non-callable is returned as-is; a callable
    is passed the thread_id only if its signature accepts a positional arg, so
    existing zero-arg test callables keep working unchanged."""
    if not callable(provider):
        return provider
    try:
        takes_arg = len(inspect.signature(provider).parameters) >= 1
    except (TypeError, ValueError):
        takes_arg = False
    return provider(thread_id) if takes_arg else provider()


def _describe(tool, arguments, context) -> str:
    """The tool's own description of this call — resolved paths, flagged
    commands — or "" when it has none.

    Best-effort: this runs for every call, not only gated ones (the record the
    chat reads is built from it too), and a tool that cannot describe a call
    must not be the reason the call does not happen.
    """
    if tool.describe_call is None:
        return ""
    try:
        return tool.describe_call(arguments, context) or ""
    except Exception:
        return ""


def _thread_id(config) -> str | None:
    """The thread_id (session_id) the graph was invoked with, from the node's
    LangGraph ``config``. Every per-session dependency resolves off this — the
    graph only ever reads the session it is running, never an app global."""
    return (config or {}).get("configurable", {}).get("thread_id")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Message], _append]
    # The model's reasoning, anchored to the message index it produced. Kept
    # out of `messages` so it is never fed back to the model, only shown and
    # logged (§4.2 context firewall).
    thinking: Annotated[list[dict], _append]
    # What each tool was actually called with — {"after", "tool", "arguments",
    # and optionally "script"/"details"} — anchored to its result's message
    # index. Same firewall and the same reason: the model made the call, so
    # echoing the script back would spend the window on it twice; the user did
    # not, and after the approval prompt is answered this is the only place the
    # script survives (see hpca.transcript.call_text).
    calls: Annotated[list[dict], _append]
    pending_tool: dict | None
    tool_rounds: int
    # The progress checklist: list of {"text": str, "done": bool}.
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
    system_prompt_fn: Callable[..., str] | None = None,
    ctx: Any = None,
    max_retries: int = 3,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    on_activity: Callable[[str, str], None] | None = None,
    max_model_len: Callable[..., int | None] | None = None,
    on_evict: Callable[[list[Message]], Any] | None = None,
    on_usage: Callable[[str, dict], None] | None = None,
    mode_fn: Callable[..., str | None] | None = None,
):
    def client(thread_id):
        """The LLM client for the session running ``thread_id``. ``llm`` may be
        a value (one client for the whole graph), a zero-arg callable, or a
        one-arg provider keyed by thread_id (per-session selection) — resolved
        per decision so two sessions' turns never share a client. Mirrors ctx."""
        return _resolve(llm, thread_id)

    # The session's interaction mode (§3.5), read per round so a mid-session
    # switch takes effect on the very next decision. None = no mode feature
    # (tests, bare graphs): behaves exactly like before.
    def mode_for(thread_id):
        return _resolve(mode_fn, thread_id)

    def window_for(thread_id):
        return _resolve(max_model_len, thread_id)

    def system_text_for(thread_id):
        # The default fallback is called zero-arg (it takes only keyword args);
        # only an app-supplied provider is resolved by thread_id.
        if system_prompt_fn is None:
            return orchestrator_system_prompt()
        return _resolve(system_prompt_fn, thread_id)

    # A turn is silent for seconds or minutes; this is what the TUI's spinner
    # names, so a wait is legible as the LLM processing or as a particular
    # tool running. Two-arg (thread_id, activity): the app routes it to the
    # right session's spinner, since a background turn must not touch the
    # visible chat.
    report = on_activity or (lambda *a: None)
    # The backend's own token count for each decision — what the context
    # meter shows. Reported per round, not per turn, because a tool-heavy
    # turn grows the prompt as it goes and that is exactly what fills a 32k
    # window. Two-arg (thread_id, usage) for the same per-session reason.
    report_usage = on_usage or (lambda *a: None)

    def _view(state: AgentState) -> list[Message]:
        """The history as the model sees it: folded once compacted."""
        messages = list(state.get("messages", []))
        compacted = state.get("compacted")
        if not compacted:
            return messages
        return [compacted["summary"]] + messages[compacted["upto"] :]

    async def _maybe_compact(state: AgentState, thread_id) -> dict:
        """Fold the older history into a summary before it overflows.

        Returns the state update (empty when nothing was compacted). The
        messages about to leave the model's view are handed to ``on_evict``
        first: the moment before context is dropped is the last chance to
        extract a durable learning from it.
        """
        view = _view(state)
        if not compact.should_compact(view, max_model_len=window_for(thread_id)):
            return {}
        messages = list(state.get("messages", []))
        already = (state.get("compacted") or {}).get("upto", 0)
        older, _ = compact.split(messages[already:])
        if not older:
            return {}
        report(thread_id, "compacting context")
        previous = (state.get("compacted") or {}).get("summary")
        # Fold the previous summary in too, so a second compaction carries
        # the early session forward instead of forgetting it.
        to_summarize = ([previous] if previous else []) + older
        try:
            summary = await compact.summarize(client(thread_id), to_summarize)
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

    async def orchestrator(state: AgentState, config) -> dict:
        thread_id = _thread_id(config)
        rounds = state.get("tool_rounds", 0)
        if max_tool_rounds > 0 and rounds >= max_tool_rounds:
            # Budget spent: don't throw away what the tools found. One last
            # call with NO tools forces the model to answer from the results
            # it already has (it often has the answer and just kept digging).
            # A non-positive budget means no cap — this branch never fires.
            return await _summarise_and_stop(state, thread_id)
        compaction = await _maybe_compact(state, thread_id)
        if compaction:
            state = {**state, **compaction}
        mode = mode_for(thread_id)
        system: Message = {
            "role": "system",
            "content": _system_text(state, mode, thread_id),
        }
        # A bare chat reply that only announces the next step ("Let me dig into
        # the source…") is the model narrating instead of acting, so feed it
        # back with a nudge and let it retry (see continue_nudge_for).
        # Bounded, and the fumbled narration is never
        # persisted — the same shape as decide()'s validation-retry.
        nudges: list[Message] = []
        for attempt in range(MAX_CONTINUE_NUDGES + 1):
            report(thread_id, "LLM processing")
            try:
                decision = await decide(
                    client(thread_id),
                    [system] + _view(state) + nudges,
                    tools,
                    max_retries=max_retries,
                )
            except DecisionError as e:
                return _final(f"I failed to produce a valid action: {e}") | compaction
            report_usage(thread_id, decision.usage)
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
            return {
                "pending_tool": {
                    "tool": decision.tool.name,
                    "arguments": decision.arguments.model_dump(),
                },
                "tool_rounds": rounds + 1,
                **thinking,
                **compaction,
            }

    async def execute_tool(state: AgentState, config) -> dict:
        thread_id = _thread_id(config)
        pending = state["pending_tool"]
        assert pending is not None
        tool = tools.get(pending["tool"])
        report(thread_id, f"running {tool.name}")
        arguments = tool.params.model_validate(pending["arguments"])
        context = _resolve(ctx, thread_id)  # per-session context provider
        if context is not None:
            # so a tool's own model calls are logged under its name
            context.current_tool = tool.name
        mode = mode_for(thread_id)
        # What this call was: recorded for every tool, not only the gated ones,
        # so the chat can show the script and the command after the fact (the
        # approval prompt is gone the moment it is answered, and in auto mode
        # there was never one). Built here because both readings — the prompt
        # below and the record — must describe the same call.
        preview = script_preview(tool.name, pending["arguments"], context)
        details = _describe(tool, arguments, context)
        call = {
            "after": len(state.get("messages", [])),
            "tool": tool.name,
            "arguments": pending["arguments"],
        }
        if preview:
            call["script"] = preview
        if details:
            call["details"] = details
        recorded = {"calls": [call]}
        # Full-auto is the one mode that waives the destructive gate (§3.5);
        # every other mode keeps §5.3's "always ask".
        destructive = tool.gates(arguments, context) and destructive_approval_required(
            mode
        )
        # Manual mode gates execution tools too (§3.5) — same
        # interrupt/resume machinery, a different question to the user.
        execution = not destructive and requires_execution_approval(mode, tool.name)
        if destructive or execution:
            payload = {
                "tool": tool.name,
                "arguments": pending["arguments"],
                "description": tool.description,
                "kind": "destructive" if destructive else "execution",
            }
            if preview:
                payload["script"] = preview
            if details:
                payload["details"] = details
            verdict = interrupt(payload)
            approved = (
                bool(verdict.get("approved"))
                if isinstance(verdict, dict)
                else bool(verdict)
            )
            # What the user typed into the "why not" box, when they gave one.
            # It rides back with the refusal so the next attempt can be a
            # corrected one rather than the same call in another shape.
            reason = (
                str(verdict.get("reason", "")) if isinstance(verdict, dict) else ""
            )
            if not approved:
                # Refused calls are recorded like any other: what the user
                # turned down is exactly what they may want to read again.
                if execution:
                    return _tool_message(skipped_message(tool.name, reason)) | recorded
                return _tool_message(denied_message(tool.name, reason)) | recorded
        try:
            output = await tool.handler(arguments, context)
            content = f"[tool result] {tool.name}: {output}"
        except Exception as e:  # surfaced to the model, never crashes the graph
            return _tool_message(
                f"[tool error] {tool.name}: {type(e).__name__}: {e}"
            ) | recorded
        update = _tool_message(content) | recorded
        if tool.name == "update_plan":
            # The checklist lives in the checkpointed state, not in the tool:
            # that is what makes it survive restarts and prompt re-injection.
            update["plan"] = [step.model_dump() for step in arguments.steps]
        return update

    def _system_text(state: AgentState, mode: str | None, thread_id) -> str:
        """This round's system prompt: the app's render plus the mode rules
        and the current plan. Appended per round, never only once, so the
        constraint survives compaction and mode switches apply immediately.

        Rendered for the running session (``thread_id``) so a background turn
        uses its own profile's memories, not the session on screen."""
        text = system_text_for(thread_id)
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

    async def _summarise_and_stop(state: AgentState, thread_id) -> dict:
        """Answer from the tool results already gathered, tools withdrawn.

        Uses the empty-registry branch of `decide`, so the model can only
        respond — the same firewalled path, just with nothing left to call.
        """
        # This call is the one that rescues the turn's findings, so it must
        # not be the one that overflows: compact first if the accumulated
        # tool results have pushed the history over the line.
        compaction = await _maybe_compact(state, thread_id)
        if compaction:
            state = {**state, **compaction}
        mode = mode_for(thread_id)
        system: Message = {
            "role": "system",
            "content": _system_text(state, mode, thread_id),
        }
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
        report(thread_id, "LLM processing")
        try:
            decision = await decide(
                client(thread_id),
                [system] + _view(state) + [budget_note],
                ToolRegistry(),  # no tools: respond-only
                max_retries=max_retries,
            )
        except DecisionError:
            return _final(
                f"I used all {max_tool_rounds} tool calls this turn without a "
                "clean finish. Tell me how to proceed."
            ) | compaction
        report_usage(thread_id, decision.usage)
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
    # What each tool was called with, anchored to its result (AgentState.calls).
    calls: list[dict] = field(default_factory=list)
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


async def compact_now(
    graph, *, session_id: str, llm, guidance: str | None = None
) -> dict | None:
    """Fold a session's history on the user's say-so (``/compact``).

    The automatic fold in the orchestrator waits for the window to fill and
    keeps a recent tail verbatim, because it fires unasked and mid-task. This
    one is asked for, so it folds *everything* not already folded: the user
    wants the room now, and the point of asking is to decide the moment
    yourself rather than have it happen mid-turn.

    ``guidance`` is the text the user typed after the command — what the
    summary must carry, or the next step it should be written for. It reaches
    the summarizer and stays in the folded view (see :mod:`hpca.agent.compact`).

    Returns ``{"folded": n, "upto": int, "summary": Message}``, or None when
    there is nothing new to fold. The stored history is never rewritten, here
    as in the automatic path: only the view the model receives changes, so the
    user can still scroll back to every message behind the summary.
    """
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    messages = list(values.get("messages", []))
    already = (values.get("compacted") or {}).get("upto", 0)
    older = messages[already:]
    if not older:
        return None
    previous = (values.get("compacted") or {}).get("summary")
    # Fold the previous summary in too, so compacting twice carries the early
    # session forward instead of forgetting it.
    summary = await compact.summarize(
        llm, ([previous] if previous else []) + older, guidance=guidance
    )
    compacted = {"upto": len(messages), "summary": summary}
    # Written as START, the way every out-of-band write enters this graph: no
    # node produced it. Naming the node explicitly is not optional — LangGraph
    # otherwise infers it from the last node that wrote, which is ambiguous on
    # a thread whose most recent write was itself an external one.
    await graph.aupdate_state(config, {"compacted": compacted}, as_node=START)
    return {"folded": len(older), **compacted}


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

    A fold that reached past ``keep`` is pulled back with the messages: the
    compaction marker is an index into this list, and one left pointing beyond
    its end would make the model's view (``[summary] + messages[upto:]``) skip
    everything typed afterwards.
    """
    config = {"configurable": {"thread_id": session_id}}
    update: dict = {"messages": {TRUNCATE_TO: keep}}
    before = await graph.aget_state(config)
    compacted = (before.values or {}).get("compacted")
    if compacted and compacted["upto"] > keep:
        update["compacted"] = {**compacted, "upto": keep}
    await graph.aupdate_state(config, update)
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
    calls = result.get("calls", [])
    interrupts = result.get("__interrupt__") or []
    plan = result.get("plan")
    if interrupts:
        return TurnResult(
            reply=None,
            interrupt=interrupts[0].value,
            messages=messages,
            thinking=thinking,
            calls=calls,
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
        calls=calls,
        plan=plan,
        first_new=first_new,
    )
