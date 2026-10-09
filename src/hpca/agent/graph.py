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
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hpca.agent import compact
from hpca.agent.history import call_message, fold_old_payloads, result_message
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
from hpca.agent.file_tools import edit_target_path
from hpca.agent.prompts import orchestrator_system_prompt
from hpca.agent.tools import ToolRegistry
from hpca.llm import STAMP_KEY, Message
from hpca.transcript import recorded_call

# Fallback per-turn tool budget for callers that do not pass one. The app
# passes llm.max_tool_rounds, whose default is -1 (no cap); a non-positive
# value here means the same — the agent works until it is done.
MAX_TOOL_ROUNDS = 30
# A turn never ends on a bare chat reply that only narrates the next step: the
# reply is fed back with a nudge and the model tries again (see
# continue_nudge_for). Bounded so a model that only ever narrates cannot loop
# forever — after this many nudges the turn ends with whatever it said.
MAX_CONTINUE_NUDGES = 2


# Sentinel for rolling a turn out of the thread (the chat rewind): the messages
# reducer is append-only, so an update cannot otherwise shrink the history.
# ``aupdate_state(config, {"messages": {TRUNCATE_TO: n}})`` keeps the first n.
TRUNCATE_TO = "__truncate_to__"

# What a stopped turn leaves behind in the thread (``stop_thread``). Addressed
# to the model, which reads it at the top of the next turn: it has to say that
# a person stopped this on purpose — otherwise the history simply looks
# unfinished and the model resumes it — without reading as an instruction to
# try the same thing again more carefully.
STOPPED_NOTE = (
    "[stopped] The user stopped this turn before it finished. Whatever was "
    "running when they stopped it did not complete, and no result for it "
    "reached this conversation. Do not pick that work back up on your own: "
    "what they say next is where they want you to go instead."
)

# The side dialog about a parked call (`answer_pending`). Addressed to the
# model in the slot the call's result will take once the user decides — the
# one position a template accepts right after a call — and saying the two
# things that keep the answer honest: nothing has run, and nothing will until
# the user says so, so there is no tool to reach for.
ASK_NOTE = (
    "[awaiting approval] {tool} has not run. The user is deciding whether to "
    "allow it and has a question about it first. Answer the question plainly "
    "and briefly: why this call, why now, what it will change, and what the "
    "alternative would be if they refuse. Do not call a tool — nothing runs "
    "until the user decides."
)
# What the model was thinking when it made the call, shown back to it under
# the note. It is the truest answer to "why" there is, and the one thing the
# conversation does not already carry: reasoning is never fed back on the main
# path. The tail, because that is where a decision gets made.
ASK_REASONING_CHARS = 4000
ASK_PREFIX = "[question about the pending call] "


def _append(left: list, right) -> list:
    if isinstance(right, dict) and TRUNCATE_TO in right:
        return left[: right[TRUNCATE_TO]]
    return left + right


def _append_messages(left: list, right) -> list:
    """`_append`, and every message that arrives without one gets a stamp.

    In the reducer because that is the one door. Messages reach a thread from
    a node returning ``{"messages": [...]}``, from `run_turn`'s opening
    payload, and from `push_event`'s `aupdate_state` — three places to forget,
    and forgetting means a chat row that cannot say when it happened.

    Already-stamped messages are left exactly as they are, which is what makes
    a fork honest: `fork_thread` re-appends the source's messages into a new
    thread, and re-dating them would make a copy of a week-old conversation
    look like it was all said just now.
    """
    if isinstance(right, dict) and TRUNCATE_TO in right:
        return left[: right[TRUNCATE_TO]]
    now = datetime.now(timezone.utc).isoformat()
    return left + [
        message if message.get(STAMP_KEY) else {**message, STAMP_KEY: now}
        for message in right
    ]


def _fold_record(upto: int, summary) -> dict:
    """The `compacted` record, stamped with the moment the fold happened.

    Stamped here rather than at either call site so the automatic path and
    `/compact` cannot disagree about the format — it is the same ISO-8601 UTC
    a message carries (`STAMP_KEY`), and the front-end is what turns it into a
    local clock. The stamp is what lets the chat say *when* the model's view
    was cut; a thread folded before it existed has none, and the row drawn for
    it says so by leaving the time off rather than inventing one.
    """
    return {
        "upto": upto,
        "summary": summary,
        STAMP_KEY: datetime.now(timezone.utc).isoformat(),
    }


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
    messages: Annotated[list[Message], _append_messages]
    # The model's reasoning, anchored to the message index it produced. Kept
    # out of `messages` so it is never fed back to the model, only shown and
    # logged (§4.2 context firewall).
    thinking: Annotated[list[dict], _append]
    # What each tool was actually called with — {"after", "tool", "arguments",
    # and optionally "script"/"details"} — anchored to the message index of the
    # call itself (the assistant message execute_tool appends, immediately
    # followed by the result). Kept out of `messages` because this is the
    # record the *user* reads, and it is never folded by age the way the
    # model's view is (hpca.agent.history.fold_old_payloads): after the
    # approval prompt is answered this is the only place the script survives,
    # and the user, who never made the call, needs it in full.
    #
    # What it does not keep is a payload the "script" block beside it already
    # carries — that would be the same lines twice in a state LangGraph
    # rewrites whole every super-step. `hpca.transcript.recorded_call` draws
    # that line, and draws it where `call_text` reads, so the record holds
    # what the chat renders and nothing else.
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


def model_view(state: dict) -> list[Message]:
    """The history as the model sees it (``build_graph``'s ``_view``).

    Module-level so a call made *about* a thread rather than *in* it —
    :func:`answer_pending` — reads the conversation exactly the way the turn
    that parked it did, and a prefix-caching backend can reuse what it
    already computed for that turn.
    """
    messages = fold_old_payloads(list(state.get("messages", [])))
    compacted = state.get("compacted")
    if not compacted:
        return messages
    return [compacted["summary"]] + messages[compacted["upto"] :]


def round_system_text(text: str, mode: str | None, plan: list[dict] | None) -> str:
    """One round's system prompt: the app's render plus the mode rules and the
    plan. Shared with :func:`answer_pending` for the reason ``model_view`` is."""
    suffix = mode_prompt_suffix(mode, plan)
    return f"{text}\n\n{suffix}" if suffix else text


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
    effort_fn: Callable[..., str | None] | None = None,
    on_step: Callable[[str, dict], None] | None = None,
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

    # The session's thinking level (hpca.thinking), read per round for the same
    # reason as the mode: /reasoning must apply from the next decision on, not
    # from the next turn. None = no dial (tests, bare graphs), and then nothing
    # about thinking is put on the wire at all.
    def effort_for(thread_id):
        return _resolve(effort_fn, thread_id)

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
    # Each tool call, and the result it returns, at the moment it happens —
    # {"kind": "call", …the call record…} then {"kind": "step", "text": …}.
    # A turn can spend minutes in tools, and an activity line saying "running
    # run_bash" does not say *what* it is running; this is what lets the chat
    # show the work as it goes instead of only once the turn is over. Two-arg
    # (thread_id, step) for the same per-session reason as the two above.
    report_step = on_step or (lambda *a: None)

    def _view(state: AgentState) -> list[Message]:
        """The history as the model sees it: folded once compacted, and with
        the payloads of all but the most recent calls described rather than
        repeated (hpca.agent.history.fold_old_payloads).

        Both folds are views over a history that is never rewritten, and they
        compose in this order for a reason: payloads go first so that what the
        summarizer reads — and what ``_maybe_compact`` measures to decide
        whether to summarize at all — is the size the window actually pays,
        not the size before the cheapest possible saving.
        """
        return model_view(state)

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
        return {"compacted": _fold_record(already + len(older), summary.message)}

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
                    effort=effort_for(thread_id),
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
                    # What the middleware had to repair to make the call valid;
                    # shown with the call rather than swallowed.
                    "repairs": decision.repairs,
                    # The backend's id for this call under the native protocol
                    # (empty otherwise). It has to be checkpointed: an approval
                    # can park the turn for as long as the user takes, and the
                    # result message is only tied to its call by this id.
                    "call_id": decision.call_id,
                },
                "tool_rounds": rounds + 1,
                **thinking,
                **compaction,
            }

    async def execute_tool(state: AgentState, config) -> dict:
        thread_id = _thread_id(config)
        pending = state["pending_tool"]
        assert pending is not None
        if pending["tool"] not in tools.names():
            # Only a user tool can vanish between the decision and this node:
            # a reload in another session, or a restart while the call was
            # parked for approval and its file no longer loads. Answered like
            # any failed call — raising here would take the turn down with it.
            content = (
                f"[tool error] {pending['tool']}: that tool is no longer "
                "loaded — the user tools changed since this call was made. "
                "Nothing ran."
            )
            report_step(thread_id, {"kind": "step", "text": content})
            return _tool_exchange(
                pending["tool"],
                pending["arguments"],
                content,
                pending.get("call_id") or "",
            )
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
        if preview is None and tool.show_call is not None:
            try:
                preview = tool.show_call(arguments, context) or None
            except Exception:  # a preview never costs the call
                preview = None
        # A repair rides with the description: it is part of what this call is,
        # and the user judging a script has to be told a line was taken out of
        # it — the preview above already shows the shortened version.
        details = "\n".join(
            part
            for part in [_describe(tool, arguments, context), *pending.get("repairs", [])]
            if part
        )
        call = {
            # The index the call's own assistant message is about to take —
            # the first of the two this node appends, so the record and the
            # message it describes share a position and the transcript can
            # render the call where it happened.
            "after": len(state.get("messages", [])),
            "tool": tool.name,
            "arguments": pending["arguments"],
        }
        if preview:
            call["script"] = preview
        if details:
            call["details"] = details
        # Stored as the chat will read it, which is not always as the model
        # made it: a payload the script block already carries whole is not
        # kept a second time here (hpca.transcript.recorded_call). The same
        # dict is what the live announcement below sends, so what the user
        # watches happen and what a re-opened session shows stay one thing.
        call = recorded_call(call)
        recorded = {"calls": [call]}
        # Full-auto is the one mode that waives the destructive gate (§3.5);
        # every other mode keeps §5.3's "always ask".
        destructive = tool.gates(arguments, context) and destructive_approval_required(
            mode
        )
        # One approval per FILE, not per diff (§3.5, auto only): once the user
        # has said yes to an edit_file on a path, further edit_file calls to
        # that same path run ungated — otherwise filling a long document
        # section by section stops for the same answer ten times. Manual mode
        # is exempt on purpose: it exists to show every call.
        if destructive and tool.name == "edit_file" and mode == "auto":
            target = edit_target_path(arguments, context)
            if target is not None and target in getattr(
                context, "approved_edit_paths", ()
            ):
                destructive = False
        # Manual mode gates execution tools too (§3.5) — same
        # interrupt/resume machinery, a different question to the user.
        execution = not destructive and requires_execution_approval(
            mode, tool.name, executes=tool.executes(arguments, context)
        )
        approved, reason = True, ""
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
            # Remember an approved edit's file so auto mode skips the gate for
            # its later edits (above). Recorded in any gating mode — approval
            # in manual carries when the user then switches to auto — but a
            # refusal records nothing: "no" means no this time, not never ask.
            if approved and destructive and tool.name == "edit_file":
                target = edit_target_path(arguments, context)
                if target is not None and hasattr(context, "approved_edit_paths"):
                    context.approved_edit_paths.add(target)
        # Announced after the gate, never before it: a parked turn must not
        # report a call the user has not answered for yet, and interrupt()
        # re-runs this node from the top on resume — announcing above would
        # then say it twice.
        report_step(thread_id, {"kind": "call", **call})
        ran = False
        if not approved:
            # Refused calls are recorded like any other: what the user turned
            # down is exactly what they may want to read again.
            content = (skipped_message if execution else denied_message)(
                tool.name, reason
            )
        else:
            try:
                output = await tool.handler(arguments, context)
                content = f"[tool result] {tool.name}: {output}"
                # A repaired call's result must SAY it was repaired — a
                # salvaged half-write that reads like a clean success leaves
                # the model believing the file is complete.
                for repair in pending.get("repairs") or []:
                    content += f"\n[repaired] {repair}"
                ran = True
            except Exception as e:  # surfaced to the model, never crashes the graph
                content = f"[tool error] {tool.name}: {type(e).__name__}: {e}"
        report_step(thread_id, {"kind": "step", "text": content})
        update = _tool_exchange(
            tool.name, pending["arguments"], content, pending.get("call_id") or ""
        ) | recorded
        if ran and tool.name == "update_plan":
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
        return round_system_text(system_text_for(thread_id), mode, state.get("plan"))

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
                effort=effort_for(thread_id),
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

    def _tool_exchange(
        tool_name: str, arguments: dict, content: str, call_id: str = ""
    ) -> dict:
        # Two messages, not one: the assistant message IS the call the model
        # emitted (its own decision envelope, big payloads elided — see
        # hpca.agent.history), and the result answers it. A history of nothing
        # but results is a run of consecutive user messages in which the model
        # has to infer its own actions from the echo in the result text; the
        # pair is the shape agent-trained models were post-trained on.
        #
        # Under the envelope protocol the result rides the user role, because
        # vLLM/Qwen templates reject mid-conversation system messages and the
        # tool role belongs to the protocol that one bypasses (§4.3). Under the
        # native protocol (``call_id`` set) it is a real tool-role message tied
        # to the call by its id. ``content`` is passed through untouched either
        # way — it is also a denial or a tool error, not only a result line.
        return {
            "messages": [
                call_message(tool_name, arguments, call_id),
                result_message(tool_name, content, call_id),
            ],
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
    # The fold record as of the end of this turn (None when nothing is folded).
    # Carried because the automatic compaction happens *inside* a turn: without
    # it the only news of a fold is the meter moving, and the caller cannot
    # tell a turn that folded from one that did not (`service.after_turn`).
    compacted: dict | None = None


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


async def propose_compaction(
    graph,
    *,
    session_id: str,
    llm,
    guidance: str | None = None,
    previous_attempt: str | None = None,
    comment: str | None = None,
) -> dict | None:
    """The fold ``/compact`` *would* make, computed and not written.

    The automatic fold in the orchestrator waits for the window to fill and
    keeps a recent tail verbatim, because it fires unasked and mid-task. This
    one is asked for, so it folds *everything* not already folded: the user
    wants the room now, and the point of asking is to decide the moment
    yourself rather than have it happen mid-turn.

    ``guidance`` is the text the user typed after the command — what the
    summary must carry, or the next step it should be written for. It reaches
    the summarizer and stays in the folded view (see :mod:`hpca.agent.compact`).
    ``previous_attempt`` and ``comment`` are a retry: the summary the user
    turned down and what they said about it.

    Nothing is written here, which is the whole point of the split — a summary
    the user has not seen yet must not already be the conversation's history.
    :func:`apply_compaction` is what lands it, once they say so.

    Returns ``{"folded": n, "upto": int, "summary": Message, "truncated":
    bool}``, or None when there is nothing new to fold. ``upto`` is a position
    in the stored history and stays meaningful while the thread grows: a turn
    that runs between the offer and the answer is simply left verbatim behind
    the summary.
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
        llm,
        ([previous] if previous else []) + older,
        guidance=guidance,
        previous=previous_attempt,
        comment=comment,
    )
    return {
        "folded": len(older),
        "upto": len(messages),
        "summary": summary.message,
        "truncated": summary.truncated,
    }


async def apply_compaction(
    graph, *, session_id: str, upto: int, summary: Message
) -> bool:
    """Land a proposed fold. False when it no longer describes this thread.

    The two ways a proposal goes stale, both from the seconds it spends on
    screen waiting for an answer: the thread was trimmed behind it (``upto``
    now points past the end of a shorter history), or something folded past
    this point in the meantime. Neither is an error worth raising — the
    conversation is intact either way, which is exactly what "nothing lands
    until the user accepts" is for — so the caller says so and stops.

    The stored history is never rewritten, here as in the automatic path: only
    the view the model receives changes, so the user can still scroll back to
    every message behind the summary.
    """
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    if upto > len(values.get("messages", [])):
        return False
    if upto <= (values.get("compacted") or {}).get("upto", 0):
        return False
    # Written as START, the way every out-of-band write enters this graph: no
    # node produced it. Naming the node explicitly is not optional — LangGraph
    # otherwise infers it from the last node that wrote, which is ambiguous on
    # a thread whose most recent write was itself an external one.
    await graph.aupdate_state(
        config, {"compacted": _fold_record(upto, summary)}, as_node=START
    )
    return True


async def compact_now(
    graph, *, session_id: str, llm, guidance: str | None = None
) -> dict | None:
    """Fold a session's history on the user's say-so, in one step.

    Propose and apply with nobody asked in between — what a front-end does
    when it has no way to hold a summary up for review, and what the tests
    drive. The interactive path is the two halves separately, with
    ``compact.proposed`` in the gap (`core.service._compact`).
    """
    proposed = await propose_compaction(
        graph, session_id=session_id, llm=llm, guidance=guidance
    )
    if proposed is None:
        return None
    await apply_compaction(
        graph,
        session_id=session_id,
        upto=proposed["upto"],
        summary=proposed["summary"],
    )
    return proposed


async def answer_pending(
    graph,
    *,
    session_id: str,
    llm,
    tools: ToolRegistry,
    system: str,
    mode: str | None = None,
    effort: str | None = None,
    turns: list[tuple[str, str]] = (),
    question: str,
    max_retries: int = 3,
) -> str:
    """The agent's answer to a question about the call it is parked on.

    Read-only, and that is the contract: nothing here writes to the thread,
    so the question and the answer never become the conversation the turn
    resumes into, and never reach a checkpoint. What the model is shown is
    the turn's own context — the same system prompt and the same view of the
    history the parked round saw — with the call it made, a note in its
    result's place saying it has not run (``ASK_NOTE``), and the questions
    asked so far with the answers given to them (``turns``).

    The first attempt offers the session's whole tool registry although no
    tool may be called, and that is on purpose: the tools are part of the
    prompt's prefix — the listing in the system message, or the backend's own
    tool channel — so offering the same ones keeps the prefix byte-identical
    to the round that parked, and a prefix-caching backend answers without
    prefilling the conversation again. A model that calls a tool anyway gets
    a second attempt with nothing to call (the respond-only branch of
    `decide`, as ``_summarise_and_stop`` uses).

    Raises LookupError when the thread is not parked on a call.
    """
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    pending = values.get("pending_tool")
    if not pending:
        raise LookupError("this session is not waiting on a call")
    tool, call_id = pending["tool"], pending.get("call_id") or ""
    note = ASK_NOTE.format(tool=tool)
    count = len(values.get("messages", []))
    reasoning = next(
        (
            str(item.get("reasoning") or "")
            for item in reversed(values.get("thinking", []) or [])
            if item.get("after") == count
        ),
        "",
    ).strip()
    if reasoning:
        note += (
            "\n\nYour reasoning when you made the call:\n"
            + reasoning[-ASK_REASONING_CHARS:]
        )
    messages: list[Message] = [
        {
            "role": "system",
            "content": round_system_text(system, mode, values.get("plan")),
        },
        *model_view(values),
        call_message(tool, pending.get("arguments") or {}, call_id),
        result_message(tool, note, call_id),
    ]
    for asked, answered in turns:
        messages.append({"role": "user", "content": ASK_PREFIX + asked})
        messages.append({"role": "assistant", "content": answered})
    messages.append({"role": "user", "content": ASK_PREFIX + question})
    for registry in (tools, ToolRegistry()):
        decision = await decide(
            llm, messages, registry, max_retries=max_retries, effort=effort
        )
        if isinstance(decision, DirectResponse):
            return decision.text
    return (
        "(The agent answered with another tool call instead of an explanation.)"
    )


async def thread_message_count(graph, *, session_id: str) -> int:
    """How many messages the thread holds right now (the point to roll back to
    before a turn appends to it)."""
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    return len((snapshot.values or {}).get("messages", []))


async def rollback_thread(graph, *, session_id: str, keep: int) -> list[Message]:
    """Drop everything after the first ``keep`` messages.

    Two callers: aborting a turn mid-flight (§ interrupt), where the
    interrupted user message and any partial tool traffic must leave the
    thread so the re-edited prompt starts from a clean history — and the chat
    rewind, where the user trims a conversation back to before it went wrong.
    Returns the surviving messages. Relies on the TRUNCATE_TO sentinel the
    reducers honour.

    Reasoning and call records anchored to trimmed messages go with them:
    anchors are message indices, so a stale one would re-attach to whatever
    future message takes that index. They are appended in anchor order, which
    is what lets a positional truncation express "every anchor past the cut".

    A fold that reached past ``keep`` is dropped outright: its summary stands
    (partly) for messages the rollback removed, and a view built from it would
    reinject exactly what the user cut away. The messages it also covered are
    still there raw, so nothing is lost — folding can be redone.
    """
    config = {"configurable": {"thread_id": session_id}}
    update: dict = {"messages": {TRUNCATE_TO: keep}}
    before = (await graph.aget_state(config)).values or {}
    for key in ("thinking", "calls"):
        anchored = before.get(key) or []
        surviving = sum(1 for record in anchored if record["after"] < keep)
        if surviving < len(anchored):
            update[key] = {TRUNCATE_TO: surviving}
    compacted = before.get("compacted")
    if compacted and compacted["upto"] > keep:
        update["compacted"] = None
    await graph.aupdate_state(config, update)
    snapshot = await graph.aget_state(config)
    return list((snapshot.values or {}).get("messages", []))


async def stop_thread(graph, *, session_id: str) -> None:
    """Close off a turn the user stopped, leaving the thread fit to run again.

    The other half of ``rollback_thread``, for the stop that keeps its work
    (`TurnScheduler.interrupt`): the messages stay where they are and what
    gets written is the fact that they stop there.

    **The note.** The exchange now ends in mid-air — a question with no
    answer, or a round of tool results with nothing said about them — and a
    model reading that back draws the obvious conclusion: something cut it
    off, so pick it up. Saying plainly what happened costs one message and
    settles it; a prompt rule about "history that looks unfinished" would have
    to be true of every turn to catch the one it is about. It rides the user
    role because that is how every machine-generated message reaches this
    model, tool results included (see ``deliver_event``).

    **The cleared ``pending_tool``.** A turn cancelled inside a tool leaves
    the checkpoint parked one step short of ``execute_tool``, still holding
    the call that never ran. Nothing replays it today — the next turn arrives
    as fresh input at START — but a thread left poised to run the very call
    the user stopped is not a thing to leave lying about. Clearing it is also
    what drops the pending task, in the case that matters: the write is
    attributed to whichever node wrote last, and for a stop inside a tool that
    is the orchestrator, whose branch is re-read against an empty
    ``pending_tool`` and routes to END. (A stop during the model call leaves
    the thread pointed at the orchestrator instead, which the next turn's
    input re-triggers anyway.) Naming START explicitly — the convention for an
    out-of-band write, see ``compact_now`` — would point *every* stopped
    thread back at the model, which is strictly worse.

    What is deliberately NOT here is a synthetic tool result closing a
    dangling call, and the reason is worth writing down because it is the
    first thing this function looks like it should do. ``execute_tool``
    appends the call and its result in ONE state update, so a cancel lands
    either before both or after both: at every point a stop can happen the
    history is already balanced, and a manufactured result would be answering
    a call that is not in the thread.
    """
    config = {"configurable": {"thread_id": session_id}}
    await graph.aupdate_state(
        config,
        {
            "messages": [{"role": "user", "content": STOPPED_NOTE}],
            "pending_tool": None,
        },
    )


async def fork_thread(
    graph, *, source_session_id: str, target_session_id: str, keep: int
) -> list[Message]:
    """Copy the first ``keep`` messages of one thread into a fresh one.

    The chat rewind's non-destructive half: the fork branches from before the
    point the conversation went wrong while the source stays whole. Values are
    copied, not checkpoints — the target must be a thread nothing has written
    to, so its reducers see the copy as the first append. Returns the copied
    messages.

    The same trimming rules as ``rollback_thread``: reasoning/call records
    anchored past the cut stay behind, and a fold reaching past it is not
    copied (its summary stands for messages the fork excludes). What is NOT
    copied on purpose: the plan checklist (it encodes the direction being
    branched away from) and the transient turn fields.
    """
    source = {"configurable": {"thread_id": source_session_id}}
    values = (await graph.aget_state(source)).values or {}
    kept = list(values.get("messages", []))[:keep]
    update: dict = {
        "messages": kept,
        "thinking": [t for t in values.get("thinking") or [] if t["after"] < keep],
        "calls": [c for c in values.get("calls") or [] if c["after"] < keep],
    }
    compacted = values.get("compacted")
    if compacted and compacted["upto"] <= keep:
        update["compacted"] = dict(compacted)
    # As START, like every out-of-band write (see compact_now): no node
    # produced it, and on a virgin thread there is no prior writer to infer.
    target = {"configurable": {"thread_id": target_session_id}}
    await graph.aupdate_state(target, update, as_node=START)
    return kept


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
    compacted = result.get("compacted")
    if interrupts:
        return TurnResult(
            reply=None,
            interrupt=interrupts[0].value,
            messages=messages,
            thinking=thinking,
            calls=calls,
            plan=plan,
            first_new=first_new,
            compacted=compacted,
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
        compacted=compacted,
    )
