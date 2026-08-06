"""Checkpointed agent state → the entries a human reads (chat window, logs).

A turn is a user message, the agent's working, and the answer. The working —
the model's reasoning, the tool calls it made and the results they returned,
interleaved in the order they happened — is folded into a single ``thinking``
entry, which the chat window shows as one collapsible box and the session log
writes as one block.

Reasoning and tool calls are anchored by the message index they produced
(``after``), never fed back to the model, and live outside ``messages`` for
exactly that reason: the model already knows what it called, and a script fed
back verbatim would cost the window twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from hpca.llm import Message

TOOL_PREFIXES = ("[tool result]", "[tool error]")
CALL_PREFIX = "[tool call]"
# The script itself, rendered as its own block below the other arguments
# rather than as a JSON list of lines nobody can read.
SCRIPT_ARG_KEYS = ("content_lines",)
# Arguments are a display aid, not a record: a pathological call must not push
# a wall of JSON into the chat (the script block has its own cap upstream).
ARGUMENTS_CHARS = 2000
# Background work reporting in on its own (§5.4). Rides the user role like
# tool results do, and is marked so the transcript does not attribute a
# process crash to the human sitting there.
EVENT_PREFIXES = ("[process ", "[job ")

USER = "user"
ASSISTANT = "assistant"
THINKING = "thinking"
ERROR = "error"
EVENT = "event"
# Memory recalled into a turn (redesign Phase 5). Shown so the user can see
# what the agent was reminded of — silent injection would make the agent's
# behavior inexplicable from the transcript alone.
RECALL = "recall"

FENCE_OPEN = "<memory-context>"
FENCE_CLOSE = "</memory-context>"


@dataclass
class Step:
    """One ordered part of a turn's working: a block of reasoning, one tool
    call, or the result it returned. The chat window reveals these as
    individually-collapsible lines when a thinking box is expanded;
    ``Entry.text`` still holds the whole box as one string, which is what the
    session log writes."""

    kind: str  # reasoning | call | step
    text: str
    tool: str = ""  # call: named here, so the label needs no parsing

    def label(self) -> str:
        """Short header for the collapsed line — the tool name for a call or a
        step (a step's is parsed from its ``[tool result] <tool>: …`` prefix),
        else "reasoning"."""
        if self.kind == "call":
            return f"{self.tool or 'tool'} (call)"
        if self.kind != "step":
            return self.kind
        for prefix in TOOL_PREFIXES:
            if self.text.startswith(prefix):
                name = self.text[len(prefix) :].strip().split(":", 1)[0].strip()
                suffix = " (error)" if prefix == "[tool error]" else ""
                return (name or "tool") + suffix
        return "step"


@dataclass
class Entry:
    kind: str  # user | assistant | thinking | error
    text: str
    steps: int = 0  # thinking: tool steps folded in
    reasoning_chars: int = 0  # thinking: how much the model thought
    # thinking: the ordered parts, kept structured so an expanded box can show
    # each one as its own collapsible element (``text`` folds them for the log).
    parts: list[Step] = field(default_factory=list)

    def summary(self) -> str:
        """One-line gist, for the collapsed box: only what is actually there."""
        parts = []
        if self.reasoning_chars:
            parts.append(f"{self.reasoning_chars:,} chars reasoning")
        if self.steps:
            parts.append(f"{self.steps} step{'' if self.steps == 1 else 's'}")
        return " · ".join(parts) or "working"


def is_tool_message(message: Message) -> bool:
    """Tool results ride the user role (§4.3); the prefix is what marks them."""
    return message["role"] == USER and str(message["content"]).startswith(
        TOOL_PREFIXES
    )


def is_event_message(message: Message) -> bool:
    """A completion the watcher delivered, not something the user typed."""
    return message["role"] == USER and str(message["content"]).startswith(
        EVENT_PREFIXES
    )


def recalled_text(message: Message) -> str:
    """What was recalled into this message, if anything.

    The fenced block lives only in the API copy (``api_content``), so the
    stored transcript keeps the user's own words; this reads it back out for
    display.
    """
    api_content = message.get("api_content")
    if not api_content or FENCE_OPEN not in api_content:
        return ""
    body = api_content.split(FENCE_OPEN, 1)[1].split(FENCE_CLOSE, 1)[0]
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("[System note:")
    ]
    return "\n".join(lines)


def call_text(call: dict) -> str:
    """One tool call as the user reads it: the tool, the arguments that say
    what it was asked to do, and the script or command it would actually run.

    The same three things the approval prompt shows for a gated call — which is
    the point: after the prompt is answered it is gone, and what ran has to
    stay somewhere the user can still open.
    """
    tool = call.get("tool") or "tool"
    script = (call.get("script") or "").strip()
    arguments = {
        key: value
        for key, value in (call.get("arguments") or {}).items()
        if not (script and key in SCRIPT_ARG_KEYS)
    }
    blocks: list[str] = []
    if arguments:
        blocks.append(_clip(json.dumps(arguments, indent=2, default=str)))
    details = (call.get("details") or "").strip()
    if details:  # resolved real paths, flagged commands (§5.3)
        blocks.append(details)
    if script:
        blocks.append(script)
    head = f"{CALL_PREFIX} {tool}:"
    return "\n".join([head, *blocks]) if blocks else head


def live_step(payload: dict) -> Step:
    """The part one in-flight announcement renders as (see the graph's
    ``on_step``): a call as it is made, or the result as it lands.

    The same rendering the finished turn gets, so a step the user opened while
    it was running reads identically once it is folded into its thinking box.
    """
    if payload.get("kind") == "call":
        return Step(
            kind="call", text=call_text(payload), tool=str(payload.get("tool") or "")
        )
    return Step(kind="step", text=str(payload.get("text", "")))


def _clip(text: str) -> str:
    if len(text) <= ARGUMENTS_CHARS:
        return text
    return text[:ARGUMENTS_CHARS] + "\n... [clipped]"


def _block(parts: list[Step]) -> str:
    return "\n\n".join(f"— {part.kind} —\n{part.text.strip()}" for part in parts)


def build_entries(
    messages: list[Message],
    thinking: list[dict] | None = None,
    calls: list[dict] | None = None,
    *,
    start: int = 0,
) -> list[Entry]:
    """Entries for ``messages[start:]``, with reasoning, tool calls and their
    results folded in.

    ``start`` selects a tail (one turn, for incremental logging) while keeping
    the absolute message indices that ``thinking`` and ``calls`` entries are
    anchored to.
    """
    reasoning_at: dict[int, list[str]] = {}
    for entry in thinking or []:
        reasoning_at.setdefault(entry["after"], []).append(entry["reasoning"])
    calls_at: dict[int, list[dict]] = {}
    for call in calls or []:
        calls_at.setdefault(call["after"], []).append(call)

    entries: list[Entry] = []
    pending: list[Step] = []  # the open thinking box
    steps = 0
    reasoning_chars = 0

    def flush() -> None:
        nonlocal pending, steps, reasoning_chars
        if pending:
            entries.append(
                Entry(
                    kind=THINKING,
                    text=_block(pending),
                    steps=steps,
                    reasoning_chars=reasoning_chars,
                    parts=list(pending),
                )
            )
        pending, steps, reasoning_chars = [], 0, 0

    def open_calls(index: int) -> None:
        """The calls made at this point, shown before the result they produced.

        A call is anchored to the index its result takes, and the two are
        written in one state update, so there is never one without the other.
        An anchor past the end of ``messages`` is a call a rolled-back turn
        left behind (see ``rollback_thread``) and is passed over here, exactly
        as an orphaned reasoning anchor is.
        """
        for call in calls_at.get(index, []):
            pending.append(
                Step(
                    kind="call",
                    text=call_text(call),
                    tool=str(call.get("tool") or ""),
                )
            )

    for index in range(start, len(messages)):
        message = messages[index]
        for reasoning in reasoning_at.get(index, []):
            if reasoning.strip():
                pending.append(Step(kind="reasoning", text=reasoning))
                reasoning_chars += len(reasoning)
        open_calls(index)
        role, content = message["role"], str(message["content"])
        if role == "system":
            continue
        if role == ASSISTANT:
            flush()  # the answer closes the box that produced it
            entries.append(Entry(kind=ASSISTANT, text=content))
        elif is_tool_message(message):
            pending.append(Step(kind="step", text=content))
            steps += 1
        elif is_event_message(message):
            flush()
            entries.append(Entry(kind=EVENT, text=content))
        else:
            flush()
            entries.append(Entry(kind=USER, text=content))
            recalled = recalled_text(message)
            if recalled:
                entries.append(Entry(kind=RECALL, text=recalled))
    flush()  # a turn interrupted for approval leaves its box open
    return entries
