"""Checkpointed agent state → the entries a human reads (chat window, logs).

A turn is a user message, the agent's working, and the answer. The working —
the model's reasoning, the tool calls it made and the results they returned,
interleaved in the order they happened — is folded into a single ``thinking``
entry, which the chat window shows as one collapsible box and the session log
writes as one block.

Reasoning and tool calls are anchored by the message index they produced
(``after``) and live outside ``messages``. The reasoning is never fed back to
the model at all; a call is, but only as the folded envelope
:mod:`hpca.agent.history` builds — a script fed back verbatim would cost the
window twice. The record here is the unfolded one, which is what the user
reads, so the message carrying the model's own copy is skipped when rendering.

What this module decides is *how much of that record is worth a human's
attention*. Two things are cut here and nowhere else:

* A call and the result it returned are ONE part, not two. They were two for as
  long as the record was written that way, which put the tool's name on the
  screen twice and left the user scrolling between a question and its answer.
  ``Step`` now holds both halves and knows whether the second one has landed.
* The framing a tool result carries for the model's benefit is taken off. A
  result says both what happened ("Created /path/x.tsv (8 lines)") and what the
  model should do next ("Change it with edit_file, not by writing it again") —
  the first is news, the second is prompt, and only the first is shown. The
  sentences are matched against :mod:`hpca.agent.hints`, which is where the
  tools get them from, so the two cannot drift apart.

Nothing here changes what the model sees. The stored messages are untouched;
this is a reading of them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from hpca.agent.hints import MODEL_HINTS
from hpca.agent.history import is_tool_call_message
from hpca.llm import STAMP_KEY, Message

TOOL_PREFIXES = ("[tool result]", "[tool error]")
# The lines themselves — a script's, or the two sides of an edit — rendered as
# their own block below the other arguments rather than as a JSON list of lines
# nobody can read. Which file was edited stays an argument; what changed does
# not, because the block already says it.
SCRIPT_ARG_KEYS = ("content_lines", "old_lines", "new_lines")
# Arguments that say how the call was made rather than what it does. A timeout
# is the harness's business; showing it puts a line of JSON on screen that
# tells the user nothing about what the agent is up to.
PLUMBING_ARG_KEYS = ("timeout_s",)
# The argument most likely to answer "on what?", best first. Only used to put a
# short target next to the tool name on the collapsed row, so a column of
# fifteen edit_file rows says which file each one touched.
TARGET_ARG_KEYS = (
    "path",
    "source_path",
    "dest_path",
    "name",
    "target",
)
# Arguments are a display aid, not a record: a pathological call must not push
# a wall of JSON into the chat (the script block has its own cap upstream).
ARGUMENTS_CHARS = 2000
# Separates the two halves of one exchange. Written rather than drawn, because
# the same string goes into the session log, which is a text file.
RESULT_RULE = "── result ──"
# Background work reporting in on its own (§5.4), and the note a stopped turn
# leaves behind it (`hpca.agent.graph.STOPPED_NOTE`). Both ride the user role
# like tool results do, and are marked so the transcript does not attribute a
# process crash — or the user's own stop gesture — to the human as something
# they said. An `event` row is the right one for the stop: it is the only kind
# that means "this happened", which is exactly what the reader scrolling back
# needs to see at the point the work breaks off.
EVENT_PREFIXES = ("[process ", "[job ", "[stopped]")
# The stop note is written for the model and reads like it — four lines
# telling it not to pick the work back up. The user does not need to be told
# that; they are the one who stopped it. Same cut as the tool-result hints
# below: the news is kept, the prompt is not.
STOPPED_PREFIX = "[stopped]"
STOPPED_TEXT = "stopped by the user"

USER = "user"
ASSISTANT = "assistant"
THINKING = "thinking"
ERROR = "error"
EVENT = "event"
# Memory recalled into a turn (redesign Phase 5). Shown so the user can see
# what the agent was reminded of — silent injection would make the agent's
# behavior inexplicable from the transcript alone.
RECALL = "recall"
# The fold boundary drawn into the conversation: everything above it reaches
# the model as a summary, everything below it verbatim. Not a thread message
# and never a rewind target — see `compaction_entry`.
COMPACTION = "compaction"

FENCE_OPEN = "<memory-context>"
FENCE_CLOSE = "</memory-context>"

# "[tool result] edit_file: " and its error twin. A tool name has no colon in
# it, so the first one ends the name.
_RESULT_PREFIX = re.compile(r"^\[tool (?:result|error)\] ([^:\n]*): ?")
# A refusal is a result too, and reads as a failure on the row even though the
# tool never raised.
_REFUSED_PREFIXES = ("DENIED by the user", "SKIPPED")
# What joins a hint to the news in front of it, taken out along with it. Half
# the hints are the back half of a sentence — "The document index is empty —
# index something with index_docs first." — and lifting one out on its own
# leaves the line ending in a dangling dash or semicolon. A full stop is
# deliberately NOT eaten: there the hint was a sentence of its own, and the stop
# closes the sentence before it, which stays.
_HINT_JOIN = r"[ \t]*[;,:—–-]?[ \t]*"


@dataclass
class Step:
    """One ordered part of a turn's working: a block of reasoning, or one tool
    exchange — the call and the result it returned, held together.

    ``text`` is the call as the user reads it and ``result`` is what came back,
    with ``done`` saying whether it has yet. A step announced while the turn is
    still running is built without a result and gains one in place
    (:meth:`attach`), so the row the user is already looking at is the row that
    fills in — nothing new appears below it and nothing jumps.

    ``kind`` is ``reasoning``, ``call``, or ``step``. The last is a result whose
    call is out of view, which happens when a log tail starts between the two;
    it keeps its own tool name, parsed back out of the text.
    """

    kind: str  # reasoning | call | step
    text: str
    tool: str = ""  # call: named here, so the label needs no parsing
    target: str = ""  # the file or key the call is about, if it has one
    result: str = ""  # what came back, framing for the model taken off
    done: bool = False  # whether the result has landed
    failed: bool = False  # the tool raised, or the user refused the call

    def attach(self, content: str) -> None:
        """Fill in the result half of an exchange announced while it ran."""
        self.result, self.failed = result_text(content)
        self.done = True

    def label(self) -> str:
        """Short header for the collapsed row: the tool, what it acted on, and
        how it went. "reasoning" for a block of thinking."""
        if self.kind == "reasoning":
            return self.kind
        name = self.tool or "tool"
        if self.target:
            name += f" · {self.target}"
        if self.failed:
            return name + " (error)"
        if self.kind == "call" and not self.done:
            return name + " …"
        return name

    def body(self) -> str:
        """Everything under the header: the call, then the result below it once
        there is one. A call still in flight shows only itself — an empty
        "result" heading would read as a tool that answered with nothing."""
        call = self.text.strip()
        result = self.result.strip()
        if self.kind != "call":
            return call
        if not self.done:
            return call
        if not call:
            return result
        if not result:
            return call
        return f"{call}\n\n{RESULT_RULE}\n{result}"


@dataclass
class Entry:
    kind: str  # user | assistant | thinking | error
    text: str
    steps: int = 0  # thinking: tool steps folded in
    reasoning_chars: int = 0  # thinking: how much the model thought
    # thinking: the ordered parts, kept structured so an expanded box can show
    # each one as its own collapsible element (``text`` folds them for the log).
    parts: list[Step] = field(default_factory=list)
    # The message index this entry IS (user/assistant/event), or -1 for
    # entries that are not one thread message (thinking folds several, recall
    # rides its user message, live rows predate the graph copy). What the chat
    # rewind (fork / roll back) uses to name its cut point.
    index: int = -1
    # When the message this entry reads was added, ISO-8601 UTC, straight off
    # `llm.STAMP_KEY`. Empty for the entries that are not one message: a
    # thinking box folds several and a live row predates the graph copy, and
    # neither can honestly name an instant. Empty too for a thread written
    # before there were stamps, which is what makes "" mean "not known"
    # rather than "the epoch".
    at: str = ""

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


def event_text(content: str) -> str:
    """An event as the user reads it.

    Only the stop note is rewritten, and only because it is the one event
    written for the model rather than about the world: a process reporting in
    says the same thing to both readers, and is passed through.
    """
    if content.startswith(STOPPED_PREFIX):
        return STOPPED_TEXT
    return content


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


def result_text(content: str) -> tuple[str, bool]:
    """One raw result message as the user reads it, and whether it went wrong.

    Three things come off. The ``[tool result] <tool>:`` prefix, because the
    row it lands on is already headed by that tool's name. The sentences the
    tools address to the model (:data:`hpca.agent.hints.MODEL_HINTS`), because
    they are instructions for the next call, not a report of what happened —
    the user reading "Created x.tsv (8 lines)" does not also need to be told
    that the agent should use edit_file next time. And the whitespace either of
    those leaves behind.

    What survives is the news: paths, counts, exit codes, output, error text,
    and a refusal saying it was refused.
    """
    failed = content.startswith("[tool error]")
    body = _RESULT_PREFIX.sub("", content, count=1)
    if body.startswith(_REFUSED_PREFIXES):
        failed = True
    for hint in MODEL_HINTS:
        without = re.sub(_HINT_JOIN + re.escape(hint), "", body)
        # A result that is *nothing but* guidance keeps it. Some tools answer a
        # malformed call with advice and no news at all, and a row that shows a
        # call and then a blank where the result goes reads as a tool that hung.
        if without.strip():
            body = without
    return "\n".join(line.rstrip() for line in body.splitlines()).strip(), failed


def result_tool(content: str) -> str:
    """The tool named by a raw result message, for a result whose call is out
    of view. "" when the message carries no such prefix."""
    match = _RESULT_PREFIX.match(content)
    return (match.group(1).strip() if match else "") or ""


def call_target(arguments: dict) -> str:
    """The one thing a call is about, short enough to sit next to the tool name.

    A path is shown by its last segment: the collapsed row has a line, and
    thirty characters of shared parent directory would spend all of it saying
    nothing that distinguishes this call from the one above.
    """
    for key in TARGET_ARG_KEYS:
        value = (arguments or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().rstrip("/").rsplit("/", 1)[-1]
    return ""


def call_arguments(arguments: dict, *, has_script: bool) -> str:
    """The arguments worth reading, one ``key: value`` line each.

    Not ``json.dumps(indent=2)``: braces, quotes and trailing commas are three
    lines of punctuation around two facts, and the user is looking for the two
    facts. Plumbing and the payload the script block already shows are left
    out — see :data:`PLUMBING_ARG_KEYS` and :data:`SCRIPT_ARG_KEYS`.
    """
    lines = []
    for key, value in (arguments or {}).items():
        if key in PLUMBING_ARG_KEYS:
            continue
        if has_script and key in SCRIPT_ARG_KEYS:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {value}")
        else:  # a list or a dict that is not the payload: one line, compact
            lines.append(f"{key}: {json.dumps(value, default=str)}")
    return clip("\n".join(lines))


def recorded_call(call: dict) -> dict:
    """One call record as it is *stored*: the arguments the chat will read
    back, and no others.

    The storage half of :func:`call_arguments`, and it lives next to it so the
    two cannot drift. The record rides in the checkpointed state, which
    LangGraph rewrites whole once per super-step — so an argument nobody
    renders is not stored once, it is stored a few hundred times over a
    session, and the payload arguments are the large ones.

    Exactly two functions ever read this dict, :func:`call_text` and
    :func:`call_target`, so what is dropped here is what those two would drop
    at paint time and nothing else:

    * :data:`PLUMBING_ARG_KEYS`, which `call_arguments` never prints.
    * :data:`SCRIPT_ARG_KEYS`, but only once the call carries a ``script``
      block — and the same emptiness test `call_text` applies, because a
      preview that is only whitespace is not a block and the raw lines are
      what gets printed instead. Where there is one, that block *is* the
      payload: the lines with their ``{key}`` references expanded, or the two
      sides of the edit as a diff, kept whole to
      ``modes.SCRIPT_PREVIEW_CHARS``, which is exactly why it is generous.

    Nothing is lost that anyone was reading. The model's own copy of the call
    keeps the arguments in full (`hpca.agent.history.call_json`), a message in
    the same state; this was the second copy of them, and the unread one.
    """
    arguments = call.get("arguments") or {}
    has_script = bool((call.get("script") or "").strip())
    kept = {
        name: value
        for name, value in arguments.items()
        if name not in PLUMBING_ARG_KEYS
        and not (has_script and name in SCRIPT_ARG_KEYS)
    }
    return call if kept == arguments else {**call, "arguments": kept}


def call_text(call: dict) -> str:
    """One tool call as the user reads it: what it does to what, and the script
    or the diff it would actually run.

    The same thing the approval prompt shows for a gated call — which is the
    point: after the prompt is answered it is gone, and what ran has to stay
    somewhere the user can still open.

    ``details`` is the tool's own account of this call, with the keys already
    resolved to real paths ("edit /work/x.tsv (replace 3 lines with 5)"). When
    there is one it stands alone: repeating the arguments underneath would say
    the same thing again in the tool's internal vocabulary.
    """
    script = (call.get("script") or "").strip()
    details = (call.get("details") or "").strip()
    blocks: list[str] = []
    if details:  # resolved real paths, flagged commands (§5.3)
        blocks.append(details)
    else:
        arguments = call_arguments(call.get("arguments") or {}, has_script=bool(script))
        if arguments:
            blocks.append(arguments)
    if script:
        blocks.append(script)
    return "\n\n".join(blocks)


def call_step(call: dict) -> Step:
    """The part one recorded call renders as, with no result on it yet."""
    return Step(
        kind="call",
        text=call_text(call),
        tool=str(call.get("tool") or ""),
        target=call_target(call.get("arguments") or {}),
    )


def result_step(content: str) -> Step:
    """The part a result renders as when its call is not in view."""
    text, failed = result_text(content)
    return Step(
        kind="step", text=text, tool=result_tool(content), done=True, failed=failed
    )


def live_step(payload: dict) -> Step:
    """The part one in-flight announcement renders as (see the graph's
    ``on_step``): a call as it is made, or the result as it lands.

    The same rendering the finished turn gets, so a step the user opened while
    it was running reads identically once it is folded into its thinking box.
    A result is only built into a part of its own here as a fallback — the chat
    attaches it to the call already on screen (:meth:`Step.attach`), which is
    what makes one exchange one row.
    """
    if payload.get("kind") == "call":
        return call_step(payload)
    return result_step(str(payload.get("text", "")))


def thinking_entry(parts: list[Step]) -> Entry:
    """The one entry a turn's working folds into.

    Public, and the only place that fold is written, because two callers have
    to agree about it exactly: :func:`build_entries` closing the box at the end
    of a turn, and the core drawing the same box *while* the turn runs (see
    ``hpca.core.scheduler``). If those two disagreed by a field, a re-opened
    session would visibly rearrange itself against what the user watched
    happen — which is the bug the old chat rebuild paid for.

    The counters are derived from the parts rather than passed in: a result
    lands by filling in the call it answers (:meth:`Step.attach`), so "how many
    steps" is exactly "how many parts have their answer", and asking the parts
    is the only version of that count which cannot drift from what is shown.
    """
    return Entry(
        kind=THINKING,
        text=_block(parts),
        steps=sum(1 for part in parts if part.kind != "reasoning" and part.done),
        reasoning_chars=sum(
            len(part.text) for part in parts if part.kind == "reasoning"
        ),
        parts=list(parts),
    )


def compaction_summary(compacted: dict | None) -> str:
    """The summary text out of a `compacted` record, or "".

    Tolerant on purpose: the record is written by the graph
    (`AgentState.compacted`) and read here two layers away, and a thread folded
    by an older build has the same key holding a bare string rather than a
    message.
    """
    if not compacted:
        return ""
    summary = compacted.get("summary")
    if isinstance(summary, dict):
        return str(summary.get("content", ""))
    return str(summary or "")


def compaction_entry(compacted: dict) -> Entry:
    """The boundary row: where the model's verbatim view begins.

    ``index`` stays -1 even though the record names one. It is deliberate and
    it is the whole safety of this row: ``index`` is what the chat rewind cuts
    at (`ui/state.py`'s rewind, `SessionFork` / `SessionRollback`), and this is
    not a message — a fork aimed at it would cut the thread at a row that is
    not in it. The boundary it draws is a *view* of the thread, not part of it.
    """
    return Entry(
        kind=COMPACTION,
        text=compaction_summary(compacted),
        at=str(compacted.get("at") or ""),
    )


def clip(text: str) -> str:
    if len(text) <= ARGUMENTS_CHARS:
        return text
    return text[:ARGUMENTS_CHARS] + "\n... [clipped]"


def _block(parts: list[Step]) -> str:
    return "\n\n".join(f"— {part.label()} —\n{part.body().strip()}" for part in parts)


def _open_call(pending: list[Step]) -> Step | None:
    """The earliest call in the open box still waiting for its result.

    Earliest, not latest: a model that made two calls before either answered
    gets its results back in the order it asked for them.
    """
    for part in pending:
        if part.kind == "call" and not part.done:
            return part
    return None


def build_entries(
    messages: list[Message],
    thinking: list[dict] | None = None,
    calls: list[dict] | None = None,
    *,
    start: int = 0,
    compacted: dict | None = None,
) -> list[Entry]:
    """Entries for ``messages[start:]``, with reasoning, tool calls and their
    results folded in.

    ``start`` selects a tail (one turn, for incremental logging) while keeping
    the absolute message indices that ``thinking`` and ``calls`` entries are
    anchored to.

    ``compacted`` is the thread's fold record, and passing it draws one extra
    row at the boundary it names: everything above reaches the model as a
    summary, everything below verbatim. Only the callers that rebuild a *whole*
    chat pass it — a tail is appended to a log that already has the row, and
    drawing it again per turn would file the same boundary once a turn.
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
            entries.append(thinking_entry(pending))
        pending, steps, reasoning_chars = [], 0, 0

    def open_calls(index: int) -> None:
        """The calls made at this point, shown before the result they produced.

        A call is anchored to the index of its own assistant message, whose
        result is the message straight after it, and record and messages are
        written in one state update, so there is never one without the other.
        An anchor past the end of ``messages`` is a call a rolled-back turn
        left behind (see ``rollback_thread``) and is passed over here, exactly
        as an orphaned reasoning anchor is.
        """
        for call in calls_at.get(index, []):
            pending.append(call_step(call))

    # Where the fold boundary goes, or -1. Read once: `upto` is an index into
    # the same list this walks, so the row lands in front of the first message
    # the model still sees whole.
    fold_at = -1
    if compacted and compaction_summary(compacted):
        upto = int(compacted.get("upto", 0) or 0)
        if start <= upto <= len(messages):
            fold_at = upto

    for index in range(start, len(messages)):
        message = messages[index]
        if index == fold_at:
            # Before anything this message contributes, and after flushing the
            # box above: the boundary separates turns, and a marker inside an
            # open thinking box would read as one of its steps.
            flush()
            entries.append(compaction_entry(compacted or {}))
        for reasoning in reasoning_at.get(index, []):
            if reasoning.strip():
                pending.append(Step(kind="reasoning", text=reasoning))
                reasoning_chars += len(reasoning)
        open_calls(index)
        role, content = message["role"], str(message["content"])
        if role == "system":
            continue
        if is_tool_call_message(message):
            # The model's own copy of a call it made (hpca.agent.history): it
            # exists so the history has the shape the model was trained on, and
            # is not an answer. The user reads the call from its anchored
            # record instead — same call, arguments unfolded — which
            # ``open_calls`` has already put in the open thinking box.
            continue
        at = str(message.get(STAMP_KEY) or "")
        if role == ASSISTANT:
            flush()  # the answer closes the box that produced it
            entries.append(Entry(kind=ASSISTANT, text=content, index=index, at=at))
        elif is_tool_message(message):
            # Onto the call it answers, so the exchange is one part. A tail
            # that begins between a call and its result has no call to land
            # on, and the result stands as its own part instead.
            call = _open_call(pending)
            if call is not None:
                call.attach(content)
            else:
                pending.append(result_step(content))
            steps += 1
        elif is_event_message(message):
            flush()
            entries.append(
                Entry(kind=EVENT, text=event_text(content), index=index, at=at)
            )
        else:
            flush()
            entries.append(Entry(kind=USER, text=content, index=index, at=at))
            recalled = recalled_text(message)
            if recalled:
                # The recall rides its user message, so it happened when that
                # message did — the one entry with no index of its own that
                # can still name an instant.
                entries.append(Entry(kind=RECALL, text=recalled, at=at))
    flush()  # a turn interrupted for approval leaves its box open
    if fold_at == len(messages):
        # Nothing was kept verbatim — every message is behind the fold. The
        # boundary is still the truth about what the model sees, so it is
        # drawn at the end rather than dropped.
        entries.append(compaction_entry(compacted or {}))
    return entries
