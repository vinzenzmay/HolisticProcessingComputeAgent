"""Validation/retry middleware around every model decision (§4.3).

The model answers with one JSON object — either a direct response or a tool
call. With constrained decoding the JSON is syntactically valid by
construction and retries only handle semantic errors (unknown tool, invalid
arguments); without it, a format instruction is appended and JSON parse errors
are retried too. Every validation failure is fed back verbatim so the model
can correct itself, a bounded number of times.

Two things here defend against the same constrained-decoding artifact, which
is worth stating once. Inside a JSON *string* every character is legal, so a
grammar cannot reject anything the model writes there. A model part-way
through a long ``list[str]`` that decides it is done with the array and starts
on the next key emits ``"timeout_s: 60"`` as one more element instead: the
array swallows the key, no error is raised anywhere, and the string surfaces
as a line of the script (a real run_bash failure — ``line 98: timeout_s::
command not found`` — after a ~100-line heredoc).

So: **array-valued arguments come last in every tool's params model**, leaving
no key for the model to reach for while it is still inside the array
(``test_middleware.TestArgumentOrder`` enforces it); and ``_strip_key_echo``
below removes such an element when one appears anyway.

The second is not a belt for the first's braces. Measured against the live 27B
backend after the reordering, a run_bash heredoc still produced a trailing
``timeout_s:`` element in 1 of 4 generations — the model repeats the key at the
end of the arguments out of habit, whether or not the grammar has one left to
give it. Ordering lowers the rate; the strip is what makes it not matter.

``edit_file`` is the one tool the ordering rule cannot save, and it is worth
being explicit about why. It needs *two* arrays — the lines out and the lines
in — so whichever comes first has a key after it, and the model reaches for it:
observed as ``new_lines**: [`` swallowed into ``old_lines`` in 6 of 12 failed
edits in one real session. That form slipped past both defences at once — the
markdown asterisks defeated the pattern, and ``[`` is not valid JSON — so the
edit was reported as a content mismatch, and the agent spent three round-trips
fixing indentation that was never wrong before abandoning ``edit_file`` for
``sed -i``. Both holes are closed below; ``edit_file`` additionally recognises
the shape itself, because a repair that fails must not fail silently.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from hpca.agent.history import ELIDED_MARKER
from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import Message, TruncatedOutput

logger = logging.getLogger("hpca.agent.middleware")

DEFAULT_MAX_RETRIES = 3
# Caps a runaway generation. Reasoning models spend this budget on thinking
# before the JSON decision, so it is roomier than a decision alone needs.
#
# Measured on the live 27B at the default (thinking OFF, hpca.config), writing
# a document through create_file: ~12 completion tokens per line — 500 for 35
# lines, 2400 for 197 — so this holds a ~330-line file, and ~270 with thinking
# turned on for a backend. That is past any specs.md worth writing in one
# call, and a longer one now degrades into a split write rather than a dead
# turn (see MAX_TRUNCATION_RETRIES). Raising it would only double how long a
# genuinely looping generation hangs before anyone finds out — and loops are
# not rare here: 1 of 4 run_bash generations in that same probe.
MAX_DECISION_TOKENS = 4096
# A cut-off decision is worth exactly one more try. Both causes are expensive
# to retry — a looping model burns the whole cap again — and the second
# attempt is told to write less, so a third would be the same answer twice.
MAX_TRUNCATION_RETRIES = 1
TRUNCATION_FEEDBACK = (
    "[validation error] Your last call was cut off at the token limit before "
    "it was finished, so nothing could be run. Do not send the same thing "
    "again — it will be cut off again. When writing a file: if it does not "
    "exist yet, create_file only a skeleton (headings, each with one "
    "placeholder line `TBD: ...`); then fill ONE section per edit_file call, "
    "at most ~80 new lines each. For anything else, make this call smaller."
)

# A salvaged write below this is not worth the confusion it costs: the model
# handles "continue from line N" well, but not "continue from line 3".
MIN_SALVAGE_LINES = 10
# Planted at the end of every salvaged write. This is what keeps a salvaged
# one-shot from silently losing its tail (measured: models declared 'done'
# with 5 of 10 sections): the marker is a `TBD` placeholder, so the tool
# results keep counting it until the model has actually replaced it with the
# missing content — the same tracking loop a skeleton write gets for free.
SALVAGE_MARKER = (
    "TBD: the write was cut off at the token limit here — replace this line "
    "with the missing rest, one section per edit_file call"
)


class DecisionError(Exception):
    """The model failed to produce a valid decision within the retry budget."""


@dataclass
class DirectResponse:
    text: str
    reasoning: str = ""  # the model's thinking, shown in the TUI's thinking box
    # The backend's own token accounting for the call that produced this
    # (prompt_tokens/completion_tokens). Measured, not estimated — it is what
    # the context meter reports.
    usage: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    tool: Tool
    arguments: BaseModel
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    # The backend's id for this call, under the native protocol. It is what
    # ties the result message back to the call, so it has to survive the trip
    # through the graph's pending_tool state. Empty under the envelope
    # protocol, which has no ids.
    call_id: str = ""
    # What had to be repaired to make this call valid (``_strip_key_echo``).
    # Carried to the call record and the approval prompt: a silent repair is
    # indistinguishable from a backend that never misbehaved, and the user
    # approving a script has to be told a line was taken out of it.
    repairs: list[str] = field(default_factory=list)

    async def execute(self, ctx: Any) -> str:
        return await self.tool.handler(self.arguments, ctx)


Decision = DirectResponse | ToolCall


def inline_refs(schema: dict) -> dict:
    """A pydantic schema with its ``$defs`` substituted in place.

    A model with a nested model (``list[MemoryOperation]``) generates
    ``{"$ref": "#/$defs/MemoryOperation"}`` plus a sibling ``$defs``. The
    ``#/`` in that pointer means *the root of the whole document* — and here
    the schema gets embedded as one branch of the decision envelope, so the
    root is the envelope, which has no ``$defs``. The backend's grammar
    compiler then rejects the request outright:

        Grammar error: Pointer '/$defs/MemoryOperation' does not exist

    Substituting the definitions removes the pointers entirely, which keeps
    every tool's arguments self-contained however deeply they nest.
    Self-referential models cannot be inlined this way and are left alone;
    no tool has one, and a broken pointer is more debuggable than a hang.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, expanding: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, expanding) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref[len("#/$defs/") :]
            if name in defs and name not in expanding:
                # Merge any siblings of $ref (e.g. a description) over the
                # definition, so local annotations are not lost.
                target = resolve(defs[name], expanding | {name})
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**target, **extra} if extra else target
            return node  # unknown or recursive: leave the pointer in place
        return {
            key: resolve(value, expanding)
            for key, value in node.items()
            if key != "$defs"
        }

    return resolve(schema, frozenset())


def decision_schema(tools: ToolRegistry) -> dict:
    """JSON schema for the decision envelope: respond | one branch per tool."""
    branches: list[dict] = [
        {
            "type": "object",
            "properties": {
                "action": {"const": "respond"},
                "response": {"type": "string"},
            },
            "required": ["action", "response"],
            "additionalProperties": False,
        }
    ]
    for tool in tools:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "action": {"const": "tool_call"},
                    "tool": {"const": tool.name},
                    "arguments": inline_refs(tool.params.model_json_schema()),
                },
                "required": ["action", "tool", "arguments"],
                "additionalProperties": False,
            }
        )
    return {"anyOf": branches}


def tool_specs(tools: ToolRegistry) -> list[dict]:
    """The tool list as the OpenAI ``tools`` array (native protocol).

    Carries exactly what ``format_instruction`` spells out in prose — name,
    description, argument schema — in the position the chat template puts it,
    which is where an agent-trained model expects to find it. The schema is
    the same ``inline_refs`` output the envelope grammar uses, so the two
    protocols cannot drift apart on what an argument is.

    The filled example goes in the description because a JSON schema saying
    ``"type": "array"`` is not enough for a 27B. Measured: on the native
    channel without it, ``edit_file`` came back with ``old_lines`` as a bare
    string rather than a list of them in 6 of 6 generations — the call was
    right, its shape was not, and the model answered in prose rather than
    correct itself on the retry. This is the same finding ``_example_args``
    exists for on the envelope side, so both protocols show the same example.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (
                    f"{tool.description}\nExample arguments: "
                    f"{json.dumps(_example_args(tool))}"
                ),
                "parameters": inline_refs(tool.params.model_json_schema()),
            },
        }
        for tool in tools
    ]


def native_instruction(tools: ToolRegistry) -> str:
    """The standing instruction when the tools ride the native channel.

    Everything ``format_instruction`` says about *shape* is the template's job
    here, so only the part it cannot express survives: that a call is a real
    action, and that answering is the other option. Deliberately short — but
    not empty, which is the v0.20.0 lesson: text that has become false is
    worth deleting, text that is merely long is not.
    """
    if len(tools) == 0:
        return "You have no tools available in this context; answer directly."
    return (
        "Call a tool when you need to act on or look at the system, and "
        "answer directly when you do not. A tool call really runs — it is "
        "not a plan or a suggestion."
    )


def format_instruction(tools: ToolRegistry) -> str:
    """Textual tool list + output format, appended to every decision request.

    Required even with constrained decoding: the schema only *constrains*
    output syntax — without this listing the model does not know which tools
    exist and claims it has none (observed on the live Qwen3.6 backend).
    """
    if len(tools) == 0:
        return (
            "Answer with exactly one JSON object and nothing else: "
            '{"action": "respond", "response": "<your answer>"}. '
            "You have no tools available in this context."
        )
    tool_lines = "\n".join(
        f'- {{"action": "tool_call", "tool": "{tool.name}", "arguments": '
        f"{json.dumps(_example_args(tool))}}} — {tool.description}"
        for tool in tools
    )
    return (
        "Answer with exactly one JSON object and nothing else.\n"
        'To answer directly: {"action": "respond", "response": "<your answer>"}\n'
        f"To call a tool:\n{tool_lines}"
    )


def _example_args(tool: Tool) -> dict:
    """A filled-in example of the tool's arguments for the textual listing.

    Nested models get an example element rather than a placeholder string:
    shown `"operations": "<the changes>"` a small model writes exactly that
    string, and the call fails validation. Showing the shape it must produce
    is the difference between a tool it can call and one it cannot.
    """
    return {
        name: _example_value(field.annotation, field.description or name)
        for name, field in tool.params.model_fields.items()
    }


def _example_value(annotation: Any, description: str) -> Any:
    origin = get_origin(annotation)
    if origin in (list, set, tuple):
        args = get_args(annotation)
        inner = args[0] if args else str
        return [_example_value(inner, description)]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {
            name: _example_value(field.annotation, field.description or name)
            for name, field in annotation.model_fields.items()
        }
    return f"<{description}>"


# A trailing array element that is really the next argument: an optionally
# quoted key, a colon, and a value. The value must be JSON — that is what the
# model was mid-way through writing — which is what keeps prose out of range:
# a heredoc ending on `name: the tool` does not match, `"name": "x"` does.
#
# The key may arrive wrapped in markdown emphasis (`**new_lines**:`). The model
# writes the key the way it writes prose, and under constrained decoding a `*`
# at an illegal position is masked away, so the pair can survive lopsided —
# hence `\**` on both sides rather than a balanced pair.
_KEY_ECHO = re.compile(
    r'^\s*\**"?(?P<key>[A-Za-z_][A-Za-z0-9_]*)"?\**\s*:\s*(?P<value>.*?),?\s*$'
)

# The value of an echoed *container* argument: the model got as far as opening
# it. An unclosed bracket is never valid JSON, so these have to be admitted by
# name — see `_is_echo`.
_OPENED_CONTAINER = {"[", "{", "[]", "{}"}


def _is_echo(element: str, siblings: dict[str, Any]) -> str:
    """The argument this element is an echo of, or "" if it is a real line.

    ``siblings`` maps each *other* field's name to its annotation, because
    what counts as a plausible value depends on the field it would fill.
    """
    match = _KEY_ECHO.match(element)
    if match is None or match["key"] not in siblings:
        return ""
    value = match["value"].strip()
    if value in _OPENED_CONTAINER:
        # `new_lines: [` — the model closed one array and opened the next
        # inside the string it was still writing. Only for a field that IS a
        # container: `count: [` names no argument this model could be filling.
        return (
            match["key"]
            if get_origin(siblings[match["key"]]) in (list, dict)
            else ""
        )
    if value:
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return ""  # `timeout_s: soon` is prose, not a swallowed argument
    return match["key"]


def _strip_key_echo(params: type[BaseModel], arguments: dict) -> list[str]:
    """Drop trailing string-array elements that echo one of the *other*
    arguments, in place. Returns one note per element dropped.

    See the module docstring for what produces them. The check is deliberately
    narrow — last element only, a sibling field's exact name, a JSON value or
    the opening bracket of a container one, and never the only element left —
    because the cost of a false positive is a line quietly missing from a file
    the user asked for. Anything dropped is reported rather than swallowed.
    """
    notes: list[str] = []
    for name, field_info in params.model_fields.items():
        if get_origin(field_info.annotation) is not list:
            continue
        value = arguments.get(name)
        if not isinstance(value, list):
            continue
        siblings = {
            other: info.annotation
            for other, info in params.model_fields.items()
            if other != name
        }
        while len(value) > 1 and isinstance(value[-1], str):
            echoed = _is_echo(value[-1], siblings)
            if not echoed:
                break
            dropped = value.pop()
            notes.append(
                f"dropped a trailing {name} element that echoed the "
                f"{echoed} argument: {dropped!r}"
            )
    for note in notes:  # a repair is also how a backend regression shows up
        logger.warning("%s: %s", params.__name__, note)
    return notes


def _elided_placeholder(value: Any) -> str:
    """A marker from the model's own history found in an argument, or "".

    Walks nested lists and dicts, because a payload one level down is where
    these turn up: the argument the model copied back was a list of lines.
    """
    if isinstance(value, str):
        found = ELIDED_MARKER.search(value)
        return found.group(0) if found else ""
    if isinstance(value, list):
        return next((hit for hit in map(_elided_placeholder, value) if hit), "")
    if isinstance(value, dict):
        return next(
            (hit for hit in map(_elided_placeholder, value.values()) if hit), ""
        )
    return ""


def _reject_elided(arguments: Any) -> None:
    """Refuse a call that carries the history's own elision marker as content.

    ``history.elide`` names every cut so the model can tell a truncated
    argument from a short one, but naming it does not stop the model copying
    the name forward as if it were the content. Writing that to a file destroys
    it, and silently: the call succeeds. Better a bounced decision, which is a
    round-trip, than a file replaced by a sentence about how long it used to be.
    """
    marker = _elided_placeholder(arguments)
    if not marker:
        return
    raise ValueError(
        f"Your arguments contain {marker!r}, which is not content — it is the "
        "placeholder this conversation puts in place of a long argument you "
        "already sent, so copying it back would write it into the file. Send "
        "the real lines. If they are too many for one call, write the first "
        "part and add the rest with edit_file."
    )


def _parse(raw: str, tools: ToolRegistry) -> Decision:
    """Parse and validate one model output; raises ValueError with feedback text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Your output was not valid JSON ({e}). Respond with exactly one "
            "JSON object as instructed."
        ) from e
    if not isinstance(data, dict):
        raise ValueError("Your output must be a single JSON object.")
    action = data.get("action")
    if action == "respond":
        response = data.get("response")
        if not isinstance(response, str):
            raise ValueError('For action "respond", "response" must be a string.')
        return DirectResponse(text=response)
    if action == "tool_call":
        try:
            tool = tools.get(str(data.get("tool")))
        except KeyError as e:
            raise ValueError(str(e)) from e
        raw_arguments = data.get("arguments") or {}
        repairs = (
            _strip_key_echo(tool.params, raw_arguments)
            if isinstance(raw_arguments, dict)
            else []
        )
        _reject_elided(raw_arguments)
        try:
            arguments = tool.params.model_validate(raw_arguments)
        except ValidationError as e:
            raise ValueError(
                f"Invalid arguments for tool {tool.name!r}:\n{e}"
            ) from e
        return ToolCall(tool=tool, arguments=arguments, repairs=repairs)
    raise ValueError(
        f'Unknown action {action!r}: use "respond" or "tool_call".'
    )


def _parse_native(response: Any, tools: ToolRegistry) -> Decision:
    """Build a decision from a native tool-calling response.

    The branch choice is the backend's, not ours: a response with tool_calls
    is a call, one without is an answer. Argument *validation* stays exactly
    where it was — the tools array constrains shape, not meaning, so a wrong
    key or a missing field still comes back through the same feedback loop.
    """
    calls = response.tool_calls or []
    if not calls:
        text = response.content or ""
        if not text.strip():
            raise ValueError(
                "You returned neither a tool call nor an answer. Call a tool, "
                "or reply with your answer as ordinary text."
            )
        return DirectResponse(text=text)
    # More than one call in a response is possible on this channel; the graph
    # runs one at a time (approval, tool-round accounting), so the rest would
    # be silently dropped. Taking the first and saying so beats both.
    function = (calls[0].get("function") or {})
    try:
        tool = tools.get(str(function.get("name")))
    except KeyError as e:
        raise ValueError(str(e)) from e
    raw = function.get("arguments") or "{}"
    if isinstance(raw, str):
        try:
            raw_arguments = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(
                f"The arguments for {tool.name!r} were not valid JSON ({e})."
            ) from e
    else:
        raw_arguments = raw
    if not isinstance(raw_arguments, dict):
        raise ValueError(f"The arguments for {tool.name!r} must be a JSON object.")
    repairs = _strip_key_echo(tool.params, raw_arguments)
    _reject_elided(raw_arguments)
    if len(calls) > 1:
        repairs.append(
            f"kept the first of {len(calls)} calls in one response; "
            "the others were dropped"
        )
    try:
        arguments = tool.params.model_validate(raw_arguments)
    except ValidationError as e:
        raise ValueError(f"Invalid arguments for tool {tool.name!r}:\n{e}") from e
    return ToolCall(
        tool=tool,
        arguments=arguments,
        repairs=repairs,
        call_id=str(calls[0].get("id") or ""),
    )


def _annotate(decision: Decision, response: Any) -> Decision:
    decision.reasoning = response.reasoning or ""
    decision.usage = response.usage or {}
    return decision


def _scalar_arg(partial: str, key: str) -> str | None:
    """A complete string-valued argument out of a JSON fragment, or None."""
    m = re.search(rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*")', partial)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def _array_start(partial: str, key: str) -> int | None:
    """Offset just past the ``[`` of ``"key": [``, or None."""
    m = re.search(rf'"{key}"\s*:\s*\[', partial)
    return m.end() if m else None


def _scan_strings(text: str, start: int) -> tuple[list[str], bool]:
    """Complete string elements of a JSON array, scanning from ``start``.

    Returns (elements, closed): closed means the ``]`` was reached, i.e. the
    array did not truncate here. A non-string element or decode failure stops
    the scan — everything before it is still complete and usable.
    """
    decoder = json.JSONDecoder()
    items: list[str] = []
    i = start
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            return items, False
        if text[i] == "]":
            return items, True
        try:
            value, i = decoder.raw_decode(text, i)
        except ValueError:
            return items, False
        if not isinstance(value, str):
            return items, False
        items.append(value)


def _trim_degenerate_tail(lines: list[str]) -> list[str]:
    """Drop a looping generation's junk tail from salvaged lines.

    The other way a generation hits max_tokens (hpca.llm.TruncatedOutput) is
    degeneration: the model repeats a near-empty element until the cap.
    Seen live: a salvaged specs write ending in ~200 lines of '  ,'. Real
    content cut short has a normal last line; a loop has a tail of dull or
    identical ones — strip both, then let MIN_SALVAGE_LINES judge the rest.
    """
    trimmed = list(lines)

    def dull(s: str) -> bool:
        return not s.strip(" \t,.|;:-_=*#")

    while trimmed and dull(trimmed[-1]):
        trimmed.pop()
    # a run of identical trailing lines is the loop's other signature
    if len(trimmed) >= 6 and len(set(trimmed[-6:])) == 1:
        repeated = trimmed[-1]
        while trimmed and trimmed[-1] == repeated:
            trimmed.pop()
    return trimmed


def _salvage_truncated_call(partial: str, tools: ToolRegistry) -> ToolCall | None:
    """A cut-off create_file/edit_file rebuilt from its complete prefix.

    A too-long file write dies at max_tokens with hundreds of perfectly good
    lines already generated; retrying regenerates (and usually re-truncates)
    all of them. Everything up to the last complete array element is real
    work, so run that and tell the model to continue — measured live
    (specs-eval, 2026-08-07), the retry path lost the whole file two times in
    three, while a salvaged prefix turns the same failure into forward
    progress. The arrays-last argument rule (module docstring) is what makes
    the prefix parseable: every scalar argument is complete before the final
    array begins. Anything else — unknown tool, truncation inside edit_file's
    old_lines, fewer than MIN_SALVAGE_LINES lines — returns None and takes
    the normal retry.
    """
    m = re.search(r'"tool"\s*:\s*"(create_file|edit_file)"', partial)
    if not m:
        return None
    tool_name = m.group(1)
    try:
        tool = tools.get(tool_name)
    except Exception:
        return None

    if tool_name == "create_file":
        dir_key = _scalar_arg(partial, "dir_key")
        name = _scalar_arg(partial, "name")
        start = _array_start(partial, "content_lines")
        if not dir_key or not name or start is None:
            return None
        lines, closed = _scan_strings(partial, start)
        # closed means the array was fine and the cut hit something else —
        # too odd a state to guess at, let the retry handle it.
        if closed:
            return None
        lines = _trim_degenerate_tail(lines)
        if len(lines) < MIN_SALVAGE_LINES:
            return None
        args: dict = {
            "dir_key": dir_key,
            "name": name,
            "content_lines": lines + [SALVAGE_MARKER],
        }
        note = (
            f"this call was cut off at the token limit; the {len(lines)} "
            f"complete lines generated before the cut were written, the rest "
            f"were lost, and a TBD marker line was appended where the file "
            f"stops (after {lines[-1]!r}). The file is INCOMPLETE: replace "
            "the marker with the missing content, one section per edit_file "
            "call, at most ~80 new lines each."
        )
    else:
        registry_key = _scalar_arg(partial, "registry_key")
        old_start = _array_start(partial, "old_lines")
        if not registry_key or old_start is None:
            return None
        old_lines, old_closed = _scan_strings(partial, old_start)
        if not old_closed or not old_lines:
            return None  # cut inside old_lines: no way to know the target
        new_start = _array_start(partial, "new_lines")
        if new_start is None or new_start <= old_start:
            return None
        new_lines, new_closed = _scan_strings(partial, new_start)
        if new_closed:
            return None
        new_lines = _trim_degenerate_tail(new_lines)
        if len(new_lines) < MIN_SALVAGE_LINES:
            return None
        args = {
            "registry_key": registry_key,
            "subpath": _scalar_arg(partial, "subpath") or "",
            "old_lines": old_lines,
            "new_lines": new_lines + [SALVAGE_MARKER],
        }
        note = (
            f"this call was cut off at the token limit; the {len(new_lines)} "
            f"complete replacement lines generated before the cut were "
            f"applied, the rest were lost, and a TBD marker line was placed "
            f"where the text stops (after {new_lines[-1]!r}). Replace the "
            "marker with the missing content, at most ~80 new lines per call."
        )

    try:
        arguments = tool.params.model_validate(args)
    except ValidationError:
        return None
    return ToolCall(tool=tool, arguments=arguments, repairs=[note])


def _no_native() -> bool:
    return False


def _retry_feedback(response: Any, error: str, native: bool) -> list[Message]:
    """What a rejected decision adds to the conversation before the retry.

    The model has to see what it wrote and why it was refused, in the shape it
    wrote it in. On the native channel that means the assistant message keeps
    its ``tool_calls`` and the complaint comes back on the tool role answering
    it — a chat template that is handed a call with no matching result renders
    a broken conversation, and some reject it outright. Only the first call is
    echoed, matching what ``_parse_native`` would have kept.
    """
    if not native:
        return [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"[validation error] {error}"},
        ]
    calls = response.tool_calls or []
    if not calls:  # nothing came back to echo — say so on the user role
        return [{"role": "user", "content": f"[validation error] {error}"}]
    call = dict(calls[0])
    call_id = str(call.get("id") or "call_retry")
    call["id"] = call_id
    return [
        {"role": "assistant", "content": response.content or "", "tool_calls": [call]},
        {
            "role": "tool",
            "content": f"[validation error] {error}",
            "tool_call_id": call_id,
            "name": str((call.get("function") or {}).get("name") or "unknown"),
        },
    ]


async def decide(
    llm: Any,
    messages: list[Message],
    tools: ToolRegistry,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tokens: int | None = MAX_DECISION_TOKENS,
) -> Decision:
    """Ask the model for a decision, validating and retrying with feedback."""
    # Native protocol: the backend's own tool channel decides the branch and
    # carries the tool list, so neither the envelope grammar nor its prose
    # listing applies. getattr keeps every caller that passes a plain fake
    # client working — there is nothing to ask, so the answer is "envelope".
    native = bool(getattr(llm, "uses_native_tools", _no_native)())
    if native:
        schema = None
    else:
        constrained = await llm.supports_constrained_decoding()
        schema = decision_schema(tools) if constrained else None
    # Backends like vLLM/Qwen only accept system messages at position 0, so
    # the tool instruction is merged into the leading system message.
    conversation = list(messages)
    instruction = native_instruction(tools) if native else format_instruction(tools)
    if conversation and conversation[0]["role"] == "system":
        merged = conversation[0]["content"] + "\n\n" + instruction
        conversation[0] = {"role": "system", "content": merged}
    else:
        conversation.insert(0, {"role": "system", "content": instruction})

    # Only passed when native, so existing callers see an unchanged call.
    offered = {"tools": tool_specs(tools)} if native else {}
    attempts = max_retries + 1
    last_error = ""
    truncations = 0
    for _ in range(attempts):
        try:
            response = await llm.chat(
                conversation, json_schema=schema, max_tokens=max_tokens, **offered
            )
        except TruncatedOutput as exc:
            # First choice: salvage. A cut-off file write's complete prefix
            # is real work — running it beats regenerating (and usually
            # re-truncating) the whole thing. The repair note rides on the
            # ToolCall so the result the model sees says the write is partial.
            salvaged = _salvage_truncated_call(getattr(exc, "partial", ""), tools)
            if salvaged is not None:
                return salvaged
            # Otherwise: nothing came back to feed the model verbatim — the
            # fragment is its own unfinished call — so the feedback says what
            # happened and what to do instead, and rides the user role like
            # every other retry (system messages are rejected after pos 0).
            if truncations >= MAX_TRUNCATION_RETRIES:
                raise
            truncations += 1
            conversation = conversation + [
                {"role": "user", "content": TRUNCATION_FEEDBACK}
            ]
            continue
        raw = response.content
        try:
            parsed = _parse_native(response, tools) if native else _parse(raw, tools)
            return _annotate(parsed, response)
        except ValueError as e:
            last_error = str(e)
            conversation = conversation + _retry_feedback(response, last_error, native)
    raise DecisionError(
        f"No valid decision after {attempts} attempts; last error: {last_error}"
    )
