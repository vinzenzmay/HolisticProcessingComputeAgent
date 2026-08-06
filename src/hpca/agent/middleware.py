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
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import Message, TruncatedOutput

logger = logging.getLogger("hpca.agent.middleware")

DEFAULT_MAX_RETRIES = 3
# Caps a runaway generation. Reasoning models spend this budget on thinking
# before the JSON decision, so it is roomier than a decision alone needs — and
# a decision that writes a file's content spends it twice over. Measured on
# the live 27B backend with thinking on: a 30-line document costs ~1200
# completion tokens, ~800 of them reasoning, so the old 4096 put a ~100-line
# specs.md right at the ceiling. Doubling buys documents of a few hundred
# lines while still bounding the loop this exists for.
MAX_DECISION_TOKENS = 8192
# A cut-off decision is worth exactly one more try. Both causes are expensive
# to retry — a looping model burns the whole cap again — and the second
# attempt is told to write less, so a third would be the same answer twice.
MAX_TRUNCATION_RETRIES = 1
TRUNCATION_FEEDBACK = (
    "[validation error] Your last call was cut off at the token limit before "
    "it was finished, so nothing could be run. Make this one smaller. If you "
    "were writing a file, write it in parts: create_file with the first part, "
    "then edit_file to add the rest. Do not simply send the same thing again."
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
_KEY_ECHO = re.compile(
    r'^\s*"?(?P<key>[A-Za-z_][A-Za-z0-9_]*)"?\s*:\s*(?P<value>.*?),?\s*$'
)


def _is_echo(element: str, siblings: set[str]) -> str:
    """The argument this element is an echo of, or "" if it is a real line."""
    match = _KEY_ECHO.match(element)
    if match is None or match["key"] not in siblings:
        return ""
    value = match["value"]
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
    narrow — last element only, a sibling field's exact name, a JSON value, and
    never the only element left — because the cost of a false positive is a
    line quietly missing from a file the user asked for. Anything dropped is
    reported rather than swallowed.
    """
    notes: list[str] = []
    for name, field_info in params.model_fields.items():
        if get_origin(field_info.annotation) is not list:
            continue
        value = arguments.get(name)
        if not isinstance(value, list):
            continue
        siblings = set(params.model_fields) - {name}
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


def _annotate(decision: Decision, response: Any) -> Decision:
    decision.reasoning = response.reasoning or ""
    decision.usage = response.usage or {}
    return decision


async def decide(
    llm: Any,
    messages: list[Message],
    tools: ToolRegistry,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tokens: int | None = MAX_DECISION_TOKENS,
) -> Decision:
    """Ask the model for a decision, validating and retrying with feedback."""
    constrained = await llm.supports_constrained_decoding()
    schema = decision_schema(tools) if constrained else None
    # Backends like vLLM/Qwen only accept system messages at position 0, so
    # the tool instruction is merged into the leading system message.
    conversation = list(messages)
    instruction = format_instruction(tools)
    if conversation and conversation[0]["role"] == "system":
        merged = conversation[0]["content"] + "\n\n" + instruction
        conversation[0] = {"role": "system", "content": merged}
    else:
        conversation.insert(0, {"role": "system", "content": instruction})

    attempts = max_retries + 1
    last_error = ""
    truncations = 0
    for _ in range(attempts):
        try:
            response = await llm.chat(
                conversation, json_schema=schema, max_tokens=max_tokens
            )
        except TruncatedOutput:
            # Nothing came back to feed the model verbatim — the fragment is
            # its own unfinished call — so the feedback says what happened and
            # what to do instead, and rides the user role like every other
            # retry (system messages are rejected after position 0).
            if truncations >= MAX_TRUNCATION_RETRIES:
                raise
            truncations += 1
            conversation = conversation + [
                {"role": "user", "content": TRUNCATION_FEEDBACK}
            ]
            continue
        raw = response.content
        try:
            return _annotate(_parse(raw, tools), response)
        except ValueError as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
    raise DecisionError(
        f"No valid decision after {attempts} attempts; last error: {last_error}"
    )
