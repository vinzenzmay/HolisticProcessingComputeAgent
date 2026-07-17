"""Validation/retry middleware around every model decision (§4.3).

The model answers with one JSON object — either a direct response or a tool
call. With constrained decoding the JSON is syntactically valid by
construction and retries only handle semantic errors (unknown tool, invalid
arguments); without it, a format instruction is appended and JSON parse errors
are retried too. Every validation failure is fed back verbatim so the model
can correct itself, a bounded number of times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import Message

DEFAULT_MAX_RETRIES = 3
# Caps a runaway generation. Reasoning models spend this budget on thinking
# before the JSON decision, so it is roomier than a decision alone needs.
MAX_DECISION_TOKENS = 4096


class DecisionError(Exception):
    """The model failed to produce a valid decision within the retry budget."""


@dataclass
class DirectResponse:
    text: str
    reasoning: str = ""  # the model's thinking, shown in the TUI's thinking box


@dataclass
class ToolCall:
    tool: Tool
    arguments: BaseModel
    reasoning: str = ""

    async def execute(self, ctx: Any) -> str:
        return await self.tool.handler(self.arguments, ctx)


Decision = DirectResponse | ToolCall


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
                    "arguments": tool.params.model_json_schema(),
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
    return {
        name: f"<{f.description or name}>"
        for name, f in tool.params.model_fields.items()
    }


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
        try:
            arguments = tool.params.model_validate(data.get("arguments") or {})
        except ValidationError as e:
            raise ValueError(
                f"Invalid arguments for tool {tool.name!r}:\n{e}"
            ) from e
        return ToolCall(tool=tool, arguments=arguments)
    raise ValueError(
        f'Unknown action {action!r}: use "respond" or "tool_call".'
    )


def _with_reasoning(decision: Decision, reasoning: str) -> Decision:
    decision.reasoning = reasoning
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
    for _ in range(attempts):
        response = await llm.chat(
            conversation, json_schema=schema, max_tokens=max_tokens
        )
        raw = response.content
        try:
            return _with_reasoning(_parse(raw, tools), response.reasoning or "")
        except ValueError as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
    raise DecisionError(
        f"No valid decision after {attempts} attempts; last error: {last_error}"
    )
