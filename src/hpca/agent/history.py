"""What one completed tool call adds to the conversation (§4.3).

The graph used to append only the *result* of a call, as a user message
("[tool result] read_file: …"), so the history was a run of consecutive user
messages and the model had to reconstruct its own actions from the echo in the
result text. That is not the shape an agent-trained model was post-trained on:
there it is an assistant message carrying the call, then the result keyed to
it. This module builds that pair.

The assistant half is the decision envelope the model itself emits — the exact
shape ``hpca.agent.middleware.decision_schema`` constrains and
``format_instruction`` documents::

    {"action": "tool_call", "tool": "read_file", "arguments": {…}}

so what lands in the history is what the model would have written, not a
paraphrase of it.

Large arguments are elided in that copy, and this is load-bearing: a
``create_file`` call carries the whole file in ``content_lines``, and echoing
it verbatim would store the file twice in the window on exactly the turns that
are already tight. The point of the assistant message is that the model can
see *what* it did — the tool, the target, the shape of the payload — not
re-read the payload it just wrote.

The result half rides the user role for the same reason it always has: vLLM /
Qwen templates reject mid-conversation system messages, and the native tool
role requires the tool-call protocol this design deliberately bypasses.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hpca.llm import Message

# A list argument longer than this is a payload, not a parameter: it is cut to
# a head plus a marker saying how much was left out.
MAX_LIST_ITEMS = 12
KEEP_LIST_ITEMS = 3
# Same for a long string argument (a here-doc script, a pasted block).
MAX_STRING_CHARS = 400

# The markers ``elide`` leaves behind, as a pattern. Naming the cut is only
# half the defence: a model that reads its own call back can still copy the
# marker forward as if it were content. It has happened — a `cat > f << EOF`
# heredoc whose body was "... 257 more lines elided ..." ran, and truncated a
# 266-line script to 46 bytes. ``middleware`` refuses any call carrying one.
ELIDED_MARKER = re.compile(r"\.\.\. \d+ more (?:lines|chars) elided \.\.\.")

CALL_ACTION = "tool_call"
# What the assistant copy of a call starts with. Cheap prefix test for
# ``is_tool_call_message`` — the transcript asks it about every message.
CALL_PREFIX = '{"action": "tool_call"'


def elide(value: Any) -> Any:
    """One argument value, cut down to its shape.

    Lists lose their tail, long strings their end, and the cut is *named* in
    place ("… 812 more lines elided …") rather than silently applied: a model
    reading its own call back must be able to tell an argument it truncated
    from one it wrote short. Applied inside lists and dicts too, so a payload
    nested one level down is not a way around the cap.
    """
    if isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            head = [elide(item) for item in value[:KEEP_LIST_ITEMS]]
            return head + [f"... {len(value) - KEEP_LIST_ITEMS} more lines elided ..."]
        return [elide(item) for item in value]
    if isinstance(value, dict):
        return {key: elide(item) for key, item in value.items()}
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        dropped = len(value) - MAX_STRING_CHARS
        return f"{value[:MAX_STRING_CHARS]}... {dropped} more chars elided ..."
    return value


def elide_arguments(arguments: dict) -> dict:
    """The call's arguments as the assistant copy carries them."""
    return {name: elide(value) for name, value in (arguments or {}).items()}


def call_json(tool_name: str, arguments: dict) -> str:
    """The decision envelope for this call, serialized as the model emits it."""
    return json.dumps(
        {
            "action": CALL_ACTION,
            "tool": tool_name,
            "arguments": elide_arguments(arguments),
        },
        ensure_ascii=False,
        default=str,
    )


def tool_call_message(tool_name: str, arguments: dict) -> Message:
    """The assistant message standing for the call the model just made."""
    return {"role": "assistant", "content": call_json(tool_name, arguments)}


def tool_result_message(tool_name: str, result: str) -> Message:
    """The user-role message carrying what the call returned."""
    return {"role": "user", "content": f"[tool result] {tool_name}: {result}"}


def native_call_message(tool_name: str, arguments: dict, call_id: str) -> Message:
    """The assistant message for a call made on the backend's own channel.

    The same pair, in the encoding the chat template understands: the call
    rides ``tool_calls`` rather than the content, and its ``id`` is what ties
    the result to it. Arguments are elided exactly as in the envelope copy —
    the reason has nothing to do with the protocol, it is that a create_file
    payload would otherwise sit in the window twice.
    """
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(
                        elide_arguments(arguments), ensure_ascii=False, default=str
                    ),
                },
            }
        ],
    }


def native_result_message(tool_name: str, content: str, call_id: str) -> Message:
    """The tool-role message answering a native call, content verbatim."""
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "name": tool_name,
    }


def call_message(tool_name: str, arguments: dict, call_id: str = "") -> Message:
    """The assistant half of an exchange, in whichever protocol is in use.

    ``call_id`` is what selects it, because only the native protocol has ids:
    it is set when the decision came back on the backend's tool channel
    (``LLMSettings.tool_protocol``) and empty otherwise.
    """
    if call_id:
        return native_call_message(tool_name, arguments, call_id)
    return tool_call_message(tool_name, arguments)


def result_message(tool_name: str, content: str, call_id: str = "") -> Message:
    """The result half, with ``content`` used exactly as given.

    The caller owns the text: what answers a call is also a denial or a tool
    error, not only a "[tool result] …" line, and this must not relabel one as
    the other. Only the role and the id change with the protocol.
    """
    if call_id:
        return native_result_message(tool_name, content, call_id)
    return {"role": "user", "content": content}


def tool_exchange(
    tool_name: str, arguments: dict, result: str, *, call_id: str = ""
) -> list[dict]:
    """The messages one completed tool call adds to the conversation."""
    return [
        call_message(tool_name, arguments, call_id),
        result_message(tool_name, f"[tool result] {tool_name}: {result}", call_id),
    ]


def call_text(message: Message) -> str:
    """A native call rendered as text, or "" for any other message.

    The native encoding puts the call in ``tool_calls`` and leaves the content
    empty, so anything that reads messages as text — the compaction
    summarizer, a log line — sees a blank where an action was. This gives it
    the call back in the envelope's own wording, which is also what the
    envelope protocol's copy of the same call reads like.
    """
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    rendered = []
    for call in calls:
        function = call.get("function") or {}
        rendered.append(f"{function.get('name')}({function.get('arguments') or '{}'})")
    return "[tool call] " + "; ".join(rendered)


def is_tool_call_message(message: Message) -> bool:
    """Is this assistant message a call the agent made, rather than an answer?

    What separates the two in the stored history — the transcript renders the
    call from its own anchored record (see :mod:`hpca.transcript`), so this
    copy is for the model only and must not surface as a reply. True for
    either protocol's copy: the native one is an assistant message carrying
    ``tool_calls``, the envelope one carries the decision object as content.
    """
    if message.get("role") != "assistant":
        return False
    if message.get("tool_calls"):
        return True
    content = str(message.get("content") or "").lstrip()
    if not content.startswith(CALL_PREFIX):
        return False
    try:
        decoded = json.loads(content)
    except ValueError:
        return False
    return isinstance(decoded, dict) and decoded.get("action") == CALL_ACTION
