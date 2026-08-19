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

What an omitted payload is replaced *by* is the load-bearing part of that. It
used to be a head of the real lines plus "... 97 more lines elided ...", which
reads exactly like content — and models copied it back: asked to rewrite a file
they had already written, they reproduced their own elided record as the new
``content_lines``, so the marker landed on disk and the file shrank to the
three kept lines. The next rewrite elided *that*, and the file converged on
four. Nothing in the tools noticed, so the model concluded its own writer was
truncating and burned a dozen rounds bisecting a bug that did not exist. So an
omission is now named as one — a descriptor inside a sentinel no file line
carries, with no copyable fragment of the payload in it — and the file tools
refuse content that carries the sentinel back (``carries_elision_marker``).

The result half rides the user role for the same reason it always has: vLLM /
Qwen templates reject mid-conversation system messages, and the native tool
role requires the tool-call protocol this design deliberately bypasses.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hpca.llm import Message

# How many of the most recent calls keep their payload in the model's view.
# This is the whole mechanism: a model that reproduces its own record does it
# *immediately* — the next call, while it is still finishing the job the record
# belongs to — so the record it reaches for is never an old one. Keeping the
# recent few intact means there is nothing wrong to copy, and the window still
# never carries more than a handful of payloads at once. Three covers
# write-then-rewrite and write-a-second-file-like-the-first; beyond that the
# model is on to something else and the payload is dead weight.
KEEP_RECENT_CALLS = 3

# What counts as a payload at all, now that folding is about age rather than
# every write. Deliberately far above the old 12-item cap: at that size the
# fold was firing on ordinary arguments — a 20-line edit, a 15-line config —
# and paying the risk of a mangled record for a few hundred characters of
# window. Only something that would genuinely cost to carry twice is worth
# replacing with a description of itself.
MAX_LIST_CHARS = 4000
# Same for a long string argument (a here-doc script, a pasted block). A string
# keeps its head, where a list does not: a cut string still reads as a
# fragment, while a head of plausible file lines reads as the file.
MAX_STRING_CHARS = 2000

# What every omission is wrapped in. A doubled angle bracket is not something a
# TSV row, a shell line or an R chunk carries, which is what makes the marker
# both unmistakable to the model and cheap to detect if it comes back anyway.
ELISION_SENTINEL = "<<HPCA:"
ELISION_CLOSE = ">>"
# The pre-0.23.3 marker, still sitting in checkpointed sessions and in the files
# already written from one. Detected so the tool guard catches those too.
_LEGACY_MARKER = re.compile(r"\.\.\. *\d+ more (?:lines|chars) elided *\.\.\.")

CALL_ACTION = "tool_call"
# What the assistant copy of a call starts with. Cheap prefix test for
# ``is_tool_call_message`` — the transcript asks it about every message.
CALL_PREFIX = '{"action": "tool_call"'


def omitted_list(value: list) -> str:
    """The descriptor that stands in for a payload list.

    Deliberately not a fragment of it: the model is told the shape of what it
    wrote — how many lines, how many characters — and where the content still
    lives, and is given nothing it could mistake for the lines themselves.
    Losing sight of the first three lines costs it little; what it wrote is one
    ``read_file`` away, and the file, unlike this record, is not a copy.
    """
    chars = sum(len(item) for item in value if isinstance(item, str))
    return (
        f"{ELISION_SENTINEL} {len(value)} lines ({chars} chars) written "
        f"earlier, not repeated here{ELISION_CLOSE}"
    )


def elide(value: Any) -> Any:
    """One argument value, cut down to its shape.

    A payload list is replaced by a descriptor of it, a long string keeps its
    head and says what follows it, and either way the cut is *named* rather
    than silently applied: a model reading its own call back must be able to
    tell an argument it truncated from one it wrote short. Applied inside lists
    and dicts too, so a payload nested one level down is not a way around the
    cap.
    """
    if isinstance(value, list):
        if sum(len(item) for item in value if isinstance(item, str)) > MAX_LIST_CHARS:
            return omitted_list(value)
        return [elide(item) for item in value]
    if isinstance(value, dict):
        return {key: elide(item) for key, item in value.items()}
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        dropped = len(value) - MAX_STRING_CHARS
        return (
            f"{value[:MAX_STRING_CHARS]}{ELISION_SENTINEL} {dropped} more "
            f"chars written earlier, not repeated here{ELISION_CLOSE}"
        )
    return value


def carries_elision_marker(text: str) -> bool:
    """Is this line something the history handed the model, not real content?

    True for the sentinel above and for the pre-0.23.3 wording, which survives
    in checkpointed sessions and in the files already written from one. The
    file tools ask this of every line they are about to write: a payload
    carrying either marker is the model quoting its own record back at itself,
    and writing it puts the marker on disk (see the module docstring).
    """
    return ELISION_SENTINEL in text or bool(_LEGACY_MARKER.search(text))


def elide_arguments(arguments: dict) -> dict:
    """The call's arguments as the assistant copy carries them."""
    return {name: elide(value) for name, value in (arguments or {}).items()}


def call_json(tool_name: str, arguments: dict) -> str:
    """The decision envelope for this call, serialized as the model emits it.

    Stored whole. Folding is a property of the *view* now, not of the record
    (see :func:`fold_old_payloads`), so what goes into the history is what the
    model actually emitted — which is also what makes the fold reversible as
    the conversation moves on.
    """
    return json.dumps(
        {"action": CALL_ACTION, "tool": tool_name, "arguments": arguments or {}},
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
    the result to it. Stored whole, like the envelope copy, and folded by the
    same view function — the protocol changes where the arguments sit, not when
    they are worth carrying.
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
                        arguments or {}, ensure_ascii=False, default=str
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


def _fold_call(message: Message) -> Message:
    """One stored call message with its payload arguments replaced by a
    description of them. Returned unchanged if it cannot be read as a call —
    a fold is an optimisation, and losing a message to a parse error would not
    be one.
    """
    if message.get("tool_calls"):
        folded = []
        for call in message["tool_calls"]:
            function = dict(call.get("function") or {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                folded.append(call)
                continue
            function["arguments"] = json.dumps(
                elide_arguments(arguments), ensure_ascii=False, default=str
            )
            folded.append({**call, "function": function})
        return {**message, "tool_calls": folded}
    try:
        envelope = json.loads(str(message.get("content") or ""))
    except ValueError:
        return message
    if not isinstance(envelope, dict) or "arguments" not in envelope:
        return message
    envelope["arguments"] = elide_arguments(envelope["arguments"])
    return {
        **message,
        "content": json.dumps(envelope, ensure_ascii=False, default=str),
    }


def fold_old_payloads(
    messages: list[Message], *, keep_recent: int = KEEP_RECENT_CALLS
) -> list[Message]:
    """The stored history as the model should see it: recent calls whole, old
    ones described.

    Folding used to happen when the call was written, which meant the model's
    record of what it had just done was already a summary by the time it
    composed its next move — and a model that is mid-job reaches for exactly
    that record. It reproduced the description as content, and the description
    went to disk. Deferring the fold makes the problem disappear rather than
    guarding against it: while the record is still being used it is intact, and
    by the time it is folded the model has moved on and only needs to know that
    something was written and roughly how much.

    The stored history is never rewritten — same rule compaction follows. This
    builds a view, so a call that folds on one turn is still whole in the
    transcript, and a payload never has to be reconstructed to be shown.
    """
    calls = [index for index, m in enumerate(messages) if is_tool_call_message(m)]
    whole = set(calls[-keep_recent:]) if keep_recent > 0 else set()
    stale = set(calls) - whole
    return [
        _fold_call(message) if index in stale else message
        for index, message in enumerate(messages)
    ]


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
