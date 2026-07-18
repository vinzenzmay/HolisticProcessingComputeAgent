"""Checkpointed agent state → the entries a human reads (chat window, logs).

A turn is a user message, the agent's working, and the answer. The working —
the model's reasoning and the tool steps it led to, interleaved in the order
they happened — is folded into a single ``thinking`` entry, which the chat
window shows as one collapsible box and the session log writes as one block.

Reasoning is anchored by the message index it produced (``after``), never fed
back to the model, and lives outside ``messages`` for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from hpca.llm import Message

TOOL_PREFIXES = ("[tool result]", "[tool error]")
# Background work reporting in on its own (§5.4). Rides the user role like
# tool results do, and is marked so the transcript does not attribute a
# process crash to the human sitting there.
EVENT_PREFIXES = ("[process ", "[job ")

USER = "user"
ASSISTANT = "assistant"
THINKING = "thinking"
ERROR = "error"
EVENT = "event"


@dataclass
class Entry:
    kind: str  # user | assistant | thinking | error
    text: str
    steps: int = 0  # thinking: tool steps folded in
    reasoning_chars: int = 0  # thinking: how much the model thought

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


def _block(parts: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"— {kind} —\n{text.strip()}" for kind, text in parts)


def build_entries(
    messages: list[Message],
    thinking: list[dict] | None = None,
    *,
    start: int = 0,
) -> list[Entry]:
    """Entries for ``messages[start:]``, with reasoning and steps folded in.

    ``start`` selects a tail (one turn, for incremental logging) while keeping
    the absolute message indices that ``thinking`` entries are anchored to.
    """
    reasoning_at: dict[int, list[str]] = {}
    for entry in thinking or []:
        reasoning_at.setdefault(entry["after"], []).append(entry["reasoning"])

    entries: list[Entry] = []
    pending: list[tuple[str, str]] = []  # the open thinking box
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
                )
            )
        pending, steps, reasoning_chars = [], 0, 0

    for index in range(start, len(messages)):
        message = messages[index]
        for reasoning in reasoning_at.get(index, []):
            if reasoning.strip():
                pending.append(("reasoning", reasoning))
                reasoning_chars += len(reasoning)
        role, content = message["role"], str(message["content"])
        if role == "system":
            continue
        if role == ASSISTANT:
            flush()  # the answer closes the box that produced it
            entries.append(Entry(kind=ASSISTANT, text=content))
        elif is_tool_message(message):
            pending.append(("step", content))
            steps += 1
        elif is_event_message(message):
            flush()
            entries.append(Entry(kind=EVENT, text=content))
        else:
            flush()
            entries.append(Entry(kind=USER, text=content))
    flush()  # a turn interrupted for approval leaves its box open
    return entries
