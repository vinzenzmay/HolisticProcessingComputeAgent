"""Struggle detection and matching (§4.4).

When a turn goes badly — retries exhausted, a job failing repeatedly, the
user aborting — that is a signal worth reflecting on, so ``turn_struggled``
triggers a self-review immediately rather than waiting for the counter (the
proposal itself is written by ``hpca.agent.reflect``).

Stored struggle notes carry a keyword line, and ``matching_struggles`` finds
the ones a new request resembles. Those matches reach the user as a warning
and the model as a fenced memory-context block, so both know the ground has
been difficult before.
"""

from __future__ import annotations

import re

from hpca.llm import Message
from hpca.profiles import Memory

STRUGGLE_KIND = "struggle"

FAILURE_MARKERS = (
    "[tool error]",
    "I failed to produce a valid action",
    "tool budget exhausted",
)


def turn_struggled(messages: list[Message], *, aborted: bool = False) -> bool:
    """Whether the last turn shows a struggle worth reflecting on (§4.4)."""
    if aborted:
        return True
    return any(
        marker in message["content"]
        for message in messages
        for marker in FAILURE_MARKERS
    )


def note_keywords(memory: Memory) -> list[str]:
    match = re.search(r"^keywords:\s*(.+)$", memory.text, re.MULTILINE)
    if not match:
        return []
    return [kw.strip().lower() for kw in match.group(1).split(",") if kw.strip()]


def matching_struggles(memories: list[Memory], text: str) -> list[Memory]:
    """Struggle notes whose keywords appear in the text (§4.4 keyword match)."""
    lowered = text.lower()
    matches = []
    for memory in memories:
        if memory.kind != STRUGGLE_KIND:
            continue
        for keyword in note_keywords(memory):
            if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lowered):
                matches.append(memory)
                break
    return matches
