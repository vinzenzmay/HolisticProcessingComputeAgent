"""Self-reflection: struggle notes (§4.4).

When a turn goes badly — retries exhausted, a job failing repeatedly, the user
aborting — the orchestrator characterizes the problem in 1–2 sentences and
proposes a struggle note for the profile. Notes are matched against future
requests by keyword/tag so the agent can warn up front ("I have struggled with
X before") and let the user decide whether to try anyway.

Like every memory write (§6.3), a proposal reaches the profile only through
user approval.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from hpca.llm import Message
from hpca.profiles import Memory

STRUGGLE_KIND = "struggle"
MAX_TRANSCRIPT_MESSAGES = 40
MAX_MESSAGE_CHARS = 400

FAILURE_MARKERS = (
    "[tool error]",
    "I failed to produce a valid action",
    "tool budget exhausted",
)

SYSTEM_PROMPT = (
    "You review a conversation where an HPC assistant struggled with a task. "
    "In 1-2 sentences, characterize what went wrong in a way that will be "
    "useful the next time a similar task comes up. Also give 2-5 short "
    "keywords (tools, file formats, operations) that identify similar future "
    "tasks. Describe the difficulty, not this session's specifics."
)

STRUGGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "description": "1-2 sentence characterization"},
        "keywords": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string"},
        },
    },
    "required": ["note", "keywords"],
    "additionalProperties": False,
}


class StruggleNote(BaseModel):
    note: str
    keywords: list[str]

    def render(self) -> str:
        """Storage form: the note plus a matchable keyword line."""
        return f"{self.note}\nkeywords: {', '.join(self.keywords)}"


class StruggleError(Exception):
    """No valid struggle note within the retry budget."""


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


def transcript(messages: list[Message]) -> str:
    lines = []
    for message in messages[-MAX_TRANSCRIPT_MESSAGES:]:
        if message["role"] == "system":
            continue
        lines.append(f"{message['role']}: {message['content'][:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines)


async def propose_struggle_note(
    llm, messages: list[Message], *, max_retries: int = 2
) -> StruggleNote:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript(messages)},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=STRUGGLE_SCHEMA,
            schema_name="struggle_note",
            max_tokens=512,
        )
        try:
            return StruggleNote.model_validate(json.loads(response.content))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
    raise StruggleError(
        f"No valid struggle note after {max_retries + 1} attempts: {last_error}"
    )
