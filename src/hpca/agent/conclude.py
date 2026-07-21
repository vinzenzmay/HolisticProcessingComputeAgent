"""`/memorize <note>`: the model turns the user's note plus the conversation
into durable memory proposals; the user approves each.

Firewalled like the explainer: the conversation transcript enters a fresh
LLM call; only validated proposals come back. Writing anything to the
profile requires per-proposal user approval in the TUI (§5.3 consistency —
a small model writes these, so review is essential).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from hpca.llm import Message
from hpca.profiles import MemoryScope

MAX_PROPOSALS = 5
MAX_TRANSCRIPT_MESSAGES = 60
MAX_MESSAGE_CHARS = 500
CONCLUDE_MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You review a conversation between a scientist and an HPC assistant and "
    "propose durable memories worth keeping for future sessions. Propose only "
    "facts that will still be true later. Choose a scope for each: "
    "'system-prompt' for something worth putting in front of the assistant on "
    "every future turn (a stable site fact, a lasting preference, a "
    "correction) — kept small, so reserve it for what is always relevant; "
    "'rag' for a situational, one-topic learning recalled only when a later "
    "request resembles it. Do NOT propose session-specific details like file "
    "names or job ids. Propose nothing if the conversation contained nothing "
    "durable."
)

PROPOSALS_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": MAX_PROPOSALS,
            "items": {
                "type": "object",
                "properties": {
                    "scope": {"enum": ["system-prompt", "rag"]},
                    "kind": {
                        "enum": ["fact", "learning", "preference", "workaround"]
                    },
                    "text": {"type": "string"},
                },
                "required": ["scope", "kind", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


class MemoryProposal(BaseModel):
    scope: MemoryScope
    kind: str
    text: str


class _Proposals(BaseModel):
    proposals: list[MemoryProposal]


class ConcludeError(Exception):
    """No valid proposal list within the retry budget."""


def transcript(messages: list[Message]) -> str:
    """Bounded plain-text transcript for the proposal prompt."""
    lines = []
    for message in messages[-MAX_TRANSCRIPT_MESSAGES:]:
        if message["role"] == "system":
            continue
        content = message["content"][:MAX_MESSAGE_CHARS]
        lines.append(f"{message['role']}: {content}")
    return "\n".join(lines)


async def propose_memories(
    llm,
    messages: list[Message],
    *,
    system_prompt_memories: str = "",
    guidance: str = "",
    max_retries: int = CONCLUDE_MAX_RETRIES,
) -> list[MemoryProposal]:
    """Memory proposals from the conversation; ``guidance`` is the user's own
    /memorize note — what they, not the model, decided is worth keeping."""
    system = SYSTEM_PROMPT
    if system_prompt_memories:
        system += (
            "\n\nAlready-known system-prompt memories (do not propose "
            "duplicates):\n" + system_prompt_memories
        )
    prompt = transcript(messages)
    if guidance:
        prompt += (
            "\n\nThe user explicitly asked to memorize the following — "
            "propose the durable memory (or memories) it implies, using the "
            f"conversation above for context:\n{guidance}"
        )
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=PROPOSALS_SCHEMA,
            schema_name="memory_proposals",
            max_tokens=1024,
        )
        try:
            return _Proposals.model_validate(
                json.loads(response.content)
            ).proposals
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
    raise ConcludeError(
        f"No valid proposals after {max_retries + 1} attempts: {last_error}"
    )
