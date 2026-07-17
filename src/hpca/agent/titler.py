"""Session titles written by the model (§3 sessions column).

A session opens named after its first message, truncated mid-word — fine as a
placeholder, useless as a label once the column holds a dozen of them. This is
a firewalled call like the explainer: the transcript goes in, one short title
comes back, and the caller stays free to overrule it by hand.

Summarising is not reasoning, so thinking is forced off here: it would cost
minutes per session (see hpca.llm) to name a chat.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from hpca.agent.conclude import transcript
from hpca.llm import Message

TITLE_MAX_CHARS = 40
TITLE_MAX_RETRIES = 1

SYSTEM_PROMPT = (
    "You name a conversation between a scientist and an HPC assistant, for a "
    "narrow sidebar listing many of them. Answer with a title of at most six "
    "words naming the task or question at hand — no quotes, no trailing "
    "period, no prefix like 'Session:' or 'Chat about'. Name what the user "
    "wanted, not what the assistant replied."
)

TITLE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "maxLength": 80}},
    "required": ["title"],
    "additionalProperties": False,
}


class _Title(BaseModel):
    title: str = Field(min_length=1)


class TitleError(Exception):
    """No usable title within the retry budget."""


def clean_title(raw: str) -> str:
    """Small models decorate; strip the decoration and keep it column-sized."""
    title = " ".join(raw.split()).strip("\"'` ").removesuffix(".")
    for prefix in ("session:", "title:", "chat about", "conversation about"):
        if title.lower().startswith(prefix):
            title = title[len(prefix) :].strip(": ")
    if len(title) > TITLE_MAX_CHARS:
        cut = title[:TITLE_MAX_CHARS].rsplit(" ", 1)[0]
        title = (cut or title[:TITLE_MAX_CHARS]).rstrip(",;:-") + "…"
    return title


async def propose_title(
    llm,
    messages: list[Message],
    *,
    max_retries: int = TITLE_MAX_RETRIES,
) -> str:
    """One short title for this conversation; raises TitleError if none."""
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript(messages)},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=TITLE_SCHEMA,
            schema_name="session_title",
            max_tokens=64,
            enable_thinking=False,
        )
        try:
            title = clean_title(_Title.model_validate(json.loads(response.content)).title)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
        else:
            if title:
                return title
            last_error = "the title was empty"
        conversation = conversation + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"[validation error] {last_error}"},
        ]
    raise TitleError(f"No usable title after {max_retries + 1} attempts: {last_error}")
