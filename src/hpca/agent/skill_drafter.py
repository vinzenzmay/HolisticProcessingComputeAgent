"""First drafts for `/skill-creator <what the skill should do>` (§5.1).

`/skill-creator` on its own opens an empty form. Given a description, the
model writes a draft first — name, description, body — and the same form
opens with those three fields filled in. Nothing is written without the user
editing and confirming, so this is a head start, not an author.

Firewalled like the titler: the request plus a bounded transcript go in, one
validated draft comes back. Drafting a procedure is writing, not reasoning,
so thinking is forced off (see hpca.llm) — the form has to appear in a couple
of seconds or the user would sooner have typed it.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from hpca.agent.conclude import transcript
from hpca.llm import Message

DRAFT_MAX_RETRIES = 1
NAME_MAX_CHARS = 40
# The schema's own cap: a description that reaches it is already unusual, so
# the ellipsis is a safety net, not the normal path — truncating the ordinary
# 15-word description would put a "…" in the form for the user to tidy up.
DESCRIPTION_MAX_CHARS = 200

SYSTEM_PROMPT = (
    "You draft a skill for an HPC assistant: a short, reusable procedure the "
    "assistant follows when a task of that kind comes up. The user has "
    "described what they want it to do; write the first draft, which they "
    "will edit before saving.\n\n"
    "Return three fields:\n"
    "- name: a short kebab-case handle, lowercase, two or three words "
    "(e.g. 'submit-slurm-job'). It becomes a slash command, so name the "
    "CLASS of work, never today's specific case.\n"
    "- description: one line of at most 25 words saying when the assistant "
    "should use this skill — the trigger, not a restatement of the name.\n"
    "- body: the procedure itself, in markdown. Numbered steps when order "
    "matters, otherwise short paragraphs or bullets. Write instructions to "
    "the assistant in the imperative ('Check the queue with squeue -u "
    "$USER'), name the concrete commands, tools and files to use, and say "
    "what to do when a step fails. Where a value changes from run to run, "
    "write a placeholder — <job-id>, the run directory — and never the "
    "actual id, path or file name from the request, not even as an example. "
    "Keep it under 30 lines: a procedure, not an essay, and no preamble "
    "about what a skill is.\n\n"
    "If a conversation is provided, mine it for the specifics — the actual "
    "commands, formats and pitfalls that came up — but generalise them into "
    "a procedure that works next time, not a log of this session. Today's "
    "job ids, directories and file names do not belong in the skill: say "
    "what to look for ('the run directory', 'the job id from sbatch') "
    "instead of naming this run's."
)

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 60},
        "description": {"type": "string", "maxLength": 200},
        "body": {"type": "string"},
    },
    "required": ["name", "description", "body"],
    "additionalProperties": False,
}


class SkillDraft(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    body: str = Field(min_length=1)


class DraftError(Exception):
    """No usable skill draft within the retry budget."""


def clean_name(raw: str) -> str:
    """A model-written name as a slash-command handle: kebab-case, bounded.

    ``skill_path`` sanitises the filename anyway, but a name with spaces or
    capitals can never be invoked as ``/<skill>`` (``_all_commands`` drops
    those), so the draft is normalised before it reaches the form.
    """
    name = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    if len(name) > NAME_MAX_CHARS:
        cut = name[:NAME_MAX_CHARS].rsplit("-", 1)[0]
        name = (cut or name[:NAME_MAX_CHARS]).strip("-")
    return name


def clean_description(raw: str) -> str:
    """One line, no decoration — the form's description field is an Input."""
    text = " ".join(raw.split()).strip("\"'` ")
    if len(text) > DESCRIPTION_MAX_CHARS:
        cut = text[:DESCRIPTION_MAX_CHARS].rsplit(" ", 1)[0]
        text = (cut or text[:DESCRIPTION_MAX_CHARS]).rstrip(",;:-") + "…"
    return text


def clean_body(raw: str) -> str:
    """Strip a fenced-code wrapper: small models like to wrap the procedure."""
    body = raw.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            body = "\n".join(lines[1:-1]).strip()
    return body


async def propose_skill(
    llm,
    request: str,
    *,
    messages: list[Message] | None = None,
    existing: str = "",
    max_retries: int = DRAFT_MAX_RETRIES,
) -> SkillDraft:
    """One skill draft for ``request``; raises DraftError if none validates.

    ``messages`` is the open conversation — the user says "based on our
    conversation" more often than not. ``existing`` is ``summarize_skills``
    output, so the draft neither duplicates a name nor re-invents a procedure
    the profile already has.
    """
    system = SYSTEM_PROMPT
    if existing:
        system += (
            "\n\nSkills this profile already has — do NOT reuse one of these "
            "names, and do not redraft a procedure that already exists:\n"
            + existing
        )
    prompt = f"Write a skill that does the following:\n{request}"
    if messages:
        prompt += (
            "\n\nThe conversation so far, for context (the skill must stand "
            "on its own without it):\n" + transcript(messages)
        )
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=DRAFT_SCHEMA,
            schema_name="skill_draft",
            max_tokens=1536,
            enable_thinking=False,
        )
        try:
            draft = SkillDraft.model_validate(json.loads(response.content))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
        else:
            name = clean_name(draft.name)
            body = clean_body(draft.body)
            if name and body:
                return SkillDraft(
                    name=name,
                    description=clean_description(draft.description),
                    body=body,
                )
            last_error = "the draft had no usable name or body"
        conversation = conversation + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"[validation error] {last_error}"},
        ]
    raise DraftError(
        f"No usable skill draft after {max_retries + 1} attempts: {last_error}"
    )
