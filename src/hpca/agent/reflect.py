"""The self-review loop (redesign Phase 4).

Periodically — not only after a visible failure — the agent looks back at the
conversation and proposes what is worth keeping: durable memories, patches to
a skill it followed, or a struggle note. Firewalled like ``conclude`` and
``struggle``: one bounded call, JSON-schema-constrained output, never fed
back into the session context.

Two departures from Hermes, both because the reviewer here is a 27B rather
than a frontier model:

* Hermes forks a full agent with tool access to curate its own library. This
  is a single call returning proposals, which the user then approves — much
  cheaper, and the failure mode of a small model with write access to its own
  memory is exactly what this whole redesign is trying to avoid.
* Hermes tells its reviewer that "nothing to save" is a missed opportunity.
  Here it is explicitly fine. An eager small reviewer producing junk every
  few turns would erode trust in the approval dialog until the user starts
  reflexively rejecting everything — which costs more than a missed learning.

The anti-capture list is adapted closely from Hermes' skill-review prompt: it
is cheap, prompt-level protection against a model poisoning its own future.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from hpca.llm import Message

MAX_TRANSCRIPT_MESSAGES = 24  # verbatim tail; older turns are digested
MAX_MESSAGE_CHARS = 400
MAX_DIGEST_MESSAGES = 40
MAX_PROPOSALS = 4
REFLECT_MAX_RETRIES = 1
# Tier 3 is unbounded by design, but the review prompt is not: enough of it
# to stop the obvious re-proposals without swamping a 27B's attention.
KNOWN_TIER3_CHARS = 2000

# Adapted from Hermes' _SKILL_REVIEW_PROMPT. Every "do not capture" line is a
# real self-poisoning mode: a small model that writes "browser tools do not
# work" once will cite that memory against itself for months, long after the
# actual problem is fixed.
ANTI_CAPTURE = (
    "Do NOT propose any of these — they become self-imposed constraints that "
    "bite you later:\n"
    "- Failures that were about this moment, not this site: a full disk, a "
    "busy queue, a typo the user already fixed.\n"
    "- Negative claims about tools or capabilities (\"X does not work here\", "
    "\"the cluster has no Y\"). These harden into refusals you will cite "
    "against yourself long after the problem is fixed. If you could not do "
    "something, the durable lesson is what DID work.\n"
    "- Anything that resolved on a retry. Then the lesson is the retry, not "
    "the failure.\n"
    "- A narrative of what happened this session. Past sessions are already "
    "searchable; memory is for what stays true.\n"
    "- File names, job ids, paths, dates, or anything stale within a week."
)

SYSTEM_PROMPT = (
    "You review a finished stretch of conversation between a scientist and "
    "an HPC assistant, and decide what — if anything — is worth keeping for "
    "future sessions.\n\n"
    "Propose only what will still be true and useful weeks from now:\n"
    "- memory (tier 1): stable facts about this site — cluster, scheduler, "
    "filesystem layout, module system, site policy.\n"
    "- memory (tier 2): user preferences, corrections the user made, and "
    "workarounds that proved necessary on this backend.\n"
    "- struggle: a difficulty worth warning about next time, in 1-2 "
    "sentences, with 2-5 keywords naming the tools, formats or operations "
    "that identify a similar future task.\n"
    "- skill_patch: a correction to a skill that was followed and turned out "
    "wrong, missing a step, or outdated. Give the skill's name and the "
    "correction.\n\n"
    "Write memories as declarative facts, never as instructions to yourself: "
    "\"the user prefers R\" is right, \"always answer in R\" is wrong.\n\n"
    + ANTI_CAPTURE
    + "\n\nProposing nothing is a perfectly good outcome, and the right one "
    "for most ordinary conversations. Only propose what you would want to "
    "read in a month."
)

SKILL_CREATION_CLAUSE = (
    "\n\nYou may also propose skill_new: a NEW skill for a class of task that "
    "came up and has no skill yet. Name it for the class of work, never for "
    "today's specific case — a name that only makes sense for this session is "
    "the wrong name. Propose one only when the procedure is genuinely "
    "reusable."
)

KINDS = ["memory", "struggle", "skill_patch", "skill_new"]

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": MAX_PROPOSALS,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"enum": KINDS},
                    "tier": {"enum": [1, 2]},
                    "text": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {"type": "string"},
                    },
                    "skill_name": {"type": "string"},
                },
                "required": ["kind", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


class Reflection(BaseModel):
    kind: Literal["memory", "struggle", "skill_patch", "skill_new"]
    text: str
    tier: Literal[1, 2] = 2
    keywords: list[str] = []
    skill_name: str = ""

    def memory_text(self) -> str:
        """Storage form. Struggle notes carry a matchable keyword line."""
        if self.kind == "struggle" and self.keywords:
            return f"{self.text}\nkeywords: {', '.join(self.keywords)}"
        return self.text

    def describe(self) -> str:
        if self.kind == "memory":
            return f"tier {self.tier} memory"
        if self.kind == "struggle":
            return "struggle note"
        if self.kind == "skill_patch":
            return f"patch to skill “{self.skill_name}”"
        return f"new skill “{self.skill_name}”"


class _Reflections(BaseModel):
    proposals: list[Reflection]


class ReflectError(Exception):
    """No valid proposal list within the retry budget."""


def _cap(text: str, limit: int) -> str:
    """Keep the most recent entries: they are the ones a review is likeliest
    to duplicate, since they came from recent work."""
    if len(text) <= limit:
        return text
    return "…\n\n" + text[-limit:]


def _line(message: Message) -> str:
    content = " ".join(str(message["content"]).split())[:MAX_MESSAGE_CHARS]
    return f"{message['role']}: {content}"


def digest(messages: list[Message], *, span: str = "recent") -> str:
    """The reviewed stretch, bounded for a small model's attention.

    ``span="recent"`` (the ordinary cadence) keeps the recent turns verbatim
    and compresses what came before: the session is still going, and the
    learnings are in what just happened.

    ``span="whole"`` is for the pre-eviction review, where the whole stretch
    is about to be discarded. Both ends stay verbatim — the beginning of a
    stretch is where the goal and the site facts are stated, and with a
    recency bias that is exactly what would be lost forever.
    """
    usable = [message for message in messages if message["role"] != "system"]
    if span == "whole" and len(usable) > MAX_DIGEST_MESSAGES:
        half = MAX_TRANSCRIPT_MESSAGES // 2
        head, tail = usable[:half], usable[-half:]
        middle = len(usable) - len(head) - len(tail)
        return "\n".join(
            [_line(m) for m in head]
            + [f"[{middle} messages omitted]"]
            + [_line(m) for m in tail]
        )
    usable = usable[-MAX_DIGEST_MESSAGES:]
    older, recent = usable[:-MAX_TRANSCRIPT_MESSAGES], usable[-MAX_TRANSCRIPT_MESSAGES:]
    lines = []
    if older:
        summary = "; ".join(
            " ".join(str(m["content"]).split())[:80]
            for m in older
            if m["role"] == "user"
        )
        lines.append(f"[earlier in this session: {summary[:600]}]")
    lines += [_line(message) for message in recent]
    return "\n".join(lines)


async def propose_reflections(
    llm,
    messages: list[Message],
    *,
    tier1: str = "",
    tier2: str = "",
    tier3: str = "",
    skills: str = "",
    allow_new_skills: bool = True,
    span: str = "recent",
    max_retries: int = REFLECT_MAX_RETRIES,
) -> list[Reflection]:
    """What this stretch of conversation is worth remembering, if anything.

    All three tiers are listed as already-known. Tier 3 especially: a
    retrieved note is *in the conversation* when the reviewer reads it, so
    without this the reviewer re-proposes what it just saw — and proposes it
    for tier 1 or 2, promoting a situational note into the always-injected
    budget that tier 3 exists to keep it out of.
    """
    system = SYSTEM_PROMPT
    if allow_new_skills:
        system += SKILL_CREATION_CLAUSE
    known = "\n\n".join(
        part for part in (tier1, tier2, _cap(tier3, KNOWN_TIER3_CHARS)) if part
    )
    if known:
        system += (
            "\n\nAlready in memory — do NOT propose these again, and do not "
            "propose near-duplicates:\n" + known
        )
    if skills:
        system += "\n\nExisting skills:\n" + skills
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": digest(messages, span=span)},
    ]
    last_error = ""
    for _ in range(max_retries + 1):
        response = await llm.chat(
            conversation,
            json_schema=REFLECTION_SCHEMA,
            schema_name="reflection",
            max_tokens=1024,
        )
        try:
            parsed = _Reflections.model_validate(json.loads(response.content))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            conversation = conversation + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"[validation error] {last_error}"},
            ]
            continue
        return [
            proposal
            for proposal in parsed.proposals
            if proposal.text.strip()
            and (
                proposal.kind in ("memory", "struggle")
                or proposal.skill_name.strip()
            )
            and (allow_new_skills or proposal.kind != "skill_new")
        ]
    raise ReflectError(
        f"No valid reflection after {max_retries + 1} attempts: {last_error}"
    )
