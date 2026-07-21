"""Memory tools: episodic recall and deferred memory flagging.

``session_search`` costs no model call: FTS5 BM25 over persisted
user/assistant messages. Discovery mode returns Hermes-style bookends
(goal → match → resolution) per hit; read mode pages through one session.

``memory`` lets the agent *flag* durable facts the moment a user states a
preference or correction. Nothing is written when the tool is called: the
flagged edits are queued and surfaced for the user to approve together at the
next ``/conclude``. Memory is only ever generated when the user asks for it —
the agent proposes, it never commits.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.episodic import MESSAGE_CHARS, EpisodicStore
from hpca.memory_ops import MemoryOp
from hpca.profiles import MemoryScope

RESULT_BUDGET = 2500

SESSION_SEARCH_DESCRIPTION = (
    "Search past sessions of this profile (full-text). Shows what was said in "
    "earlier conversations — it is NOT evidence about the current state of "
    "files, jobs, or the cluster. Modes: query= finds matching sessions "
    "(goal, matching snippet, resolution each); session_id= (optionally "
    "around=<turn number>) reads one session's messages."
)


class SessionSearchParams(BaseModel):
    query: str = Field(
        default="",
        description="Full-text search over past sessions (omit when reading "
        "one session)",
    )
    session_id: str = Field(
        default="",
        description="Read this session's messages instead of searching",
    )
    around: int | None = Field(
        default=None,
        description="With session_id: center the excerpt on this turn number",
    )


def _clip_to_budget(lines: list[str]) -> str:
    text = "\n".join(lines)
    if len(text) <= RESULT_BUDGET:
        return text
    return text[: RESULT_BUDGET - 22] + "\n… [result truncated]"


async def session_search(args: SessionSearchParams, ctx: ToolContext) -> str:
    store: EpisodicStore | None = ctx.episodic
    if store is None or not store.fts_available:
        return (
            "Episodic search is unavailable (sqlite without FTS5 on this "
            "system)."
        )
    # Cross-profile recall is off by default: profiles exist to isolate what
    # each context learns and sees (redesign, resolved question 2).
    profile = None if ctx.settings.memory.cross_profile_search else ctx.profile
    if args.session_id:
        rows = store.window(args.session_id, around=args.around, profile=profile)
        if not rows:
            return f"No messages found for session {args.session_id!r}."
        lines = [f"Session {args.session_id} messages:"]
        for row in rows:
            content = " ".join(str(row["content"]).split())
            if len(content) > MESSAGE_CHARS:
                content = content[: MESSAGE_CHARS - 1] + "…"
            lines.append(f"[{row['turn_no']}] {row['role']}: {content}")
        if args.around is None:
            lines.append(
                "(session tail; pass around=<turn number> to read elsewhere)"
            )
        return _clip_to_budget(lines)
    if not args.query.strip():
        return "Give a query to search, or a session_id to read."
    hits = store.search(
        args.query, profile=profile, exclude_session_id=ctx.session_id
    )
    if not hits:
        return f"No past-session matches for {args.query!r} in this profile."
    lines = []
    for hit in hits:
        label = f"profile {hit.profile}, " if profile is None else ""
        lines += [
            f"Session “{hit.title}” ({label}{hit.created_at}, "
            f"id {hit.session_id}, match at turn {hit.turn_no}):",
            f"  goal: {hit.goal or '—'}",
            f"  match: {hit.snippet}",
            f"  resolution: {hit.resolution or '—'}",
        ]
    lines.append(
        "(use session_search with session_id= and around=<turn> for context)"
    )
    return _clip_to_budget(lines)


MEMORY_DESCRIPTION = (
    "Flag durable facts worth keeping in this profile's memory. Nothing is "
    "saved now: what you flag is collected and reviewed together when the user "
    "runs /conclude, and the user approves every entry then. Choose a scope: "
    "'system-prompt' for something worth putting in front of the assistant on "
    "every future turn (a stable site fact, a lasting preference, a "
    "correction) — kept small, so reserve it for what is always relevant; "
    "'rag' for a situational, one-topic learning recalled only when a later "
    "request resembles it. Put ALL changes in ONE call via the operations "
    "array. Address an existing entry by a short unique substring of its text; "
    "use 'demote' to move a system-prompt entry to rag."
)


class MemoryOperation(BaseModel):
    op: str = Field(description="add, replace, remove, or demote")
    scope: str = Field(
        default="system-prompt",
        description="system-prompt (injected every turn) or rag (retrieved "
        "only when relevant)",
    )
    match: str = Field(
        default="",
        description="replace/remove/demote: a unique substring of the entry",
    )
    text: str = Field(default="", description="add/replace: the new entry text")


class MemoryParams(BaseModel):
    operations: list[MemoryOperation] = Field(
        description="The changes to flag together"
    )


def _scope_of(raw: str) -> MemoryScope:
    try:
        return MemoryScope(raw.strip().lower())
    except ValueError:
        return MemoryScope.SYSTEM_PROMPT


async def memory(args: MemoryParams, ctx: ToolContext) -> str:
    """Flag a memory batch for review at the next /conclude. The tool never
    writes: it queues the proposal and reports that it was noted."""
    if ctx.queue_memory_edits is None:
        return "Memory flagging is not available in this context."
    operations = [
        MemoryOp(
            op=o.op.strip().lower(),
            scope=_scope_of(o.scope),
            match=o.match,
            text=o.text,
        )
        for o in args.operations
    ]
    return ctx.queue_memory_edits(operations)


def add_memory_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="session_search",
            description=SESSION_SEARCH_DESCRIPTION,
            params=SessionSearchParams,
            handler=session_search,
        )
    )
    registry.register(
        Tool(
            name="memory",
            description=MEMORY_DESCRIPTION,
            params=MemoryParams,
            handler=memory,
        )
    )
    return registry
