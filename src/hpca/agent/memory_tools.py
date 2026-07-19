"""Memory tools: episodic recall (Phase 2) and curated memory edits (Phase 3).

``session_search`` costs no model call: FTS5 BM25 over persisted
user/assistant messages. Discovery mode returns Hermes-style bookends
(goal → match → resolution) per hit; read mode pages through one session.

``memory`` lets the agent propose edits to the profile's curated tiers the
moment a user states a preference or correction, instead of waiting for
``/conclude``. Unlike Hermes, which lets the model write directly, every
batch here goes through the approval dialog — a 27B is not trusted to
maintain its own memory unsupervised — and the tool result reports what the
user actually approved.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.episodic import MESSAGE_CHARS, EpisodicStore
from hpca.memory_ops import MemoryOp

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
    "Save durable facts to this profile's memory, which is injected into "
    "every future session. Make ALL changes in ONE call via the operations "
    "array: the batch is applied together and the size budget is checked "
    "only on the final result, so one call can remove or shorten stale "
    "entries AND add a new one. Tier 1 is for stable site facts (cluster, "
    "scheduler, filesystem layout), tier 2 for learnings, user preferences "
    "and workarounds. Address an existing entry by a short unique substring "
    "of its text. The user approves every change."
)


class MemoryOperation(BaseModel):
    op: str = Field(description="add, replace, or remove")
    tier: int = Field(default=2, description="1 for site facts, 2 for learnings")
    match: str = Field(
        default="",
        description="replace/remove: a short unique substring of the entry",
    )
    text: str = Field(default="", description="add/replace: the new entry text")


class MemoryParams(BaseModel):
    operations: list[MemoryOperation] = Field(
        description="The changes to apply together"
    )


async def memory(args: MemoryParams, ctx: ToolContext) -> str:
    """The write itself happens in the TUI, which owns the approval dialog and
    the profile file; the tool only validates and hands the batch over."""
    if ctx.propose_memory_edits is None:
        return "Memory editing is not available in this context."
    operations = [
        MemoryOp(op=o.op.strip().lower(), tier=o.tier, match=o.match, text=o.text)
        for o in args.operations
    ]
    return await ctx.propose_memory_edits(operations)


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
