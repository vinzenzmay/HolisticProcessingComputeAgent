"""Memory tools (redesign Phase 2): episodic recall via ``session_search``.

Recall costs no model call: FTS5 BM25 over persisted user/assistant messages.
Discovery mode returns Hermes-style bookends (goal → match → resolution) per
hit; read mode pages through one session's messages. Output is char-bounded
like every other tool, so a hit-rich search cannot flood a 27B's context.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.episodic import MESSAGE_CHARS, EpisodicStore

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


def add_memory_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="session_search",
            description=SESSION_SEARCH_DESCRIPTION,
            params=SessionSearchParams,
            handler=session_search,
        )
    )
    return registry
