"""Skill tool (§5.1): fetch a user-defined procedure on demand."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry


class ReadSkillParams(BaseModel):
    name: str = Field(description="Name of the skill to read")


async def read_skill(args: ReadSkillParams, ctx: ToolContext) -> str:
    if not ctx.skills:
        return "There are no skills configured for this profile."
    for skill in ctx.skills:
        if skill.name == args.name:
            return f"Skill {skill.name!r} — {skill.description}\n\n{skill.body}"
    available = ", ".join(s.name for s in ctx.skills)
    return f"No skill named {args.name!r}. Available skills: {available}"


def add_skill_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="read_skill",
            description="Read the full procedure of a user-defined skill",
            params=ReadSkillParams,
            handler=read_skill,
        )
    )
    return registry
