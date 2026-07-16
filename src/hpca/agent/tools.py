"""Plugin-style tool registry (§5.1).

Every tool couples a pydantic parameter schema with an async handler. Agents
receive a *subset* registry (few tools per agent, §1); destructive tools are
flagged so the HITL gate (§5.3) can intercept them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

Handler = Callable[[BaseModel, Any], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    params: type[BaseModel]
    handler: Handler
    destructive: bool = False
    # Conditional destructiveness (e.g. move only when the target exists).
    # Must be a side-effect-free, deterministic predicate: the graph node
    # re-runs it when resuming from an interrupt.
    is_destructive_call: Callable[[BaseModel, Any], bool] | None = None
    # Human-readable description of one call with *resolved real paths* for
    # the confirmation modal (§5.3). Same purity rules as above.
    describe_call: Callable[[BaseModel, Any], str] | None = None

    def gates(self, arguments: BaseModel, ctx: Any) -> bool:
        """Whether this specific call needs the HITL gate (§5.3)."""
        if self.destructive:
            return True
        if self.is_destructive_call is not None:
            return self.is_destructive_call(arguments, ctx)
        return False


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(
                f"Unknown tool {name!r}. Available tools: {', '.join(self.names())}"
            )
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """Restricted view for one agent; unknown names fail at wiring time."""
        return ToolRegistry({name: self.get(name) for name in names})

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
