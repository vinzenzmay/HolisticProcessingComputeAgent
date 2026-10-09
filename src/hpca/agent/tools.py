"""Plugin-style tool registry (§5.1).

Every tool couples a pydantic parameter schema with an async handler. Agents
receive a *subset* registry (few tools per agent, §1); destructive tools are
flagged so the HITL gate (§5.3) can intercept them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    # The body of what is being decided, when a call has one its arguments do
    # not show — source, a diff. It fills the approval's scrolling block under
    # describe_call's sentence, as a script does for run_bash, and is kept with
    # the call for the chat. Same purity rules as above.
    show_call: Callable[[BaseModel, Any], str | None] | None = None
    # The file a user tool was loaded from (hpca.user_tools), None for every
    # tool HPCA ships. Set by the loader, never by the file: it is what lets
    # manual mode treat code nobody reviewed as an execution tool.
    user_file: Path | None = None
    # Whether a call runs code nobody has reviewed, so manual mode (§3.5) asks
    # before it the way it asks before run_bash. Same purity rules as above.
    runs_code: Callable[[BaseModel, Any], bool] | None = None

    def executes(self, arguments: BaseModel, ctx: Any) -> bool:
        """Whether manual mode treats this call as an execution tool's. A user
        tool always is: what it does is whatever its file says."""
        if self.user_file is not None:
            return True
        return self.runs_code is not None and self.runs_code(arguments, ctx)

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

    def remove(self, name: str) -> Tool | None:
        """Take a tool out of the registry; returns it, or None if absent.

        Only hot-reloading user tools does this. The registry is read on every
        decision, so a removal reaches the next round of every session — and a
        call already decided on finds its tool gone (see the graph's
        execute_tool)."""
        return self._tools.pop(name, None)

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """Restricted view for one agent; unknown names fail at wiring time."""
        return ToolRegistry({name: self.get(name) for name in names})

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
