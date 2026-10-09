"""Self-modification tools (§5.1): the agent tries and loads its own tools.

The loading is `hpca.user_tools`; these are the model's two doors to it, and
the shipped ``new-tool`` skill is the procedure that walks through them:

- ``check_user_tool`` loads a file into a staging area the model never decides
  from, and can call one of its tools once, as a test. With no file it says
  where things are — the HPCA source to learn conventions from, and the user
  tools directory to write into — which is step one of the skill.
- ``reload_user_tools`` swaps the directory into the live registry. It always
  asks (outside full-auto): it is the moment new code enters the agent's own
  process, and ``create_file`` never asked about a new file. The approval
  shows the source of every new file and the diff of every changed one.

A test call is gated by what it calls, not by the check: a tool already checked
in its current form asks only if *it* would (its own ``destructive`` flag or
predicate), and one that has not been asks, because what it does is unknown.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.paths import resolve_path
from hpca.user_tools import UserTools, describe_error, preview, summary


class CheckUserToolParams(BaseModel):
    file: str = Field(
        default="",
        description="The tool file: its name in the user tools directory "
        "(e.g. 'fairshare.py') or its full path. Empty: report where the HPCA "
        "source and the user tools directory are, and what is loaded",
    )
    tool: str = Field(
        default="",
        description="A tool the file registers, to call once as a test. "
        "Empty: only load and check the file",
    )
    arguments: str = Field(
        default="{}",
        description="The test call's arguments, as a JSON object",
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _object_as_json(cls, value):
        # The schema says string, the natural thing to send is an object;
        # taking both costs nothing and saves a validation round trip.
        return json.dumps(value) if isinstance(value, dict) else value


class ReloadUserToolsParams(BaseModel):
    pass


def tool_file(value: str, user_tools: UserTools, ctx: ToolContext | None) -> Path:
    """The user tools file an argument names, or ValueError saying why not.

    Only files *in* the directory: that is the only place a reload loads from,
    so passing a check on a file anywhere else would be a promise nothing keeps.
    """
    directory = user_tools.directory
    text = value.strip()
    given = Path(text).expanduser()
    if not given.is_absolute() and given.parent == Path("."):
        path = directory / (given.name if given.suffix else f"{given.name}.py")
    else:
        resolved = resolve_path(text, ctx.workdir if ctx is not None else Path.cwd())
        if resolved.parent.resolve() != directory.resolve():
            raise ValueError(
                f"{resolved} is not in the user tools directory {directory}. "
                "Tools load only from there — create the file in that directory."
            )
        path = directory / resolved.name
    if path.suffix != ".py":
        raise ValueError(f"{path.name}: a user tool file is a .py file")
    if path.name.startswith(("_", ".")):
        raise ValueError(
            f"{path.name}: files starting with '_' or '.' are never loaded — "
            "rename it"
        )
    return path


def _arguments(text: str) -> dict:
    raw = json.loads(text.strip() or "{}")
    if not isinstance(raw, dict):
        raise ValueError("arguments must be a JSON object, e.g. {\"user\": \"me\"}")
    return raw


def add_selfmod_tools(registry: ToolRegistry, user_tools: UserTools) -> ToolRegistry:
    async def check_user_tool(args: CheckUserToolParams, ctx: ToolContext) -> str:
        if not args.file.strip():
            return user_tools.status()
        try:
            path = tool_file(args.file, user_tools, ctx)
        except ValueError as e:
            return str(e)
        if not path.is_file():
            return f"There is no {path} yet — write it with create_file first."
        loaded = await user_tools.check(path)
        if not loaded.ok:
            return (
                f"{path.name} does NOT load. Fix these and check again:\n"
                + "\n".join(f"- {problem}" for problem in loaded.problems)
            )
        summary = f"{path.name} loads cleanly and registers: " + ", ".join(
            f"{tool.name} (destructive)" if tool.destructive else tool.name
            for tool in loaded.tools
        )
        live = user_tools.live_files().get(path)
        if live is None:
            state = "Not loaded into the agent yet — reload_user_tools loads it."
        elif live.digest == loaded.digest:
            state = "This exact version is already loaded and callable."
        else:
            state = (
                "An older version is loaded — reload_user_tools replaces it "
                "with this one."
            )
        name = args.tool.strip()
        if not name:
            return (
                f"{summary}. {state} To test a tool first, call check_user_tool "
                "again with tool and arguments."
            )
        tool = next((t for t in loaded.tools if t.name == name), None)
        if tool is None:
            return f"{summary}. It registers no tool named {name!r}."
        try:
            params = tool.params.model_validate(_arguments(args.arguments))
        except (ValueError, ValidationError) as e:
            return (
                f"{summary}. The test call was NOT made — its arguments do not "
                f"fit {name}'s parameters:\n{e}"
            )
        call = f"{name}({json.dumps(params.model_dump(mode='json'))})"
        try:
            output = await tool.handler(params, ctx)
        except (Exception, SystemExit) as e:
            return f"{summary}. Test call {call} raised {describe_error(e, path)}"
        note = (
            ""
            if isinstance(output, str)
            else f"\n[note] it returned a {type(output).__name__}, not a str — "
            "a tool's result is text the model reads; return a string"
        )
        return (
            f"{summary}. {state}\nTest call {call} returned (exactly what the "
            f"model would read):\n{output}{note}"
        )

    def test_call_gates(args: CheckUserToolParams, ctx: ToolContext) -> bool:
        name = args.tool.strip()
        if not args.file.strip() or not name:
            return False
        try:
            path = tool_file(args.file, user_tools, ctx)
        except ValueError:
            return False  # refused before anything runs
        tool = user_tools.checked_tool(path, name)
        if tool is None:
            return True  # not checked as it is now: what it does is unknown
        try:
            params = tool.params.model_validate(_arguments(args.arguments))
        except (ValueError, ValidationError):
            return False  # refused in validation, nothing runs
        try:
            return tool.gates(params, ctx)
        except Exception:
            return True

    def describe_test_call(args: CheckUserToolParams, ctx: ToolContext) -> str:
        name = args.tool.strip()
        if not args.file.strip() or not name:
            return ""
        try:
            path = tool_file(args.file, user_tools, ctx)
        except ValueError:
            return ""
        tool = user_tools.checked_tool(path, name)
        if tool is None:
            known = "It has not been checked in this form, so what it changes is unknown."
        elif tool.destructive:
            known = "Its author marked it destructive."
        else:
            known = ""
        return (
            f"Runs {name} from {path} once, as a test, before it is loaded "
            f"into the agent — arguments {args.arguments}. {known}"
        ).rstrip()

    async def reload_user_tools(args: ReloadUserToolsParams, ctx: ToolContext) -> str:
        report = await user_tools.reload()
        return report.render()

    registry.register(
        Tool(
            name="check_user_tool",
            description="Check a user tool file without loading it into the "
            "agent: whether it loads, and (given tool and arguments) what one "
            "test call returns. With no file: where the HPCA source and the "
            "user tools directory are. To build a new tool, follow the "
            "new-tool skill",
            params=CheckUserToolParams,
            handler=check_user_tool,
            is_destructive_call=test_call_gates,
            describe_call=describe_test_call,
            runs_code=lambda args, ctx: bool(args.file.strip()),
        )
    )
    registry.register(
        Tool(
            name="reload_user_tools",
            description="Load the user tools directory into the running agent: "
            "new and changed tool files become callable tools at once, deleted "
            "ones go away. Check each file with check_user_tool first",
            params=ReloadUserToolsParams,
            handler=reload_user_tools,
            destructive=True,
            describe_call=lambda args, ctx: summary(user_tools.pending()),
            show_call=lambda args, ctx: preview(user_tools.pending()) or None,
        )
    )
    return registry
