"""Core tool suite (§5.1): script creation with mandatory syntax gate, script
execution through the tracked runner, bounded file reading, path listing.

Handlers return strings for the model; exceptions (unknown registry keys,
key conflicts) propagate and are surfaced as ``[tool error]`` messages by the
graph — the message text is written for the model to act on.
"""

from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.checks import syntax_check
from hpca.registry import RegistryError

SCRIPT_SUFFIX = {"bash": ".sh", "python": ".py", "R": ".R", "snakemake": ".smk"}
INTERPRETER = {
    ".sh": ["bash"],
    ".py": [sys.executable],
    ".R": ["Rscript"],
    ".smk": ["snakemake", "-s"],
}
KEY_PATTERN = r"^[a-z0-9_.-]+$"


class CreateScriptParams(BaseModel):
    kind: Literal["bash", "python", "R", "snakemake"] = Field(
        description="Script language"
    )
    registry_key: str = Field(
        pattern=KEY_PATTERN, description="New registry key for the script"
    )
    # An array of lines, not one string: the live model reliably fills string
    # arrays but mangles \n escapes in long strings under guided decoding.
    content_lines: list[str] = Field(
        min_length=1,
        description="Script content as an array of lines, one string per line",
    )


async def create_script(args: CreateScriptParams, ctx: ToolContext) -> str:
    ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.scripts_dir / f"{args.registry_key}{SCRIPT_SUFFIX[args.kind]}"
    if args.registry_key in ctx.registry:  # fail before writing anything
        raise RegistryError(
            f"Key {args.registry_key!r} is already registered; pick a different key"
        )
    lines = args.content_lines
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) == 1 and nonempty[0].lstrip().startswith("#!"):
        # A one-line "script" whose only line is a shebang comments itself
        # out entirely; syntax checks would pass vacuously.
        return (
            "Script NOT created: the whole script is a single shebang line, so "
            "it would do nothing. Put each script line into its own "
            "content_lines array element and call create_script again."
        )
    path.write_text("\n".join(lines) + "\n")
    check = await syntax_check(args.kind, path)
    if not check.ok:
        path.unlink(missing_ok=True)
        return (
            f"Script NOT created: {check.checker} found syntax errors — fix "
            f"the script and call create_script again:\n{check.errors}"
        )
    ctx.registry.register(args.registry_key, path)
    note = f" ({check.errors})" if check.skipped else ""
    return (
        f"Created script {args.registry_key!r} ({args.kind}); "
        f"syntax check ok{note}. Start it with start_script."
    )


class ReadFileParams(BaseModel):
    registry_key: str = Field(description="Registry key of the file to read")
    max_lines: int = Field(
        default=100, ge=10, le=500, description="Line budget for the output"
    )


async def read_file(args: ReadFileParams, ctx: ToolContext) -> str:
    path = ctx.registry.resolve(args.registry_key)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) <= args.max_lines:
        return "\n".join(lines)
    # §4.3 output size control: head/tail, never the full dump
    head = args.max_lines // 2
    tail = args.max_lines - head
    return "\n".join(
        lines[:head]
        + [f"... [{len(lines) - head - tail} lines omitted] ..."]
        + lines[-tail:]
    )


class StartScriptParams(BaseModel):
    registry_key: str = Field(description="Registry key of the script to run")
    args: str = Field(default="", description="Command-line arguments, space-separated")


async def start_script(args: StartScriptParams, ctx: ToolContext) -> str:
    path = ctx.registry.resolve(args.registry_key)
    interpreter = INTERPRETER.get(path.suffix)
    if interpreter is None:
        raise ValueError(
            f"Cannot start {args.registry_key!r}: unknown script type {path.suffix!r}"
        )
    argv = interpreter + [str(path)] + (args.args.split() if args.args else [])
    record = await ctx.runner.start(argv, name=args.registry_key)
    stdout_key = ctx.registry.register_auto(
        record.stdout_path, hint=f"{args.registry_key}_stdout"
    )
    stderr_key = ctx.registry.register_auto(
        record.stderr_path, hint=f"{args.registry_key}_stderr"
    )
    return (
        f"Started {args.registry_key!r} (pid {record.pid}). It runs in the "
        f"background; logs: {stdout_key}, {stderr_key} (use read_file to check)."
    )


class ListPathsParams(BaseModel):
    pass


async def list_paths(args: ListPathsParams, ctx: ToolContext) -> str:
    paths = ctx.registry.list()
    if not paths:
        return "No paths registered yet."
    return "\n".join(f"{key}: {path}" for key, path in sorted(paths.items()))


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="create_script",
            description="Create a bash/python/R/snakemake script (syntax-checked)",
            params=CreateScriptParams,
            handler=create_script,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a registered file (head/tail truncated)",
            params=ReadFileParams,
            handler=read_file,
        )
    )
    registry.register(
        Tool(
            name="start_script",
            description="Run a registered script as a tracked background process",
            params=StartScriptParams,
            handler=start_script,
        )
    )
    registry.register(
        Tool(
            name="list_paths",
            description="List all registered path keys",
            params=ListPathsParams,
            handler=list_paths,
        )
    )
    return registry
