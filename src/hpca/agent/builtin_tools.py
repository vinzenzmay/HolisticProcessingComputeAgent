"""Core tool suite (§5.1): script creation with mandatory syntax gate, script
execution through the tracked runner, bounded file reading, path listing.

Scripts run two ways, both through the tracked runner (§5.1 — there is no
free-form shell tool; a script is the unit of execution). ``run_script``
waits and hands the output back, which is how the agent looks around the
system: find a file, check a program exists, list conda environments.
``start_script`` is for work that outlives the turn.

Handlers return strings for the model; exceptions (unknown registry keys,
key conflicts) propagate and are surfaced as ``[tool error]`` messages by the
graph — the message text is written for the model to act on.
"""

from __future__ import annotations

import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.checks import syntax_check
from hpca.registry import RegistryError
from hpca.verify_code import format_gate_failure, format_gate_warnings, verify_script

SCRIPT_SUFFIX = {"bash": ".sh", "python": ".py", "R": ".R", "snakemake": ".smk"}
RUN_TIMEOUT_DEFAULT = 60
RUN_TIMEOUT_MAX = 600
RUN_OUTPUT_LINES = 60  # per stream, before the model is pointed at the log
RUN_OUTPUT_CHARS = 4000
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


def _strict_bash(lines: list[str]) -> list[str]:
    """Prepend `set -euo pipefail` so a failing command aborts the script.

    Default bash marches past a failed command, so a script that runs a tool
    and then echoes "Done" exits 0 even when the tool failed — the runner
    reports success and the agent reports success, both wrong (this bit a real
    sniffles run). Strict mode makes the failure the script's exit code.
    Scripts that already opt in are left alone.
    """
    if any(
        line.strip().startswith("set -e") or "set -euo" in line for line in lines
    ):
        return lines
    strict = "set -euo pipefail"
    if lines and lines[0].lstrip().startswith("#!"):
        return [lines[0], strict, *lines[1:]]
    return [strict, *lines]


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
    if args.kind == "bash":
        # execution scripts fail loudly; run_bash (exploration) stays lenient
        lines = _strict_bash(lines)
    content = "\n".join(lines) + "\n"
    path.write_text(content)
    check = await syntax_check(args.kind, path)
    if not check.ok:
        path.unlink(missing_ok=True)
        return (
            f"Script NOT created: {check.checker} found syntax errors — fix "
            f"the script and call create_script again:\n{check.errors}"
        )
    warnings: list[str] = []
    if check.skipped:
        warnings.append(check.errors)
    if ctx.symbols is not None and ctx.symbols.count() > 0:
        # semantic code-vs-docs gate (§5.2): mismatches block, gaps only warn
        reports = verify_script(args.kind, content, index=ctx.symbols)
        mismatches = [r for r in reports if r.status == "mismatch"]
        if mismatches:
            path.unlink(missing_ok=True)
            return format_gate_failure(mismatches)
        not_indexed = [r for r in reports if r.status == "not_indexed"]
        if not_indexed:
            warnings.append(format_gate_warnings(not_indexed))
    ctx.registry.register(args.registry_key, path)
    note = f" ({'; '.join(warnings)})" if warnings else ""
    strict = (
        " Runs fail-fast (set -euo pipefail): a failed command stops the "
        "script, so do not print success unconditionally." if args.kind == "bash"
        else ""
    )
    return (
        f"Created script {args.registry_key!r} ({args.kind}); "
        f"syntax check ok{note}. Start it with start_script.{strict}"
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


class RunScriptParams(BaseModel):
    registry_key: str = Field(description="Registry key of the script to run")
    args: str = Field(default="", description="Arguments, space-separated")
    timeout_s: int = Field(
        default=RUN_TIMEOUT_DEFAULT,
        ge=1,
        le=RUN_TIMEOUT_MAX,
        description=f"Seconds to wait before killing it (max {RUN_TIMEOUT_MAX})",
    )


def _tail(text: str, stream: str, log_key: str | None) -> str:
    """Bound one stream for the prompt, pointing at the log for the rest."""
    lines = text.splitlines()
    clipped = lines[-RUN_OUTPUT_LINES:]
    body = "\n".join(clipped)[-RUN_OUTPUT_CHARS:]
    if not body.strip():
        return ""
    omitted = len(lines) - len(clipped)
    if omitted > 0:
        where = f"read_file {log_key!r} for all of it" if log_key else (
            "re-run with a tighter filter for the rest"
        )
        note = f"\n[... {omitted} earlier {stream} lines omitted; {where}]"
    else:
        note = ""
    return f"{stream}:\n{body}{note}"


async def run_script(args: RunScriptParams, ctx: ToolContext) -> str:
    """Run a script and wait for it, returning what it printed.

    The same tracked runner as start_script — the process shows in the TUI and
    is killable — but awaited, so the output comes back in this tool result
    instead of a log the model would have to poll.
    """
    path = ctx.registry.resolve(args.registry_key)
    interpreter = INTERPRETER.get(path.suffix)
    if interpreter is None:
        raise ValueError(
            f"Cannot run {args.registry_key!r}: unknown script type {path.suffix!r}"
        )
    argv = interpreter + [str(path)] + (args.args.split() if args.args else [])
    record = await ctx.runner.start(
        argv, name=args.registry_key, timeout_s=args.timeout_s
    )
    record = await ctx.runner.wait(record.pid)
    stdout_key = ctx.registry.register_auto(
        record.stdout_path, hint=f"{args.registry_key}_stdout"
    )
    stderr_key = ctx.registry.register_auto(
        record.stderr_path, hint=f"{args.registry_key}_stderr"
    )
    parts = [
        _tail(record.stdout_path.read_text(errors="replace"), "stdout", stdout_key),
        _tail(record.stderr_path.read_text(errors="replace"), "stderr", stderr_key),
    ]
    output = "\n\n".join(part for part in parts if part) or "(no output)"
    if record.state == "killed":
        return (
            f"TIMED OUT after {args.timeout_s}s and was killed. Narrow the "
            f"script (fewer directories, -maxdepth, pipe through head) or "
            f"raise timeout_s, then run it again.\n\n{output}"
        )
    status = (
        "exit 0"
        if record.exit_code == 0
        else f"FAILED with exit code {record.exit_code}"
    )
    return f"{args.registry_key} finished ({status}).\n\n{output}"


class RunBashParams(BaseModel):
    # An array of lines, not one string: the live model reliably fills string
    # arrays but mangles \n escapes in long strings under guided decoding.
    content_lines: list[str] = Field(
        min_length=1,
        description="Bash script content as an array of lines, one per line",
    )
    timeout_s: int = Field(
        default=RUN_TIMEOUT_DEFAULT,
        ge=1,
        le=RUN_TIMEOUT_MAX,
        description=f"Seconds to wait before killing it (max {RUN_TIMEOUT_MAX})",
    )


async def run_bash(args: RunBashParams, ctx: ToolContext) -> str:
    """Write a throwaway bash script, syntax-check it, run it, wait, and return
    its output — all in one call.

    This is the look-around workhorse: find a file, check a program, read a BAM
    header, list conda envs. create_script + run_script does the same thing in
    two steps and is for scripts worth keeping (submitted to Slurm, re-run);
    this is for the one-shot check you would otherwise pay two tool rounds for.
    """
    ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
    name = f"bash_{time.time_ns()}"  # throwaway, unique on disk, never registered
    path = ctx.scripts_dir / f"{name}.sh"
    lines = args.content_lines
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) == 1 and nonempty[0].lstrip().startswith("#!"):
        return (
            "NOT run: the whole script is a single shebang line, so it would do "
            "nothing. Put each command on its own content_lines element."
        )
    path.write_text("\n".join(lines) + "\n")
    check = await syntax_check("bash", path)
    if not check.ok:
        path.unlink(missing_ok=True)
        return (
            f"NOT run: {check.checker} found syntax errors — fix the script and "
            f"call run_bash again:\n{check.errors}"
        )
    record = await ctx.runner.start(
        ["bash", str(path)], name=name, timeout_s=args.timeout_s
    )
    record = await ctx.runner.wait(record.pid)
    parts = [
        _tail(record.stdout_path.read_text(errors="replace"), "stdout", name),
        _tail(record.stderr_path.read_text(errors="replace"), "stderr", name),
    ]
    output = "\n\n".join(part for part in parts if part) or "(no output)"
    if record.state == "killed":
        return (
            f"TIMED OUT after {args.timeout_s}s and was killed. Narrow it "
            f"(fewer directories, -maxdepth, pipe through head) or raise "
            f"timeout_s, then run it again.\n\n{output}"
        )
    if record.exit_code == 0:
        return f"ran (exit 0).\n\n{output}"
    return (
        f"ran (FAILED, exit {record.exit_code}). Read the error, fix the "
        f"script, and call run_bash again.\n\n{output}"
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
            name="run_bash",
            description=(
                "Write and run a one-shot bash script in a single call, return "
                "its output (the tool for looking around: find files, check a "
                "program, read a BAM header, list conda envs)"
            ),
            params=RunBashParams,
            handler=run_bash,
        )
    )
    registry.register(
        Tool(
            name="run_script",
            description=(
                "Run a registered script, wait for it, and return its output "
                "(use this to look around: find files, check a program exists)"
            ),
            params=RunScriptParams,
            handler=run_script,
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
