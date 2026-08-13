"""Core tool suite (§5.1): script creation with mandatory syntax gate, script
execution through the tracked runner, bounded file reading, path listing.

Scripts run two ways, both through the tracked runner (§5.1 — there is no
free-form shell tool; a script is the unit of execution), split on the one
axis the model cannot get back by itself: *when the result arrives*.
``run_bash`` writes a throwaway script inline and waits — the look-around
workhorse (find a file, check a program exists, list conda environments) and,
via ``{key}`` expansion, the way a *registered* script is run synchronously
too. ``start_background_script`` runs a registered script in the background,
for work that outlives the turn; it alone reports back as a completion event
(§5.4), because ``run_bash`` already handed its output over.

The retired third tool was ``run_script`` (registered script, waits). Splitting
on where the script came from bought nothing — its description advertised the
same look-around job as ``run_bash``, so the model had two plausible tools for
one move — while ``{key}`` expansion keeps the §4.3 "tools take keys, never
literal paths" rule intact instead of carving an exception into it.

Handlers return strings for the model; exceptions (unknown registry keys,
key conflicts) propagate and are surfaced as ``[tool error]`` messages by the
graph — the message text is written for the model to act on.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field, field_validator

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.checks import syntax_check
from hpca.registry import RegistryError, registered_note
from hpca.verify_code import format_gate_failure, format_gate_warnings, verify_script

SCRIPT_SUFFIX = {"bash": ".sh", "python": ".py", "R": ".R", "snakemake": ".smk"}
# Which checker a file on disk answers to, read off its suffix — how edit_file
# decides whether the content it is about to write is a script §5.2 must gate.
# Anything else (.txt, .yaml, .csv) has no checker and is left to itself.
KIND_BY_SUFFIX = {suffix: kind for kind, suffix in SCRIPT_SUFFIX.items()}
RUN_TIMEOUT_DEFAULT = 60
RUN_TIMEOUT_MAX = 600
RUN_OUTPUT_LINES = 60  # per stream, before the model is pointed at the log
RUN_OUTPUT_CHARS = 4000
# The same bound, applied to what goes *in*. run_bash always accepted a script
# of any size, and what the live model does with that is not write a longer
# look-around: asked for a design document with only run_bash available, it
# put all 5-8k characters of it into ONE array element — the long-string case
# `content_lines` exists to avoid — and nothing checked it beyond `bash -n`,
# because run_bash skips §5.2's code-vs-docs gate that create_script faces.
# 2000 characters is several times the longest genuine look-around (twenty
# bounded finds is ~1200) and far below any document.
RUN_SCRIPT_MAX_CHARS = 2000
INTERPRETER = {
    ".sh": ["bash"],
    ".py": [sys.executable],
    ".R": ["Rscript"],
    ".smk": ["snakemake", "-s"],
}
KEY_CHARS = r"[a-z0-9_.-]+"
KEY_PATTERN = rf"^{KEY_CHARS}$"

# A `{key}` in a run_bash line is a registry reference, expanded to the
# registered absolute path before anything else looks at the script. The
# lookbehind keeps `${VAR}` out; the character class keeps `{a,b}` brace
# expansion and `awk '{print $1}'` out (comma, space and `$` are all excluded).
# `{print}` still matches by shape, which is why expansion substitutes only
# keys that are actually registered and leaves everything else untouched —
# there is no way to tell a bare awk body from a typo'd key by shape alone.
_KEY_REF = re.compile(rf"(?<![$\\]){{({KEY_CHARS})}}")
MAX_KEYS_IN_NOTE = 30


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


def expand_keys(lines: list[str], ctx: object) -> tuple[list[str], list[str]]:
    """Expand ``{key}`` references to registered paths.

    Returns the expanded lines and the brace references that matched nothing,
    which the caller reports only if the run then fails — an unmatched
    ``{print}`` in an awk body is not an error, a typo'd key is, and the exit
    code is what tells them apart.

    The path is substituted raw, not shell-quoted: the model writes ``{ref}``
    where it would otherwise write the literal path, and quoting would break
    the equally common ``"{ref}"``. Pure apart from registry reads, as the
    gating predicates that call it require.
    """
    registry = getattr(ctx, "registry", None)
    unresolved: list[str] = []

    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if registry is not None and key in registry:
            return str(registry.resolve(key))
        unresolved.append(match.group(0))
        return match.group(0)

    return [_KEY_REF.sub(substitute, line) for line in lines], unresolved


def _unresolved_note(unresolved: list[str], ctx: ToolContext) -> str:
    """Told to the model only on failure — see ``expand_keys``."""
    if not unresolved:
        return ""
    refs = ", ".join(sorted(set(unresolved)))
    known = ", ".join(sorted(ctx.registry.list())[:MAX_KEYS_IN_NOTE]) or "(none)"
    return (
        f"\n\nNote: {refs} did not match any registry key and was passed "
        f"through unchanged. If you meant a registered path, the keys are: "
        f"{known}"
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


async def check_script_content(
    kind: str, path: Path, content: str, ctx: ToolContext, *, refusal: str
) -> tuple[str, list[str]]:
    """§5.2's mandatory gate on one script's content: the syntax check first,
    then the semantic code-vs-docs check against the indexed symbols.

    Returns the model-facing refusal (empty when the content passes) and the
    warnings a successful result should carry. ``path`` must already hold
    ``content`` — the checkers read the file — and what happens to it either
    way is the caller's: create_script unlinks a script it will not keep,
    edit_file checks a scratch copy so the real file is never left broken.

    Shared rather than duplicated because the gate is what makes §5.2
    *mandatory*: a second way to put content into a script file would be a
    second way around it.
    """
    check = await syntax_check(kind, path)
    if not check.ok:
        return (
            f"{refusal}: {check.checker} found syntax errors — fix them and "
            f"try again:\n{check.errors}"
        ), []
    warnings: list[str] = []
    if check.skipped:
        warnings.append(check.errors)
    if ctx.symbols is not None:
        # Learn the flags of the external programs this script drives before
        # judging it. Without this the gate below has nothing to check for
        # exactly the tools that matter (minimap2, samtools, ...), because
        # index_docs is explicit-only and nothing ever calls it (§5.2, §5.6).
        from hpca.agent.doc_tools import autoindex_script_commands

        await autoindex_script_commands(kind, content, ctx)
    if ctx.symbols is not None and ctx.symbols.count() > 0:
        # semantic code-vs-docs gate (§5.2): mismatches block, gaps only warn
        reports = verify_script(kind, content, index=ctx.symbols)
        mismatches = [r for r in reports if r.status == "mismatch"]
        if mismatches:
            return format_gate_failure(mismatches, refusal=refusal), []
        not_indexed = [r for r in reports if r.status == "not_indexed"]
        if not_indexed:
            warnings.append(format_gate_warnings(not_indexed))
    return "", warnings


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
    refused, warnings = await check_script_content(
        args.kind, path, content, ctx, refusal="Script NOT created"
    )
    if refused:
        path.unlink(missing_ok=True)  # never keep a script that failed the gate
        return refused
    ctx.registry.register(args.registry_key, path)
    note = f" ({'; '.join(warnings)})" if warnings else ""
    strict = (
        " Runs fail-fast (set -euo pipefail): a failed command stops the "
        "script, so do not print success unconditionally." if args.kind == "bash"
        else ""
    )
    return (
        f"Created script {args.registry_key!r} ({args.kind}); "
        f"syntax check ok{note}. Start it with start_background_script, or run "
        f"it now with run_bash: {{{args.registry_key}}} expands to its "
        f"path.{strict}"
    )


class ReadFileParams(BaseModel):
    registry_key: str = Field(
        description="Registry key or absolute path of the file to read"
    )
    subpath: str = Field(
        default="",
        description=(
            "Path relative to registry_key when it names a directory, e.g. "
            "'src/main.py'. Leave empty to read the key itself."
        ),
    )
    start_line: int = Field(
        default=1,
        ge=1,
        description=(
            "1-indexed line to start reading from; page through a long file "
            "by repeating the call with the start_line the previous result "
            "suggested"
        ),
    )
    # Default 200, not 100: measured on the live 27B, a mid-file edit in a
    # 700-line file cost 7 paging reads at 100 lines a page — the model
    # follows the continuation hint faithfully, so the page size is the whole
    # cost. 200 halves it while a page stays ~2k tokens.
    max_lines: int = Field(
        default=200, ge=10, le=500, description="Line budget for the output"
    )


def _list_dir(path: Path, key: str, max_lines: int) -> str:
    # A directory key is a dead end for read_text; instead of a raw
    # IsADirectoryError, list it and point at the subpath route so the model
    # can descend without registering every file first (live-session thrash).
    entries = sorted(
        p.name + ("/" if p.is_dir() else "") for p in path.iterdir()
    )
    shown = entries[:max_lines]
    omitted = len(entries) - len(shown)
    tail = f"\n... [{omitted} more] ..." if omitted > 0 else ""
    body = "\n".join(shown) or "(empty)"
    return (
        f"{key!r} is a directory, not a file. Contents:\n{body}{tail}\n"
        f"Read one with read_file(registry_key={key!r}, subpath='<name>')."
    )


async def read_file(args: ReadFileParams, ctx: ToolContext) -> str:
    path, key = ctx.registry.resolve_or_register(args.registry_key)
    # Taken before the subpath descent below rebinds `key` to the inner file's
    # own key: the note is about the argument the model passed, and a plain
    # key + subpath call must read exactly as it always did.
    note = registered_note(args.registry_key, key)
    # A file's content is what the model copies edit_file's old_lines out of,
    # so the note goes on its own line after it, never appended to a line.
    tail = f"\n{note.strip()}" if note else ""
    if not path.exists():
        # A key may be registered ahead of the thing it names (register_path
        # accepts a path that is not there yet), and a file registered earlier
        # can be moved or deleted from under it. Say so plainly: without this
        # the read raises a bare FileNotFoundError at the model.
        return (
            f"Nothing at {path} (registered as {key!r}). It was registered "
            "before anything was created there, or it has since moved or been "
            "deleted — create it, or register the path that is really there."
        )
    if args.subpath:
        # Descend into a registered directory. Reject escapes and keep the
        # resolved file addressable next turn via its own auto-registered key.
        candidate = (path / args.subpath).resolve()
        if not candidate.is_relative_to(path.resolve()):
            return (
                f"subpath {args.subpath!r} escapes {args.registry_key!r}; "
                "use a path inside the directory."
            )
        if not candidate.exists():
            return (
                f"No such file: {args.subpath!r} under {args.registry_key!r}. "
                f"Call read_file(registry_key={args.registry_key!r}) to list it."
            )
        path = candidate
        key = ctx.registry.register_auto(path, hint=path.name)
    if path.is_dir():
        return _list_dir(path, key, args.max_lines) + tail
    lines = path.read_text(errors="replace").splitlines()
    total = len(lines)
    if args.start_line > total:
        return (
            f"start_line={args.start_line} is past the end of {key!r}: it has "
            f"{total} lines. Call read_file again with start_line <= {total}."
        )
    # §4.3 output size control: a contiguous window, never the full dump.
    # Head/tail with the middle omitted made the region a model wanted to
    # edit literally unreadable (and the omission marker got copied into
    # edit_file old_lines); paging with start_line replaces it.
    start = args.start_line - 1
    end = start + args.max_lines
    body = "\n".join(lines[start:end])
    if end >= total:
        return body + tail
    return (
        f"{body}\n... [file continues: lines {end + 1}-{total}; call "
        f"read_file again with start_line={end + 1}, and max_lines up to 500 "
        f"to see more per call]{tail}"
    )


class StartBackgroundScriptParams(BaseModel):
    registry_key: str = Field(description="Registry key of the script to run")
    args: str = Field(default="", description="Command-line arguments, space-separated")


async def start_background_script(
    args: StartBackgroundScriptParams, ctx: ToolContext
) -> str:
    path = ctx.registry.resolve(args.registry_key)
    interpreter = INTERPRETER.get(path.suffix)
    if interpreter is None:
        raise ValueError(
            f"Cannot start {args.registry_key!r}: unknown script type {path.suffix!r}"
        )
    # A model that does not get an immediate result readily starts the script
    # twice; both copies then write the same outputs, and the corrupted result
    # is far worse than the wasted CPU (seen in a live session: two sniffles
    # runs onto one VCF). The wait is now honest — §5.4 reports the exit.
    running = ctx.runner.running_named(args.registry_key)
    if running is not None:
        return (
            f"NOT started: {args.registry_key!r} is already running (pid "
            f"{running}), and a second copy would write the same output files. "
            "Wait for it — you will be told when it finishes — or kill it first."
        )
    argv = interpreter + [str(path)] + (args.args.split() if args.args else [])
    record = await ctx.runner.start(argv, name=args.registry_key, background=True)
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



class RunBashParams(BaseModel):
    # timeout_s before the lines, because nothing may follow a long array —
    # see the argument-order rule in hpca.agent.middleware.
    timeout_s: int = Field(
        default=RUN_TIMEOUT_DEFAULT,
        ge=1,
        le=RUN_TIMEOUT_MAX,
        description=f"Seconds to wait before killing it (max {RUN_TIMEOUT_MAX})",
    )
    # An array of lines, not one string: the live model reliably fills string
    # arrays but mangles \n escapes in long strings under guided decoding.
    content_lines: list[str] = Field(
        min_length=1,
        description=(
            "Bash script content as an array of lines, one per line "
            f"(a short look-around script, at most {RUN_SCRIPT_MAX_CHARS} "
            "characters — write files with create_file, not with this)"
        ),
    )

    @field_validator("content_lines")
    @classmethod
    def _short_enough_to_be_a_look_around(cls, lines: list[str]) -> list[str]:
        """Refuse a script that is really a file being written.

        A validator rather than a check in the handler, so the call never
        becomes a pending tool call: the model gets this back inside the same
        decision and can call the right tool instead, and manual mode never
        asks the user to approve a script that was going to be refused. The
        message has to carry the whole route out, because it is the only thing
        the model gets.
        """
        size = sum(len(line) + 1 for line in lines)
        if size <= RUN_SCRIPT_MAX_CHARS:
            return lines
        raise ValueError(
            f"this script is {size} characters and run_bash takes at most "
            f"{RUN_SCRIPT_MAX_CHARS}: it is for looking around, not for "
            "writing files. To WRITE a file — notes, a specs document, a "
            "config — call create_file with the content as content_lines. To "
            "RUN real work, call create_script (it is syntax- and "
            "docs-checked), then start_background_script, or run_bash with "
            "{its_key}. If the content is too long for one call, write the "
            "first part with create_file, then add each further part with "
            "edit_file: put the file's current last line in old_lines, and "
            "that same line followed by the new lines in new_lines. Or, if "
            "this really is a look-around, make it shorter."
        )


# ----------------------------------------------------- run_bash destructiveness
# run_bash runs arbitrary bash, so — unlike the file tools — its destructiveness
# lives in the script text, not in structured arguments. This is a best-effort
# heuristic that lets a genuinely destructive look-around command trip the §5.3
# gate (so plan/auto mode still pause on it) while benign look-around runs
# unattended. It is deliberately NOT airtight: the trash/backup layer is the
# real net. It matches only the *leading* command of each `;`/`|`/`&`-separated
# segment, so a path, pattern, or comment merely mentioning `rm` does not trip
# it. Like the other is_destructive_call predicates it must stay pure and
# deterministic — the graph re-runs it when a parked turn resumes.
DESTRUCTIVE_COMMANDS = frozenset(
    {
        "rm", "rmdir", "dd", "mkfs", "shred", "truncate", "fdisk",
        "mkswap", "wipefs", "chmod", "chown", "mv", "scancel",
    }
)
# Words that stand in front of the real command without being it.
_COMMAND_WRAPPERS = frozenset(
    {"sudo", "command", "nohup", "time", "env", "exec", "xargs", "nice", "ionice"}
)
_SEGMENT_SPLIT = re.compile(r"[;&|\n]+")  # command separators (|, ||, &&, ;, &)
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")  # leading VAR=val


def _segment_command(segment: str) -> str:
    """The command a single shell segment would run, or '' if none.

    Leading ``VAR=val`` assignments and wrapper words (``sudo``, ``xargs``, …)
    are skipped so ``find . | sudo rm`` resolves to ``rm``.
    """
    for token in segment.split():
        if _ASSIGNMENT.match(token):
            continue
        base = token.rsplit("/", 1)[-1]  # /bin/rm -> rm
        if base in _COMMAND_WRAPPERS:
            continue
        return base
    return ""


def _bash_commands(content_lines: list[str]) -> list[str]:
    commands: list[str] = []
    for raw in content_lines:
        line = raw.split("#", 1)[0]  # drop inline comments (heuristic)
        for segment in _SEGMENT_SPLIT.split(line):
            cmd = _segment_command(segment)
            if cmd:
                commands.append(cmd)
    return commands


def _bash_flagged(content_lines: list[str]) -> list[str]:
    return sorted(
        {
            cmd
            for cmd in _bash_commands(content_lines)
            if cmd in DESTRUCTIVE_COMMANDS or cmd.startswith("mkfs.")
        }
    )


def _bash_is_destructive(args: RunBashParams, ctx: object = None) -> bool:
    # Judged on the *expanded* script: a `{key}` standing in for /bin/rm would
    # otherwise walk past the gate as an unrecognised word.
    lines, _ = expand_keys(args.content_lines, ctx)
    return bool(_bash_flagged(lines))


def _describe_bash(args: RunBashParams, ctx: object = None) -> str:
    lines, _ = expand_keys(args.content_lines, ctx)
    flagged = _bash_flagged(lines)
    if not flagged:
        return ""
    return "Flagged command(s): " + ", ".join(flagged)


def _tail(text: str, stream: str, register: Callable[[], str]) -> str:
    """Bound one stream for the prompt, pointing at the log for the rest.

    ``register`` is called only when something was actually cut. A look-around
    command whose output fits needs no registry key, and minting one per run
    would fill the registry the model reasons over with logs it never reads.
    """
    lines = text.splitlines()
    kept = lines[-RUN_OUTPUT_LINES:]
    body = "\n".join(kept)
    cut_head = len(body) > RUN_OUTPUT_CHARS  # long lines, few of them
    body = body[-RUN_OUTPUT_CHARS:]
    if not body.strip():
        return ""
    omitted = len(lines) - len(kept)
    if omitted > 0 or cut_head:
        what = (
            f"{omitted} earlier {stream} lines"
            if omitted > 0
            else f"the start of {stream}"
        )
        note = f"\n[... {what} omitted; read_file {register()!r} for all of it]"
    else:
        note = ""
    return f"{stream}:\n{body}{note}"


# How many distinct failing lines to quote, and how many neighbours each keeps.
# A script without `set -e` can fail on every line; the first few are the ones
# worth reading, and the rest are usually the same mistake repeated.
CITED_LINES = 5
CITED_CONTEXT = 1


def _cited_lines(script_lines: list[str], stderr: str, path: Path) -> str:
    """The script lines bash's messages point at, numbered, with neighbours.

    Without this a failure says "line 98" to a model that cannot see line 98 —
    the script is a throwaway file it never reads back — so its only available
    fix is to rewrite the whole thing. For the hundred-line heredoc that is
    both the most expensive move and the one most likely to reproduce whatever
    broke it.

    Quoted whatever the exit code, because run_bash is deliberately lenient
    (no ``set -euo pipefail``, unlike create_script): a command that fails
    mid-script leaves the exit code to whatever ran last, so "exit 0" and a
    broken line are not mutually exclusive.

    Only bash's own messages about *this* script count — they carry its path,
    and `awk: line 3` or a Python traceback's "line 12" number something else
    entirely.
    """
    marker = re.compile(rf"{re.escape(str(path))}: line (\d+):")
    numbers = sorted(
        {
            int(match)
            for match in marker.findall(stderr)
            if 1 <= int(match) <= len(script_lines)
        }
    )[:CITED_LINES]
    if not numbers:
        return ""
    wanted = {
        neighbour
        for number in numbers
        for neighbour in range(number - CITED_CONTEXT, number + CITED_CONTEXT + 1)
        if 1 <= neighbour <= len(script_lines)
    }
    width = len(str(max(wanted)))
    quoted: list[str] = []
    previous = 0
    for number in sorted(wanted):
        if previous and number > previous + 1:
            quoted.append("  ...")  # a jump, not a run of lines
        quoted.append(
            f"{'>' if number in numbers else ' '} {number:{width}} | "
            f"{script_lines[number - 1]}"
        )
        previous = number
    return "The script lines bash's messages point at:\n" + "\n".join(quoted)


def _run_output(record, ctx: ToolContext, hint: str) -> str:
    """Both streams, bounded, each registering its log only if it was cut."""
    parts = [
        _tail(
            path.read_text(errors="replace"),
            stream,
            lambda p=path, s=stream: ctx.registry.register_auto(
                p, hint=f"{hint}_{s}"
            ),
        )
        for stream, path in (
            ("stdout", record.stdout_path),
            ("stderr", record.stderr_path),
        )
    ]
    return "\n\n".join(part for part in parts if part) or "(no output)"


async def run_bash(args: RunBashParams, ctx: ToolContext) -> str:
    """Write a throwaway bash script, syntax-check it, run it, wait, and return
    its output — all in one call.

    This is the look-around workhorse: find a file, check a program, read a BAM
    header, list conda envs. It is also how a *registered* script is run
    synchronously — write ``{key}`` and it expands to the path — so there is
    one tool for "run this and tell me what it said", whatever the script is.
    Work that outlives the turn goes to start_background_script instead.
    """
    ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
    name = f"bash_{time.time_ns()}"  # throwaway, unique on disk, never registered
    path = ctx.scripts_dir / f"{name}.sh"
    lines, unresolved = expand_keys(args.content_lines, ctx)
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
    cited = _cited_lines(
        lines, record.stderr_path.read_text(errors="replace"), path
    )
    output = "\n\n".join(
        part for part in [_run_output(record, ctx, "bash"), cited] if part
    )
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
        f"{_unresolved_note(unresolved, ctx)}"
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
            description=(
                "Read a registered file, paged: returns up to max_lines from "
                "start_line and tells you where to continue if the file goes "
                "on. If the key is a directory, lists it; pass subpath to "
                "read a file inside it."
            ),
            params=ReadFileParams,
            handler=read_file,
        )
    )
    registry.register(
        Tool(
            name="run_bash",
            description=(
                "Run bash and wait for its output: look around (find files, "
                "check a program, read a BAM header, list conda envs) or run a "
                "registered script by writing {registry_key}, which expands to "
                "its path. Use for anything you want the result of now"
            ),
            params=RunBashParams,
            handler=run_bash,
            is_destructive_call=_bash_is_destructive,
            describe_call=_describe_bash,
        )
    )
    registry.register(
        Tool(
            name="start_background_script",
            description=(
                "Run a registered script as a tracked background process for "
                "work that outlives this turn (a pipeline, a long tool run). "
                "Returns a pid immediately, NOT the output; you are told when "
                "it finishes. For output now, use run_bash"
            ),
            params=StartBackgroundScriptParams,
            handler=start_background_script,
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
