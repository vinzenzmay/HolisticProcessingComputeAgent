"""Core tool suite (§5.1): script creation with mandatory syntax gate, script
execution through the tracked runner, bounded file reading, script listing.

Scripts run two ways, both through the tracked runner (§5.1 — there is no
free-form shell tool; a script is the unit of execution), split on the one
axis the model cannot get back by itself: *when the result arrives*.
``run_bash`` writes a throwaway script inline and waits — the look-around
workhorse (find a file, check a program exists, list conda environments) and,
via ``{name}`` expansion, the way a *kept* script is run synchronously
too. ``start_background_script`` runs a kept script in the background,
for work that outlives the turn; it alone reports back as a completion event
(§5.4), because ``run_bash`` already handed its output over.

The retired third tool was ``run_script`` (kept script, waits). Splitting on
where the script came from bought nothing — its description advertised the same
look-around job as ``run_bash``, so the model had two plausible tools for one
move — while ``{name}`` expansion keeps a kept script reachable without making
the model retype the scripts dir.

A script's *name* is the one handle here that is not a path, and it is not an
indirection either: it is the file's own name in ``ctx.scripts_dir``, so
``script_path`` answers with a directory lookup and nothing is stored anywhere.

Handlers return strings for the model; exceptions propagate and are surfaced as
``[tool error]`` messages by the graph — the message text is written for the
model to act on.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field, field_validator

from hpca.agent import hints
from hpca.agent.context import ToolContext
from hpca.agent.history import carries_elision_marker
from hpca.agent.tools import Tool, ToolRegistry
from hpca.checks import syntax_check
from hpca.filetail import read_head, read_tail
from hpca.paths import PathError, resolve_path
from hpca.verify_code import format_gate_failure, format_gate_warnings, verify_script

SCRIPT_SUFFIX = {"bash": ".sh", "python": ".py"}
# Which checker a file on disk answers to, read off its suffix — how edit_file
# decides whether the content it is about to write is a script §5.2 must gate.
# Anything else (.txt, .yaml, .csv) has no checker and is left to itself.
KIND_BY_SUFFIX = {suffix: kind for kind, suffix in SCRIPT_SUFFIX.items()}
RUN_TIMEOUT_DEFAULT = 60
RUN_TIMEOUT_MAX = 600
RUN_OUTPUT_LINES = 60  # per stream, before the model is pointed at the log
RUN_OUTPUT_CHARS = 4000
# How much of a log is read to produce that. It is a window taken from the end
# of the file, not the file — a look-around that prints a gigabyte used to
# cost a gigabyte of memory and a frozen UI to keep four kilobytes of it.
#
# Eight times the char cap, because the window has to be wider than what is
# kept on three counts: RUN_OUTPUT_CHARS characters are up to four times that
# many bytes in UTF-8, the seek lands mid-character (the decode replaces the
# fragment, and the replacement is discarded with the rest of the front), and
# RUN_OUTPUT_LINES lines have to fit too. They do at any ordinary line width;
# where they do not, the lines are long enough that the char cap decides the
# output anyway and the answer is the same either way.
RUN_OUTPUT_WINDOW = RUN_OUTPUT_CHARS * 8
# The same bound, applied to what goes *in*. run_bash always accepted a script
# of any size, and what the live model does with that is not write a longer
# look-around: asked for a design document with only run_bash available, it
# put all 5-8k characters of it into ONE array element — the long-string case
# `content_lines` exists to avoid — and nothing checked it beyond `bash -n`,
# because run_bash skips §5.2's code-vs-docs gate that create_script faces.
# 2000 characters is several times the longest genuine look-around (twenty
# bounded finds is ~1200) and far below any document.
RUN_SCRIPT_MAX_CHARS = 2000
INTERPRETER = {".sh": ["bash"], ".py": [sys.executable]}
KEY_CHARS = r"[a-z0-9_.-]+"
KEY_PATTERN = rf"^{KEY_CHARS}$"

# A `{name}` in a run_bash line is a kept script, expanded to its absolute
# path before anything else looks at the script. The lookbehind keeps `${VAR}`
# out; the character class keeps `{a,b}` brace expansion and `awk '{print $1}'`
# out (comma, space and `$` are all excluded). `{print}` still matches by
# shape, which is why expansion substitutes only names that really are scripts
# and leaves everything else untouched — there is no way to tell a bare awk
# body from a typo'd name by shape alone.
_KEY_REF = re.compile(rf"(?<![$\\]){{({KEY_CHARS})}}")
MAX_KEYS_IN_NOTE = 30


class CreateScriptParams(BaseModel):
    # "Script language" read as a menu of the languages this tool supports,
    # which is how a model that has already chosen the tool sees it — so it
    # picked the nearer of the two for a file that is neither. The field is
    # read at the moment the kind is decided, which is the last moment the
    # mistake is still cheap, so it says the pair is closed and where else to
    # go. The tool description says it earlier and louder; both are needed,
    # because they are read at different decisions.
    kind: Literal["bash", "python"] = Field(
        description=(
            "Script language — bash or python are the only two this tool "
            "writes. A file in any other language (a Snakefile, a Makefile, "
            "a nextflow .nf, an R script) is not a script here: write it with "
            "create_file instead"
        )
    )
    # "without a suffix" alone lost to every script the model has ever read,
    # all of which are written `something.sh` — so it sent one, and the suffix
    # `kind` implies was appended to it (`validate_hg002.sh.sh`). The example is
    # here because a shape shown beats a prohibition stated;
    # ``strip_script_suffix`` is the net under it, for when this loses too.
    name: str = Field(
        pattern=KEY_PATTERN,
        description=(
            "Name for the script, with NO suffix — the .sh or .py follows "
            "from kind and is added for you. Write 'validate_hg002', never "
            "'validate_hg002.sh'"
        ),
    )
    # An array of lines, not one string: the live model reliably fills string
    # arrays but mangles \n escapes in long strings under guided decoding.
    content_lines: list[str] = Field(
        min_length=1,
        description="Script content as an array of lines, one string per line",
    )


def strip_script_suffix(name: str) -> str:
    """``validate_hg002.sh`` -> ``validate_hg002``; anything else unchanged.

    The name a script is *called* carries no suffix — create_script appends the
    one its kind implies. But every script the model has ever seen written down
    is written `something.sh`, and "without a suffix" in a field description
    does not outweigh that: it sends `validate_hg002.sh`, the suffix is appended
    to what it sent, and the file lands as `validate_hg002.sh.sh`. Nothing then
    breaks loudly — the doubled name is self-consistent, so every lookup keeps
    working and the model keeps reading `.sh.sh` back out of its own results.
    That is worse than a refusal, because it is a mistake with no feedback edge.

    Stripping is safe because a name that genuinely wants to end in `.sh` cannot
    be told apart from this slip by shape, and the slip is the overwhelmingly
    likelier reading. Both suffixes go, not only the one the chosen kind
    implies: `name='x.py', kind='bash'` is the same slip about the same field.
    """
    path = Path(name)
    return path.stem if path.suffix in KIND_BY_SUFFIX and path.stem else name


def script_path(name: str, ctx: object) -> Path | None:
    """The kept script called ``name``, or None.

    A script's name IS its file name in the scripts dir, so there is nothing to
    look up: the two suffixes are tried in turn. That is the whole of what the
    registry did for scripts, minus the table.

    A name that arrives carrying its own suffix resolves too, so a model that
    writes `{validate_hg002.sh}` in a run_bash line reaches the script
    create_script kept as `validate_hg002`. The literal form is tried first,
    which is what keeps a `validate_hg002.sh.sh` written before
    ``strip_script_suffix`` existed reachable under the only name it ever had.
    """
    scripts_dir = getattr(ctx, "scripts_dir", None)
    if scripts_dir is None:
        return None
    stripped = strip_script_suffix(name)
    candidates = (name,) if stripped == name else (name, stripped)
    for candidate_name in candidates:
        for suffix in SCRIPT_SUFFIX.values():
            candidate = Path(scripts_dir) / f"{candidate_name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def script_names(ctx: object) -> list[str]:
    """The kept scripts, by name, for the notes that list what does exist."""
    scripts_dir = getattr(ctx, "scripts_dir", None)
    if scripts_dir is None or not Path(scripts_dir).is_dir():
        return []
    suffixes = set(SCRIPT_SUFFIX.values())
    return sorted(
        {
            entry.stem
            for entry in Path(scripts_dir).iterdir()
            if entry.suffix in suffixes and not entry.stem.startswith("bash_")
        }
    )


def expand_keys(lines: list[str], ctx: object) -> tuple[list[str], list[str]]:
    """Expand ``{name}`` references to the paths of kept scripts.

    Returns the expanded lines and the brace references that matched nothing,
    which the caller reports only if the run then fails — an unmatched
    ``{print}`` in an awk body is not an error, a typo'd script name is, and the
    exit code is what tells them apart.

    The path is substituted raw, not shell-quoted: the model writes ``{ref}``
    where it would otherwise write the literal path, and quoting would break
    the equally common ``"{ref}"``. Pure apart from a directory listing, as the
    gating predicates that call it require.
    """
    unresolved: list[str] = []

    def substitute(match: re.Match) -> str:
        path = script_path(match.group(1), ctx)
        if path is not None:
            return str(path)
        unresolved.append(match.group(0))
        return match.group(0)

    return [_KEY_REF.sub(substitute, line) for line in lines], unresolved


def _unresolved_note(unresolved: list[str], ctx: ToolContext) -> str:
    """Told to the model only on failure — see ``expand_keys``."""
    if not unresolved:
        return ""
    refs = ", ".join(sorted(set(unresolved)))
    known = ", ".join(script_names(ctx)[:MAX_KEYS_IN_NOTE]) or "(none)"
    return (
        f"\n\nNote: {refs} did not match any script and was passed through "
        f"unchanged. If you meant a kept script, they are: {known}. For a file "
        f"path, write the path itself."
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
    # The name the script is kept under, which is not always the one that was
    # sent: a `.sh`/`.py` the model wrote out itself would otherwise be doubled
    # against the suffix the kind implies (see ``strip_script_suffix``). Done
    # here rather than in the params model so the whole function — the taken
    # check, the file, and what the result calls the script — speaks one name.
    name = strip_script_suffix(args.name)
    path = ctx.scripts_dir / f"{name}{SCRIPT_SUFFIX[args.kind]}"
    # Fail before writing anything, so a late refusal cannot leave a half-made
    # script behind. A name is free exactly when no kept script answers to it —
    # which now needs no table to decide, only the directory.
    taken = script_path(args.name, ctx)
    if taken is not None:
        raise PathError(
            f"Script {name!r} already exists ({taken}); "
            f"{hints.SCRIPT_NAME_EXISTS}"
        )
    for number, line in enumerate(args.content_lines, 1):
        # The third way content reaches disk, and open to the same failure as
        # create_file: what the session history leaves in place of an omitted
        # payload (see ``hpca.agent.history``) reads enough like content that a
        # model asked to rewrite a script it already wrote hands its own elided
        # record back as the new lines. Refuse before the file is written —
        # once the placeholder is on disk the script is gone, and every rewrite
        # from then on elides what is left and shrinks it further.
        if carries_elision_marker(line):
            # The line itself is deliberately not quoted back; see the same
            # refusal in hpca.agent.file_tools for the measurement that settled
            # it. Quoting the placeholder returns it to the context, and the
            # model composes its next call out of the refusal it just read.
            return (
                f"Script NOT created: line {number} of content_lines is not "
                "script content, it is a placeholder the session history left "
                "in place of a payload it did not keep, so what you sent is "
                "your own record of an earlier call rather than the script. "
                f"Nothing was written. {hints.ELISION_REWRITE_SCRIPT}"
            )
    lines = args.content_lines
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) == 1 and nonempty[0].lstrip().startswith("#!"):
        # A one-line "script" whose only line is a shebang comments itself
        # out entirely; syntax checks would pass vacuously.
        return (
            "Script NOT created: the whole script is a single shebang line, so "
            f"it would do nothing. {hints.SCRIPT_SHEBANG_ONLY}"
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
    note = f" ({'; '.join(warnings)})" if warnings else ""
    strict = f" {hints.BASH_STRICT_MODE}" if args.kind == "bash" else ""
    return (
        f"Created script {name!r} at {path} ({args.kind}); "
        f"syntax check ok{note}. Start it with start_background_script, or run "
        f"it now with run_bash: {{{name}}} expands to its path.{strict}"
    )


class ReadFileParams(BaseModel):
    path: str = Field(
        description="Path of the file to read; a directory is listed instead"
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


def _list_dir(path: Path, max_lines: int) -> str:
    # A directory is a dead end for read_text; instead of a raw
    # IsADirectoryError, list it, which is the answer the model wanted often
    # enough that it is worth not costing a second call.
    entries = sorted(
        p.name + ("/" if p.is_dir() else "") for p in path.iterdir()
    )
    shown = entries[:max_lines]
    omitted = len(entries) - len(shown)
    tail = f"\n... [{omitted} more] ..." if omitted > 0 else ""
    body = "\n".join(shown) or "(empty)"
    return (
        f"{path} is a directory, not a file. Contents:\n{body}{tail}\n"
        f"Read one with read_file on its full path."
    )


def _read_lines(path: Path) -> list[str]:
    return path.read_text(errors="replace").splitlines()


async def read_file(args: ReadFileParams, ctx: ToolContext) -> str:
    try:
        path = resolve_path(args.path, ctx.workdir)
    except PathError as exc:
        return str(exc)
    if not path.exists():
        # Say so plainly: without this the read raises a bare
        # FileNotFoundError at the model, which reads as a crash rather than
        # as an answer it can act on.
        return f"Nothing at {path}. {hints.PATH_NOT_FOUND}"
    if path.is_dir():
        return _list_dir(path, args.max_lines)
    # Whole-file, unlike the run_bash tail above, and it has to be: the page
    # this returns is addressed by line number and its continuation hint
    # quotes the file's total, neither of which a window from one end knows.
    # So this one only gets off the loop — a slow read is then a slow tool
    # call rather than a frozen UI.
    lines = await asyncio.to_thread(_read_lines, path)
    total = len(lines)
    if args.start_line > total:
        return (
            f"start_line={args.start_line} is past the end of {path}: it has "
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
        return body
    return (
        f"{body}\n... [file continues: lines {end + 1}-{total}; call "
        f"read_file again with start_line={end + 1}, and max_lines up to 500 "
        f"to see more per call]"
    )


class StartBackgroundScriptParams(BaseModel):
    name: str = Field(description="Name of the script to run, as create_script took it")
    args: str = Field(default="", description="Command-line arguments, space-separated")


async def start_background_script(
    args: StartBackgroundScriptParams, ctx: ToolContext
) -> str:
    path = script_path(args.name, ctx)
    if path is None:
        known = ", ".join(script_names(ctx)[:MAX_KEYS_IN_NOTE]) or "(none)"
        return (
            f"NOT started: there is no script called {args.name!r}. "
            f"The scripts you have kept are: {known}"
        )
    interpreter = INTERPRETER.get(path.suffix)
    if interpreter is None:
        raise ValueError(
            f"Cannot start {args.name!r}: unknown script type {path.suffix!r}"
        )
    # A model that does not get an immediate result readily starts the script
    # twice; both copies then write the same outputs, and the corrupted result
    # is far worse than the wasted CPU (seen in a live session: two sniffles
    # runs onto one VCF). The wait is now honest — §5.4 reports the exit.
    running = ctx.runner.running_named(args.name)
    if running is not None:
        return (
            f"NOT started: {args.name!r} is already running (pid "
            f"{running}), and a second copy would write the same output files. "
            f"{hints.SCRIPT_ALREADY_RUNNING}"
        )
    argv = interpreter + [str(path)] + (args.args.split() if args.args else [])
    record = await ctx.runner.start(argv, name=args.name, background=True)
    return (
        f"Started {args.name!r} (pid {record.pid}). It runs in the "
        f"background; logs: {record.stdout_path}, {record.stderr_path} "
        "(use read_file to check)."
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
            f"writing files. {hints.RUN_BASH_IS_NOT_A_WRITER}"
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


def _tail(text: str, stream: str, log_path: Path, *, whole: bool = True) -> str:
    """Bound one stream for the prompt, pointing at the log for the rest.

    The log path is named only when something was actually cut: a look-around
    command whose output fits needs no pointer, and printing one per run fills
    the context with paths the model never reads.

    ``text`` is what :func:`read_tail` returned and ``whole`` is its verdict on
    whether that is the entire stream. When it is not, the count of earlier
    lines is the one thing this cannot say — the window carries no evidence
    about what precedes it, and counting means reading the gigabyte that was
    deliberately not read — so the note drops the number and keeps the
    sentence. Everything the model does with the note (there is more, it is at
    this path) is unchanged by that.
    """
    lines = text.splitlines()
    kept = lines[-RUN_OUTPUT_LINES:]
    body = "\n".join(kept)
    cut_head = len(body) > RUN_OUTPUT_CHARS  # long lines, few of them
    body = body[-RUN_OUTPUT_CHARS:]
    if not body.strip():
        return ""
    omitted = len(lines) - len(kept)
    if omitted > 0 or cut_head or not whole:
        if not whole:
            what = f"earlier {stream} lines"
        elif omitted > 0:
            what = f"{omitted} earlier {stream} lines"
        else:
            what = f"the start of {stream}"
        note = f"\n[... {what} omitted; read_file {str(log_path)!r} for all of it]"
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

    ``stderr`` is what :func:`_read_run_streams` gathered, which for a large
    stderr is both ends of it and not the middle. Both ends, because a bounded
    read has to choose one and the messages live at either: a script without
    `set -e` prints its first failure at the front and keeps going, while a
    script that dies at its last line prints there. The two windows overlap
    for any stderr small enough — costing nothing, since what is collected
    here is a *set* of line numbers.
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


@dataclass(frozen=True)
class _RunStreams:
    """What one finished run's logs had to say, already read.

    Held as text rather than as paths so that every read a run_bash answer
    needs happens in one place, off the loop, and nothing downstream is
    tempted to open a log again while composing a string.
    """

    stdout: str
    stdout_whole: bool
    stderr: str
    stderr_whole: bool
    stderr_head: str  # empty unless the stderr tail fell short of the start


async def _read_run_streams(record) -> _RunStreams:
    """Every read a run_bash answer needs, bounded and off the event loop.

    Bounded because the answer keeps a few kilobytes however much was printed;
    off the loop because the UI and the agent share one, so a synchronous read
    here is a freeze the user watches happen — and the logs live in the app
    dir, which on a cluster node is NFS, where even a bounded read costs a
    network round trip.

    stderr's head is fetched only when its tail did not already reach the
    start of the file, so the ordinary run — where stderr is a line or two —
    still costs exactly two reads.
    """
    stdout, stderr = await asyncio.gather(
        asyncio.to_thread(read_tail, record.stdout_path, RUN_OUTPUT_WINDOW),
        asyncio.to_thread(read_tail, record.stderr_path, RUN_OUTPUT_WINDOW),
    )
    stderr_text, stderr_whole = stderr
    head = ""
    if not stderr_whole:
        head = await asyncio.to_thread(
            read_head, record.stderr_path, RUN_OUTPUT_WINDOW
        )
    return _RunStreams(
        stdout=stdout[0],
        stdout_whole=stdout[1],
        stderr=stderr_text,
        stderr_whole=stderr_whole,
        stderr_head=head,
    )


def _run_output(streams: _RunStreams, record) -> str:
    """Both streams, bounded, each naming its log file only if it was cut."""
    parts = [
        _tail(text, stream, path, whole=whole)
        for text, whole, stream, path in (
            (streams.stdout, streams.stdout_whole, "stdout", record.stdout_path),
            (streams.stderr, streams.stderr_whole, "stderr", record.stderr_path),
        )
    ]
    return "\n\n".join(part for part in parts if part) or "(no output)"


async def run_bash(args: RunBashParams, ctx: ToolContext) -> str:
    """Write a throwaway bash script, syntax-check it, run it, wait, and return
    its output — all in one call.

    This is the look-around workhorse: find a file, check a program, read a BAM
    header, list conda envs. It is also how a *kept* script is run
    synchronously — write ``{name}`` and it expands to its path — so there is
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
            f"nothing. {hints.RUN_BASH_SHEBANG_ONLY}"
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
    streams = await _read_run_streams(record)
    seen_stderr = "\n".join(
        part for part in (streams.stderr_head, streams.stderr) if part
    )
    cited = _cited_lines(lines, seen_stderr, path)
    output = "\n\n".join(
        part for part in [_run_output(streams, record), cited] if part
    )
    if record.state == "killed":
        return (
            f"TIMED OUT after {args.timeout_s}s and was killed. "
            f"{hints.RUN_BASH_TIMED_OUT}\n\n{output}"
        )
    if record.exit_code == 0:
        return f"ran (exit 0).\n\n{output}"
    return (
        f"ran (FAILED, exit {record.exit_code}). {hints.RUN_BASH_FAILED}"
        f"\n\n{output}"
        f"{_unresolved_note(unresolved, ctx)}"
    )


class ListScriptsParams(BaseModel):
    pass


async def list_scripts(args: ListScriptsParams, ctx: ToolContext) -> str:
    """The scripts this session has kept, by name.

    What is left of ``list_paths`` once paths are just paths: a script's name is
    the one handle in the system that is not a path, so it is the one thing
    still worth being able to list.
    """
    names = script_names(ctx)
    if not names:
        return "No scripts kept yet."
    return "\n".join(f"{name}: {script_path(name, ctx)}" for name in names)


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="create_script",
            description=(
                "Create a bash or python script (syntax-checked) — those two "
                "languages ONLY. Anything else that runs — a Snakefile, a "
                "Makefile, a nextflow .nf, an R script — is not a script "
                "here: write it with create_file at the path it needs, then "
                "create_script a short bash script that calls it"
            ),
            params=CreateScriptParams,
            handler=create_script,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read a file by path, paged: returns up to max_lines from "
                "start_line and tells you where to continue if the file goes "
                "on. A directory is listed instead."
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
                "kept script by writing {name}, which expands to its path. "
                "Use for anything you want the result of now"
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
                "Run a kept script (by name) as a tracked background process for "
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
            name="list_scripts",
            description="List the scripts kept in this session, by name",
            params=ListScriptsParams,
            handler=list_scripts,
        )
    )
    return registry
