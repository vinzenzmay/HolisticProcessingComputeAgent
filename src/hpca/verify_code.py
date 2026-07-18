"""Semantic verification gate: code vs indexed docs (§5.2).

Syntax checks miss the small model's dominant failure mode — plausible but
wrong API usage (invented CLI flags, misspelled kwargs). This gate extracts
used APIs deterministically (ast / tokenization, no LLM) and checks them
against the symbol table. Mechanical mismatches block with an actionable
message; ``not_indexed`` is a warning, never a block — index coverage is
always partial (§5.2.4).
"""

from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass

from hpca.symbols import SymbolIndex

BASH_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "do", "done", "while", "until",
    "case", "esac", "in", "function", "return", "exit", "set", "export",
    "local", "readonly", "shift", "trap", "source", "true", "false", "echo",
    "cd", "read", "[", "[[", "{", "}", "!",
}
SEGMENT_SEPARATORS = {"|", "||", "&&", ";", "&"}
PYTHON_BUILTINS = frozenset(dir(__builtins__)) | {"print", "range", "len"}

# Wrappers that run *another* program; the flags after them belong to that
# program, not the wrapper. ENVIRONMENT_TOOL_GUIDANCE actively tells the agent
# to write `conda run -n <env> <tool> ...`, so without unwrapping the gate reads
# every such line as a call to `conda` and checks minimap2's flags against it.
ENV_RUNNERS = {"conda", "mamba", "micromamba"}
RUNNER_VALUE_FLAGS = {"-n", "--name", "-p", "--prefix", "--cwd"}
TRANSPARENT_PREFIXES = {"env", "time", "nohup", "nice", "stdbuf", "exec"}


@dataclass
class SymbolReport:
    symbol: str
    status: str  # confirmed | mismatch | not_indexed
    detail: str = ""


# --------------------------------------------------------------- extraction

BashUsage = tuple[str, list[str], list[str]]  # command, subcommand words, flags


def _tokenize(line: str) -> list[str]:
    """Words plus shell operators, with operators as their own tokens.

    ``shlex.split`` leaves ``a|b`` as a single token, so an unspaced pipeline
    collapsed into one command and the downstream command's flags were checked
    against the upstream one — ``grep -v x f|minimap2 -x map-ont`` asked grep
    about ``-x``. ``punctuation_chars`` splits on operators while still
    respecting quoting, so ``grep 'a|b'`` stays one word.
    """
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return line.split()  # unbalanced quotes: best effort


def extract_bash(content: str) -> list[BashUsage]:
    usages: list[BashUsage] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = _tokenize(line)
        segment: list[str] = []
        for token in tokens + ["|"]:
            # _tokenize already isolates operators, so a trailing "|" inside a
            # token is data (awk -F'|'), not a pipeline separator
            if token in SEGMENT_SEPARATORS:
                if segment:
                    usage = _parse_segment(segment)
                    if usage:
                        usages.append(usage)
                segment = []
            else:
                segment.append(token)
    return usages


def _strip_assignments(tokens: list[str]) -> list[str]:
    while tokens and ("=" in tokens[0] and not tokens[0].startswith("-")):
        tokens = tokens[1:]  # leading VAR=value assignments
    return tokens


def _unwrap_prefixes(tokens: list[str]) -> list[str]:
    """Drop wrapper commands so flags are attributed to the program that owns them."""
    while tokens:
        head = basename(tokens[0])
        if head in TRANSPARENT_PREFIXES:
            tokens = _strip_assignments(tokens[1:])
            continue
        if head in ENV_RUNNERS and len(tokens) > 1 and tokens[1] == "run":
            tokens = tokens[2:]
            while tokens and tokens[0].startswith("-"):
                takes_value = (
                    tokens[0] in RUNNER_VALUE_FLAGS
                )  # `-n env`; `--name=env` carries its own value
                tokens = tokens[2:] if takes_value else tokens[1:]
            continue
        break
    return tokens


def basename(executable: str) -> str:
    """Index key for a command written as a bare name or an absolute path."""
    return executable.rsplit("/", 1)[-1]


def _parse_segment(tokens: list[str]) -> BashUsage | None:
    tokens = _unwrap_prefixes(_strip_assignments(tokens))
    if not tokens:
        return None
    command = tokens[0]
    if command in BASH_KEYWORDS or command.startswith(("$", "(", ">", "<")):
        return None
    flags: list[str] = []
    subcommands: list[str] = []
    for token in tokens[1:]:
        if token.startswith(">") or token.startswith("<"):
            break
        if token == "--":
            break  # end-of-options marker; everything after it is an operand
        if token == "-":
            continue  # stdin/stdout placeholder, e.g. `samtools sort -o out.bam -`
        if token.startswith("-"):
            flags.append(token.split("=", 1)[0])
        elif not flags and not subcommands and token.isalpha():
            subcommands.append(token)  # e.g. "view" in "samtools view"
    return (command, subcommands, flags)


PythonCall = tuple[str, list[str]]  # function name, kwarg names


def extract_python(content: str) -> list[PythonCall]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    calls: list[PythonCall] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            kwargs = [kw.arg for kw in node.keywords if kw.arg]
            calls.append((name, kwargs))
    return calls


# --------------------------------------------------------------------- gate


def verify_script(kind: str, content: str, *, index: SymbolIndex) -> list[SymbolReport]:
    if kind == "bash":
        return _verify_bash(content, index)
    if kind == "python":
        return _verify_python(content, index)
    return []  # R / snakemake: extraction is best-effort, deferred


def _flag_matches(flag: str, known: set[str]) -> bool:
    """Whether a flag token as written is consistent with a command's flag set.

    Short options may carry their value attached (``sort -k1,1``, ``samtools
    view -q20``) or be clustered (``-bh``), so an exact-match test rejects
    correct scripts. Matching on the leading short option is deliberately
    permissive: the failure this gate exists to catch is an *invented flag
    name*, and blocking a valid command is far more costly than letting a
    malformed value through to the tool's own error message.
    """
    if flag in known:
        return True
    if flag.startswith("--") or len(flag) <= 2:
        return False
    return f"-{flag[1]}" in known


def _verify_bash(content: str, index: SymbolIndex) -> list[SymbolReport]:
    reports: list[SymbolReport] = []
    unindexed_seen: set[str] = set()
    for executable, subcommands, flags in extract_bash(content):
        command = basename(executable)  # /path/to/envs/bio/bin/minimap2 -> minimap2
        display = command
        key = command
        if not index.has_command(key) and subcommands:
            candidate = f"{command}-{subcommands[0]}"
            if index.has_command(candidate):
                key = candidate
                display = f"{command} {subcommands[0]}"
        if not index.has_command(key):
            if flags and command not in unindexed_seen:
                unindexed_seen.add(command)
                reports.append(
                    SymbolReport(
                        symbol=command,
                        status="not_indexed",
                        detail="command not in the symbol index; flags unchecked",
                    )
                )
            continue
        known = set(index.flags_for(key))
        for flag in flags:
            if _flag_matches(flag, known):
                reports.append(
                    SymbolReport(symbol=f"{display} {flag}", status="confirmed")
                )
            else:
                reports.append(
                    SymbolReport(
                        symbol=f"{display} {flag}",
                        status="mismatch",
                        detail=(
                            f"{flag} is not a documented flag of {display}. "
                            f"Documented flags: {', '.join(sorted(known))}"
                        ),
                    )
                )
    return reports


def _verify_python(content: str, index: SymbolIndex) -> list[SymbolReport]:
    reports: list[SymbolReport] = []
    for name, kwargs in extract_python(content):
        if not kwargs:
            continue  # nothing checkable; stay quiet on print(x) etc.
        params = index.kwargs_for(name)
        if params is None:
            if name not in PYTHON_BUILTINS:
                reports.append(
                    SymbolReport(
                        symbol=f"{name}(...)",
                        status="not_indexed",
                        detail="function not in the symbol index; kwargs unchecked",
                    )
                )
            continue
        unknown = [kw for kw in kwargs if kw not in params]
        if unknown:
            signature = index.lookup(name)
            rendered = signature[0].signature if signature else ", ".join(params)
            reports.append(
                SymbolReport(
                    symbol=f"{name}(...)",
                    status="mismatch",
                    detail=(
                        f"unknown keyword argument(s) {', '.join(unknown)}; "
                        f"indexed signature is {rendered}"
                    ),
                )
            )
        else:
            reports.append(SymbolReport(symbol=f"{name}(...)", status="confirmed"))
    return reports


def commands_needing_docs(
    kind: str, content: str, *, index: SymbolIndex
) -> list[tuple[str, str]]:
    """Commands used with flags that the index cannot check yet.

    Feeds the on-demand indexing step in ``create_script``: an unindexed
    command means the gate silently passes exactly the usage most likely to be
    hallucinated. Returns ``(executable, subcommand)`` with the executable as
    written in the script — possibly an absolute path — so the caller can run
    it to fetch help text. Deduplicated by command name.
    """
    if kind != "bash":
        return []
    seen: set[str] = set()
    pending: list[tuple[str, str]] = []
    for executable, subcommands, flags in extract_bash(content):
        if not flags:
            continue  # nothing to check, so nothing to look up
        command = basename(executable)
        subcommand = subcommands[0] if subcommands else ""
        if command in seen:
            continue
        if index.has_command(command):
            continue
        if subcommand and index.has_command(f"{command}-{subcommand}"):
            continue
        seen.add(command)
        pending.append((executable, subcommand))
    return pending


def format_gate_failure(mismatches: list[SymbolReport]) -> str:
    lines = [
        "Script NOT created: the verification gate found API usage that "
        "contradicts the indexed documentation — fix and retry:"
    ]
    lines += [f"- {r.symbol}: {r.detail}" for r in mismatches]
    return "\n".join(lines)


def format_gate_warnings(not_indexed: list[SymbolReport]) -> str:
    names = ", ".join(r.symbol for r in not_indexed)
    return f"Warning: not indexed, unverified: {names}."
