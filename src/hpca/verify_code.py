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


@dataclass
class SymbolReport:
    symbol: str
    status: str  # confirmed | mismatch | not_indexed
    detail: str = ""


# --------------------------------------------------------------- extraction

BashUsage = tuple[str, list[str], list[str]]  # command, subcommand words, flags


def extract_bash(content: str) -> list[BashUsage]:
    usages: list[BashUsage] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            tokens = line.split()
        segment: list[str] = []
        for token in tokens + ["|"]:
            if token in SEGMENT_SEPARATORS or token.endswith(("|", ";")):
                if segment:
                    usage = _parse_segment(segment)
                    if usage:
                        usages.append(usage)
                segment = []
            else:
                segment.append(token)
    return usages


def _parse_segment(tokens: list[str]) -> BashUsage | None:
    while tokens and ("=" in tokens[0] and not tokens[0].startswith("-")):
        tokens = tokens[1:]  # leading VAR=value assignments
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


def _verify_bash(content: str, index: SymbolIndex) -> list[SymbolReport]:
    reports: list[SymbolReport] = []
    unindexed_seen: set[str] = set()
    for command, subcommands, flags in extract_bash(content):
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
        known = index.flags_for(key)
        for flag in flags:
            if flag in known:
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
