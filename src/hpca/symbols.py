"""Exact-match symbol table (§5.6.1) — no embeddings involved.

Function/class signatures come from indexed Python source via ``ast``; CLI
flags come from man-page OPTIONS sections. This is what the verification gate
(§5.2) and ``lookup_symbol`` query: "does ``samtools view -e`` exist?" is an
exact-match question.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

SECTION_RE = re.compile(r"^[A-Z][A-Z /-]+$")
FLAG_ENTRY_RE = re.compile(r"^\s+(-{1,2}[A-Za-z0-9][\w.-]*)")
FLAG_TOKEN_RE = re.compile(r"-{1,2}[A-Za-z0-9][\w.-]*")
METAVAR_RE = re.compile(r"^[A-Z][A-Z0-9_|.\[\]]*$|^<[^>]+>$")


@dataclass
class Symbol:
    name: str
    kind: str  # function | method | class | cli-flag | cli
    parent: str = ""
    signature: str = ""
    params: list[str] = field(default_factory=list)
    source: str = ""
    lineno: int = 0
    doc: str = ""


# ------------------------------------------------------------------- python


def _render_signature(name: str, args: ast.arguments) -> tuple[str, list[str]]:
    parts: list[str] = []
    params: list[str] = []

    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg in ("self", "cls"):
            continue
        params.append(arg.arg)
        parts.append(
            arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}"
        )
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(arg.arg)
        parts.append(
            arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}"
        )
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return f"{name}({', '.join(parts)})", params


def _first_doc_line(node) -> str:
    doc = ast.get_docstring(node)
    return doc.splitlines()[0] if doc else ""


def parse_python_module(text: str, *, module: str, source: str) -> list[Symbol]:
    """Public functions, classes, and methods of one module; [] on bad syntax."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols: list[Symbol] = []

    def add_function(node, kind: str, parent: str) -> None:
        if node.name.startswith("_"):
            return
        signature, params = _render_signature(node.name, node.args)
        symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                parent=parent,
                signature=signature,
                params=params,
                source=source,
                lineno=node.lineno,
                doc=_first_doc_line(node),
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(node, "function", module)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="class",
                    parent=module,
                    signature=node.name,
                    source=source,
                    lineno=node.lineno,
                    doc=_first_doc_line(node),
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_function(item, "method", f"{module}.{node.name}")
    return symbols


# ---------------------------------------------------------------- man pages


def _parse_flag_entries(
    text: str, *, command: str, source: str, section_gated: bool
) -> list[Symbol]:
    """Shared flag scanner for man pages and ``--help`` output.

    ``section_gated`` selects the man-page dialect: only lines inside an
    all-caps OPTIONS section count. ``--help`` output has no such headers (it
    uses "Options:", "Alignment:", or no header at all), so help mode scans
    every indented flag entry instead.
    """
    symbols: list[Symbol] = []
    in_options = not section_gated
    entry_symbols: list[Symbol] = []  # flags of the entry being read
    doc_lines: list[str] = []

    def flush() -> None:
        nonlocal entry_symbols, doc_lines
        doc = " ".join(" ".join(doc_lines).split())
        for symbol in entry_symbols:
            symbol.doc = doc
        entry_symbols, doc_lines = [], []

    for lineno, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if (
            section_gated
            and line
            and not line[0].isspace()
            and SECTION_RE.match(stripped)
        ):
            flush()
            in_options = "OPTION" in stripped
            continue
        if not in_options:
            continue
        if FLAG_ENTRY_RE.match(line):
            flush()
            # split the flag part ("-o, --output FILE") from an inline doc
            tokens = stripped.split()
            flags: list[str] = []
            rest_index = 0
            for i, token in enumerate(tokens):
                bare = token.rstrip(",")
                if FLAG_TOKEN_RE.fullmatch(bare):
                    flags.append(bare)
                    rest_index = i + 1
                elif METAVAR_RE.match(token) and flags:
                    rest_index = i + 1  # metavar like FILE / INT
                else:
                    break
            doc_lines = [" ".join(tokens[rest_index:])]
            for flag in flags:
                symbol = Symbol(
                    name=flag,
                    kind="cli-flag",
                    parent=command,
                    signature=stripped,
                    source=source,
                    lineno=lineno,
                )
                symbols.append(symbol)
                entry_symbols.append(symbol)
        elif entry_symbols and stripped:
            doc_lines.append(stripped)
    flush()
    return symbols


def parse_manpage_flags(text: str, *, command: str) -> list[Symbol]:
    """Extract flags from the OPTIONS section(s) of rendered man-page text."""
    return _parse_flag_entries(
        text, command=command, source=f"man:{command}", section_gated=True
    )


def parse_help_flags(text: str, *, command: str) -> list[Symbol]:
    """Extract flags from ``<cmd> --help`` output.

    Many bioinformatics tools ship no man page inside a conda env, so help
    text is the only machine-readable flag list available.
    """
    return _parse_flag_entries(
        text, command=command, source=f"help:{command}", section_gated=False
    )


# -------------------------------------------------------------------- store


class SymbolIndex:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, symbols: list[Symbol]) -> None:
        self._conn.executemany(
            "INSERT INTO symbols (name, kind, parent, signature, params, "
            "source, lineno, doc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    s.name,
                    s.kind,
                    s.parent,
                    s.signature,
                    json.dumps(s.params),
                    s.source,
                    s.lineno,
                    s.doc,
                )
                for s in symbols
            ],
        )
        self._conn.commit()

    def lookup(self, name: str, *, kind: str | None = None) -> list[Symbol]:
        query = "SELECT * FROM symbols WHERE name = ?"
        args: list = [name]
        if kind:
            query += " AND kind = ?"
            args.append(kind)
        return [self._to_symbol(row) for row in self._conn.execute(query, args)]

    def flags_for(self, command: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM symbols WHERE kind = 'cli-flag' AND parent = ?",
            (command,),
        ).fetchall()
        return [row["name"] for row in rows]

    def kwargs_for(self, function_name: str) -> list[str] | None:
        row = self._conn.execute(
            "SELECT params FROM symbols WHERE name = ? "
            "AND kind IN ('function', 'method') LIMIT 1",
            (function_name,),
        ).fetchone()
        return json.loads(row["params"]) if row else None

    def has_command(self, command: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM symbols WHERE parent = ? AND kind = 'cli-flag' LIMIT 1",
            (command,),
        ).fetchone()
        return row is not None

    def clear_source(self, source: str) -> None:
        self._conn.execute("DELETE FROM symbols WHERE source = ?", (source,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

    @staticmethod
    def _to_symbol(row: sqlite3.Row) -> Symbol:
        return Symbol(
            name=row["name"],
            kind=row["kind"],
            parent=row["parent"] or "",
            signature=row["signature"] or "",
            params=json.loads(row["params"] or "[]"),
            source=row["source"],
            lineno=row["lineno"] or 0,
            doc=row["doc"] or "",
        )


def index_python_source(index: SymbolIndex, root: Path) -> int:
    """Index every parseable .py under root; returns files indexed."""
    root = Path(root)
    indexed = 0
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent)
        module = ".".join(relative.with_suffix("").parts)
        source = str(path)
        index.clear_source(source)
        symbols = parse_python_module(
            path.read_text(errors="replace"), module=module, source=source
        )
        if symbols:
            index.add(symbols)
            indexed += 1
    return indexed
