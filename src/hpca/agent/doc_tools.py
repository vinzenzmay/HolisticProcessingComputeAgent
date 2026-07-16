"""Documentation tools (§5.1, §5.6): exact lookup, man pages, source, indexing.

``index_docs`` is the explicit indexing command (§5.6 — never implicit).
``ask_docs`` is the orchestrator's door to the doc-researcher (§4.2): the
question goes into a firewalled sub-loop; only the bounded, cited answer
returns.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.symbols import index_python_source, parse_manpage_flags

MANPAGE_MAX_LINES = 400
SOURCE_MAX_LINES = 200
OVERSTRIKE_RE = re.compile(".\x08")


async def fetch_manpage(name: str) -> str | None:
    """Rendered man-page text (overstrike stripped), or None if unavailable."""
    env = dict(os.environ, MANWIDTH="80")
    try:
        proc = await asyncio.create_subprocess_exec(
            "man", "-P", "cat", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
    except FileNotFoundError:
        return None
    stdout, _ = await proc.communicate()
    if proc.returncode != 0 or not stdout:
        return None
    return OVERSTRIKE_RE.sub("", stdout.decode(errors="replace"))


class LookupSymbolParams(BaseModel):
    name: str = Field(description="Exact symbol: function name or CLI flag")
    kind: str = Field(
        default="", description="Optional filter: function|method|class|cli-flag"
    )


async def lookup_symbol(args: LookupSymbolParams, ctx: ToolContext) -> str:
    if ctx.symbols is None:
        return "No symbol index available; run index_docs first."
    matches = ctx.symbols.lookup(args.name, kind=args.kind or None)
    if not matches:
        return (
            f"{args.name!r} is NOT in the symbol index "
            "(absence here does not prove the symbol does not exist)."
        )
    lines = []
    for m in matches[:10]:
        lines.append(
            f"{m.kind} {m.signature or m.name} — parent: {m.parent} "
            f"[source: {m.source}]" + (f" — {m.doc}" if m.doc else "")
        )
    return "\n".join(lines)


class ReadManpageParams(BaseModel):
    name: str = Field(description="Man page name, e.g. 'grep' or 'samtools-view'")


async def read_manpage(args: ReadManpageParams, ctx: ToolContext) -> str:
    text = await fetch_manpage(args.name)
    if text is None:
        return f"No man page found for {args.name!r} on this host."
    lines = text.splitlines()
    if len(lines) > MANPAGE_MAX_LINES:
        head = MANPAGE_MAX_LINES // 2
        tail = MANPAGE_MAX_LINES - head
        lines = (
            lines[:head]
            + [f"... [{len(lines) - head - tail} lines omitted] ..."]
            + lines[-tail:]
        )
    return "\n".join(lines)


class ReadSourceParams(BaseModel):
    registry_key: str = Field(description="Registry key of the source file")
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=SOURCE_MAX_LINES, ge=1)


async def read_source(args: ReadSourceParams, ctx: ToolContext) -> str:
    path = ctx.registry.resolve(args.registry_key)
    lines = path.read_text(errors="replace").splitlines()
    end = min(args.end_line, args.start_line + SOURCE_MAX_LINES - 1, len(lines))
    excerpt = lines[args.start_line - 1 : end]
    numbered = [
        f"{number}: {line}"
        for number, line in enumerate(excerpt, start=args.start_line)
    ]
    return "\n".join(numbered) or "(empty range)"


class IndexDocsParams(BaseModel):
    what: Literal["python_source", "manpages"] = Field(
        description="What to index"
    )
    target: str = Field(
        description="python_source: registry key of a source directory; "
        "manpages: space-separated command names, e.g. 'grep samtools-view'"
    )


async def index_docs(args: IndexDocsParams, ctx: ToolContext) -> str:
    if ctx.symbols is None:
        raise RuntimeError("No symbol index configured in this session")
    if args.what == "python_source":
        root = ctx.registry.resolve(args.target)
        files = index_python_source(ctx.symbols, root)
        return f"Indexed {files} Python files from {args.target!r}."
    indexed, missing = [], []
    for command in args.target.split():
        text = await fetch_manpage(command)
        if text is None:
            missing.append(command)
            continue
        symbols = parse_manpage_flags(text, command=command)
        ctx.symbols.clear_source(f"man:{command}")
        ctx.symbols.add(symbols)
        indexed.append(f"{command} ({len(symbols)} flags)")
    parts = []
    if indexed:
        parts.append("Indexed man pages: " + ", ".join(indexed) + ".")
    if missing:
        parts.append("No man page found for: " + ", ".join(missing) + ".")
    return " ".join(parts) or "Nothing indexed."


class AskDocsParams(BaseModel):
    question: str = Field(
        description="A focused technical question about a tool, API, or flag"
    )


async def ask_docs(args: AskDocsParams, ctx: ToolContext) -> str:
    from hpca.agent.researcher import research

    if ctx.llm is None:
        return "Documentation research is unavailable (no LLM in this context)."
    return await research(
        ctx.llm, args.question, ctx, tier1=ctx.tier1_text
    )


RESEARCH_TOOL_NAMES = ["lookup_symbol", "read_manpage", "read_source"]


def add_doc_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="lookup_symbol",
            description="Exact lookup of a function/flag in the symbol index",
            params=LookupSymbolParams,
            handler=lookup_symbol,
        )
    )
    registry.register(
        Tool(
            name="read_manpage",
            description="Read a man page (bounded)",
            params=ReadManpageParams,
            handler=read_manpage,
        )
    )
    registry.register(
        Tool(
            name="read_source",
            description="Read a line range of a registered source file",
            params=ReadSourceParams,
            handler=read_source,
        )
    )
    registry.register(
        Tool(
            name="index_docs",
            description="Index Python source or man pages into the symbol index",
            params=IndexDocsParams,
            handler=index_docs,
        )
    )
    return registry


def add_ask_docs(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="ask_docs",
            description="Answer a technical question from indexed docs/man "
            "pages (cited); use for any tool/API/flag question",
            params=AskDocsParams,
            handler=ask_docs,
        )
    )
    return registry
