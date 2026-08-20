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
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from hpca.agent import hints
from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.embeddings import EmbeddingError
from hpca.paths import resolve_path
from hpca.rag import chunk_text, embed_fitting
from hpca.symbols import index_python_source, parse_help_flags, parse_manpage_flags
from hpca.verify_code import basename, commands_needing_docs

MANPAGE_MAX_LINES = 400
SOURCE_MAX_LINES = 200
SEARCH_TOP_K = 5
DOC_SUFFIXES = {".md", ".txt", ".rst", ".text"}
OVERSTRIKE_RE = re.compile(".\x08")

HELP_TIMEOUT_SECONDS = 10
HELP_MAX_CHARS = 200_000
# Below this, assume the parse failed rather than that the tool has no flags.
# A half-parsed flag list is worse than none: it turns correct scripts into
# gate failures. Falling short here leaves the command unindexed, which only
# warns — the pre-existing behaviour.
MIN_PARSED_FLAGS = 4
MAX_AUTOINDEX_PROBES = 6
# Probing runs the program with --help. Fetching a man page never does, so
# these stay reachable by the man path; only the exec probe is refused.
NEVER_EXECUTE = {
    "rm", "rmdir", "dd", "mkfs", "shred", "mv", "cp", "chmod", "chown",
    "truncate", "fdisk", "mkswap", "wipefs", "sbatch", "srun", "scancel",
}


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


def safe_to_execute(executable: str, *, scripts_dir: Path | None = None) -> bool:
    """Whether probing this command with ``--help`` may run it.

    Deliberately not a judgement about *where* the program lives. An earlier
    version demanded a ``bin/`` directory as a proxy for "packaged software"
    and was wrong in both directions: it refused real tools installed at
    ``/software/<tool>-<version>/<tool>``, an ordinary HPC layout, while
    happily running whatever a user had dropped in ``~/bin``. Location
    describes where something was put, not what it does, so the restriction
    cost real coverage — unindexed commands are unverifiable, which is the
    problem this whole path exists to fix — and bought no guarantee.

    Two exact rules remain. Known-destructive commands are never run; their
    man pages still cover them, and fetching one executes nothing. Neither is
    anything the agent itself wrote: a generated script has no ``--help`` to
    read, and running one before the approval gate (§5.3) has seen it would
    invert that gate.
    """
    if basename(executable) in NEVER_EXECUTE:
        return False
    resolved = shutil.which(executable)
    if resolved is None:
        return False  # not on PATH and not an executable path: nothing to run
    if scripts_dir is not None:
        try:
            Path(resolved).resolve().relative_to(Path(scripts_dir).resolve())
        except ValueError:
            return True  # outside the agent's own scripts, which is the norm
        return False
    return True


async def fetch_help(executable: str, subcommand: str = "") -> str | None:
    """``<cmd> [sub] --help`` text, or None if the program yielded nothing.

    Output is read from both streams and regardless of exit status: plenty of
    bioinformatics tools print their usage to stderr and exit non-zero
    (minimap2, bwa), which a returncode check would throw away.
    """
    argv = [executable] + ([subcommand] if subcommand else [])
    for flag in ("--help", "-h"):
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, flag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ, COLUMNS="80", MANWIDTH="80"),
            )
        except (OSError, ValueError):
            return None  # not executable / not found — no point trying -h
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), HELP_TIMEOUT_SECONDS
            )
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            await proc.wait()
            return None  # give up on a hanging tool rather than hang again on -h
        text = (stdout + stderr).decode(errors="replace")[:HELP_MAX_CHARS]
        if text.strip():
            return text
    return None


async def learn_command(executable: str, subcommand: str, ctx: ToolContext) -> str | None:
    """Index one CLI's flags on demand; returns the index key, or None.

    Both sources are unioned rather than tried in order, because each misses
    flags the other has: ``sort``/``head``/``sed`` document no OPTIONS section
    a man parser can find, while ``find``'s man page lists 9 flags and hides
    ``-maxdepth`` in EXPRESSION, where ``--help`` does list it. Missing a real
    flag is what turns a correct script into a gate failure, so coverage wins.
    The subcommand-qualified key is tried first because ``samtools --help``
    lists subcommands rather than ``samtools view``'s flags.
    """
    if ctx.symbols is None:
        return None
    name = basename(executable)
    candidates = [(f"{name}-{subcommand}", subcommand)] if subcommand else []
    candidates.append((name, ""))
    for key, sub in candidates:
        if ctx.symbols.has_command(key):
            return key
        symbols = []
        manpage = await fetch_manpage(key)
        if manpage is not None:
            symbols += parse_manpage_flags(manpage, command=key)
        if safe_to_execute(executable, scripts_dir=ctx.scripts_dir):
            help_text = await fetch_help(executable, sub)
            if help_text is not None:
                symbols += parse_help_flags(help_text, command=key)
        unique = list({symbol.name: symbol for symbol in reversed(symbols)}.values())
        if len(unique) >= MIN_PARSED_FLAGS:
            ctx.symbols.clear_source(f"man:{key}")
            ctx.symbols.clear_source(f"help:{key}")
            ctx.symbols.add(unique)
            return key
    return None


async def autoindex_script_commands(
    kind: str, content: str, ctx: ToolContext
) -> list[str]:
    """Learn the flags of every external program a script drives (§5.2).

    Runs before the verification gate so that gate has something to check.
    Deterministic on purpose: the model never decides whether to look a tool
    up, because under instruction load a small model reliably decides not to.
    Commands that cannot be learned are remembered as failures so a session
    probes each one at most once.
    """
    if ctx.symbols is None:
        return []
    pending = commands_needing_docs(kind, content, index=ctx.symbols)
    learned: list[str] = []
    for executable, subcommand in pending[:MAX_AUTOINDEX_PROBES]:
        name = basename(executable)
        if name in ctx.doc_probe_failed:
            continue
        key = await learn_command(executable, subcommand, ctx)
        if key is None:
            ctx.doc_probe_failed.add(name)
        else:
            learned.append(key)
    return learned


class LookupSymbolParams(BaseModel):
    name: str = Field(description="Exact symbol: function name or CLI flag")
    kind: str = Field(
        default="", description="Optional filter: function|method|class|cli-flag"
    )


async def lookup_symbol(args: LookupSymbolParams, ctx: ToolContext) -> str:
    if ctx.symbols is None:
        return f"No symbol index available; {hints.NO_SYMBOL_INDEX}"
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
    path: str = Field(description="Path of the source file")
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=SOURCE_MAX_LINES, ge=1)


async def read_source(args: ReadSourceParams, ctx: ToolContext) -> str:
    path = resolve_path(args.path, ctx.workdir)
    if not path.is_file():
        return f"Nothing to read at {path}. {hints.PATH_NOT_FOUND}"
    lines = path.read_text(errors="replace").splitlines()
    end = min(args.end_line, args.start_line + SOURCE_MAX_LINES - 1, len(lines))
    excerpt = lines[args.start_line - 1 : end]
    numbered = [
        f"{number}: {line}"
        for number, line in enumerate(excerpt, start=args.start_line)
    ]
    return "\n".join(numbered) or "(empty range)"


class SearchDocsParams(BaseModel):
    query: str = Field(description="Prose question or topic to search for")


async def search_docs(args: SearchDocsParams, ctx: ToolContext) -> str:
    """Semantic retrieval over indexed docs (§5.6.2)."""
    if ctx.rag is None or ctx.embedder is None:
        return f"Semantic search is not configured; {hints.NO_SEMANTIC_SEARCH}"
    if ctx.rag.count() == 0:
        return f"The document index is empty — {hints.EMPTY_DOC_INDEX}"
    try:
        vectors = await ctx.embedder.embed([args.query])
    except EmbeddingError as e:
        return f"Semantic search unavailable (embedding backend error: {e})."
    hits = ctx.rag.query(vectors[0], k=SEARCH_TOP_K)
    parts = []
    for hit in hits:
        parts.append(f"── {hit.source} (distance {hit.distance:.3f}) ──\n{hit.text}")
    return "\n\n".join(parts)


async def _rag_index_text(ctx: ToolContext, source: str, text: str) -> int | str:
    """Chunk+embed+store one document; chunk count, or an error string.

    Embedding goes through ``embed_fitting`` so a chunk the model finds too
    long is split rather than taken as a verdict on the whole document (§5.6.2).
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0
    try:
        chunks, vectors = await embed_fitting(ctx.embedder, chunks)
    except EmbeddingError as e:
        return f"embedding backend error: {e}"
    if not chunks:
        return 0
    ctx.rag.clear_source(source)
    ctx.rag.add(source, chunks, vectors)
    return len(chunks)


class IndexDocsParams(BaseModel):
    what: Literal["python_source", "manpages", "docs_dir"] = Field(
        description="What to index"
    )
    target: str = Field(
        description="python_source/docs_dir: path of a directory; "
        "manpages: space-separated command names, e.g. 'grep samtools-view'"
    )


async def index_docs(args: IndexDocsParams, ctx: ToolContext) -> str:
    if ctx.symbols is None:
        raise RuntimeError("No symbol index configured in this session")
    rag_ready = ctx.rag is not None and ctx.embedder is not None

    if args.what == "python_source":
        root = resolve_path(args.target, ctx.workdir)
        files = index_python_source(ctx.symbols, root)
        return f"Indexed {files} Python files from {root}."

    if args.what == "docs_dir":
        if not rag_ready:
            return (
                "Cannot index docs_dir: semantic search is not configured "
                "(no embedding backend)."
            )
        root = resolve_path(args.target, ctx.workdir)
        indexed, problems = 0, []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in DOC_SUFFIXES or not path.is_file():
                continue
            outcome = await _rag_index_text(
                ctx, str(path), path.read_text(errors="replace")
            )
            if isinstance(outcome, str):
                problems.append(f"{path.name}: {outcome}")
            elif outcome:
                indexed += 1
        message = f"Indexed {indexed} documents from {args.target!r} for search."
        if problems:
            # The count leads, and the examples follow it. Reporting only the
            # first three read as a footnote on a success when in fact most of
            # a manual had been dropped, which is how a 100-of-1597 index came
            # to be announced as done.
            message += (
                f" {len(problems)} could NOT be indexed"
                + (f" (of {indexed + len(problems)} found)" if indexed else "")
                + ": "
                + "; ".join(problems[:3])
                + ("; ..." if len(problems) > 3 else "")
            )
        return message

    indexed, missing, searchable = [], [], 0
    for command in args.target.split():
        text = await fetch_manpage(command)
        if text is None:
            missing.append(command)
            continue
        symbols = parse_manpage_flags(text, command=command)
        ctx.symbols.clear_source(f"man:{command}")
        ctx.symbols.add(symbols)
        indexed.append(f"{command} ({len(symbols)} flags)")
        if rag_ready:
            outcome = await _rag_index_text(ctx, f"man:{command}", text)
            if isinstance(outcome, int):
                searchable += outcome
    parts = []
    if indexed:
        parts.append("Indexed man pages: " + ", ".join(indexed) + ".")
    if searchable:
        parts.append(f"Also indexed {searchable} chunks for semantic search.")
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
    return await research(ctx.llm, args.question, ctx)


RESEARCH_TOOL_NAMES = ["lookup_symbol", "read_manpage", "read_source", "search_docs"]


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
            name="search_docs",
            description="Semantic search over indexed documentation (prose "
            "questions)",
            params=SearchDocsParams,
            handler=search_docs,
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
