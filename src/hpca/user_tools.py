"""User tools: Python files in the app dir that add tools to the running agent.

HPCA's own tools live in its source and change whenever HPCA is updated. A
tool a user writes for themselves — or has the agent write, through the
shipped ``new-tool`` skill — has to outlive that, so it lives in
``<app_dir>/tools/`` and never in the checkout, where an update would conflict
with it or erase it.

A tool file is an ordinary module with one entry point, the same shape as the
``add_*_tools`` functions HPCA's own tool modules end with::

    def register(registry: ToolRegistry) -> None:
        registry.register(Tool(name=..., description=..., params=..., handler=...))

Every ``*.py`` in the directory is a candidate; files starting with ``_`` or
``.`` are skipped (scratch copies, editor backups). Loading is *hot*:
`UserTools.reload` re-reads the directory and swaps the result into the live
registry in place. The registry is read on every decision, so the next round of
every session sees the change without a graph rebuild — the same trick the
memory service uses to add ``read_skill`` late.

What a file has to pass, because a tool that cannot be put on the wire breaks
every decision of every session, not just its own calls:

- ``register`` registers at least one `Tool` and raises nothing;
- each name is an identifier, and never a built-in tool's — a user ``edit_file``
  would take the trash backup and the approval gate with it — nor a name another
  user file already took;
- the description is not empty, the handler is ``async def`` (the graph awaits
  it), and the parameter model renders into both tool protocols.

A file that fails keeps its *previous* version loaded, if it had one: a broken
edit in the middle of a session must not take away the working tool it was
meant to improve. The report says so, every time.

**Startup loads only what a reload approved.** A reload is the one moment a
person says yes to new code — the agent's ``reload_user_tools`` behind its
approval gate, or the user's own ``/reload-tools`` — and it records the digest
of every file it let in, in ``.approved.json`` next to them. Startup loads a
file only while its bytes still hash to that digest; anything else on disk is
listed as waiting, never executed. Without this, a draft the agent wrote with
``create_file`` (which never asks) would run inside the agent on the next start,
and the approval would be a formality a restart walks around. A tool written by
hand costs one ``/reload-tools``.

`check` is the same load into a staging area the model never sees, so a file
can be tried — and one of its tools called once — before it is let in.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel

import hpca
from hpca.agent.tools import Tool, ToolRegistry
from hpca.config import app_dir

USER_TOOLS_SUBDIR = "tools"
# Module names the files are imported under. They have to stay in sys.modules
# while their tools are live: pydantic resolves a parameter model's
# annotations through ``sys.modules[cls.__module__]``, and a file written with
# ``from __future__ import annotations`` has nothing but annotations. A checked
# file gets its own prefix so that trying a new version never replaces the
# module the live version's models resolve through.
LIVE_PREFIX = "hpca_user_tool__"
CHECK_PREFIX = "hpca_user_tool_check__"
# The function-name rule both tool protocols accept, narrowed to an identifier
# so a name reads the same in a JSON envelope, a native call and a log line.
TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# The digests of the file versions a reload let in; startup loads nothing else.
APPROVED_FILE = ".approved.json"
# How much source the reload approval shows. A tool file is a page or two; the
# cap is there for the file that is not, and the cut is said out loud.
PREVIEW_CHARS = 20_000


def user_tools_dir() -> Path:
    """Where user tools live: ``<app_dir>/tools``. Computed per call, like
    ``app_dir()`` itself, so a test's ``$HPCA_HOME`` is honoured."""
    return app_dir() / USER_TOOLS_SUBDIR


def hpca_source() -> tuple[Path, Path | None]:
    """The running HPCA's package dir, and the checkout around it if there is one.

    The package is always readable — it is the code running right now. The
    checkout (``src/`` layout, a ``pyproject.toml`` two levels up) exists for an
    editable install only, and is where the design doc and tests are.
    """
    package = Path(hpca.__file__).resolve().parent
    root = package.parent.parent
    return package, root if (root / "pyproject.toml").is_file() else None


def tool_files(directory: Path) -> list[Path]:
    """The files `reload` loads, in the order it loads them."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix == ".py"
        and path.is_file()
        and not path.name.startswith(("_", "."))
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _digest_of(path: Path) -> str | None:
    try:
        return _digest(path.read_bytes())
    except OSError:
        return None


@dataclass
class LoadedFile:
    """One file's load: the tools it registered, or why it registered none.

    ``problems`` empty means every tool in ``tools`` passed; any problem means
    none of them may go live — a file is let in whole or not at all.
    """

    path: Path
    digest: str = ""
    source: str = ""
    tools: list[Tool] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    # Left unexecuted at startup because no reload approved this version.
    unapproved: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems

    def names(self) -> list[str]:
        return [tool.name for tool in self.tools]


def _where(error: BaseException, path: Path) -> str:
    """The line of the user's file an error came from, if it came from there.

    The model fixes a tool by editing that file, so a traceback through
    pydantic or importlib is noise; the last frame in the file is the answer.
    """
    if isinstance(error, SyntaxError) and error.lineno:
        text = (error.text or "").strip()
        return f" (line {error.lineno}: {text})" if text else f" (line {error.lineno})"
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if Path(frame.filename) == path
    ]
    if not frames:
        return ""
    frame = frames[-1]
    line = (frame.line or "").strip()
    return f" (line {frame.lineno}: {line})" if line else f" (line {frame.lineno})"


def describe_error(error: BaseException, path: Path) -> str:
    return f"{type(error).__name__}: {error}{_where(error, path)}"


def check_tool(tool: object, reserved: Mapping[str, str]) -> list[str]:
    """Why this tool may not go live; empty when it may.

    ``reserved`` maps every name the tool may not take to who holds it, so the
    refusal can say "a built-in tool" or "the tool in other.py".
    """
    if not isinstance(tool, Tool):
        return [
            f"registered a {type(tool).__name__}, not an hpca.agent.tools.Tool"
        ]
    label = f"tool {tool.name!r}"
    problems = []
    if not isinstance(tool.name, str) or not TOOL_NAME.match(tool.name):
        problems.append(
            f"{label}: a name is letters, digits and underscores, starting "
            "with a letter or underscore, at most 64 characters"
        )
    elif tool.name in reserved:
        problems.append(
            f"{label}: that name is taken by {reserved[tool.name]} — a user "
            "tool cannot replace another tool; pick a different name"
        )
    if not str(tool.description or "").strip():
        problems.append(
            f"{label}: the description is empty, and it is what the model "
            "chooses tools by"
        )
    if not inspect.iscoroutinefunction(tool.handler):
        problems.append(
            f"{label}: the handler must be an `async def` taking (args, ctx) — "
            "the agent awaits it"
        )
    if not (isinstance(tool.params, type) and issubclass(tool.params, BaseModel)):
        problems.append(f"{label}: params must be a pydantic BaseModel subclass")
    else:
        # Exactly what every decision will do with it, on both protocols —
        # a schema that cannot be rendered fails here instead of there.
        from hpca.agent.middleware import (
            decision_schema,
            format_instruction,
            tool_specs,
        )

        alone = ToolRegistry({tool.name: tool})
        try:
            json.dumps(tool_specs(alone))
            json.dumps(decision_schema(alone))
            format_instruction(alone)
        except Exception as e:
            problems.append(
                f"{label}: its parameters cannot be turned into a tool schema "
                f"({type(e).__name__}: {e})"
            )
    return problems


class _Staging(ToolRegistry):
    """The registry a file's ``register`` is handed: the real thing, with the
    one check the real one leaves to its callers — HPCA's own code never
    registers anything but a `Tool`, and a user file's mistake should read as
    one rather than as an AttributeError from inside the registry."""

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(
                "registry.register() takes an hpca.agent.tools.Tool, not a "
                f"{type(tool).__name__}"
            )
        return super().register(tool)


def load_file(
    path: Path,
    *,
    reserved: Mapping[str, str],
    prefix: str = LIVE_PREFIX,
    approved: str | None = None,
) -> LoadedFile:
    """Import one tool file and collect what its ``register`` registers.

    Compiled from the bytes that were hashed, not re-read by an importer, so
    the digest is of the code that ran — which is what makes ``approved``
    (the digest a reload recorded, "" for none) a check on the code that would
    run rather than on the file a moment earlier. Never raises for anything
    the file does — ``SystemExit`` included, since a file calling ``sys.exit``
    must not take the agent with it.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        return LoadedFile(path, problems=[f"cannot read the file: {e}"])
    loaded = LoadedFile(path, _digest(data), data.decode("utf-8", errors="replace"))
    if approved is not None and approved != loaded.digest:
        loaded.unapproved = True
        loaded.problems.append(
            (
                "changed since it was last approved"
                if approved
                else "never approved"
            )
            + " — not loaded. A reload approves it: /reload-tools, or the "
            "agent's reload_user_tools"
        )
        return loaded
    module_name = prefix + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        loaded.problems.append("not importable as a Python module")
        return loaded
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    staging = _Staging()
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
        register = getattr(module, "register", None)
        if not callable(register):
            loaded.problems.append(
                "the file defines no `register(registry)` function — that is "
                "the one entry point a tool file has"
            )
        else:
            register(staging)
    except (Exception, SystemExit) as e:
        loaded.problems.append(describe_error(e, path))
    if loaded.problems:
        _restore(module_name, previous)
        return loaded
    loaded.tools = list(staging)
    if not loaded.tools:
        loaded.problems.append("register() registered no tools")
    for tool in loaded.tools:
        loaded.problems.extend(check_tool(tool, reserved))
    if loaded.problems:
        _restore(module_name, previous)
        loaded.tools = []
        return loaded
    for tool in loaded.tools:
        tool.user_file = path
    return loaded


def _restore(module_name: str, previous) -> None:
    """Put back the module a failed load displaced, so the version still live
    keeps resolving through sys.modules."""
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous


@dataclass
class ReloadReport:
    """What a reload changed, in the terms the model and the user act on."""

    directory: Path
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[LoadedFile] = field(default_factory=list)
    # Per failed file, the tools whose previous version stayed live.
    kept: dict[Path, list[str]] = field(default_factory=dict)
    # Why the approval record could not be written, if it could not.
    unrecorded: str = ""

    def render(self) -> str:
        lines = []
        live = self.added + self.changed + self.unchanged
        if live:
            parts = (
                [f"{name} (new)" for name in self.added]
                + [f"{name} (changed)" for name in self.changed]
                + [f"{name} (unchanged)" for name in self.unchanged]
            )
            lines.append(f"User tools loaded from {self.directory}: {', '.join(parts)}.")
        else:
            lines.append(f"No user tools are loaded (directory: {self.directory}).")
        if self.removed:
            lines.append(
                f"Removed (their file is gone or no longer defines them): "
                f"{', '.join(self.removed)}."
            )
        for loaded in self.failed:
            lines.append(f"{loaded.path.name} did NOT load:")
            lines.extend(f"  - {problem}" for problem in loaded.problems)
            kept = self.kept.get(loaded.path)
            if kept:
                lines.append(
                    f"  Its previous version stays loaded: {', '.join(kept)}."
                )
        if self.unrecorded:
            lines.append(
                f"The approval could not be recorded ({self.unrecorded}), so "
                "the next start will not load these tools."
            )
        return "\n".join(lines)


class UserTools:
    """The user tools live in one registry, and what it takes to change them.

    One per service, holding the registry every session decides from. Loads
    run off the loop (the app dir is NFS on a cluster, and a file's top level
    is arbitrary code); the swap into the registry runs on it, in one
    synchronous step, so no decision ever sees half a reload.
    """

    def __init__(self, registry: ToolRegistry, directory: Path | None = None) -> None:
        self._registry = registry
        self._directory = directory
        # What is live, per file. A failed file's entry is its last good load.
        self._files: dict[Path, LoadedFile] = {}
        # The files the last load refused, so the status can say why.
        self._failed: dict[Path, LoadedFile] = {}
        # The last `check` of each file — what `checked_tool` answers from.
        self._checked: dict[Path, LoadedFile] = {}
        self._lock = asyncio.Lock()

    @property
    def directory(self) -> Path:
        return self._directory if self._directory is not None else user_tools_dir()

    def live_files(self) -> dict[Path, LoadedFile]:
        return dict(self._files)

    def owned(self) -> set[str]:
        return {name for loaded in self._files.values() for name in loaded.names()}

    def builtins(self) -> dict[str, str]:
        """Every name in the registry that no user file holds."""
        owned = self.owned()
        return {
            name: "a built-in tool"
            for name in self._registry.names()
            if name not in owned
        }

    def reserved_for(self, path: Path) -> dict[str, str]:
        """Every name ``path`` may not take: the built-ins, and the tools other
        user files hold. Its own live tools are not in it — re-checking a file
        must not collide with its previous self."""
        return self.builtins() | {
            name: f"the user tool in {loaded.path.name}"
            for file, loaded in self._files.items()
            if file != path
            for name in loaded.names()
        }

    # ------------------------------------------------------------------ load

    def stage(
        self,
        builtins: Mapping[str, str],
        approved: Mapping[str, str] | None = None,
    ) -> list[LoadedFile]:
        """Load every file on disk, touching nothing live. Safe off the loop
        because ``builtins`` was read from the registry on it. Collisions
        between user files are `apply`'s, settled in file order once every
        file's load is known. With ``approved`` (name → digest), a file runs
        only if it is the version recorded there."""
        return [
            load_file(
                path,
                reserved=builtins,
                approved=None if approved is None else approved.get(path.name, ""),
            )
            for path in tool_files(self.directory)
        ]

    # ------------------------------------------------------------- approval

    def approved(self) -> dict[str, str]:
        """What the last reload approved, name → digest. A missing or
        unreadable record approves nothing: failing closed is the point."""
        try:
            data = json.loads((self.directory / APPROVED_FILE).read_text())
            files = data.get("files", {})
            return {str(k): str(v) for k, v in files.items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def _record_approval(self) -> None:
        """Write down every live version. A failed file whose previous version
        stayed live keeps that version's digest: true, and harmless — the
        broken bytes on disk do not match it, so the next start skips them."""
        record = {
            "version": 1,
            "files": {path.name: f.digest for path, f in sorted(self._files.items())},
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / APPROVED_FILE
        scratch = target.with_name(f"{APPROVED_FILE}.tmp")
        scratch.write_text(json.dumps(record, indent=2) + "\n")
        scratch.replace(target)

    def apply(self, staged: list[LoadedFile]) -> ReloadReport:
        """Swap a staged load into the registry. Synchronous on purpose."""
        report = ReloadReport(self.directory)
        previous = self._files
        before = {name: path for path, f in previous.items() for name in f.names()}
        builtins = {
            name for name in self._registry.names() if name not in before
        }
        live: dict[Path, LoadedFile] = {}
        claimed: dict[str, Path] = {}

        def clash(loaded: LoadedFile) -> list[str]:
            return [
                f"tool {name!r}: that name is taken by "
                + (
                    "a built-in tool"
                    if name in builtins
                    else f"the user tool in {claimed[name].name}"
                )
                + " — pick a different name"
                for name in loaded.names()
                if name in builtins or name in claimed
            ]

        for loaded in staged:
            if loaded.ok:
                loaded.problems.extend(clash(loaded))
            if not loaded.ok:
                report.failed.append(loaded)
                old = previous.get(loaded.path)
                if old is not None and not clash(old):
                    live[loaded.path] = old
                    claimed.update({name: old.path for name in old.names()})
                    report.kept[loaded.path] = old.names()
                continue
            live[loaded.path] = loaded
            claimed.update({name: loaded.path for name in loaded.names()})

        for name in before:
            self._registry.remove(name)
        for loaded in live.values():
            for tool in loaded.tools:
                self._registry.register(tool)
        self._files = live
        self._failed = {loaded.path: loaded for loaded in report.failed}

        for path, loaded in live.items():
            if path in report.kept:
                continue
            old = previous.get(path)
            for name in loaded.names():
                if name not in before:
                    report.added.append(name)
                elif old is not None and old.digest == loaded.digest:
                    report.unchanged.append(name)
                else:
                    report.changed.append(name)
        report.removed = sorted(set(before) - set(claimed))
        return report

    def load_now(self) -> ReloadReport:
        """The startup load: approved versions only. Synchronous, because
        nothing else is running yet and the first turn should find the tools
        already there."""
        return self.apply(self.stage(self.builtins(), self.approved()))

    async def reload(self) -> ReloadReport:
        """Load everything on disk and approve what got in. Callers are the
        approval: the agent's tool behind its gate, or the user's command."""
        async with self._lock:
            staged = await asyncio.to_thread(self.stage, self.builtins())
            report = self.apply(staged)
            try:
                await asyncio.to_thread(self._record_approval)
            except OSError as e:
                report.unrecorded = str(e)
            return report

    # ----------------------------------------------------------------- check

    async def check(self, path: Path) -> LoadedFile:
        """Load one file into the staging area, never into the registry."""
        reserved = self.reserved_for(path)
        loaded = await asyncio.to_thread(
            load_file, path, reserved=reserved, prefix=CHECK_PREFIX
        )
        self._checked[path] = loaded
        return loaded

    def checked_tool(self, path: Path, name: str) -> Tool | None:
        """A tool from the last check of ``path`` — only while the file on disk
        is still the one that was checked. Side-effect free, so an approval
        predicate can ask it."""
        loaded = self._checked.get(path)
        if loaded is None or not loaded.ok:
            return None
        if _digest_of(path) != loaded.digest:
            return None
        return next((tool for tool in loaded.tools if tool.name == name), None)

    # ---------------------------------------------------------------- status

    def pending(self) -> list[tuple[Path, str, str]]:
        """What a reload would change on disk: (path, "new"|"changed"|
        "removed", its source or diff). Read-only — the reload approval is
        built from it, and an approval predicate may be asked twice."""
        changes = []
        on_disk = tool_files(self.directory)
        for path in on_disk:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            text = data.decode("utf-8", errors="replace")
            old = self._files.get(path)
            if old is None:
                changes.append((path, "new", text))
            elif old.digest != _digest(data):
                diff = "".join(
                    difflib.unified_diff(
                        old.source.splitlines(keepends=True),
                        text.splitlines(keepends=True),
                        fromfile=f"{path.name} (loaded)",
                        tofile=f"{path.name} (on disk)",
                    )
                )
                changes.append((path, "changed", diff))
        for path in self._files:
            if path not in on_disk:
                changes.append((path, "removed", ""))
        return changes

    def status(self) -> str:
        """Where things are and what is loaded — the first thing the new-tool
        skill asks for."""
        package, checkout = hpca_source()
        lines = [
            f"HPCA package (the code running now): {package}",
            (
                f"HPCA checkout: {checkout}"
                if checkout is not None
                else "HPCA checkout: none found — this is an installed package, "
                "so only the package dir above is readable"
            ),
            f"User tools directory: {self.directory}"
            + ("" if self.directory.is_dir() else " (does not exist yet)"),
        ]
        if self._files:
            lines.append("Loaded user tools:")
            lines.extend(
                f"- {path.name}: {', '.join(loaded.names())}"
                for path, loaded in sorted(self._files.items())
            )
        else:
            lines.append("Loaded user tools: none")
        waiting = []
        for path, kind, _ in self.pending():
            failed = self._failed.get(path)
            if failed is not None and failed.digest == _digest_of(path):
                kind = f"did not load: {failed.problems[0]}"
            waiting.append(f"{path.name} ({kind})")
        if waiting:
            lines.append(
                "Not loaded yet (reload_user_tools would pick these up): "
                + ", ".join(waiting)
            )
        return "\n".join(lines)


def summary(changes: list[tuple[Path, str, str]]) -> str:
    """The reload approval's one sentence: which files, and what loading means."""
    if not changes:
        return "Reloads the user tools; nothing has changed on disk since the last load."
    files = ", ".join(f"{path.name} ({kind})" for path, kind, _ in changes)
    return (
        f"Loads into the running agent: {files}. The code runs inside HPCA's "
        "own process from then on."
    )


def preview(changes: list[tuple[Path, str, str]]) -> str:
    """The pending changes as the reload approval shows them: the full source
    of a new file, the diff of a changed one. Cut out of the middle when it
    must be, never off the end — the end of a file is where a handler returns."""
    parts = []
    for path, kind, text in changes:
        parts.append(f"=== {path.name} ({kind}) ===")
        if text:
            parts.append(text.rstrip())
    out = "\n".join(parts)
    if len(out) <= PREVIEW_CHARS:
        return out
    half = PREVIEW_CHARS // 2
    head = out[:half].rsplit("\n", 1)[0]
    tail = out[len(out) - half :].split("\n", 1)[-1]
    omitted = len(out) - len(head) - len(tail)
    return f"{head}\n... [{omitted:,} characters omitted] ...\n{tail}"
