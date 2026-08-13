"""File operation tools (§5.1, §5.3): all destructive paths trash-backed.

``register_path`` names a path up front, and a key stays the cheap way to
refer to one; but every key-taking argument here also takes a literal absolute
path (``PathRegistry.resolve_or_register``, which registers it in passing), so
a path the model just saw in `ls` output does not need a round-trip before it
can be used. Deletions and overwrites are HITL-gated; a plain move/copy to a
fresh target is not (conditional ``is_destructive_call``).

``restore_file`` is the other half of the trash (§5.3): without it the backup
a deletion writes is only reachable by the user digging through the app dir by
hand, which is exactly what happened the first time someone asked for a file
back. It takes the original path rather than a registry key — the key is gone,
dropped by the deletion that created the backup.

``create_file`` and ``edit_file`` are the two tools here that write content,
and they split "new file" from "change a file" rather than overlapping.
``edit_file`` exists for the cost: without it the only way to change a
400-line script is to re-send all 400 lines through ``create_script``, paying
for the file twice in one window. ``create_file`` exists because prose had no
tool at all — ``create_script`` writes only into the scripts dir under a
language suffix — so a specs.md had to go through a ``cat << 'EOF'`` heredoc
in run_bash.

Both are held to the same rules as everything else that writes: a script's
content faces §5.2's syntax and code-vs-docs gate before it lands, checked on
a scratch copy so a refusal leaves nothing broken behind. ``edit_file``
overwrites, so it also backs the previous content up and gates; ``create_file``
refuses an existing path instead, which is what keeps it out of §5.3
entirely.
"""

from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.registry import registered_note
from hpca.trash import TrashEntry

# Enough for the model to recognise the file the user means without pasting a
# week of deletions into the context.
TRASH_LIST_LIMIT = 20


def _require_trash(ctx: ToolContext):
    if ctx.trash is None:
        raise RuntimeError("Trash manager is not configured in this session")
    return ctx.trash


def _descend(base: Path, key: str, subpath: str) -> Path:
    """Resolve a path inside a registered directory by relative subpath.

    Mirrors read_file: a directory key is a legitimate handle, and subpath
    reaches a file inside it without registering every entry first. Raises
    ValueError (model-facing) on an escaping or missing subpath so the handler
    returns the message into the retry loop.
    """
    if not subpath:
        return base
    candidate = (base / subpath).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ValueError(
            f"subpath {subpath!r} escapes {key!r}; use a path inside the directory."
        )
    if not candidate.exists():
        raise ValueError(
            f"No such path: {subpath!r} under {key!r}. "
            f"Read {key!r} to list what is there."
        )
    return candidate


def _resolved(
    ctx: ToolContext, value: str, subpath: str, *, register: bool = True
) -> tuple[Path, str]:
    """``(path, key)`` for a key-or-literal-path argument, descended by subpath.

    Every key-taking argument in this module goes through
    ``PathRegistry.resolve_or_register``; ``register=False`` is what the gating
    predicates and describe helpers pass, so they resolve a literal path
    without writing to the registry (see the method's docstring).
    """
    base, key = ctx.registry.resolve_or_register(value, register=register)
    return _descend(base, key, subpath), key


def _source(ctx: ToolContext, key: str, subpath: str, *, register: bool = True) -> Path:
    """Resolve a registry key or literal path, then descend into it by subpath."""
    return _resolved(ctx, key, subpath, register=register)[0]


class RegisterPathParams(BaseModel):
    key: str = Field(
        description=(
            "Registry key for this path; reusing a key is allowed only when "
            "nothing exists at the path it points at (fixing a typo)"
        )
    )
    path: str = Field(
        description="Absolute path exactly as the user wrote it — copy verbatim"
    )


async def register_path(args: RegisterPathParams, ctx: ToolContext) -> str:
    """Give a path a key, whether or not anything is there yet.

    A missing path used to be refused outright, on the reasoning that it is
    almost always a typo. It is also, just as often, a path that does not
    exist *yet* — the output directory a run will write, the specs.md the user
    just asked for — and refusing left the model with no key for it, hence no
    way to name it in the call that would have created it. So it registers
    either way, and the result says which case this is: a key pointing at
    nothing is a real state (read_file and create_file report it in the same
    terms), not a silent one that turns into a confusing failure later.

    The cost of taking the typo is that the key is spent, and there is no
    unregister tool — so the registry lets a second call repoint a key whose
    path does not exist (nothing is behind it to protect). The result says so
    explicitly, naming the path that was dropped, because "Registered ..."
    alone reads the same whether the correction landed or was ignored.
    """
    path = Path(args.path)
    replaced = ctx.registry.register(args.key, path)
    note = (
        f" It previously pointed at {replaced}, where nothing exists, so that "
        "registration is gone."
        if replaced is not None
        else ""
    )
    if not path.exists():
        return (
            f"Registered {args.key!r}, but nothing exists at {path} yet.{note} "
            "If the user meant a path that is already there, check the spelling "
            "against their message; otherwise create it before reading it."
        )
    kind = "directory" if path.is_dir() else "file"
    return f"Registered {kind} as {args.key!r}.{note}"


class DeleteFileParams(BaseModel):
    registry_key: str = Field(
        description="Registry key or absolute path of the file to delete"
    )
    subpath: str = Field(
        default="",
        description=(
            "Path relative to registry_key when it names a directory, e.g. "
            "'logs/run.err'. Leave empty to delete the key itself."
        ),
    )


def _delete_resolvable(args: DeleteFileParams, ctx: ToolContext) -> bool:
    # Gate only calls that can actually run; unresolvable keys (and bad
    # subpaths) go straight to the error-feedback loop instead of asking the
    # user to approve a dud.
    try:
        _source(ctx, args.registry_key, args.subpath, register=False)
        return True
    except Exception:
        return False


def _describe_delete(args: DeleteFileParams, ctx: ToolContext) -> str:
    path = _source(ctx, args.registry_key, args.subpath, register=False)
    size = path.stat().st_size if path.exists() else 0
    backed_up = ctx.trash is not None and size < ctx.trash.backup_limit_bytes
    backup_note = (
        "will be recoverable from trash"
        if backed_up
        else "NO BACKUP (file exceeds the backup size limit) — irreversible"
    )
    return f"rm {path}\n({size} bytes; {backup_note})"


def _describe_move(args: MoveFileParams, ctx: ToolContext) -> str:
    source, _ = ctx.registry.resolve_or_register(args.source_key, register=False)
    target = _move_target(args, ctx)
    note = " (OVERWRITES existing file — old file goes to trash)" if target.exists() else ""
    return f"mv {source} → {target}{note}"


def _describe_copy(args: CopyFileParams, ctx: ToolContext) -> str:
    source, _ = ctx.registry.resolve_or_register(args.source_key, register=False)
    target = _copy_target(args, ctx)
    note = " (OVERWRITES existing file — old file goes to trash)" if target.exists() else ""
    return f"cp {source} → {target}{note}"


async def delete_file(args: DeleteFileParams, ctx: ToolContext) -> str:
    trash = _require_trash(ctx)
    try:
        path = _source(ctx, args.registry_key, args.subpath)
    except ValueError as exc:
        return str(exc)
    entry = trash.trash(path)
    # No registered_note here, unlike the other tools: a literal path handed to
    # delete_file does get a key on the way in, and the sweep below drops it
    # again in the same call, so reporting it would name a key that is gone.
    # Drop every key pointing at the deleted path (the directory key survives
    # when only a subpath was removed); keys are few, so a scan is fine.
    for key, registered in ctx.registry.list().items():
        if registered == path:
            ctx.registry.remove(key)
    if entry.method == "none":
        return (
            f"Deleted {path} WITHOUT backup (file was above the backup size "
            "limit); this cannot be undone."
        )
    # Naming the tool, not just the trash: the model has to know recovery is
    # something it can DO, or it tells the user to go dig in the app dir.
    return (
        f"Deleted {path} (backed up via {entry.method}; recoverable with "
        f"restore_file if the user asks for it back)."
    )


class RestoreFileParams(BaseModel):
    path: str = Field(
        default="",
        description=(
            "Original path of the deleted file, as delete_file reported it "
            "(its file name alone also works). Leave empty to list what is in "
            "the trash."
        ),
    )


def _local_time(trashed_at: str) -> str:
    try:
        return f"{datetime.fromisoformat(trashed_at).astimezone():%Y-%m-%d %H:%M}"
    except ValueError:
        return trashed_at


def _newest_first(entries: list[TrashEntry]) -> list[TrashEntry]:
    # trashed_at is an ISO UTC stamp, so it sorts lexicographically. Sorting on
    # it rather than on the entry directory's name keeps the order right for
    # entries written by any past version.
    return sorted(entries, key=lambda entry: entry.trashed_at, reverse=True)


def _trash_listing(entries: list[TrashEntry]) -> str:
    if not entries:
        return "The trash is empty — there is nothing to restore."
    shown = entries[:TRASH_LIST_LIMIT]
    lines = [
        f"{entry.original_path} (deleted {_local_time(entry.trashed_at)})"
        + ("" if entry.trashed_path else " — NO backup, not restorable")
        for entry in shown
    ]
    if len(entries) > len(shown):
        lines.append(f"... and {len(entries) - len(shown)} older entries")
    return "In the trash, newest first:\n" + "\n".join(lines)


def _matching_entries(entries: list[TrashEntry], wanted: str) -> list[TrashEntry]:
    """Trash entries for ``wanted``, newest first.

    Widening rather than exact-only: the model may quote the path it saw, the
    file name alone, or a fragment of a long path, and a restore that fails on
    a near-miss costs the user their file.
    """
    name = Path(wanted).name
    for candidates in (
        [e for e in entries if str(e.original_path) == wanted],
        [e for e in entries if e.original_path.name == name],
        [e for e in entries if wanted in str(e.original_path)],
    ):
        if candidates:
            return candidates
    return []


async def restore_file(args: RestoreFileParams, ctx: ToolContext) -> str:
    """Put a trashed file back where it was (§5.3 recovery).

    Never gated: restore only ever *creates* a file, and refuses outright when
    something already sits at the original path, so there is nothing for the
    user to approve. Ambiguity is handed back to the model instead of guessed
    at — restoring the wrong file over a name collision is not recoverable in
    turn.
    """
    trash = _require_trash(ctx)
    entries = _newest_first(trash.list())
    wanted = args.path.strip()
    if not wanted:
        return _trash_listing(entries)
    matches = _matching_entries(entries, wanted)
    if not matches:
        return f"Nothing in the trash matches {wanted!r}.\n{_trash_listing(entries)}"
    if len({entry.original_path for entry in matches}) > 1:
        return (
            f"{wanted!r} matches several deleted files — call restore_file "
            f"again with the full path of the one you want:\n"
            f"{_trash_listing(matches)}"
        )
    entry = matches[0]  # the most recent deletion of that path
    if entry.trashed_path is None:
        return (
            f"Cannot restore {entry.original_path}: it was deleted without a "
            "backup (it was above the backup size limit), so no copy was kept. "
            "Tell the user it is not recoverable from HPCA's trash."
        )
    try:
        restored = trash.restore(entry)
    except FileExistsError:
        return (
            f"Cannot restore {entry.original_path}: a file already exists "
            "there. Move or rename that file first with move_file, then "
            "restore again — the backup is still in the trash. Undoing an edit "
            "always lands here, because the backup and the edited file share a "
            "path. Do not delete the file to clear the path: that trashes it "
            "under the same path too, and the restore would then bring back "
            "what you were undoing."
        )
    except OSError as exc:
        return f"Could not restore {entry.original_path}: {exc}"
    key = ctx.registry.register_auto(restored, hint=restored.name)
    return f"Restored {restored}, registered as {key!r}."


class EditFileParams(BaseModel):
    registry_key: str = Field(
        description="Registry key or absolute path of the file to edit"
    )
    subpath: str = Field(
        default="",
        description=(
            "Path relative to registry_key when it names a directory, e.g. "
            "'config/run.yaml'. Leave empty to edit the key itself."
        ),
    )
    # Line arrays for the same reason create_script takes them: the live model
    # fills string arrays reliably and mangles \n escapes in long strings.
    old_lines: list[str] = Field(
        min_length=1,
        description=(
            "The exact consecutive lines to replace, copied from read_file "
            "without their line numbers — indentation and spacing included. "
            "Include enough surrounding lines that they occur only once."
        ),
    )
    new_lines: list[str] = Field(
        default_factory=list,
        description=(
            "The lines to put in their place, one string per line. Send an "
            "empty array to delete the old lines."
        ),
    )


def _find_runs(haystack: list[str], needle: list[str]) -> list[int]:
    """Every index where ``needle`` occurs as a run of whole lines.

    Whole lines, not a substring of the file: the model is asked for lines, so
    matching them as lines is what makes "occurs twice" mean what it says — and
    keeps ``fi`` from matching inside ``pipefail``.
    """
    span = len(needle)
    return [
        index
        for index in range(len(haystack) - span + 1)
        if haystack[index : index + span] == needle
    ]


# The repair-and-fuzz layer below exists to absorb model imperfection inside
# the tool instead of bouncing errors back into the retry loop: the live model
# (a mid-size Qwen behind constrained decoding) reliably produces *almost*
# right arguments — an embedded "\n" inside one array element, line numbers
# copied straight from read_file's numbered listing, a smart quote where the
# file has ASCII — and every bounce costs a round-trip plus, usually, a full
# re-read of the file.

def _split_embedded_newlines(lines: list[str]) -> list[str]:
    """One string per line, even when the model packed several into one.

    The params ask for arrays of lines, but a model that thinks in blocks
    sometimes sends ["a\\nb"] for two lines. The intent is unambiguous, so
    split rather than refuse.
    """
    out: list[str] = []
    for line in lines:
        out.extend(line.split("\n"))
    return out


# A line-number prefix as read_file (or `cat -n`, or an editor) renders it:
# optional indent, digits, then ":", a tab, or "|" — e.g. "12: x", "12\tx",
# " 12 | x".
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*(?::|\t|\|)\s?")


def _strip_line_number_prefixes(lines: list[str]) -> list[str] | None:
    """The lines without a copied numbered-listing prefix, or None.

    Only fires when EVERY element carries the prefix — one numbered line in
    ten is genuine content (a dict entry, a timestamp), ten in ten is a model
    copying a numbered listing. The caller additionally tries the unstripped
    lines first, so genuine content that actually matches is never mangled.
    """
    if not all(_LINE_NUMBER_PREFIX.match(line) for line in lines):
        return None
    return [_LINE_NUMBER_PREFIX.sub("", line, count=1) for line in lines]


# Fuzzy normalization, per line (modelled on PI's normalizeForFuzzyMatch):
# smart quotes → ASCII quotes, Unicode dashes (U+2010..U+2015, minus U+2212)
# → "-", special spaces (NBSP and friends) → " ". Applied after NFKC, which
# already folds most compatibility characters.
_FUZZY_TRANSLATE = str.maketrans(
    {
        # smart single quotes U+2018..U+201B
        **dict.fromkeys(map(ord, "\u2018\u2019\u201a\u201b"), "'"),
        # smart double quotes U+201C..U+201F
        **dict.fromkeys(map(ord, "\u201c\u201d\u201e\u201f"), '"'),
        # hyphen, non-breaking hyphen, figure/en/em dash, horizontal bar, minus
        **dict.fromkeys(
            map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-"
        ),
        # NBSP, en/em-family spaces, narrow NBSP, math space, ideographic space
        **dict.fromkeys(
            map(
                ord,
                "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
                "\u2009\u200a\u202f\u205f\u3000",
            ),
            " ",
        ),
    }
)


def _fuzzy_line(line: str) -> str:
    """A line as fuzzy matching sees it. Leading whitespace survives on
    purpose: indentation is meaning (Python), trailing whitespace never is."""
    return (
        unicodedata.normalize("NFKC", line).translate(_FUZZY_TRANSLATE).rstrip()
    )


# The matching ladder, strictest first: exact; trailing whitespace stripped;
# full fuzzy. A unique match at ANY level applies the edit — asking the model
# to fix a trailing space it cannot even see is a wasted round-trip.
_MATCH_LEVELS = (None, str.rstrip, _fuzzy_line)


def _match_ladder(file_lines: list[str], old_lines: list[str]) -> list[int]:
    """Hit indices at the strictest level that matches at all.

    Ambiguity is judged at the level that matched: two exact occurrences are
    two occurrences, and never get "rescued" by a looser level seeing three.
    """
    for transform in _MATCH_LEVELS:
        if transform is None:
            hay, needle = file_lines, old_lines
        else:
            hay = [transform(line) for line in file_lines]
            needle = [transform(line) for line in old_lines]
        hits = _find_runs(hay, needle)
        if hits:
            return hits
    return []


def _near_miss(file_lines: list[str], old_lines: list[str]) -> str:
    """Where the lines nearly match, for the usual whitespace near-miss.

    A model that mis-indents its copy gets back "line 12" instead of a flat
    "not found", which is the difference between fixing the call and re-reading
    the whole file.
    """
    stripped = [line.strip() for line in file_lines]
    wanted = [line.strip() for line in old_lines]
    hits = _find_runs(stripped, wanted)
    if not hits:
        return ""
    where = ", ".join(f"line {index + 1}" for index in hits[:5])
    return (
        f" The same text ignoring leading/trailing whitespace is at {where} — "
        "read the file again and copy the indentation exactly."
    )


def _edit_target(args: EditFileParams, ctx: ToolContext) -> Path:
    # Read-only: the handler resolves for itself (it needs the key back), so
    # every caller left here is a predicate or a preview.
    return _source(ctx, args.registry_key, args.subpath, register=False)


def edit_target_path(args: EditFileParams, ctx: ToolContext) -> Path | None:
    """The file an edit_file call would touch, or None when it cannot resolve.

    The graph's per-file approval memory (§3.5 auto mode) keys on this: two
    calls that resolve to the same path are edits of the same file, whatever
    mix of key and subpath the model used to name it. Swallows resolution
    errors — an unresolvable call never skips a gate, it just fails normally.
    """
    try:
        return _edit_target(args, ctx).resolve()
    except Exception:
        return None


def _edit_resolvable(args: EditFileParams, ctx: ToolContext) -> bool:
    # Overwriting a file's content is destructive (§5.3), so every edit that
    # can actually run gates; a dud key goes to the error-feedback loop instead
    # of asking the user to approve something that will not happen.
    try:
        return _edit_target(args, ctx).is_file()
    except Exception:
        return False


def _lines(count: int) -> str:
    return f"{count} line" if count == 1 else f"{count} lines"


def _tbd_note(content: str) -> str:
    """The unfinished-work reminder for a skeleton-then-fill write.

    The guidance teaches `TBD: ...` placeholder lines; this is the tool's
    side of that protocol. A 27B filling ten sections loses count (measured:
    sections left as TBD, or the model answering 'done' with two remaining),
    and the fix that costs nothing until it matters is the result naming what
    is left after every write.
    """
    remaining = [
        (i + 1, line.strip())
        for i, line in enumerate(content.split("\n"))
        if line.lstrip().startswith("TBD")
    ]
    if not remaining:
        return ""
    line_no, text = remaining[0]
    return (
        f" {len(remaining)} placeholder line(s) still to fill, next at "
        f"line {line_no}: {text!r}."
    )


def _change(args: EditFileParams) -> str:
    """The size of the change, in the one phrasing the prompt and the result
    both use: "3 lines → 2 lines", or "3 lines deleted" when nothing goes back.
    Counted after embedded-\\n repair, so the numbers match what lands."""
    old = _split_embedded_newlines(list(args.old_lines))
    new = _split_embedded_newlines(list(args.new_lines))
    if not new:
        return f"{_lines(len(old))} deleted"
    return f"{_lines(len(old))} → {_lines(len(new))}"


def _read_for_edit(path: Path) -> tuple[str, str, str]:
    """(LF-normalized text, bom, dominant line ending) of ``path``.

    Matching happens on LF-normalized, BOM-less text — the model copies lines
    out of read_file and never sees a BOM or a \\r — and the write restores
    both, so an edit does not silently re-terminate a CRLF file.
    """
    raw = path.read_bytes().decode()
    bom, text = ("\ufeff", raw[1:]) if raw.startswith("\ufeff") else ("", raw)
    # Dominant = first: a file that opens with \r\n is a CRLF file, whatever a
    # stray later line does (mirrors PI's detectLineEnding).
    crlf, lf = text.find("\r\n"), text.find("\n")
    ending = "\r\n" if crlf != -1 and lf != -1 and crlf < lf else "\n"
    return text.replace("\r\n", "\n").replace("\r", "\n"), bom, ending


def _describe_edit(args: EditFileParams, ctx: ToolContext) -> str:
    path = _edit_target(args, ctx)
    size = path.stat().st_size if path.is_file() else 0
    backed_up = ctx.trash is not None and size < ctx.trash.backup_limit_bytes
    backup_note = (
        "the current file is copied to trash first"
        if backed_up
        else "NO BACKUP (file exceeds the backup size limit) — irreversible"
    )
    return f"edit {path}\n({_change(args)}; {backup_note})"


def edit_preview(arguments: dict, ctx: object = None) -> str:
    """One edit as the user judges it: the lines out, then the lines in.

    The whole file is not the call — the change is — so this is what the
    approval prompt and the chat's call box show (see ``modes.script_preview``).
    Reads nothing and never raises: it runs on every call, and again whenever a
    parked turn resumes.
    """
    out = [f"- {line}" for line in arguments.get("old_lines") or []]
    into = [f"+ {line}" for line in arguments.get("new_lines") or []]
    return "\n".join(out + into)


async def edit_file(args: EditFileParams, ctx: ToolContext) -> str:
    """Replace an exact run of lines in a registered file.

    The point is the file that stays: a one-line fix to a 400-line script costs
    the two lines, not the script twice over (once read, once rewritten). What
    it must not become is a way around the rest of §5: the previous content
    goes to the trash exactly as a deletion's would, and when the file is a
    script the edited content faces §5.2's gate before it is written — checked
    on a scratch copy, so a refused edit leaves the real file untouched rather
    than briefly broken.
    """
    from hpca.agent.builtin_tools import KIND_BY_SUFFIX, check_script_content

    try:
        path, key = _resolved(ctx, args.registry_key, args.subpath)
    except ValueError as exc:
        return str(exc)
    if path.is_dir():
        return (
            f"NOT edited: {args.registry_key!r} is a directory. Pass subpath to "
            "edit a file inside it."
        )
    try:
        text, bom, ending = _read_for_edit(path)
    except (UnicodeDecodeError, OSError) as exc:
        return f"NOT edited: {path} could not be read as text ({type(exc).__name__})."

    # Repair before matching: embedded \n split always (the intent is
    # unambiguous), numbered-prefix stripping only as a fallback when the
    # lines as sent match nothing — genuine "12: x" content that IS in the
    # file matches first and is never mangled.
    old_lines = _split_embedded_newlines(list(args.old_lines))
    new_lines = _split_embedded_newlines(list(args.new_lines))
    file_lines = text.split("\n")
    hits = _match_ladder(file_lines, old_lines)
    if not hits:
        stripped = _strip_line_number_prefixes(old_lines)
        if stripped is not None:
            hits = _match_ladder(file_lines, stripped)
            if hits:
                old_lines = stripped
    if not hits:
        return (
            f"NOT edited: those lines are not in {path}."
            f"{_near_miss(file_lines, old_lines)}"
        )
    if len(hits) > 1:
        where = ", ".join(f"line {index + 1}" for index in hits[:5])
        return (
            f"NOT edited: those lines occur {len(hits)} times in {path} ({where}), "
            "so which one you mean is ambiguous. Call edit_file again with "
            "enough surrounding lines to pick out the one you want."
        )
    # A unique fallback-level match applies: the file's own lines outside the
    # run are untouched (whole-line replacement, so nothing gets normalized
    # that the edit did not touch), and new_lines land verbatim.
    start = hits[0]
    after = start + len(old_lines)
    edited = "\n".join(file_lines[:start] + new_lines + file_lines[after:])
    if edited == text:
        return (
            f"NOT edited: the replacement produces identical content — "
            "old_lines and new_lines are the same. If you meant to change "
            "something else, re-read the file and edit that."
        )

    warnings: list[str] = []
    kind = KIND_BY_SUFFIX.get(path.suffix)
    if kind is not None:
        # §5.2 on a scratch copy: the checkers read a file, and the real one
        # must not spend even a moment holding content the gate would refuse.
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        scratch = ctx.scripts_dir / f"edit_{time.time_ns()}{path.suffix}"
        scratch.write_text(edited)
        try:
            refused, warnings = await check_script_content(
                kind, scratch, edited, ctx, refusal="NOT edited"
            )
        finally:
            scratch.unlink(missing_ok=True)
        if refused:
            return refused

    entry = _require_trash(ctx).backup(path)
    # Write back what the file was, not what matching needed: dominant line
    # ending and BOM restored, so an edit never re-terminates a CRLF file.
    out = edited.replace("\n", ending) if ending != "\n" else edited
    path.write_bytes((bom + out).encode())
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    # One short line on success (§ result-message diet): the standing undo
    # lecture cost tokens on every edit; the trash behavior itself is
    # unchanged, and only the irreversible case still warns.
    no_backup = (
        ""
        if entry.trashed_path
        else " NO backup was kept (the file is above the backup size limit), "
        "so this cannot be undone."
    )
    return (
        f"Edited {path} at line {start + 1}: {_change(args)}{extra}"
        f"{registered_note(args.registry_key, key)}."
        f"{no_backup}{_tbd_note(edited)}"
    )


class CreateFileParams(BaseModel):
    dir_key: str = Field(
        description=(
            "Registry key or absolute path of the directory to create the "
            "file in"
        )
    )
    name: str = Field(
        description=(
            "Name for the new file, e.g. 'specs.md'. May name a subdirectory "
            "of it too, e.g. 'docs/specs.md'"
        )
    )
    # Last, and an array of lines, for the two reasons the module and
    # hpca.agent.middleware give: long strings get their \n escapes mangled,
    # and nothing may follow a long array.
    content_lines: list[str] = Field(
        min_length=1,
        description="File content as an array of lines, one string per line",
    )


async def create_file(args: CreateFileParams, ctx: ToolContext) -> str:
    """Write a new text file into a registered directory.

    The gap this fills is prose. ``create_script`` writes only into the scripts
    dir under a language suffix, and ``edit_file`` needs a file to already
    exist, so the one way to author a specs.md was a ``cat << 'EOF'`` heredoc
    through run_bash — a hundred lines of documentation squeezed through bash
    quoting, where a single stray line costs the whole file (that is exactly
    how the swallowed-argument bug in hpca.agent.middleware surfaced).

    It creates and does not overwrite: an existing path is refused and pointed
    at edit_file. That keeps it non-destructive by construction — nothing for
    §5.3 to gate, so writing a document does not stop for approval — and keeps
    "change a file" as one tool rather than two ways in.
    """
    from hpca.agent.builtin_tools import KIND_BY_SUFFIX, check_script_content

    base, dir_key = ctx.registry.resolve_or_register(args.dir_key)
    # Only a path that IS something else disqualifies the key. One that is not
    # there yet does not: register_path takes a directory before it exists, and
    # the write below already makes missing parents, so refusing here would
    # bounce a call the tool can simply carry out — the retry loop is where the
    # backend gets confused, which is the one place not to send it.
    if base.exists() and not base.is_dir():
        return (
            f"NOT created: {args.dir_key!r} is a file, not a directory. Give "
            "the key of the directory the file belongs in."
        )
    name = args.name.strip()
    path = (base / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(base.resolve()):
        return (
            f"NOT created: {name!r} points outside {args.dir_key!r}. Give a "
            "name relative to it, and register another directory if the file "
            "belongs somewhere else."
        )
    if path.exists():
        return (
            f"NOT created: {path} already exists. To change it call edit_file "
            "with the lines to replace; to replace it wholesale, delete_file "
            "first (the old version stays recoverable from the trash)."
        )
    # Same embedded-\n repair as edit_file: the array is lines, but a model
    # that packs a block into one element still means the lines it contains.
    content_lines = _split_embedded_newlines(list(args.content_lines))
    content = "\n".join(content_lines) + "\n"

    warnings: list[str] = []
    kind = KIND_BY_SUFFIX.get(path.suffix)
    if kind is not None:
        # §5.2 on a scratch copy, as edit_file does: a second way to put
        # content into a script file must not be a second way around the gate,
        # and a refused file must never have existed at the real path.
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        scratch = ctx.scripts_dir / f"new_{time.time_ns()}{path.suffix}"
        scratch.write_text(content)
        try:
            refused, warnings = await check_script_content(
                kind, scratch, content, ctx, refusal="NOT created"
            )
        finally:
            scratch.unlink(missing_ok=True)
        if refused:
            return refused

    made_dirs = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    key = ctx.registry.register_auto(path, hint=path.stem)
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    # Say when a directory had to be made: writing a file is one thing, and
    # creating the tree it sits in is another the user may not have asked for.
    made = f" (created {path.parent} on the way)" if made_dirs else ""
    return (
        f"Created {path} ({_lines(len(content_lines))}){made}, registered as "
        f"{key!r}{extra}. Change it with edit_file, not by writing it "
        f"again.{_tbd_note(content)}{registered_note(args.dir_key, dir_key)}"
    )


class MoveFileParams(BaseModel):
    source_key: str = Field(
        description="Registry key or absolute path of the file to move"
    )
    subpath: str = Field(
        default="",
        description=(
            "Path relative to source_key when it names a directory. Leave "
            "empty to move the key itself."
        ),
    )
    dest_dir_key: str = Field(
        description="Registry key or absolute path of the target directory"
    )
    new_name: str = Field(
        default="", description="Optional new file name; default keeps the name"
    )


def _move_target(args: MoveFileParams, ctx: ToolContext) -> Path:
    # Predicate/describe path only — read-only, as their purity rule requires.
    source = _source(ctx, args.source_key, args.subpath, register=False)
    dest_dir, _ = ctx.registry.resolve_or_register(args.dest_dir_key, register=False)
    return dest_dir / (args.new_name or source.name)


def _move_overwrites(args: MoveFileParams, ctx: ToolContext) -> bool:
    try:
        return _move_target(args, ctx).exists()
    except Exception:
        return False  # resolution errors surface in the handler instead


async def move_file(args: MoveFileParams, ctx: ToolContext) -> str:
    import shutil

    try:
        source, source_key = _resolved(ctx, args.source_key, args.subpath)
        dest_dir, dest_key = ctx.registry.resolve_or_register(args.dest_dir_key)
    except ValueError as exc:
        return str(exc)
    target = dest_dir / (args.new_name or source.name)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    shutil.move(str(source), target)
    # source_key/dest_key, not the raw arguments: a literal path was given a key
    # on the way in, and it is that key reassign and the result talk about.
    keys = registered_note(args.dest_dir_key, dest_key)
    if args.subpath:
        # source_key still names the directory; register the moved file freshly.
        key = ctx.registry.register_auto(target, hint=target.name)
        return (
            f"Moved {source_key!r}/{args.subpath} to {target}, registered as "
            f"{key!r}{note}.{registered_note(args.source_key, source_key)}{keys}"
        )
    ctx.registry.reassign(source_key, target)
    return f"Moved {source_key!r} to {target}{note}.{keys}"


class CopyFileParams(BaseModel):
    source_key: str = Field(
        description="Registry key or absolute path of the file to copy"
    )
    subpath: str = Field(
        default="",
        description=(
            "Path relative to source_key when it names a directory. Leave "
            "empty to copy the key itself."
        ),
    )
    dest_dir_key: str = Field(
        description="Registry key or absolute path of the target directory"
    )
    new_name: str = Field(
        default="", description="Optional new file name; default keeps the name"
    )


def _copy_target(args: CopyFileParams, ctx: ToolContext) -> Path:
    # Predicate/describe path only — read-only, as their purity rule requires.
    source = _source(ctx, args.source_key, args.subpath, register=False)
    dest_dir, _ = ctx.registry.resolve_or_register(args.dest_dir_key, register=False)
    return dest_dir / (args.new_name or source.name)


def _copy_overwrites(args: CopyFileParams, ctx: ToolContext) -> bool:
    try:
        return _copy_target(args, ctx).exists()
    except Exception:
        return False


async def copy_file(args: CopyFileParams, ctx: ToolContext) -> str:
    import shutil

    try:
        source, source_key = _resolved(ctx, args.source_key, args.subpath)
        dest_dir, dest_key = ctx.registry.resolve_or_register(args.dest_dir_key)
    except ValueError as exc:
        return str(exc)
    target = dest_dir / (args.new_name or source.name)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    shutil.copy2(source, target)
    # source_key, not the raw argument: a literal path hints off its own key,
    # not off the 90 characters the model typed.
    hint = f"{source.name}_copy" if args.subpath else f"{source_key}_copy"
    key = ctx.registry.register_auto(target, hint=hint)
    keys = registered_note(args.source_key, source_key) + registered_note(
        args.dest_dir_key, dest_key
    )
    return (
        f"Copied {source_key!r} to {target}, registered as {key!r}{note}.{keys}"
    )


def add_file_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="register_path",
            description=(
                "Register a path (the user's, or one you found) under a key; "
                "call it again with the same key to correct a mistyped path"
            ),
            params=RegisterPathParams,
            handler=register_path,
        )
    )
    registry.register(
        Tool(
            name="delete_file",
            description=(
                "Delete a registered file (trash-backed); pass subpath to "
                "delete a file inside a registered directory"
            ),
            params=DeleteFileParams,
            handler=delete_file,
            is_destructive_call=_delete_resolvable,
            describe_call=_describe_delete,
        )
    )
    registry.register(
        Tool(
            name="restore_file",
            description=(
                "Restore a file deleted earlier (by delete_file, or overwritten "
                "by move/copy) from the trash, using its original path; call "
                "with an empty path to list what can be restored"
            ),
            params=RestoreFileParams,
            handler=restore_file,
        )
    )
    registry.register(
        Tool(
            name="create_file",
            description=(
                "Write a NEW text file — specs, notes, a README, a config — "
                "into a registered directory, one array element per line. Use "
                "this instead of echoing or heredoc'ing a file through "
                "run_bash. Past ~150 lines send a skeleton (headings + one "
                "`TBD: ...` line each) and fill per section with edit_file. "
                "It will not overwrite: to change a file that exists, call "
                "edit_file"
            ),
            params=CreateFileParams,
            handler=create_file,
        )
    )
    registry.register(
        Tool(
            name="edit_file",
            description=(
                "Change part of a registered text file in place: give the "
                "exact lines to replace and what to put there, instead of "
                "rewriting the whole file. Pass subpath to edit a file inside "
                "a registered directory. Read the file first and copy the "
                "lines from it exactly"
            ),
            params=EditFileParams,
            handler=edit_file,
            is_destructive_call=_edit_resolvable,
            describe_call=_describe_edit,
        )
    )
    registry.register(
        Tool(
            name="move_file",
            description=(
                "Move a registered file into a registered directory; pass "
                "subpath to move a file inside a registered directory"
            ),
            params=MoveFileParams,
            handler=move_file,
            is_destructive_call=_move_overwrites,
            describe_call=_describe_move,
        )
    )
    registry.register(
        Tool(
            name="copy_file",
            description=(
                "Copy a registered file into a registered directory; pass "
                "subpath to copy a file inside a registered directory"
            ),
            params=CopyFileParams,
            handler=copy_file,
            is_destructive_call=_copy_overwrites,
            describe_call=_describe_copy,
        )
    )
    return registry
