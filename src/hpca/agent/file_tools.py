"""File operation tools (§5.1, §5.3): all destructive paths trash-backed.

``register_path`` is the single entry point for user-mentioned paths into the
registry — the one place the model must echo a literal path (copied from the
user's message). Everything else is key-based. Deletions and overwrites are
HITL-gated; a plain move/copy to a fresh target is not (conditional
``is_destructive_call``).

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

import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
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


def _source(ctx: ToolContext, key: str, subpath: str) -> Path:
    """Resolve a registry key, then optionally descend into it by subpath."""
    return _descend(ctx.registry.resolve(key), key, subpath)


class RegisterPathParams(BaseModel):
    key: str = Field(description="New registry key for this path")
    path: str = Field(
        description="Absolute path exactly as the user wrote it — copy verbatim"
    )


async def register_path(args: RegisterPathParams, ctx: ToolContext) -> str:
    path = Path(args.path)
    if not path.exists():
        return (
            f"Path NOT registered: {path} does not exist. Check the spelling "
            "against the user's message, or ask the user."
        )
    ctx.registry.register(args.key, path)
    kind = "directory" if path.is_dir() else "file"
    return f"Registered {kind} as {args.key!r}."


class DeleteFileParams(BaseModel):
    registry_key: str = Field(description="Registry key of the file to delete")
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
        _source(ctx, args.registry_key, args.subpath)
        return True
    except Exception:
        return False


def _describe_delete(args: DeleteFileParams, ctx: ToolContext) -> str:
    path = _source(ctx, args.registry_key, args.subpath)
    size = path.stat().st_size if path.exists() else 0
    backed_up = ctx.trash is not None and size < ctx.trash.backup_limit_bytes
    backup_note = (
        "will be recoverable from trash"
        if backed_up
        else "NO BACKUP (file exceeds the backup size limit) — irreversible"
    )
    return f"rm {path}\n({size} bytes; {backup_note})"


def _describe_move(args: MoveFileParams, ctx: ToolContext) -> str:
    source = ctx.registry.resolve(args.source_key)
    target = _move_target(args, ctx)
    note = " (OVERWRITES existing file — old file goes to trash)" if target.exists() else ""
    return f"mv {source} → {target}{note}"


def _describe_copy(args: CopyFileParams, ctx: ToolContext) -> str:
    source = ctx.registry.resolve(args.source_key)
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
    registry_key: str = Field(description="Registry key of the file to edit")
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
    return _source(ctx, args.registry_key, args.subpath)


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


def _change(args: EditFileParams) -> str:
    """The size of the change, in the one phrasing the prompt and the result
    both use: "3 lines → 2 lines", or "3 lines deleted" when nothing goes back."""
    if not args.new_lines:
        return f"{_lines(len(args.old_lines))} deleted"
    return f"{_lines(len(args.old_lines))} → {_lines(len(args.new_lines))}"


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
        path = _edit_target(args, ctx)
    except ValueError as exc:
        return str(exc)
    if path.is_dir():
        return (
            f"NOT edited: {args.registry_key!r} is a directory. Pass subpath to "
            "edit a file inside it."
        )
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError) as exc:
        return f"NOT edited: {path} could not be read as text ({type(exc).__name__})."

    file_lines = text.split("\n")
    hits = _find_runs(file_lines, list(args.old_lines))
    if not hits:
        return (
            f"NOT edited: those lines are not in {path}."
            f"{_near_miss(file_lines, list(args.old_lines))}"
        )
    if len(hits) > 1:
        where = ", ".join(f"line {index + 1}" for index in hits[:5])
        return (
            f"NOT edited: those lines occur {len(hits)} times in {path} ({where}), "
            "so which one you mean is ambiguous. Call edit_file again with "
            "enough surrounding lines to pick out the one you want."
        )
    start = hits[0]
    after = start + len(args.old_lines)
    edited = "\n".join(file_lines[:start] + list(args.new_lines) + file_lines[after:])

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
    path.write_text(edited)
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    # Naming the undo, not just the backup: a copy the model does not know how
    # to reach is a copy the user is told to go dig for by hand.
    note = (
        "The version before this edit is in the trash: to undo, move the "
        "edited file aside with move_file, then restore_file this path."
        if entry.trashed_path
        else "NO backup was kept (the file is above the backup size limit), so "
        "this cannot be undone."
    )
    return f"Edited {path} at line {start + 1}: {_change(args)}{extra}. {note}"


class CreateFileParams(BaseModel):
    dir_key: str = Field(
        description="Registry key of the directory to create the file in"
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

    base = ctx.registry.resolve(args.dir_key)
    if not base.is_dir():
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
    content = "\n".join(args.content_lines) + "\n"

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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    key = ctx.registry.register_auto(path, hint=path.stem)
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    return (
        f"Created {path} ({_lines(len(args.content_lines))}), registered as "
        f"{key!r}{extra}. Change it with edit_file, not by writing it again."
    )


class MoveFileParams(BaseModel):
    source_key: str = Field(description="Registry key of the file to move")
    subpath: str = Field(
        default="",
        description=(
            "Path relative to source_key when it names a directory. Leave "
            "empty to move the key itself."
        ),
    )
    dest_dir_key: str = Field(description="Registry key of the target directory")
    new_name: str = Field(
        default="", description="Optional new file name; default keeps the name"
    )


def _move_target(args: MoveFileParams, ctx: ToolContext) -> Path:
    source = _source(ctx, args.source_key, args.subpath)
    dest_dir = ctx.registry.resolve(args.dest_dir_key)
    return dest_dir / (args.new_name or source.name)


def _move_overwrites(args: MoveFileParams, ctx: ToolContext) -> bool:
    try:
        return _move_target(args, ctx).exists()
    except Exception:
        return False  # resolution errors surface in the handler instead


async def move_file(args: MoveFileParams, ctx: ToolContext) -> str:
    import shutil

    try:
        source = _source(ctx, args.source_key, args.subpath)
    except ValueError as exc:
        return str(exc)
    target = _move_target(args, ctx)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    shutil.move(str(source), target)
    if args.subpath:
        # source_key still names the directory; register the moved file freshly.
        key = ctx.registry.register_auto(target, hint=target.name)
        return f"Moved {args.source_key!r}/{args.subpath} to {target}, registered as {key!r}{note}."
    ctx.registry.reassign(args.source_key, target)
    return f"Moved {args.source_key!r} to {target}{note}."


class CopyFileParams(BaseModel):
    source_key: str = Field(description="Registry key of the file to copy")
    subpath: str = Field(
        default="",
        description=(
            "Path relative to source_key when it names a directory. Leave "
            "empty to copy the key itself."
        ),
    )
    dest_dir_key: str = Field(description="Registry key of the target directory")
    new_name: str = Field(
        default="", description="Optional new file name; default keeps the name"
    )


def _copy_target(args: CopyFileParams, ctx: ToolContext) -> Path:
    source = _source(ctx, args.source_key, args.subpath)
    dest_dir = ctx.registry.resolve(args.dest_dir_key)
    return dest_dir / (args.new_name or source.name)


def _copy_overwrites(args: CopyFileParams, ctx: ToolContext) -> bool:
    try:
        return _copy_target(args, ctx).exists()
    except Exception:
        return False


async def copy_file(args: CopyFileParams, ctx: ToolContext) -> str:
    import shutil

    try:
        source = _source(ctx, args.source_key, args.subpath)
    except ValueError as exc:
        return str(exc)
    target = _copy_target(args, ctx)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    shutil.copy2(source, target)
    hint = f"{source.name}_copy" if args.subpath else f"{args.source_key}_copy"
    key = ctx.registry.register_auto(target, hint=hint)
    return f"Copied {args.source_key!r} to {target}, registered as {key!r}{note}."


def add_file_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="register_path",
            description="Register a path (the user's, or one you found) under a new key",
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
                "run_bash. It will not overwrite: to change a file that "
                "exists, call edit_file"
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
