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
"""

from __future__ import annotations

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
            "there. Move or rename that file first, then restore again — the "
            "backup is still in the trash."
        )
    except OSError as exc:
        return f"Could not restore {entry.original_path}: {exc}"
    key = ctx.registry.register_auto(restored, hint=restored.name)
    return f"Restored {restored}, registered as {key!r}."


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
