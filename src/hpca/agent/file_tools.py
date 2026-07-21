"""File operation tools (§5.1, §5.3): all destructive paths trash-backed.

``register_path`` is the single entry point for user-mentioned paths into the
registry — the one place the model must echo a literal path (copied from the
user's message). Everything else is key-based. Deletions and overwrites are
HITL-gated; a plain move/copy to a fresh target is not (conditional
``is_destructive_call``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry


def _require_trash(ctx: ToolContext):
    if ctx.trash is None:
        raise RuntimeError("Trash manager is not configured in this session")
    return ctx.trash


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


def _delete_resolvable(args: DeleteFileParams, ctx: ToolContext) -> bool:
    # Gate only calls that can actually run; unresolvable keys go straight to
    # the error-feedback loop instead of asking the user to approve a dud.
    try:
        ctx.registry.resolve(args.registry_key)
        return True
    except Exception:
        return False


def _describe_delete(args: DeleteFileParams, ctx: ToolContext) -> str:
    path = ctx.registry.resolve(args.registry_key)
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
    path = ctx.registry.resolve(args.registry_key)
    entry = trash.trash(path)
    ctx.registry.remove(args.registry_key)
    if entry.method == "none":
        return (
            f"Deleted {path} WITHOUT backup (file was above the backup size "
            "limit); this cannot be undone."
        )
    return f"Deleted {path} (recoverable from trash via {entry.method})."


class MoveFileParams(BaseModel):
    source_key: str = Field(description="Registry key of the file to move")
    dest_dir_key: str = Field(description="Registry key of the target directory")
    new_name: str = Field(
        default="", description="Optional new file name; default keeps the name"
    )


def _move_target(args: MoveFileParams, ctx: ToolContext) -> Path:
    source = ctx.registry.resolve(args.source_key)
    dest_dir = ctx.registry.resolve(args.dest_dir_key)
    return dest_dir / (args.new_name or source.name)


def _move_overwrites(args: MoveFileParams, ctx: ToolContext) -> bool:
    try:
        return _move_target(args, ctx).exists()
    except Exception:
        return False  # resolution errors surface in the handler instead


async def move_file(args: MoveFileParams, ctx: ToolContext) -> str:
    import shutil

    source = ctx.registry.resolve(args.source_key)
    target = _move_target(args, ctx)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    shutil.move(str(source), target)
    ctx.registry.reassign(args.source_key, target)
    return f"Moved {args.source_key!r} to {target}{note}."


class CopyFileParams(BaseModel):
    source_key: str = Field(description="Registry key of the file to copy")
    dest_dir_key: str = Field(description="Registry key of the target directory")
    new_name: str = Field(
        default="", description="Optional new file name; default keeps the name"
    )


def _copy_target(args: CopyFileParams, ctx: ToolContext) -> Path:
    source = ctx.registry.resolve(args.source_key)
    dest_dir = ctx.registry.resolve(args.dest_dir_key)
    return dest_dir / (args.new_name or source.name)


def _copy_overwrites(args: CopyFileParams, ctx: ToolContext) -> bool:
    try:
        return _copy_target(args, ctx).exists()
    except Exception:
        return False


async def copy_file(args: CopyFileParams, ctx: ToolContext) -> str:
    import shutil

    source = ctx.registry.resolve(args.source_key)
    target = _copy_target(args, ctx)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    shutil.copy2(source, target)
    key = ctx.registry.register_auto(target, hint=f"{args.source_key}_copy")
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
            description="Delete a registered file (trash-backed)",
            params=DeleteFileParams,
            handler=delete_file,
            is_destructive_call=_delete_resolvable,
            describe_call=_describe_delete,
        )
    )
    registry.register(
        Tool(
            name="move_file",
            description="Move a registered file into a registered directory",
            params=MoveFileParams,
            handler=move_file,
            is_destructive_call=_move_overwrites,
            describe_call=_describe_move,
        )
    )
    registry.register(
        Tool(
            name="copy_file",
            description="Copy a registered file into a registered directory",
            params=CopyFileParams,
            handler=copy_file,
            is_destructive_call=_copy_overwrites,
            describe_call=_describe_copy,
        )
    )
    return registry
