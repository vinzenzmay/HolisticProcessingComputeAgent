"""Deletion recovery via hardlinks (§5.3).

Deletions never copy data: the file is hardlinked into
``<app_dir>/trash/<timestamp>/`` (zero extra space on the same filesystem —
Lustre/GPFS support this) before the original is unlinked. A real copy is the
fallback across filesystems. Files above the backup limit get *no* backup
(quota!) — the confirmation modal must say so. Each trash entry directory
carries a ``meta.json`` for listing, restore, and TTL cleanup.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class TrashEntry:
    original_path: Path
    trashed_path: Path | None  # None: file was above the backup limit
    method: str  # hardlink | copy | none
    trashed_at: str


class TrashManager:
    def __init__(self, trash_dir: Path, *, backup_limit_bytes: int) -> None:
        self._trash_dir = trash_dir
        self.backup_limit_bytes = backup_limit_bytes

    def trash(self, path: Path) -> TrashEntry:
        """Back up (if under the limit) and remove the file."""
        path = Path(path)
        size = path.stat().st_size  # raises FileNotFoundError for ghosts
        entry_dir = self._trash_dir / f"{time.time_ns()}"
        entry_dir.mkdir(parents=True)
        trashed_path: Path | None = entry_dir / path.name
        if size >= self.backup_limit_bytes:
            method = "none"
            trashed_path = None
        else:
            try:
                os.link(path, trashed_path)
                method = "hardlink"
            except OSError:  # cross-filesystem trash dir
                shutil.copy2(path, trashed_path)
                method = "copy"
        entry = TrashEntry(
            original_path=path,
            trashed_path=trashed_path,
            method=method,
            trashed_at=datetime.now(timezone.utc).isoformat(),
        )
        (entry_dir / "meta.json").write_text(
            json.dumps(
                {
                    "original_path": str(path),
                    "trashed_path": str(trashed_path) if trashed_path else None,
                    "method": method,
                    "trashed_at": entry.trashed_at,
                }
            )
        )
        path.unlink()
        return entry

    def list(self) -> list[TrashEntry]:
        entries = []
        if not self._trash_dir.exists():
            return entries
        for meta_file in sorted(self._trash_dir.glob("*/meta.json")):
            try:
                meta = json.loads(meta_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            entries.append(
                TrashEntry(
                    original_path=Path(meta["original_path"]),
                    trashed_path=Path(meta["trashed_path"])
                    if meta.get("trashed_path")
                    else None,
                    method=meta["method"],
                    trashed_at=meta["trashed_at"],
                )
            )
        return entries

    def restore(self, entry: TrashEntry) -> Path:
        if entry.trashed_path is None:
            raise FileNotFoundError(
                f"{entry.original_path} was deleted without backup (size limit)"
            )
        if entry.original_path.exists():
            raise FileExistsError(
                f"Cannot restore: {entry.original_path} already exists"
            )
        entry.original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry.trashed_path), entry.original_path)
        shutil.rmtree(entry.trashed_path.parent, ignore_errors=True)
        return entry.original_path

    def cleanup(self, ttl_days: int) -> int:
        """Remove entries older than the TTL; returns how many were removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        removed = 0
        if not self._trash_dir.exists():
            return removed
        for meta_file in self._trash_dir.glob("*/meta.json"):
            try:
                trashed_at = datetime.fromisoformat(
                    json.loads(meta_file.read_text())["trashed_at"]
                )
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
            if trashed_at < cutoff:
                shutil.rmtree(meta_file.parent, ignore_errors=True)
                removed += 1
        return removed
