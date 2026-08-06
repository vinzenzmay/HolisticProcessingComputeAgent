"""Tests for hpca.trash: hardlink-based deletion recovery (§5.3)."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from hpca.trash import TrashManager


@pytest.fixture
def manager(tmp_path):
    return TrashManager(tmp_path / "trash", backup_limit_bytes=1024 * 1024)


@pytest.fixture
def victim(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("precious content")
    return path


class TestTrash:
    def test_file_removed_and_content_preserved(self, manager, victim):
        entry = manager.trash(victim)
        assert not victim.exists()
        assert entry.method == "hardlink"
        assert entry.trashed_path.read_text() == "precious content"
        assert entry.original_path == victim

    def test_hardlink_not_copy(self, manager, victim):
        original_inode = victim.stat().st_ino
        entry = manager.trash(victim)
        assert entry.trashed_path.stat().st_ino == original_inode

    def test_copy_fallback_on_cross_device(self, manager, victim, monkeypatch):
        def exdev(*args, **kwargs):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", exdev)
        entry = manager.trash(victim)
        assert entry.method == "copy"
        assert not victim.exists()
        assert entry.trashed_path.read_text() == "precious content"

    def test_oversized_file_gets_no_backup(self, tmp_path):
        manager = TrashManager(tmp_path / "trash", backup_limit_bytes=4)
        big = tmp_path / "big.bin"
        big.write_text("way more than four bytes")
        entry = manager.trash(big)
        assert entry.method == "none"
        assert not big.exists()
        assert entry.trashed_path is None

    def test_missing_file_raises(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.trash(tmp_path / "ghost.txt")


class TestBackup:
    """``backup`` keeps the file — the copy is for an in-place overwrite."""

    def test_file_stays_and_content_is_kept(self, manager, victim):
        entry = manager.backup(victim)
        assert victim.exists()
        assert entry.trashed_path.read_text() == "precious content"
        assert entry.original_path == victim

    def test_a_real_copy_not_a_hardlink(self, manager, victim):
        # A hardlink shares the inode, so writing the edited content through
        # the original would rewrite the "backup" along with it.
        entry = manager.backup(victim)
        assert entry.method == "copy"
        assert entry.trashed_path.stat().st_ino != victim.stat().st_ino
        victim.write_text("edited in place")
        assert entry.trashed_path.read_text() == "precious content"

    def test_backup_is_listed_and_restorable(self, manager, victim):
        manager.backup(victim)
        entries = manager.list()
        assert [e.original_path for e in entries] == [victim]
        victim.unlink()  # the edited file moved aside
        assert manager.restore(entries[0]) == victim
        assert victim.read_text() == "precious content"

    def test_oversized_file_gets_no_backup(self, tmp_path):
        manager = TrashManager(tmp_path / "trash", backup_limit_bytes=4)
        big = tmp_path / "big.bin"
        big.write_text("way more than four bytes")
        entry = manager.backup(big)
        assert entry.method == "none"
        assert entry.trashed_path is None
        assert big.exists()

    def test_missing_file_raises(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.backup(tmp_path / "ghost.txt")


class TestListRestore:
    def test_list_entries(self, manager, victim):
        manager.trash(victim)
        entries = manager.list()
        assert len(entries) == 1
        assert entries[0].original_path == victim

    def test_restore_puts_file_back(self, manager, victim):
        entry = manager.trash(victim)
        restored = manager.restore(entry)
        assert restored == victim
        assert victim.read_text() == "precious content"
        assert manager.list() == []

    def test_restore_refuses_to_overwrite(self, manager, victim):
        entry = manager.trash(victim)
        victim.write_text("new file at old path")
        with pytest.raises(FileExistsError):
            manager.restore(entry)
        assert victim.read_text() == "new file at old path"


class TestCleanup:
    def _age_entry(self, manager, entry, days):
        meta_file = entry.trashed_path.parent / "meta.json"
        meta = json.loads(meta_file.read_text())
        old = datetime.now(timezone.utc) - timedelta(days=days)
        meta["trashed_at"] = old.isoformat()
        meta_file.write_text(json.dumps(meta))

    def test_old_entries_removed(self, manager, victim, tmp_path):
        entry = manager.trash(victim)
        other = tmp_path / "fresh.txt"
        other.write_text("fresh")
        manager.trash(other)
        self._age_entry(manager, entry, days=10)
        removed = manager.cleanup(ttl_days=7)
        assert removed == 1
        remaining = manager.list()
        assert len(remaining) == 1
        assert remaining[0].original_path == other

    def test_fresh_entries_kept(self, manager, victim):
        manager.trash(victim)
        assert manager.cleanup(ttl_days=7) == 0
        assert len(manager.list()) == 1
