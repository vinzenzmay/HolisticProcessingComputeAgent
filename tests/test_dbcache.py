"""Tests for hpca.dbcache: node-local sqlite working copies.

The app's databases live in an NFS home on a cluster node, where every sqlite
call is a network round-trip. DbCache keeps the working copies on node-local
storage and syncs them back to home. See specs/specs-db-local-cache.md.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import struct
import subprocess
import sys

import pytest

from hpca import dbcache
from hpca.dbcache import (
    DB_NAMES,
    DbCache,
    copy_database,
    fingerprint,
    local_dir_for,
    local_root,
)


def make_db(path, rows=("a",)):
    """A small WAL database with a ``notes`` table holding ``rows``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS notes (text TEXT)")
    conn.executemany("INSERT INTO notes (text) VALUES (?)", [(r,) for r in rows])
    conn.commit()
    conn.close()


def read_notes(path):
    conn = sqlite3.connect(path)
    try:
        return [row[0] for row in conn.execute("SELECT text FROM notes")]
    finally:
        conn.close()


def make_churned_db(path, rows=2000):
    """A database most of whose pages are free: filled, then emptied.

    What checkpoints.db looks like after a few sessions — sqlite keeps the
    pages a delete released on the free list and never shrinks the file.
    """
    make_db(path, rows=("kept",))
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO notes (text) VALUES (?)", [("x" * 400,)] * rows
    )
    conn.commit()
    conn.execute("DELETE FROM notes WHERE text LIKE 'x%'")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()


def free_pages(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA freelist_count").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def forget_compact_memo():
    """``_COMPACT_UNSUPPORTED`` is module state keyed by path; tmp_path makes
    every test's paths unique, but a shared set still leaks across a run."""
    dbcache._COMPACT_UNSUPPORTED.clear()
    yield
    dbcache._COMPACT_UNSUPPORTED.clear()


class TestLocalRoot:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_LOCAL_DIR", str(tmp_path / "scratch"))
        monkeypatch.setenv("TMPDIR", str(tmp_path / "ignored"))
        assert local_root() == tmp_path / "scratch"

    def test_falls_back_to_the_system_temp_dir(self, monkeypatch, tmp_path):
        # $TMPDIR is what Slurm sets per job, and gettempdir() honours it.
        monkeypatch.delenv("HPCA_LOCAL_DIR", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path / "jobtmp"))
        assert local_root() == tmp_path / "jobtmp"

    def test_configured_override_beats_the_temp_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HPCA_LOCAL_DIR", raising=False)
        monkeypatch.setenv("TMPDIR", str(tmp_path / "jobtmp"))
        assert local_root(configured=str(tmp_path / "cfg")) == tmp_path / "cfg"

    def test_env_override_beats_the_configured_one(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_LOCAL_DIR", str(tmp_path / "env"))
        assert local_root(configured=str(tmp_path / "cfg")) == tmp_path / "env"

    def test_user_home_is_expanded(self, monkeypatch):
        monkeypatch.setenv("HPCA_LOCAL_DIR", "~/scratch")
        assert str(local_root()) == os.path.expanduser("~/scratch")


class TestLocalDirFor:
    def test_is_keyed_to_the_app_dir(self, tmp_path):
        one = local_dir_for(tmp_path / "homeA", root=tmp_path / "tmp")
        two = local_dir_for(tmp_path / "homeB", root=tmp_path / "tmp")
        assert one != two

    def test_is_stable_for_one_app_dir(self, tmp_path):
        # Crash recovery depends on the next run finding the same directory.
        one = local_dir_for(tmp_path / "home", root=tmp_path / "tmp")
        two = local_dir_for(tmp_path / "home", root=tmp_path / "tmp")
        assert one == two

    def test_lives_under_the_root(self, tmp_path):
        local = local_dir_for(tmp_path / "home", root=tmp_path / "tmp")
        assert local.parent == tmp_path / "tmp"

    def test_name_carries_the_uid_so_users_do_not_collide_in_shared_tmp(self, tmp_path):
        local = local_dir_for(tmp_path / "home", root=tmp_path / "tmp")
        assert str(os.getuid()) in local.name


class TestCopyDatabase:
    def test_copies_committed_rows(self, tmp_path):
        make_db(tmp_path / "src.db", rows=("a", "b"))
        copy_database(tmp_path / "src.db", tmp_path / "dst.db")
        assert read_notes(tmp_path / "dst.db") == ["a", "b"]

    def test_copies_while_a_writer_holds_the_source_open(self, tmp_path):
        # The live checkpointer and RagStore connections are never quiesced.
        src = tmp_path / "src.db"
        make_db(src, rows=("a",))
        holder = sqlite3.connect(src)
        holder.execute("INSERT INTO notes (text) VALUES ('b')")
        holder.commit()
        try:
            copy_database(src, tmp_path / "dst.db")
        finally:
            holder.close()
        assert read_notes(tmp_path / "dst.db") == ["a", "b"]

    def test_overwrites_an_existing_destination_completely(self, tmp_path):
        make_db(tmp_path / "src.db", rows=("new",))
        make_db(tmp_path / "dst.db", rows=("old", "older", "oldest"))
        copy_database(tmp_path / "src.db", tmp_path / "dst.db")
        assert read_notes(tmp_path / "dst.db") == ["new"]

    def test_creates_missing_parent_directories(self, tmp_path):
        make_db(tmp_path / "src.db")
        copy_database(tmp_path / "src.db", tmp_path / "deep" / "nested" / "dst.db")
        assert (tmp_path / "deep" / "nested" / "dst.db").exists()

    def test_leaves_no_wal_sidecar_beside_the_destination(self, tmp_path):
        # Home should hold one standalone file: another node may read it.
        make_db(tmp_path / "src.db")
        copy_database(tmp_path / "src.db", tmp_path / "dst.db")
        assert not (tmp_path / "dst.db-wal").exists()

    def test_a_failing_copy_leaves_the_destination_untouched(self, tmp_path):
        # The copy lands in a temp file renamed over the destination only
        # once complete — an interrupted sync must not tear the only other
        # copy (seen in the field as a home file with its first page zeroed).
        make_db(tmp_path / "dst.db", rows=("old",))
        (tmp_path / "src.db").write_bytes(b"this is not a database")

        with pytest.raises(sqlite3.DatabaseError):
            copy_database(tmp_path / "src.db", tmp_path / "dst.db")

        assert read_notes(tmp_path / "dst.db") == ["old"]
        assert not list(tmp_path.glob("*.backup-tmp"))

    def test_a_leftover_temp_file_does_not_block_the_copy(self, tmp_path):
        # A hard kill mid-copy leaves the temp file behind; the next copy
        # must replace it, not trip over it.
        make_db(tmp_path / "src.db", rows=("a",))
        (tmp_path / "dst.db.backup-tmp").write_bytes(b"half-written garbage")
        copy_database(tmp_path / "src.db", tmp_path / "dst.db")
        assert read_notes(tmp_path / "dst.db") == ["a"]

    def test_a_corrupt_destination_is_quarantined_not_buried(self, tmp_path):
        make_db(tmp_path / "src.db", rows=("fresh",))
        (tmp_path / "dst.db").write_bytes(b"\x00" * 4096 + b"salvageable")

        aside = copy_database(tmp_path / "src.db", tmp_path / "dst.db")

        assert aside is not None and aside.name.startswith("dst.db.corrupt-")
        assert aside.read_bytes() == b"\x00" * 4096 + b"salvageable"
        assert read_notes(tmp_path / "dst.db") == ["fresh"]

    def test_an_intact_destination_is_not_quarantined(self, tmp_path):
        make_db(tmp_path / "src.db", rows=("new",))
        make_db(tmp_path / "dst.db", rows=("old",))
        assert copy_database(tmp_path / "src.db", tmp_path / "dst.db") is None
        assert not list(tmp_path.glob("*.corrupt-*"))

    def test_an_empty_destination_is_not_condemned(self, tmp_path):
        # sqlite treats a zero-byte file as a database with no tables yet;
        # only a non-empty file without the magic is evidence of corruption.
        make_db(tmp_path / "src.db", rows=("a",))
        (tmp_path / "dst.db").write_bytes(b"")
        assert copy_database(tmp_path / "src.db", tmp_path / "dst.db") is None
        assert not list(tmp_path.glob("*.corrupt-*"))

    def test_copies_sqlite_vec_virtual_tables(self, tmp_path):
        # rag.db holds vec0 tables; the backup API is page-level, so it does
        # not need the extension loaded on either connection.
        sqlite_vec = pytest.importorskip("sqlite_vec")
        src = tmp_path / "rag.db"
        conn = sqlite3.connect(src)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE vecs USING vec0(embedding float[4])")
        conn.execute(
            "INSERT INTO vecs (rowid, embedding) VALUES (1, ?)",
            (struct.pack("4f", 1.0, 0.0, 0.0, 0.0),),
        )
        conn.commit()
        conn.close()

        copy_database(src, tmp_path / "copy.db")

        check = sqlite3.connect(tmp_path / "copy.db")
        check.enable_load_extension(True)
        sqlite_vec.load(check)
        check.enable_load_extension(False)
        hits = check.execute(
            "SELECT rowid FROM vecs WHERE embedding MATCH ? AND k = 1",
            (struct.pack("4f", 1.0, 0.0, 0.0, 0.0),),
        ).fetchall()
        check.close()
        assert [row[0] for row in hits] == [1]


@pytest.fixture
def home(tmp_path):
    d = tmp_path / "home"
    d.mkdir()
    return d


@pytest.fixture
def local(tmp_path):
    return tmp_path / "node-local"


class TestAcquire:
    def test_reports_active_and_redirects_paths(self, home, local):
        cache = DbCache(home, local_dir=local)
        assert cache.acquire() is True
        assert cache.active
        assert cache.path_for("hpca.db") == local / "hpca.db"

    def test_seeds_existing_databases_from_home(self, home, local):
        make_db(home / "hpca.db", rows=("kept",))
        DbCache(home, local_dir=local).acquire()
        assert read_notes(local / "hpca.db") == ["kept"]

    def test_does_not_invent_databases_that_home_does_not_have(self, home, local):
        DbCache(home, local_dir=local).acquire()
        assert not (local / "hpca.db").exists()

    def test_seeds_every_managed_database(self, home, local):
        for name in DB_NAMES:
            make_db(home / name, rows=(name,))
        DbCache(home, local_dir=local).acquire()
        for name in DB_NAMES:
            assert read_notes(local / name) == [name]

    def test_the_working_dir_is_private_to_this_user(self, home, local):
        # /tmp is shared on a cluster node, and these databases hold the whole
        # conversation history.
        DbCache(home, local_dir=local).acquire()
        assert stat.S_IMODE(local.stat().st_mode) == 0o700

    def test_an_inherited_working_dir_is_locked_down_too(self, home, local):
        # A crash left it behind; it may have been created world-readable by
        # an older version, or by a umask that allowed it.
        local.mkdir(parents=True)
        local.chmod(0o755)
        DbCache(home, local_dir=local).acquire()
        assert stat.S_IMODE(local.stat().st_mode) == 0o700

    def test_inactive_cache_hands_back_the_home_paths(self, home, local):
        cache = DbCache(home, local_dir=local, enabled=False)
        assert cache.acquire() is False
        assert not cache.active
        assert cache.path_for("hpca.db") == home / "hpca.db"

    def test_disabled_cache_writes_no_lease_and_no_local_dir(self, home, local):
        DbCache(home, local_dir=local, enabled=False).acquire()
        assert not (home / "db.lease").exists()
        assert not local.exists()

    def test_refuses_a_local_dir_that_is_the_app_dir(self, home):
        # Copying a file onto itself must never be attempted.
        cache = DbCache(home, local_dir=home)
        assert cache.acquire() is False
        assert "same" in cache.reason.lower()

    def test_falls_back_when_the_local_dir_cannot_be_created(self, home, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        cache = DbCache(home, local_dir=blocker / "sub")
        assert cache.acquire() is False
        assert cache.path_for("hpca.db") == home / "hpca.db"
        assert cache.reason

    def test_unknown_database_name_is_rejected(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        with pytest.raises(KeyError):
            cache.path_for("secrets.db")


class TestRecovery:
    def test_recovers_local_databases_a_crash_left_behind(self, home, local):
        # A previous run died before syncing: local is newer than home.
        make_db(home / "hpca.db", rows=("old",))
        make_db(local / "hpca.db", rows=("old", "unsynced"))

        DbCache(home, local_dir=local).acquire()

        assert read_notes(home / "hpca.db") == ["old", "unsynced"]

    def test_recovered_state_is_what_the_new_run_starts_from(self, home, local):
        make_db(home / "hpca.db", rows=("old",))
        make_db(local / "hpca.db", rows=("old", "unsynced"))

        cache = DbCache(home, local_dir=local)
        cache.acquire()

        assert read_notes(cache.path_for("hpca.db")) == ["old", "unsynced"]

    def test_recovers_a_database_home_never_had(self, home, local):
        make_db(local / "rag.db", rows=("only-local",))
        DbCache(home, local_dir=local).acquire()
        assert read_notes(home / "rag.db") == ["only-local"]


class TestSync:
    def test_writes_local_changes_back_to_home(self, home, local):
        make_db(home / "hpca.db", rows=("a",))
        cache = DbCache(home, local_dir=local)
        cache.acquire()

        conn = sqlite3.connect(cache.path_for("hpca.db"))
        conn.execute("INSERT INTO notes (text) VALUES ('b')")
        conn.commit()
        conn.close()
        cache.sync()

        assert read_notes(home / "hpca.db") == ["a", "b"]

    def test_creates_a_database_home_does_not_have_yet(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"), rows=("first-run",))
        cache.sync()
        assert read_notes(home / "hpca.db") == ["first-run"]

    def test_is_a_no_op_when_inactive(self, home, local):
        make_db(home / "hpca.db", rows=("untouched",))
        cache = DbCache(home, local_dir=local, enabled=False)
        cache.acquire()
        cache.sync()
        assert read_notes(home / "hpca.db") == ["untouched"]

    def test_one_unreadable_database_does_not_stop_the_others(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        (local / "hpca.db").write_bytes(b"this is not a database")
        make_db(local / "rag.db", rows=("fine",))

        cache.sync()  # must not raise

        assert read_notes(home / "rag.db") == ["fine"]

    def test_reports_success(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"))
        assert cache.sync() is True

    def test_reports_failure_without_raising(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        (local / "hpca.db").write_bytes(b"this is not a database")
        assert cache.sync() is False

    def test_an_unwritable_lease_does_not_raise(self, home, local):
        # Home briefly unreachable must not take the app down.
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        (home / "db.lease").unlink()
        (home / "db.lease").mkdir()  # a directory: the write cannot succeed
        assert cache.sync() is False

    def test_refreshes_the_lease_heartbeat(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        before = json.loads((home / "db.lease").read_text())["heartbeat"]
        cache.sync()
        after = json.loads((home / "db.lease").read_text())["heartbeat"]
        assert after >= before


class TestRelease:
    def test_syncs_before_letting_go(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"), rows=("final",))
        cache.release()
        assert read_notes(home / "hpca.db") == ["final"]

    def test_removes_the_working_dir_so_the_next_start_skips_recovery(
        self, home, local
    ):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"))
        cache.release()
        assert not local.exists()

    def test_drops_the_lease(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        cache.release()
        assert not (home / "db.lease").exists()

    def test_leaves_paths_pointing_at_home_afterwards(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        cache.release()
        assert not cache.active
        assert cache.path_for("hpca.db") == home / "hpca.db"

    def test_keeps_the_working_dir_when_the_final_sync_fails(self, home, local):
        # Deleting it would destroy the only surviving copy; leaving it lets
        # the next start recover, exactly as after a crash.
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(local / "hpca.db", rows=("only-copy",))
        (local / "checkpoints.db").write_bytes(b"this is not a database")

        cache.release()

        assert read_notes(local / "hpca.db") == ["only-copy"]

    def test_a_failing_sync_does_not_stop_the_release(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()

        def boom(**kwargs):
            raise OSError("home unreachable")

        cache.sync = boom
        cache.release()

        assert not cache.active

    def test_is_idempotent(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        cache.release()
        cache.release()

    def test_is_a_no_op_when_never_acquired(self, home, local):
        DbCache(home, local_dir=local).release()
        assert not local.exists()


class TestLease:
    def test_records_this_process(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        lease = json.loads((home / "db.lease").read_text())
        assert lease["pid"] == os.getpid()
        assert lease["local_dir"] == str(local)

    def test_a_second_instance_falls_back_to_home(self, home, local, tmp_path):
        first = DbCache(home, local_dir=local)
        first.acquire()

        second = DbCache(home, local_dir=tmp_path / "other-local")
        assert second.acquire() is False
        assert second.path_for("hpca.db") == home / "hpca.db"
        assert "another" in second.reason.lower()

    def test_the_second_instance_does_not_touch_the_first_working_dir(
        self, home, local, tmp_path
    ):
        make_db(home / "hpca.db", rows=("a",))
        first = DbCache(home, local_dir=local)
        first.acquire()
        conn = sqlite3.connect(first.path_for("hpca.db"))
        conn.execute("INSERT INTO notes (text) VALUES ('from-first')")
        conn.commit()
        conn.close()

        DbCache(home, local_dir=tmp_path / "other").acquire()
        first.sync()

        assert read_notes(home / "hpca.db") == ["a", "from-first"]

    def test_a_fallback_instance_leaves_the_lease_alone_on_release(
        self, home, local, tmp_path
    ):
        first = DbCache(home, local_dir=local)
        first.acquire()
        second = DbCache(home, local_dir=tmp_path / "other")
        second.acquire()

        second.release()

        lease = json.loads((home / "db.lease").read_text())
        assert lease["local_dir"] == str(local)

    def test_a_dead_pid_on_this_host_is_taken_over_at_once(self, home, local):
        (home / "db.lease").write_text(
            json.dumps(
                {
                    "host": DbCache(home, local_dir=local).host,
                    "pid": _dead_pid(),
                    "started_at": 0.0,
                    "heartbeat": 0.0,
                    "local_dir": str(local),
                }
            )
        )
        assert DbCache(home, local_dir=local).acquire() is True

    def test_a_stale_lease_from_another_host_is_taken_over(self, home, local):
        (home / "db.lease").write_text(
            json.dumps(
                {
                    "host": "some-other-node",
                    "pid": 1,
                    "started_at": 0.0,
                    "heartbeat": 0.0,  # 1970: far older than stale_after_s
                    "local_dir": "/tmp/elsewhere",
                }
            )
        )
        assert DbCache(home, local_dir=local).acquire() is True

    def test_a_fresh_lease_from_another_host_is_respected(self, home, local):
        import time

        (home / "db.lease").write_text(
            json.dumps(
                {
                    "host": "some-other-node",
                    "pid": 1,
                    "started_at": time.time(),
                    "heartbeat": time.time(),
                    "local_dir": "/tmp/elsewhere",
                }
            )
        )
        assert DbCache(home, local_dir=local).acquire() is False

    def test_a_corrupt_lease_is_ignored(self, home, local):
        (home / "db.lease").write_text("{not json")
        assert DbCache(home, local_dir=local).acquire() is True


def _dead_pid() -> int:
    """A pid that is certainly not running: run a child and reap it.

    Not os.fork(): forking a multi-threaded pytest process is deprecated and
    can deadlock in the child.
    """
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait()
    return child.pid


class TestRoundTrip:
    def test_two_sequential_runs_accumulate(self, home, local):
        first = DbCache(home, local_dir=local)
        first.acquire()
        make_db(first.path_for("hpca.db"), rows=("run1",))
        first.release()

        second = DbCache(home, local_dir=local)
        second.acquire()
        conn = sqlite3.connect(second.path_for("hpca.db"))
        conn.execute("INSERT INTO notes (text) VALUES ('run2')")
        conn.commit()
        conn.close()
        second.release()

        assert read_notes(home / "hpca.db") == ["run1", "run2"]

    def test_a_crashed_run_loses_nothing_the_next_start_can_see(self, home, local):
        crashed = DbCache(home, local_dir=local)
        crashed.acquire()
        make_db(crashed.path_for("hpca.db"), rows=("written-then-killed",))
        # No release(): the process was killed. Its lease and working dir stay,
        # and the lease goes stale as soon as the pid is gone.
        _orphan_the_lease(home)

        restarted = DbCache(home, local_dir=local)
        assert restarted.acquire() is True
        assert read_notes(home / "hpca.db") == ["written-then-killed"]
        assert read_notes(restarted.path_for("hpca.db")) == ["written-then-killed"]


def _orphan_the_lease(home) -> None:
    """Repoint the lease at a dead pid, as a killed process would leave it."""
    lease = json.loads((home / "db.lease").read_text())
    lease["pid"] = _dead_pid()
    (home / "db.lease").write_text(json.dumps(lease))


def _torn(payload: bytes = b"leftover page data" * 200) -> bytes:
    """What the field corruption looked like: exactly the first 4096-byte
    page zeroed, every later page still carrying data."""
    return b"\x00" * 4096 + payload


class TestCorruptHomeCopy:
    """A torn write on the network filesystem can destroy home's copy while
    the local one stays good. That file used to block seed, recovery and
    every sync-back at once ("file is not a database"); now it is moved
    aside and life goes on."""

    def test_recovery_replaces_it_with_the_surviving_local_copy(
        self, home, local
    ):
        (home / "hpca.db").write_bytes(_torn())
        make_db(local / "hpca.db", rows=("unsynced",))

        cache = DbCache(home, local_dir=local)
        assert cache.acquire() is True

        assert read_notes(home / "hpca.db") == ["unsynced"]
        assert list(home.glob("hpca.db.corrupt-*"))

    def test_sync_back_replaces_it_with_the_local_copy(self, home, local):
        make_db(home / "hpca.db", rows=("a",))
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        (home / "hpca.db").write_bytes(_torn())  # torn under a running app

        conn = sqlite3.connect(cache.path_for("hpca.db"))
        conn.execute("INSERT INTO notes (text) VALUES ('b')")
        conn.commit()
        conn.close()

        assert cache.sync() is True
        assert read_notes(home / "hpca.db") == ["a", "b"]

    def test_seed_moves_it_aside_and_starts_afresh(self, home, local):
        # No local copy survives to prefer, so there is nothing to seed
        # from: the app starts this database empty, and the torn file is
        # kept for offline recovery instead of blocking every later sync.
        (home / "hpca.db").write_bytes(_torn())

        cache = DbCache(home, local_dir=local)
        assert cache.acquire() is True

        assert not (local / "hpca.db").exists()
        assert not (home / "hpca.db").exists()
        assert list(home.glob("hpca.db.corrupt-*"))

    def test_the_torn_bytes_are_kept_for_offline_recovery(self, home, local):
        (home / "hpca.db").write_bytes(_torn(b"the old sessions"))
        make_db(local / "hpca.db", rows=("unsynced",))

        DbCache(home, local_dir=local).acquire()

        (aside,) = home.glob("hpca.db.corrupt-*")
        assert aside.read_bytes() == _torn(b"the old sessions")

    def test_the_user_is_told_once(self, home, local):
        (home / "hpca.db").write_bytes(_torn())
        make_db(local / "hpca.db", rows=("unsynced",))

        cache = DbCache(home, local_dir=local)
        cache.acquire()

        warnings = cache.drain_warnings()
        assert any("hpca.db" in w and "corrupt" in w for w in warnings)
        assert cache.drain_warnings() == []


class TestCompactCopy:
    """``compact=True``: rebuild rather than duplicate. See COMPACT_DB_NAMES."""

    def test_a_rebuilt_copy_drops_the_free_list(self, tmp_path):
        src = tmp_path / "checkpoints.db"
        make_churned_db(src)
        assert free_pages(src) > 0  # the state the rebuild exists to fix

        copy_database(src, tmp_path / "dst.db", compact=True)

        assert free_pages(tmp_path / "dst.db") == 0
        assert (tmp_path / "dst.db").stat().st_size < src.stat().st_size

    def test_a_rebuilt_copy_keeps_every_row(self, tmp_path):
        make_db(tmp_path / "src.db", rows=("a", "b", "c"))
        copy_database(tmp_path / "src.db", tmp_path / "dst.db", compact=True)
        assert read_notes(tmp_path / "dst.db") == ["a", "b", "c"]

    def test_a_rebuilt_copy_has_no_wal_sidecar_either(self, tmp_path):
        make_db(tmp_path / "src.db")
        copy_database(tmp_path / "src.db", tmp_path / "dst.db", compact=True)
        assert not (tmp_path / "dst.db-wal").exists()

    def test_a_page_copy_still_carries_the_free_list(self, tmp_path):
        # The default, and what rag.db keeps getting: page for page.
        src = tmp_path / "rag.db"
        make_churned_db(src)
        copy_database(src, tmp_path / "dst.db")
        assert free_pages(tmp_path / "dst.db") == free_pages(src)

    def test_a_rebuild_carries_vec0_tables_across(self, tmp_path):
        # Not the reason rag.db stays on the page copy: VACUUM copies a
        # virtual table's shadow tables and its schema row without ever
        # instantiating the module, so a rebuild would be correct here — it
        # would just be expensive work for a file with no free pages.
        sqlite_vec = pytest.importorskip("sqlite_vec")
        src = tmp_path / "rag.db"
        conn = sqlite3.connect(src)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE vecs USING vec0(embedding float[4])")
        conn.execute(
            "INSERT INTO vecs (rowid, embedding) VALUES (1, ?)",
            (struct.pack("4f", 1.0, 0.0, 0.0, 0.0),),
        )
        conn.commit()
        conn.close()

        copy_database(src, tmp_path / "dst.db", compact=True)

        check = sqlite3.connect(tmp_path / "dst.db")
        check.enable_load_extension(True)
        sqlite_vec.load(check)
        check.enable_load_extension(False)
        hits = check.execute(
            "SELECT rowid FROM vecs WHERE embedding MATCH ? AND k = 1",
            (struct.pack("4f", 1.0, 0.0, 0.0, 0.0),),
        ).fetchall()
        check.close()
        assert [row[0] for row in hits] == [1]

    def test_a_failing_rebuild_falls_back_to_the_page_copy(
        self, tmp_path, monkeypatch
    ):
        def refuse(source, tmp):
            raise sqlite3.OperationalError("no")

        monkeypatch.setattr(dbcache, "_vacuum_into", refuse)
        make_db(tmp_path / "src.db", rows=("kept",))

        copy_database(tmp_path / "src.db", tmp_path / "dst.db", compact=True)

        assert read_notes(tmp_path / "dst.db") == ["kept"]

    def test_a_failing_rebuild_is_not_retried_every_sync(
        self, tmp_path, monkeypatch
    ):
        attempts = []

        def refuse(source, tmp):
            attempts.append(tmp)
            raise sqlite3.OperationalError("no")

        monkeypatch.setattr(dbcache, "_vacuum_into", refuse)
        make_db(tmp_path / "src.db")

        copy_database(tmp_path / "src.db", tmp_path / "dst.db", compact=True)
        copy_database(tmp_path / "src.db", tmp_path / "dst.db", compact=True)

        assert len(attempts) == 1
        assert str(tmp_path / "src.db") in dbcache._COMPACT_UNSUPPORTED

    def test_a_corrupt_source_still_raises(self, tmp_path):
        (tmp_path / "src.db").write_bytes(b"this is not a database")
        with pytest.raises(sqlite3.DatabaseError):
            copy_database(tmp_path / "src.db", tmp_path / "dst.db", compact=True)
        assert not (tmp_path / "dst.db").exists()
        assert not (tmp_path / "dst.db.backup-tmp").exists()


class TestFingerprint:
    def test_a_write_changes_it(self, tmp_path):
        make_db(tmp_path / "a.db")
        before = fingerprint(tmp_path / "a.db")
        conn = sqlite3.connect(tmp_path / "a.db")
        conn.execute("INSERT INTO notes (text) VALUES ('b')")
        conn.commit()
        conn.close()
        assert fingerprint(tmp_path / "a.db") != before

    def test_reading_does_not(self, tmp_path):
        make_db(tmp_path / "a.db")
        before = fingerprint(tmp_path / "a.db")
        assert read_notes(tmp_path / "a.db") == ["a"]
        assert fingerprint(tmp_path / "a.db") == before

    def test_a_missing_database_has_one_too(self, tmp_path):
        assert fingerprint(tmp_path / "nothing.db") == (None, None)

    def test_a_wal_sidecar_appearing_is_a_change(self, tmp_path):
        make_db(tmp_path / "a.db")
        before = fingerprint(tmp_path / "a.db")
        (tmp_path / "a.db-wal").write_bytes(b"")
        assert fingerprint(tmp_path / "a.db") != before


class TestSyncSkipsUnchanged:
    def test_an_unchanged_database_is_not_copied_again(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"), rows=("synced",))
        cache.sync()
        assert read_notes(home / "hpca.db") == ["synced"]

        # Behind the cache's back: a second sync that actually copied would
        # overwrite this, so its survival is the proof the copy was skipped.
        (home / "hpca.db").unlink()
        make_db(home / "hpca.db", rows=("untouched-by-the-second-sync",))
        assert cache.sync() is True

        assert read_notes(home / "hpca.db") == ["untouched-by-the-second-sync"]

    def test_a_changed_database_is_synced_again(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"), rows=("first",))
        cache.sync()

        conn = sqlite3.connect(cache.path_for("hpca.db"))
        conn.execute("INSERT INTO notes (text) VALUES ('second')")
        conn.commit()
        conn.close()
        cache.sync()

        assert read_notes(home / "hpca.db") == ["first", "second"]

    def test_the_first_sync_of_a_run_copies_even_with_nothing_written(
        self, home, local
    ):
        # What replaces a home copy that is mostly free list with a compacted
        # one on a run that never happens to write to it.
        make_churned_db(home / "checkpoints.db")
        before = (home / "checkpoints.db").stat().st_size
        cache = DbCache(home, local_dir=local)
        cache.acquire()

        cache.sync()

        assert (home / "checkpoints.db").stat().st_size < before
        assert read_notes(home / "checkpoints.db") == ["kept"]

    def test_a_skipped_database_still_refreshes_the_lease(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"))
        cache.sync()
        first = json.loads((home / "db.lease").read_text())["heartbeat"]

        cache.sync()

        assert json.loads((home / "db.lease").read_text())["heartbeat"] >= first

    def test_a_failed_sync_is_retried_on_the_next_tick(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        (local / "hpca.db").write_bytes(b"this is not a database")
        assert cache.sync() is False

        (local / "hpca.db").unlink()
        make_db(local / "hpca.db", rows=("repaired",))
        assert cache.sync() is True
        assert read_notes(home / "hpca.db") == ["repaired"]

    def test_force_copies_regardless(self, home, local):
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"), rows=("local",))
        cache.sync()
        (home / "hpca.db").unlink()
        make_db(home / "hpca.db", rows=("stale",))

        cache.sync(force=True)

        assert read_notes(home / "hpca.db") == ["local"]

    def test_release_syncs_even_when_nothing_changed(self, home, local):
        # The working dir is about to be deleted, so the final sync has no
        # surviving copy to fall back on and must never skip.
        cache = DbCache(home, local_dir=local)
        cache.acquire()
        make_db(cache.path_for("hpca.db"), rows=("final",))
        cache.sync()
        (home / "hpca.db").unlink()
        make_db(home / "hpca.db", rows=("stale",))

        cache.release()

        assert read_notes(home / "hpca.db") == ["final"]


class TestSeedCompacts:
    def test_seeding_leaves_the_free_list_behind(self, home, local):
        make_churned_db(home / "checkpoints.db")
        DbCache(home, local_dir=local).acquire()
        assert free_pages(local / "checkpoints.db") == 0
        assert read_notes(local / "checkpoints.db") == ["kept"]

    def test_rag_db_is_seeded_page_for_page(self, home, local):
        # Not in COMPACT_DB_NAMES: its vec0 tables need the extension loaded
        # before a rebuild could re-create them.
        make_churned_db(home / "rag.db")
        DbCache(home, local_dir=local).acquire()
        assert free_pages(local / "rag.db") == free_pages(home / "rag.db")

    def test_recovery_compacts_too(self, home, local):
        local.mkdir(parents=True)
        make_churned_db(local / "checkpoints.db")
        before = (local / "checkpoints.db").stat().st_size

        DbCache(home, local_dir=local).acquire()

        assert (home / "checkpoints.db").stat().st_size < before
        assert read_notes(home / "checkpoints.db") == ["kept"]
