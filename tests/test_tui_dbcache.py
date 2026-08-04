"""The app runs its databases from node-local storage and syncs home.

On a cluster node ``$HOME`` is NFS and every sqlite call is a network
round-trip, which is what makes the TUI lag while the agent works. The app
opens hpca.db, checkpoints.db and rag.db from a node-local working dir
instead, syncs them back on a timer, and syncs-and-cleans-up on exit.
See specs-db-local-cache.md and hpca.dbcache.
"""

import json
import logging
import sqlite3
import subprocess
import sys
import time

import pytest

from hpca.config import Settings
from hpca.dbcache import DB_NAMES
from hpca.llm import ChatResponse
from hpca.tui.app import HpcaApp


class FakeLLM:
    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content="")

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_LOCAL_DIR", str(tmp_path / "node-local"))
    return tmp_path / "node-local"


def db_file(conn) -> str:
    """The file behind an open sqlite connection."""
    return conn.execute("PRAGMA database_list").fetchone()[2]


def working_dir(app):
    return app._dbcache.path_for("hpca.db").parent


class TestPlacement:
    async def test_app_tables_open_from_the_working_dir(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert app._dbcache.active
            assert db_file(app._conn).startswith(str(local))

    async def test_the_working_dir_is_not_home(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert working_dir(app) != home

    async def test_checkpoints_live_there_too(self, home, local):
        # The checkpointer writes on every graph step: the heaviest writer.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert app._dbcache.path_for("checkpoints.db").parent == working_dir(app)
            assert app._dbcache.path_for("checkpoints.db").exists()

    async def test_rag_store_lives_there_too(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert db_file(app.rag_store._conn).startswith(str(local))

    async def test_home_holds_no_live_databases_while_running(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert not (home / "hpca.db").exists()

    async def test_the_rest_of_the_app_dir_stays_in_home(self, home, local):
        # Only the databases move; profiles, logs and settings do not.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert not any(
                p.name in ("profiles", "chatlogs", "settings.json")
                for p in working_dir(app).iterdir()
            )


class TestExit:
    async def test_databases_are_written_back_to_home(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        assert (home / "hpca.db").exists()
        assert (home / "checkpoints.db").exists()

    async def test_a_session_created_in_one_run_is_in_home_afterwards(
        self, home, local
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await app.start_new_session()
            await pilot.pause()
        conn = sqlite3.connect(home / "hpca.db")
        count = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
        conn.close()
        assert count >= 1

    async def test_the_working_dir_is_removed(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            path = working_dir(app)
        assert not path.exists()

    async def test_the_lease_is_dropped(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert (home / "db.lease").exists()
        assert not (home / "db.lease").exists()

    async def test_a_second_run_sees_the_first_run_sessions(self, home, local):
        first = HpcaApp(llm=FakeLLM())
        async with first.run_test(size=(120, 30)) as pilot:
            await first.start_new_session()
            await pilot.pause()
            created = first.active_session.session_id

        second = HpcaApp(llm=FakeLLM())
        async with second.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            ids = [s.session_id for s in second.session_store.list(profile="default")]
        assert created in ids


class TestSync:
    async def test_sync_writes_the_current_state_to_home(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await app._sync_db_cache()
            conn = sqlite3.connect(home / "hpca.db")
            count = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
            conn.close()
        assert count >= 1

    async def test_a_sync_timer_is_registered(self, home, local, monkeypatch):
        seen = []
        original = HpcaApp.set_interval

        def record(self, interval, callback=None, **kwargs):
            seen.append((interval, getattr(callback, "__name__", "")))
            return original(self, interval, callback, **kwargs)

        monkeypatch.setattr(HpcaApp, "set_interval", record)
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        assert (60, "_sync_db_cache") in seen

    async def test_the_interval_comes_from_settings(self, home, local, monkeypatch):
        seen = []
        original = HpcaApp.set_interval

        def record(self, interval, callback=None, **kwargs):
            seen.append((interval, getattr(callback, "__name__", "")))
            return original(self, interval, callback, **kwargs)

        monkeypatch.setattr(HpcaApp, "set_interval", record)
        settings = Settings()
        settings.database.sync_interval_s = 5
        app = HpcaApp(settings, llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        assert (5, "_sync_db_cache") in seen

    async def test_interval_zero_means_sync_on_exit_only(
        self, home, local, monkeypatch
    ):
        seen = []
        original = HpcaApp.set_interval

        def record(self, interval, callback=None, **kwargs):
            seen.append(getattr(callback, "__name__", ""))
            return original(self, interval, callback, **kwargs)

        monkeypatch.setattr(HpcaApp, "set_interval", record)
        settings = Settings()
        settings.database.sync_interval_s = 0
        app = HpcaApp(settings, llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        assert "_sync_db_cache" not in seen
        assert (home / "hpca.db").exists()  # exit still syncs

    async def test_overlapping_syncs_are_skipped(self, home, local):
        # An NFS sync can outlast its tick; two at once must not interleave.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            calls = []
            app._dbcache.sync = lambda: calls.append(1)
            app._syncing_db_cache = True
            await app._sync_db_cache()
            assert calls == []

    async def test_sync_failure_is_reported_but_not_fatal(self, home, local):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            def boom():
                raise OSError("home unreachable")

            app._dbcache.sync = boom
            await app._sync_db_cache()
            await pilot.pause()
            assert app.is_running


@pytest.fixture
def fresh_dbcache_logger():
    """Detach the process-global handler so it re-binds to this test's app dir.

    The handler is attached on first use and holds the app dir resolved then —
    one per process, which is right for the app and wrong across tests.
    """
    logger = logging.getLogger("hpca.dbcache")
    saved = list(logger.handlers)
    logger.handlers.clear()
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:] = saved


class TestLogging:
    async def test_dbcache_logging_never_reaches_the_terminal(
        self, home, local, fresh_dbcache_logger
    ):
        # logging's last-resort handler writes WARNING+ to stderr, and a sync
        # failure logs at ERROR — straight through the TUI's display.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert fresh_dbcache_logger.propagate is False
            assert any(
                isinstance(h, logging.FileHandler)
                for h in fresh_dbcache_logger.handlers
            )

    async def test_it_leaves_its_trail_in_the_app_dir(
        self, home, local, fresh_dbcache_logger
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            fresh_dbcache_logger.error("probe")
        assert "probe" in (home / "dbcache.log").read_text()


class TestFallback:
    async def test_disabled_setting_keeps_the_databases_in_home(self, home, local):
        settings = Settings()
        settings.database.local_cache = False
        app = HpcaApp(settings, llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert not app._dbcache.active
            assert db_file(app._conn) == str(home / "hpca.db")
            assert not local.exists()

    async def test_a_second_instance_runs_from_home(self, home, local, tmp_path):
        # A live lease from another instance: private copies would erase each
        # other's sessions on sync, so this one must share the home file.
        (home / "db.lease").write_text(
            json.dumps(
                {
                    "host": "another-node",
                    "pid": 1,
                    "started_at": time.time(),
                    "heartbeat": time.time(),
                    "local_dir": str(tmp_path / "theirs"),
                }
            )
        )
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert not app._dbcache.active
            assert db_file(app._conn) == str(home / "hpca.db")

    async def test_the_user_is_told_why(self, home, local, tmp_path):
        (home / "db.lease").write_text(
            json.dumps(
                {
                    "host": "another-node",
                    "pid": 1,
                    "started_at": time.time(),
                    "heartbeat": time.time(),
                    "local_dir": str(tmp_path / "theirs"),
                }
            )
        )
        messages = []
        app = HpcaApp(llm=FakeLLM())
        original = HpcaApp.notify
        app.notify = lambda message, **kw: (
            messages.append(message),
            original(app, message, **kw),
        )[1]
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
        assert any("another HPCA instance" in m for m in messages)

    async def test_it_still_works_end_to_end_from_home(self, home, local):
        settings = Settings()
        settings.database.local_cache = False
        app = HpcaApp(settings, llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert len(app.session_store.list(profile="default")) >= 1
        conn = sqlite3.connect(home / "hpca.db")
        count = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
        conn.close()
        assert count >= 1


class TestCrashRecovery:
    async def test_a_working_dir_left_behind_is_recovered_on_the_next_start(
        self, home, local
    ):
        killed = HpcaApp(llm=FakeLLM())
        async with killed.run_test(size=(120, 30)) as pilot:
            await killed.start_new_session()
            await pilot.pause()
            created = killed.active_session.session_id
            # Simulate a hard kill: no final sync, no cleanup, lease left over
            # pointing at a pid that is gone by the next start.
            killed._dbcache.active = False
        lease = json.loads((home / "db.lease").read_text())
        lease["pid"] = _dead_pid()
        (home / "db.lease").write_text(json.dumps(lease))
        assert not (home / "hpca.db").exists()  # nothing reached home

        restarted = HpcaApp(llm=FakeLLM())
        async with restarted.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            ids = [s.session_id for s in restarted.session_store.list(profile="default")]
        assert created in ids

    async def test_every_managed_database_is_recovered(self, home, local):
        killed = HpcaApp(llm=FakeLLM())
        async with killed.run_test(size=(120, 30)):
            killed._dbcache.active = False
        lease = json.loads((home / "db.lease").read_text())
        lease["pid"] = _dead_pid()
        (home / "db.lease").write_text(json.dumps(lease))

        restarted = HpcaApp(llm=FakeLLM())
        async with restarted.run_test(size=(120, 30)):
            pass
        for name in DB_NAMES:
            assert (home / name).exists(), name


def _dead_pid() -> int:
    """A pid that is certainly not running — what a killed run leaves behind."""
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait()
    return child.pid


class TestCorruptHomeCopy:
    async def test_the_user_is_told_at_startup(self, home, local):
        # A torn write destroyed home's copy (first page zeroed). The app
        # must quarantine it, start the database afresh and say so as a
        # toast — dbcache.log alone proved too easy to miss.
        (home / "hpca.db").write_bytes(b"\x00" * 4096 + b"leftovers" * 100)
        messages = []
        app = HpcaApp(llm=FakeLLM())
        original = HpcaApp.notify
        app.notify = lambda message, **kw: (
            messages.append(message),
            original(app, message, **kw),
        )[1]

        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()

        assert any("corrupt" in m for m in messages)
        assert list(home.glob("hpca.db.corrupt-*"))
        # The exit sync then writes a fresh, healthy copy under the old name.
        sqlite3.connect(home / "hpca.db").execute("PRAGMA schema_version")
