"""The app runs its databases from node-local storage and syncs home.

On a cluster node ``$HOME`` is NFS and every sqlite call is a network
round-trip, which is what makes the TUI lag while the agent works. The app
opens hpca.db, checkpoints.db and rag.db from a node-local working dir
instead, syncs them back on a timer, and syncs-and-cleans-up on exit.
See specs-db-local-cache.md and hpca.dbcache.
"""
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from hpca.config import Settings
from hpca.dbcache import DB_NAMES
from hpca.llm import ChatResponse
from hpca.tui.app import (
    DB_SYNC_DONE_MESSAGE,
    DB_SYNC_INTERRUPT_MESSAGE,
    DB_SYNC_WAIT_MESSAGE,
    HpcaApp,
)


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

    async def test_a_first_run_creates_the_app_dir_it_logs_into(
        self, monkeypatch, tmp_path, local, fresh_dbcache_logger
    ):
        # First launch on a fresh account: nothing has written settings yet, so
        # the app dir does not exist — and this logger is the first thing
        # startup touches. Opening a FileHandler under a missing directory took
        # the app down before the UI existed.
        virgin = tmp_path / "never-run"
        monkeypatch.setenv("HPCA_HOME", str(virgin))
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert app.is_running
        assert (virgin / "dbcache.log").exists()


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


class TestExitMessage:
    """Quitting on a cluster node ends in a copy home over NFS, which looks
    like a hang: the TUI is gone and the shell prompt is not back. The app
    leaves the alt screen first and says what the wait is for.
    """

    @pytest.fixture
    def terminal(self, monkeypatch):
        """Make the headless test driver look like a real terminal, and record
        what reaches it — in order, application-mode stop included."""
        from textual.drivers.headless_driver import HeadlessDriver

        seen: list[str] = []
        monkeypatch.setattr(
            HeadlessDriver, "is_headless", property(lambda self: False)
        )
        monkeypatch.setattr(
            HeadlessDriver, "write", lambda self, data: seen.append(data)
        )
        monkeypatch.setattr(
            HeadlessDriver,
            "stop_application_mode",
            lambda self: seen.append("<stop-application-mode>"),
        )
        return seen

    async def test_the_wait_is_explained_on_the_terminal(
        self, home, local, terminal
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert app._dbcache.active
        assert any(DB_SYNC_WAIT_MESSAGE in text for text in terminal)

    async def test_the_message_survives_the_tui(self, home, local, terminal):
        # Textual dispatches Unmount with the alt screen still up, so the
        # message would be wiped out with it unless application mode stops
        # first. Anything after it is on the terminal the user is left with.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        stopped = terminal.index("<stop-application-mode>")
        said = next(
            i for i, text in enumerate(terminal) if DB_SYNC_WAIT_MESSAGE in text
        )
        assert stopped < said

    async def test_it_is_said_before_the_copy_starts(self, home, local, terminal):
        # Saying it afterwards would be no help at all.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            cache = app._dbcache
            original = cache.release
            seen_at_release: list[list[str]] = []

            def release():
                seen_at_release.append(list(terminal))
                original()

            cache.release = release
        assert any(DB_SYNC_WAIT_MESSAGE in text for text in seen_at_release[0])

    async def test_and_the_end_of_the_wait_is_announced_too(
        self, home, local, terminal
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        assert any(DB_SYNC_DONE_MESSAGE in text for text in terminal)
        assert terminal.index(
            next(t for t in terminal if DB_SYNC_DONE_MESSAGE in t)
        ) > terminal.index(next(t for t in terminal if DB_SYNC_WAIT_MESSAGE in t))

    async def test_nothing_is_said_when_nothing_is_copied(
        self, home, local, terminal
    ):
        # Running straight from home has no exit copy to wait for.
        settings = Settings()
        settings.database.local_cache = False
        app = HpcaApp(settings, llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            assert not app._dbcache.active
        assert not any(DB_SYNC_WAIT_MESSAGE in text for text in terminal)

    async def test_the_tui_still_says_nothing_to_a_real_terminal_run(
        self, home, local
    ):
        # Without the terminal fixture the driver is headless: no writes, so
        # the test suite's own output stays clean.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 30)):
            pass
        assert not app._write_to_terminal("anything")


class TestSyncInterrupt:
    """The reflex the message exists to head off. Leaving application mode
    gives Ctrl+C its meaning back, so the first press must not kill the copy.
    """

    def guarded_app(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            HpcaApp,
            "_write_to_terminal",
            lambda self, text: bool(seen.append(text)) or True,
        )
        return HpcaApp(llm=FakeLLM()), seen

    def test_the_first_ctrl_c_is_answered_not_obeyed(self, home, local, monkeypatch):
        app, seen = self.guarded_app(monkeypatch)
        with app._sync_interrupt_guard(True):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)  # let the handler run
        assert any(DB_SYNC_INTERRUPT_MESSAGE in text for text in seen)

    def test_a_second_ctrl_c_still_aborts(self, home, local, monkeypatch):
        # A hung NFS mount must never become a trap.
        app, _ = self.guarded_app(monkeypatch)
        before = signal.getsignal(signal.SIGINT)
        with app._sync_interrupt_guard(True):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)
            assert signal.getsignal(signal.SIGINT) is before

    def test_the_handler_is_handed_back_afterwards(self, home, local, monkeypatch):
        app, _ = self.guarded_app(monkeypatch)
        before = signal.getsignal(signal.SIGINT)
        with app._sync_interrupt_guard(True):
            pass
        assert signal.getsignal(signal.SIGINT) is before

    def test_nothing_is_guarded_when_nothing_was_said(
        self, home, local, monkeypatch
    ):
        # Without the local cache the terminal is still Textual's; touching
        # signal handling there would be meddling.
        app, _ = self.guarded_app(monkeypatch)
        before = signal.getsignal(signal.SIGINT)
        with app._sync_interrupt_guard(False):
            assert signal.getsignal(signal.SIGINT) is before
