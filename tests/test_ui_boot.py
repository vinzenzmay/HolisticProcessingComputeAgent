"""The entry point: an `AgentService` in the UI's own process, and the order
it is taken down in.

Two halves, deliberately. `TestShutdownOrder` drives `Core.stop` over recorded
doubles, because the claim being made is about *sequence* — checkpointer, DbIO,
sqlite, rag, then the copy home — and a real database cannot say whether it was
closed third or fourth. The rest builds the real thing (with a fake LLM, no
terminal and a tmp home) and checks that the databases are where they should
be, that a command reaches the service, and that the wait message lands on a
terminal the user can still read.

`specs/specs-ui-acceptance.md` records the same guarantees under "Node-local
databases"; `tests/test_tui_dbcache.py` asserts them of the Textual app.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import subprocess
import sys
import time

import pytest

from hpca import protocol
from hpca.config import Settings
from hpca.llm import ChatResponse
from hpca.transport import InProcessConnection
from hpca.core import boot as core_boot
from hpca.ui import boot
from hpca.ui.boot import (
    DB_SYNC_DONE_MESSAGE,
    DB_SYNC_INTERRUPT_MESSAGE,
    DB_SYNC_WAIT_MESSAGE,
    Core,
    start,
)
from hpca.ui.screen import EXIT_MODES, Screen


class FakeLLM:
    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content="")

    async def supports_constrained_decoding(self):
        return True

    async def close(self):
        pass


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_LOCAL_DIR", str(tmp_path / "node-local"))
    return tmp_path / "node-local"


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    """A dev box may well have `sbatch` on it; a test must not care."""
    monkeypatch.setattr("hpca.core.boot._detect_slurm", lambda settings: None)


# ------------------------------------------------------------ the order alone


class Recorder:
    """A double that writes its name down when it is closed."""

    def __init__(self, order: list[str], name: str) -> None:
        self.order = order
        self.name = name

    def close(self) -> None:
        self.order.append(self.name)

    async def aclose(self) -> None:
        self.order.append(self.name)

    async def __aexit__(self, *exc) -> None:
        self.order.append(self.name)


class FakeService:
    def __init__(self, order: list[str], *, connected: bool = True) -> None:
        self.order = order
        self.queue: asyncio.Queue = asyncio.Queue()
        self.started = 0
        self._connected = connected

    def subscribe(self):
        return self.queue

    def unsubscribe(self, queue):
        pass

    async def stop(self):
        self.order.append("service")

    def start_timers(self):
        pass

    async def startup(self):
        self.started += 1
        if isinstance(self._connected, Exception):
            raise self._connected
        return self._connected


class FakeCache:
    """A `DbCache` as far as `Core.stop` is concerned."""

    def __init__(self, order: list[str], *, active: bool = True) -> None:
        self.order = order
        self.active = active
        self.released = False
        self.syncs = 0

    def release(self) -> None:
        self.released = True
        self.order.append("sync")

    def sync(self) -> bool:
        self.syncs += 1
        return True

    def drain_warnings(self):
        return []


class Aio(Recorder):
    """DbIO's close is a coroutine; sqlite's is not."""

    async def close(self) -> None:
        self.order.append(self.name)


def build_core(
    order: list[str], *, active: bool = True, service=None, **kwargs
) -> Core:
    return Core(
        service if service is not None else FakeService(order),
        InProcessConnection(),
        **kwargs,
        dbcache=FakeCache(order, active=active),
        dbio=Aio(order, "dbio"),
        db=Recorder(order, "sqlite"),
        saver_ctx=Recorder(order, "checkpointer"),
        rag=Recorder(order, "rag"),
    )


class TestShutdownOrder:
    async def test_it_is_the_order_the_textual_app_settled_on(self):
        # Load-bearing, not tidiness: the sync must copy a quiesced database,
        # and nothing may open a file inside the working dir after it is gone.
        order: list[str] = []
        await build_core(order).stop(say=lambda text: None)
        assert order == [
            "service",
            "checkpointer",
            "dbio",
            "sqlite",
            "rag",
            "sync",
        ]

    async def test_the_wait_is_explained_before_the_copy_starts(self):
        # Saying it afterwards would be no help at all.
        said: list[str] = []
        order: list[str] = []
        core = build_core(order)
        core.dbcache.order = said  # the copy appends here too, in sequence
        await core.stop(say=said.append)
        assert DB_SYNC_WAIT_MESSAGE in said[0]
        assert said.index("sync") > 0

    async def test_and_the_end_of_the_wait_is_announced(self):
        said: list[str] = []
        await build_core([]).stop(say=said.append)
        assert any(DB_SYNC_DONE_MESSAGE in text for text in said)

    async def test_nothing_is_said_when_nothing_is_copied(self):
        # Running straight from home has no exit copy to wait for.
        said: list[str] = []
        await build_core([], active=False).stop(say=said.append)
        assert said == []

    async def test_stopping_twice_closes_nothing_twice(self):
        order: list[str] = []
        core = build_core(order)
        await core.stop(say=lambda text: None)
        await core.stop(say=lambda text: None)
        assert order.count("sqlite") == 1

    async def test_a_failing_close_does_not_strand_the_databases(self):
        # The sync home is the one step that must happen whatever else broke:
        # everything before it is a resource, and the copy is the user's data.
        order: list[str] = []
        core = build_core(order)

        async def explode(*exc):
            raise RuntimeError("the checkpointer is wedged")

        core.saver_ctx.__aexit__ = explode
        await core.stop(say=lambda text: None)
        assert "sync" in order


class TestPeriodicSync:
    async def test_the_timer_only_runs_when_there_is_something_to_copy(self):
        core = build_core([], active=False)
        core.sync_interval = 0.01
        core.run()
        await asyncio.sleep(0.05)
        assert core.dbcache.syncs == 0
        await core.stop(say=lambda text: None)

    async def test_and_does_when_there_is(self):
        core = build_core([])
        core.sync_interval = 0.01
        core.run()
        await asyncio.sleep(0.05)
        assert core.dbcache.syncs > 0
        await core.stop(say=lambda text: None)

    async def test_an_overlapping_sync_is_skipped(self):
        # On NFS one copy can outlast its interval, and two backups of the same
        # file at once is pointless work.
        core = build_core([])
        core._syncing = True
        await core.sync()
        assert core.dbcache.syncs == 0

    async def test_a_failing_sync_is_reported_and_not_fatal(self):
        core = build_core([])
        said: list[str] = []
        core._notify = lambda text, severity="information": said.append(text)

        def explode():
            raise OSError("home is unreachable")

        core.dbcache.sync = explode
        await core.sync()
        assert any("failed" in text for text in said)


    async def test_an_interval_of_zero_means_the_exit_copy_only(self):
        # `settings.database.sync_interval_s = 0` is the documented way to say
        # "copy home when I leave and not before"; the timer is what has to
        # honour it, and it is the same guard `dbcache.active` goes through.
        core = build_core([])
        cache = core.dbcache  # `_final_sync` lets go of it on the way out
        core.sync_interval = 0
        core.run()
        await asyncio.sleep(0.05)
        assert cache.syncs == 0
        await core.stop(say=lambda text: None)
        assert cache.released, "and the copy still happens on the way out"

    async def test_the_interval_comes_from_the_settings_file(self, home, local):
        settings = Settings.load()
        settings.database.sync_interval_s = 12
        built = await Core.start(settings=settings, llm=FakeLLM())
        try:
            assert built.sync_interval == 12
        finally:
            await built.stop(say=lambda text: None)


class TestTheNotices:
    """What happened before there was a UI to be told about it.

    A declined local cache or a quarantined corrupt copy is decided in
    `Core.start`, minutes before the first frame; it reaches the user as a
    toast like everything else, which means it has to go on the wire.
    """

    async def test_they_arrive_as_toasts_after_hello(self):
        ui_end, core_end = InProcessConnection.pair()
        core = build_core([], notices=["another instance holds the lease"])
        core.wire = core_end
        core.run()
        try:
            events = aiter(ui_end)
            assert (await anext(events)).type == protocol.Hello.TYPE
            notice = protocol.parse(await anext(events))
            assert isinstance(notice, protocol.Notify)
            assert notice.text.startswith("Databases: ")
            assert "another instance holds the lease" in notice.text
            assert notice.severity == "warning"
        finally:
            await core.stop(say=lambda text: None)

    async def test_a_clean_start_says_nothing_about_the_databases(self):
        ui_end, core_end = InProcessConnection.pair()
        core = build_core([])
        core.wire = core_end
        core.run()
        try:
            events = aiter(ui_end)
            assert (await anext(events)).type == protocol.Hello.TYPE
            await core.service.queue.put(protocol.Notify(text="something else"))
            assert "Databases:" not in protocol.parse(await anext(events)).text
        finally:
            await core.stop(say=lambda text: None)

    async def test_a_declined_cache_is_one(self, home, monkeypatch):
        # The decision itself: local mode off in settings is a reason, and a
        # reason is what becomes the notice.
        settings = Settings.load()
        settings.database.local_cache = False
        cache, notices = core_boot._open_db_cache(settings, home)
        assert notices == [], "declining what was never asked for is not news"
        settings.database.local_cache = True
        monkeypatch.setattr(
            "hpca.dbcache.DbCache.acquire", lambda self: False
        )
        cache, notices = core_boot._open_db_cache(settings, home)
        assert notices, "a cache that could not be taken is"


class TestTheLoggers:
    """Records into a file, and none of them onto a terminal in raw mode.

    One WARNING reaching logging's last-resort handler lands in the middle of
    a frame and stays there until the next full paint, so both of these are
    about the *absence* of output as much as the presence of a file.
    """

    def test_a_file_logger_writes_where_it_says_and_nowhere_else(self, home):
        log = boot.file_logger("hpca.test.dbcache", "dbcache.log")
        log.warning("the lease was not free")
        assert "the lease was not free" in (home / "dbcache.log").read_text()
        assert log.propagate is False, "or it reaches stderr as well"

    def test_and_attaching_it_twice_does_not_double_the_records(self, home):
        boot.file_logger("hpca.test.twice", "twice.log")
        log = boot.file_logger("hpca.test.twice", "twice.log")
        log.warning("once")
        assert (home / "twice.log").read_text().count("once") == 1

    def test_the_quiet_terminal_takes_the_package_off_stderr(self, home):
        package = logging.getLogger("hpca")
        with boot.quiet_terminal():
            assert package.propagate is False
            logging.getLogger("hpca.something").warning("into the file")
        assert "into the file" in (home / "ui.log").read_text()

    def test_and_gives_the_logger_back_on_the_way_out(self, home):
        # A UI is a guest in someone's process — this suite reads records off
        # the root logger, and a front-end that permanently silenced `hpca`
        # would take those assertions with it.
        package = logging.getLogger("hpca")
        before = (list(package.handlers), package.propagate, package.level)
        with boot.quiet_terminal():
            pass
        assert (list(package.handlers), package.propagate, package.level) == before

    def test_an_unwritable_app_dir_is_not_a_reason_not_to_start(
        self, home, monkeypatch
    ):
        monkeypatch.setattr(
            "logging.FileHandler",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only")),
        )
        with boot.quiet_terminal():
            pass  # must not raise


class TestSyncInterrupt:
    async def test_the_first_ctrl_c_is_answered_not_obeyed(self):
        # By this point the terminal is out of raw mode, so the impatient press
        # the message exists to prevent really would kill the copy.
        said: list[str] = []
        order: list[str] = []
        core = build_core(order)
        release = core.dbcache.release

        def slow_release():
            # The copy runs on a worker thread; the signal is delivered to the
            # main one, which is parked on `to_thread` and free to take it.
            os.kill(os.getpid(), signal.SIGINT)
            for _ in range(200):
                if any(DB_SYNC_INTERRUPT_MESSAGE in t for t in said):
                    break
                time.sleep(0.01)
            release()

        core.dbcache.release = slow_release
        await core.stop(say=said.append)
        assert any(DB_SYNC_INTERRUPT_MESSAGE in text for text in said)
        assert "sync" in order  # and the copy still finished

    async def test_the_handler_is_put_back_afterwards(self):
        before = signal.getsignal(signal.SIGINT)
        await build_core([]).stop(say=lambda text: None)
        assert signal.getsignal(signal.SIGINT) is before

    async def test_nothing_is_guarded_when_nothing_was_said(self):
        before = signal.getsignal(signal.SIGINT)
        await build_core([], active=False).stop(say=lambda text: None)
        assert signal.getsignal(signal.SIGINT) is before


class TestStartupPass:
    """The once-only work `tui/app.py` did on mount, and what it answers with.

    The core does all of it (`AgentService.startup`); what belongs here is the
    one part of it that is a *screen* — nothing answering opens manage-LLMs,
    because a front-end is the only thing that can open one.
    """

    async def opened(self, **kwargs):
        """Run one core's startup to completion; returns whether it asked for
        the screen, and the service it ran."""
        order: list[str] = []
        asked: list[int] = []
        service = FakeService(order, **kwargs)
        core = build_core(
            order, service=service, on_no_backend=lambda: asked.append(1)
        )
        core.run()
        for _ in range(20):
            await asyncio.sleep(0)
        await core.stop(say=lambda text: None)
        return bool(asked), service

    async def test_the_service_gets_its_one_startup_pass(self):
        # Trash, curator, auto-connect and the check — all of it had exactly
        # one caller, and this is now that caller.
        _, service = await self.opened()
        assert service.started == 1

    async def test_nothing_answering_opens_the_screen_that_fixes_it(self):
        asked, _ = await self.opened(connected=False)
        assert asked

    async def test_a_backend_that_answers_leaves_the_user_alone(self):
        asked, _ = await self.opened(connected=True)
        assert not asked

    async def test_a_startup_that_explodes_is_not_a_failed_run(self):
        # Best-effort throughout: discovery, a curator pass and an NFS sweep
        # are none of them worth refusing to start over.
        asked, service = await self.opened(connected=RuntimeError("boom"))
        assert service.started == 1
        assert not asked


# ------------------------------------------------------------ the real thing


@pytest.fixture
async def core(home, local):
    built = await Core.start(llm=FakeLLM())
    try:
        yield built
    finally:
        await built.stop(say=lambda text: None)


class TestPlacement:
    async def test_the_databases_open_from_the_working_dir(self, core, local):
        assert core.dbcache.active
        path = core.db.execute("PRAGMA database_list").fetchone()[2]
        assert path.startswith(str(local))

    async def test_checkpoints_live_there_too(self, core, local):
        assert str(core.dbcache.path_for("checkpoints.db")).startswith(str(local))
        assert core.dbcache.path_for("checkpoints.db").exists()

    async def test_and_so_does_the_rag_store(self, core, local):
        # `build_service` has no parameter for it and would have opened it in
        # home; boot hands it over through `extras` instead.
        assert core.service._deps.extras["rag"] is core.rag
        opened = core.rag._conn.execute("PRAGMA database_list").fetchone()[2]
        assert opened.startswith(str(local))

    async def test_home_is_written_back_on_exit(self, home, local):
        built = await Core.start(llm=FakeLLM())
        await built.stop(say=lambda text: None)
        assert (home / "hpca.db").exists()
        assert (home / "checkpoints.db").exists()

    async def test_and_the_working_dir_is_gone(self, home, local):
        built = await Core.start(llm=FakeLLM())
        working = built.dbcache.path_for("hpca.db").parent
        await built.stop(say=lambda text: None)
        assert not working.exists()


class TestTheWire:
    @pytest.fixture
    async def wired(self, home, local):
        ui_end, core_end = InProcessConnection.pair()
        built = await Core.start(llm=FakeLLM(), wire=core_end)
        built.run()
        try:
            yield built, ui_end
        finally:
            await built.stop(say=lambda text: None)

    async def test_the_core_says_hello_first(self, wired):
        _, ui_end = wired
        env = await asyncio.wait_for(anext(aiter(ui_end)), 2.0)
        assert env.type == protocol.Hello.TYPE

    async def test_a_command_reaches_the_service(self, wired):
        # End to end over the wire: the sidebar the UI asks for is answered by
        # the real `AgentService`, with no terminal and no LLM involved.
        core, ui_end = wired
        events = aiter(ui_end)
        await anext(events)  # hello
        await ui_end.send(protocol.SessionList())
        assert await _wait_for(events, protocol.SessionRows.TYPE)

    async def test_a_frame_that_is_not_a_command_is_dropped_not_fatal(self, wired):
        # A peer sending junk down this must not be able to end the session —
        # the same bargain `transport` makes one layer down.
        core, ui_end = wired
        events = aiter(ui_end)
        await anext(events)  # hello
        await ui_end.send(protocol.Hello())  # an event, on the command channel
        await ui_end.send(protocol.SessionList())
        assert await _wait_for(events, protocol.SessionRows.TYPE)


async def _wait_for(events, type_name: str, timeout: float = 2.0):
    async with asyncio.timeout(timeout):
        while True:
            env = await anext(events)
            if env.type == type_name:
                return env


class TestTheWholeRun:
    async def test_it_starts_draws_and_shuts_down(self, home, local):
        # The M3 acceptance, hermetically: a frame is painted, ^C ends the run,
        # and the terminal is put back before anything is printed on it.
        read, write = os.pipe()
        out = io.StringIO()
        screen = Screen(fd=read, out=out)
        run = asyncio.ensure_future(
            start(llm=FakeLLM(), screen=screen, say=out.write)
        )
        await asyncio.sleep(0.2)
        os.write(write, b"\x03")  # quit
        code = await asyncio.wait_for(run, 20.0)
        os.close(read), os.close(write)

        painted = out.getvalue()
        assert code == 0
        assert "HPCA" in painted  # the header was drawn
        assert EXIT_MODES in painted
        # The wait message must survive the UI: printed while the alternate
        # screen was still up, it would be wiped out along with it.
        assert painted.index(EXIT_MODES) < painted.index(DB_SYNC_WAIT_MESSAGE)
        assert painted.index(DB_SYNC_WAIT_MESSAGE) < painted.index(
            DB_SYNC_DONE_MESSAGE
        )

    async def test_nothing_answering_puts_manage_llms_on_the_screen(
        self, home, local, monkeypatch
    ):
        """The whole wire, end to end: the core probes, finds nothing, says so
        — and the front-end this file builds opens the screen that fixes it.

        Both halves were green on their own for a milestone
        (`specs/specs-ui-coverage.md` §3.1): `auto_connect` had ten tests and no
        caller, and the UI had a manage-LLMs screen nothing could open.
        """
        from hpca.core.backends import NO_BACKEND_MESSAGE
        from hpca.core.service import AgentService
        from hpca.ui.app import RowUI
        from hpca.ui.overlays import LlmOverlay

        async def nothing_answers(*args, **kwargs):
            return []

        monkeypatch.setattr(AgentService, "startup_backend_check", True)
        monkeypatch.setattr("hpca.core.backends.probe_endpoint", nothing_answers)

        made: list[RowUI] = []

        class Watched(RowUI):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                made.append(self)

        monkeypatch.setattr("hpca.ui.app.RowUI", Watched)

        read, write = os.pipe()
        out = io.StringIO()
        screen = Screen(fd=read, out=out)
        run = asyncio.ensure_future(
            start(llm=FakeLLM(), screen=screen, say=lambda text: None)
        )
        try:
            def said() -> bool:
                return bool(made) and any(
                    NO_BACKEND_MESSAGE in toast.text for toast in made[0].toasts
                )

            def opened() -> bool:
                return any(isinstance(x, LlmOverlay) for x in made[0].overlays)

            for _ in range(400):
                await asyncio.sleep(0.01)
                # The warning is emitted before the screen is asked for and
                # delivered after it, so waiting for the later of the two is
                # what makes this deterministic rather than lucky.
                if said() and opened():
                    break
            assert made, "no UI was built"
            # Anywhere on the stack, not necessarily on top: the manage-LLMs
            # screen scans as it opens, and an empty scan legitimately pushes
            # the tunnel recipe over it. Asserting the top screen would make
            # this pass or fail on whether the sweep beat the assertion.
            assert opened()
            # And it says why it is there, rather than appearing unbidden.
            assert said()
        finally:
            os.write(write, b"\x1b")  # close the screen, then quit
            await asyncio.sleep(0.2)
            os.write(write, b"\x03")
            await asyncio.wait_for(run, 20.0)
            os.close(read), os.close(write)

    async def test_the_databases_are_home_afterwards(self, home, local):
        read, write = os.pipe()
        screen = Screen(fd=read, out=io.StringIO())
        run = asyncio.ensure_future(
            start(llm=FakeLLM(), screen=screen, say=lambda text: None)
        )
        await asyncio.sleep(0.2)
        os.write(write, b"\x03")
        await asyncio.wait_for(run, 20.0)
        os.close(read), os.close(write)
        assert (home / "hpca.db").exists()


# ------------------------------------------------------------ the entry point


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )


class TestArgv:
    # §4.2 item 10. The rule survived the deletion of the second front-end and
    # matters more without it, because M11's `--serve` is the reason it exists:
    # a core process that imported the UI on its way to deciding it was not one
    # would pay for a terminal driver to run a graph nobody is watching.
    def test_importing_the_entry_point_costs_no_front_end(self):
        result = run_python(
            "import sys, hpca.__main__; "
            "assert 'hpca.ui.app' not in sys.modules, 'the UI came with it'; "
            "assert 'textual' not in sys.modules, 'Textual came back'"
        )
        assert result.returncode == 0, result.stderr

    def test_and_parsing_argv_does_not_either(self):
        result = run_python(
            "import sys, hpca.__main__ as m; "
            "m.build_parser().parse_args(['--profile', 'hpc']); "
            "assert 'hpca.ui.app' not in sys.modules, 'the parser dragged it in'"
        )
        assert result.returncode == 0, result.stderr

    def test_the_retired_new_ui_flag_is_gone(self):
        # It selected the row UI while there were two. Left as an assertion
        # rather than deleted: `hpca --new-ui` is in a shell history or two,
        # and argparse rejecting it is the answer, not quietly accepting it.
        import pytest

        from hpca.__main__ import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--new-ui"])

    def test_a_profile_can_be_named(self):
        from hpca.__main__ import build_parser

        assert build_parser().parse_args(["--profile", "hpc"]).profile == "hpc"

    def test_it_refuses_to_draw_without_a_terminal(self, monkeypatch, capsys):
        from hpca.__main__ import main

        monkeypatch.setattr("sys.stdin", io.StringIO())
        assert main([]) == 2
        assert "terminal" in capsys.readouterr().err
