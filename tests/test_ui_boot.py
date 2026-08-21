"""The entry point: an `AgentService` in the UI's own process, and the order
it is taken down in.

Two halves, deliberately. `TestShutdownOrder` drives `Core.stop` over recorded
doubles, because the claim being made is about *sequence* — checkpointer, DbIO,
sqlite, rag, then the copy home — and a real database cannot say whether it was
closed third or fourth. The rest builds the real thing (with a fake LLM, no
terminal and a tmp home) and checks that the databases are where they should
be, that a command reaches the service, and that the wait message lands on a
terminal the user can still read.

`specs-ui-acceptance.md` records the same guarantees under "Node-local
databases"; `tests/test_tui_dbcache.py` asserts them of the Textual app.
"""

from __future__ import annotations

import asyncio
import io
import os
import signal
import subprocess
import sys
import time

import pytest

from hpca import protocol
from hpca.llm import ChatResponse
from hpca.transport import InProcessConnection
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
    monkeypatch.setattr("hpca.ui.boot._detect_slurm", lambda settings: None)


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
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.queue: asyncio.Queue = asyncio.Queue()

    def subscribe(self):
        return self.queue

    def unsubscribe(self, queue):
        pass

    async def stop(self):
        self.order.append("service")

    def start_timers(self):
        pass


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


def build_core(order: list[str], *, active: bool = True) -> Core:
    return Core(
        FakeService(order),
        InProcessConnection(),
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
    def test_importing_the_entry_point_costs_neither_front_end(self):
        # §4.2 item 10: `main()` argparses before importing either side, so a
        # core process never imports the UI and the UI never imports Textual.
        result = run_python(
            "import sys, hpca.__main__; "
            "assert 'textual' not in sys.modules, 'Textual came with it'; "
            "assert 'hpca.ui.app' not in sys.modules, 'the row UI came with it'"
        )
        assert result.returncode == 0, result.stderr

    def test_and_parsing_argv_does_not_either(self):
        result = run_python(
            "import sys, hpca.__main__ as m; "
            "m.build_parser().parse_args(['--new-ui']); "
            "assert 'textual' not in sys.modules, 'the parser dragged it in'"
        )
        assert result.returncode == 0, result.stderr

    def test_the_new_ui_is_opt_in(self):
        from hpca.__main__ import build_parser

        assert build_parser().parse_args([]).new_ui is False
        assert build_parser().parse_args(["--new-ui"]).new_ui is True

    def test_a_profile_can_be_named(self):
        from hpca.__main__ import build_parser

        assert build_parser().parse_args(["--profile", "hpc"]).profile == "hpc"

    def test_it_refuses_to_draw_without_a_terminal(self, monkeypatch, capsys):
        from hpca.__main__ import main

        monkeypatch.setattr("sys.stdin", io.StringIO())
        assert main(["--new-ui"]) == 2
        assert "terminal" in capsys.readouterr().err
