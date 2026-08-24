"""Tests for hpca.coreproc: starting the core child, and where it listens.

The core runs as a child process (specs/specs-core-process.md §2). What is testable
without a real core is precisely the fiddly part: a socket path that fits in
``sun_path`` and is not reachable by the rest of a shared login node, a
handshake that fails loudly instead of hanging, and a stop() that ends in
SIGKILL rather than a wedged child.

Every supervisor test here spawns a *stub* core — a ``python -c`` script that
prints a handshake line and sleeps — never ``python -m hpca``, which would
import langgraph and cost the suite seconds per test (§9).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import socket
import stat
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from hpca import coreproc
from hpca.coreproc import (
    HANDSHAKE_VERSION,
    CoreProcessError,
    CoreSupervisor,
    IdleShutdown,
    clear_stale_socket,
    default_core_argv,
    socket_is_live,
    socket_path_for,
)

# --------------------------------------------------------------------- stubs


def stub(body: str, *args: str) -> list[str]:
    """A fake core: a ``python -c`` body plus its argv. See the module docstring
    for why no test may spawn the real one."""
    src = "import json, os, signal, sys, time\n" + textwrap.dedent(body).strip()
    return [sys.executable, "-c", src, *args]


# argv[1] is the socket to claim, argv[2] the version to claim; dedented here
# so the composed bodies below stay at one indentation level.
_WRITE_HELLO = textwrap.dedent(
    """
    hello = {"ready": True, "pid": os.getpid(), "socket": sys.argv[1],
             "version": int(sys.argv[2])}
    sys.stdout.write(json.dumps(hello) + chr(10))
    sys.stdout.flush()
    sys.stdout.close()
    """
).strip()

_SLEEP = "\ntime.sleep(30)\n"


def ready_stub(socket_path: Path, *, version: int = HANDSHAKE_VERSION) -> list[str]:
    """Hands shake correctly, then waits to be stopped."""
    return stub(_WRITE_HELLO + _SLEEP, str(socket_path), str(version))


def deaf_stub(socket_path: Path) -> list[str]:
    """Hands shake, then ignores SIGTERM — stop() has to SIGKILL this child."""
    body = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" + _WRITE_HELLO
    return stub(body + _SLEEP, str(socket_path), str(HANDSHAKE_VERSION))


def noisy_stub(socket_path: Path, text: str) -> list[str]:
    """Writes to stderr before the handshake, to prove stderr lands in the log."""
    body = "sys.stderr.write(sys.argv[3] + chr(10))\nsys.stderr.flush()\n"
    return stub(
        body + _WRITE_HELLO + _SLEEP,
        str(socket_path),
        str(HANDSHAKE_VERSION),
        text,
    )


def silent_stub() -> list[str]:
    """Never says anything: the ready timeout is the only way out."""
    return stub("time.sleep(30)")


def line_stub(text: str) -> list[str]:
    """Prints one arbitrary line as its handshake, then waits."""
    return stub(
        """
        sys.stdout.write(sys.argv[1] + chr(10))
        sys.stdout.flush()
        time.sleep(30)
        """,
        text,
    )


@pytest.fixture
def short_root():
    """A socket root short enough not to trip the sun_path fallback.

    ``tmp_path`` cannot be it: pytest nests it several directories deep, deeper
    still under xdist, so a socket under it can exceed 100 bytes on its own.
    That is the condition ``test_a_long_root_falls_back`` exists to check, and
    it must not become the accidental condition for every other test here.
    """
    root = Path(tempfile.mkdtemp(prefix="hpca-t-", dir="/tmp"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def app_dir(short_root, tmp_path, monkeypatch):
    """An app dir whose socket lands under the test's root, not the real one."""
    monkeypatch.setenv("HPCA_LOCAL_DIR", str(short_root))
    home = tmp_path / "home"
    home.mkdir()
    return home


# ---------------------------------------------------------------- socket path


class TestSocketPathFor:
    def test_creates_a_private_directory(self, tmp_path, short_root):
        path = socket_path_for(tmp_path / "home", local_root=short_root)
        assert path.name == "core.sock"
        assert path.parent.is_dir()
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_tightens_a_directory_a_loose_umask_made(self, tmp_path, short_root):
        # mkdir's mode is masked by umask; the chmod afterwards is what makes
        # 0700 actually true.
        old = os.umask(0o022)
        try:
            path = socket_path_for(tmp_path / "home", local_root=short_root)
        finally:
            os.umask(old)
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_fits_in_sun_path(self, tmp_path, short_root):
        path = socket_path_for(tmp_path / "home", local_root=short_root)
        assert len(str(path).encode()) <= coreproc.MAX_SOCKET_PATH_BYTES
        assert coreproc.MAX_SOCKET_PATH_BYTES < 108  # the kernel's real limit

    def test_is_stable_for_one_app_dir(self, tmp_path, short_root):
        first = socket_path_for(tmp_path / "home", local_root=short_root)
        second = socket_path_for(tmp_path / "home", local_root=short_root)
        assert first == second

    def test_differs_per_app_dir(self, tmp_path, short_root):
        one = socket_path_for(tmp_path / "a", local_root=short_root)
        two = socket_path_for(tmp_path / "b", local_root=short_root)
        assert one != two

    def test_defaults_to_the_node_local_root(self, tmp_path, short_root, monkeypatch):
        monkeypatch.setenv("HPCA_LOCAL_DIR", str(short_root / "scratch"))
        path = socket_path_for(tmp_path / "home")
        assert path.parent.parent == short_root / "scratch"

    def test_a_long_root_falls_back(self, tmp_path, monkeypatch):
        # A Slurm $TMPDIR can be long enough to overrun sun_path on its own.
        monkeypatch.setattr(coreproc, "FALLBACK_ROOT", tmp_path / "fb")
        long_root = tmp_path / ("d" * 120)
        path = socket_path_for(tmp_path / "home", local_root=long_root)
        assert path.parent.parent == tmp_path / "fb"
        assert not path.is_relative_to(long_root)
        assert not long_root.exists()  # the rejected root is not created

    def test_the_real_fallback_root_is_short_enough(self):
        # A fallback that did not fit would be a bug nobody notices until
        # Slurm hands out a long $TMPDIR on a busy day.
        example = coreproc.FALLBACK_ROOT / "hpca-0123abcd" / coreproc.SOCKET_NAME
        assert len(str(example).encode()) <= coreproc.MAX_SOCKET_PATH_BYTES

    def test_refuses_a_world_readable_directory(self, tmp_path, short_root):
        # The socket runs bash on request: a directory anyone can enter is a
        # directory anyone can plant their own listener in.
        path = socket_path_for(tmp_path / "home", local_root=short_root)
        path.parent.chmod(0o755)
        with pytest.raises(CoreProcessError, match="mode"):
            socket_path_for(tmp_path / "home", local_root=short_root)

    def test_refuses_a_directory_owned_by_someone_else(
        self, tmp_path, short_root, monkeypatch
    ):
        socket_path_for(tmp_path / "home", local_root=short_root)
        mine = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: mine + 1)
        with pytest.raises(CoreProcessError, match="owned by uid"):
            socket_path_for(tmp_path / "home", local_root=short_root)

    def test_refuses_a_symlink(self, tmp_path, short_root):
        path = socket_path_for(tmp_path / "home", local_root=short_root)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir(mode=0o700)
        path.parent.rmdir()
        path.parent.symlink_to(elsewhere)
        with pytest.raises(CoreProcessError, match="not a directory"):
            socket_path_for(tmp_path / "home", local_root=short_root)

    def test_refuses_a_plain_file(self, tmp_path, short_root):
        path = socket_path_for(tmp_path / "home", local_root=short_root)
        path.parent.rmdir()
        path.parent.write_text("not a directory")
        with pytest.raises(CoreProcessError, match="not a directory"):
            socket_path_for(tmp_path / "home", local_root=short_root)


# --------------------------------------------------------------- stale socket


class TestStaleSocket:
    def test_a_missing_socket_is_not_live(self, tmp_path):
        assert socket_is_live(tmp_path / "core.sock") is False
        assert clear_stale_socket(tmp_path / "core.sock") is False

    def test_a_listening_socket_is_live_and_kept(self, tmp_path):
        path = tmp_path / "core.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(path))
            server.listen(1)
            assert socket_is_live(path) is True
            with pytest.raises(CoreProcessError, match="already listening"):
                clear_stale_socket(path)
            assert path.exists()
        finally:
            server.close()

    def test_a_leftover_socket_is_removed(self, tmp_path):
        path = tmp_path / "core.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)
        server.close()  # what a crashed core leaves: the file, no listener
        assert path.exists()
        assert socket_is_live(path) is False
        assert clear_stale_socket(path) is True
        assert not path.exists()


# ----------------------------------------------------------------- supervisor


class TestDefaultArgv:
    def test_runs_the_serve_entry_point_with_our_interpreter(self):
        argv = default_core_argv(Path("/run/hpca-abc/core.sock"))
        assert argv == [
            sys.executable,
            "-m",
            "hpca",
            "--serve",
            "--socket",
            "/run/hpca-abc/core.sock",
        ]


class TestHandshake:
    async def test_start_returns_the_socket_path(self, app_dir):
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(app_dir=app_dir, argv0=ready_stub(want))
        try:
            assert await sup.start() == want
            assert sup.socket_path == want
            assert sup.pid is not None
            assert sup.returncode is None
        finally:
            await sup.stop(timeout_s=2)

    async def test_the_child_gets_its_own_session(self, app_dir):
        # Without start_new_session the core sits in the terminal's foreground
        # process group, and Ctrl-C in the UI kills it mid-turn.
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(app_dir=app_dir, argv0=ready_stub(want))
        try:
            await sup.start()
            assert os.getsid(sup.pid) == sup.pid
            assert os.getsid(sup.pid) != os.getsid(os.getpid())
        finally:
            await sup.stop(timeout_s=2)

    async def test_stderr_lands_in_the_log_not_the_terminal(self, app_dir, tmp_path):
        want = socket_path_for(app_dir)
        log = tmp_path / "core.log"
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=noisy_stub(want, "a traceback"), log_path=log
        )
        try:
            await sup.start()
        finally:
            await sup.stop(timeout_s=2)
        assert "a traceback" in log.read_text()

    def test_the_log_defaults_to_the_app_dir(self, app_dir):
        assert CoreSupervisor(app_dir=app_dir).log_path == app_dir / "core.log"

    async def test_a_silent_core_times_out_and_is_killed(self, app_dir):
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=silent_stub(), ready_timeout_s=0.05
        )
        with pytest.raises(CoreProcessError, match="did not report ready"):
            await sup.start()
        # The point of the timeout is that nothing is left running.
        assert sup.returncode == -signal.SIGTERM

    async def test_the_error_points_at_the_log(self, app_dir):
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=silent_stub(), ready_timeout_s=0.05
        )
        with pytest.raises(CoreProcessError) as caught:
            await sup.start()
        assert str(sup.log_path) in str(caught.value)

    async def test_garbage_instead_of_json_raises(self, app_dir):
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=line_stub("Traceback (most recent call last)")
        )
        with pytest.raises(CoreProcessError, match="not JSON"):
            await sup.start()
        assert sup.returncode is not None

    async def test_ready_false_raises(self, app_dir):
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=line_stub('{"ready": false, "version": 1}')
        )
        with pytest.raises(CoreProcessError, match="did not report ready"):
            await sup.start()

    async def test_a_version_mismatch_raises(self, app_dir):
        # The check exists for a stale core left by a crashed run.
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=ready_stub(want, version=HANDSHAKE_VERSION + 1)
        )
        with pytest.raises(CoreProcessError, match="version"):
            await sup.start()
        assert sup.returncode is not None

    async def test_a_core_that_exits_immediately_raises(self, app_dir):
        sup = CoreSupervisor(app_dir=app_dir, argv0=stub("raise SystemExit(3)"))
        with pytest.raises(CoreProcessError, match="exited without reporting ready"):
            await sup.start()

    async def test_a_missing_executable_raises(self, app_dir):
        sup = CoreSupervisor(app_dir=app_dir, argv0=["/nonexistent/hpca-core"])
        with pytest.raises(CoreProcessError, match="could not start the core"):
            await sup.start()
        assert await sup.stop(timeout_s=1) is None

    async def test_a_second_start_is_refused(self, app_dir):
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(app_dir=app_dir, argv0=ready_stub(want))
        try:
            await sup.start()
            with pytest.raises(CoreProcessError, match="already started"):
                await sup.start()
        finally:
            await sup.stop(timeout_s=2)


class TestStop:
    async def test_reaps_a_well_behaved_child(self, app_dir):
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(app_dir=app_dir, argv0=ready_stub(want))
        await sup.start()
        assert await sup.stop(timeout_s=2) == -signal.SIGTERM
        assert sup.returncode == -signal.SIGTERM

    async def test_kills_a_child_that_ignores_sigterm(self, app_dir):
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(app_dir=app_dir, argv0=deaf_stub(want))
        await sup.start()
        assert await sup.stop(timeout_s=0.05) == -signal.SIGKILL

    async def test_is_idempotent(self, app_dir):
        want = socket_path_for(app_dir)
        sup = CoreSupervisor(app_dir=app_dir, argv0=ready_stub(want))
        await sup.start()
        first = await sup.stop(timeout_s=2)
        second = await sup.stop(timeout_s=2)
        assert first == second == -signal.SIGTERM

    async def test_without_a_start_returns_none(self, app_dir):
        assert await CoreSupervisor(app_dir=app_dir).stop() is None

    async def test_after_a_failed_start_is_quiet(self, app_dir):
        sup = CoreSupervisor(
            app_dir=app_dir, argv0=silent_stub(), ready_timeout_s=0.05
        )
        with pytest.raises(CoreProcessError):
            await sup.start()
        assert await sup.stop(timeout_s=2) == -signal.SIGTERM


# --------------------------------------------------------------- idle shutdown


def idle_flag():
    """An IdleShutdown handler plus the event it sets."""
    fired = asyncio.Event()

    async def on_idle() -> None:
        fired.set()

    return fired, on_idle


class TestIdleShutdown:
    async def test_fires_after_the_grace_window(self):
        fired, on_idle = idle_flag()
        idle = IdleShutdown(0.05, on_idle)
        idle.client_connected()
        idle.client_disconnected()
        await asyncio.wait_for(fired.wait(), 2)

    async def test_does_not_fire_while_a_client_is_attached(self):
        fired, on_idle = idle_flag()
        idle = IdleShutdown(0.05, on_idle)
        idle.client_connected()
        await asyncio.sleep(0.15)
        assert not fired.is_set()
        assert not idle.pending
        idle.cancel()

    async def test_a_reconnect_inside_the_window_cancels_it(self):
        fired, on_idle = idle_flag()
        idle = IdleShutdown(0.1, on_idle)
        idle.client_connected()
        idle.client_disconnected()
        await asyncio.sleep(0.02)
        idle.client_connected()  # the UI restarted in time
        await asyncio.sleep(0.2)
        assert not fired.is_set()
        assert not idle.pending
        idle.cancel()

    async def test_the_last_client_leaving_is_what_arms_it(self):
        fired, on_idle = idle_flag()
        idle = IdleShutdown(0.05, on_idle)
        idle.client_connected()
        idle.client_connected()
        idle.client_disconnected()
        assert not idle.pending
        idle.client_disconnected()
        assert idle.pending
        await asyncio.wait_for(fired.wait(), 2)

    async def test_cancel_stops_a_running_window(self):
        fired, on_idle = idle_flag()
        idle = IdleShutdown(0.05, on_idle)
        idle.client_disconnected()
        assert idle.pending
        idle.cancel()
        assert not idle.pending
        await asyncio.sleep(0.15)
        assert not fired.is_set()

    async def test_churn_does_not_leak_timers(self):
        fired, on_idle = idle_flag()
        idle = IdleShutdown(30.0, on_idle)  # long enough never to fire here
        before = len(asyncio.all_tasks())
        for _ in range(20):
            idle.client_connected()
            idle.client_disconnected()
        await asyncio.sleep(0.01)  # let the cancellations settle
        live = [task for task in asyncio.all_tasks() if not task.done()]
        assert len(live) <= before + 1  # this test's own task, plus one timer
        idle.cancel()
        assert not fired.is_set()

    async def test_extra_disconnects_do_not_disarm_it(self):
        # A connection can be reported closed twice; a negative count would
        # leave the core running with nobody attached.
        fired, on_idle = idle_flag()
        idle = IdleShutdown(0.05, on_idle)
        idle.client_disconnected()
        idle.client_disconnected()
        idle.client_connected()
        await asyncio.sleep(0.15)
        assert not fired.is_set()
        idle.cancel()

    async def test_a_failing_handler_does_not_escape(self, caplog):
        async def boom() -> None:
            raise RuntimeError("no")

        idle = IdleShutdown(0.05, boom)
        idle.client_disconnected()
        await asyncio.sleep(0.15)
        assert "idle shutdown handler failed" in caplog.text
        assert not idle.pending
