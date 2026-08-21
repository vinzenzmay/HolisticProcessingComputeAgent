"""Tests for hpca.ui.run: the one asyncio loop that owns stdin, the connection
and SIGWINCH.

Hermetic, and it costs nothing to be: `Screen` takes a file descriptor and an
output stream, so a pipe stands in for the terminal and a `StringIO` collects
the frames. The core on the other end is an `InProcessConnection` peer, exactly
as `test_ui_client.py` drives the client.

What is checked here is the *shell*: that a byte on stdin reaches `handle`,
that an event repaints, that a resize forces a whole frame, that the escape
timeout still tells a pressed escape from half an arrow key, and that the
terminal comes back however the loop ends.
"""

from __future__ import annotations

import asyncio
import io
import os
import signal

import pytest

from hpca import protocol
from hpca.transport import InProcessConnection
from hpca.ui.app import RowUI
from hpca.ui.client import UIClient
from hpca.ui.keys import PASTE_END, PASTE_START
from hpca.ui.run import Loop, drive, terminal_size
from hpca.ui.screen import EXIT_MODES, Screen
from tests.ui_harness import Peer

SIZE = (80, 24)


class FakeUI:
    """Everything the loop asks of a `RowUI`, and nothing else.

    Deliberately not the real one: the loop's job is to deliver keys and
    events and to decide when to paint, and a double makes each of those a
    one-line assertion instead of a search through a frame.
    """

    def __init__(self) -> None:
        self.keys: list[tuple[str, int, int]] = []
        self.frames: list[tuple[list[str], bool]] = []
        self.invalidated = 0
        self.frame_ms = 0.0
        self.text = "hello"
        self.stale: float | None = None

    def render(self, width: int, height: int) -> list[str]:
        return [f"{self.text} {width}x{height}"]

    def handle(self, key: str, width: int, height: int) -> bool:
        self.keys.append((key, width, height))
        return key != "quit"

    def invalidate(self) -> None:
        self.invalidated += 1

    def next_wake(self) -> float | None:
        return self.stale


class Harness:
    """A loop running on a pipe, with a way to wait for the next frame.

    `settle` counts frames rather than sleeping a fixed time: every path into
    the loop ends in exactly one paint, so "the frame after this keystroke" is
    a thing the test can name.
    """

    def __init__(self, ui, *, client=None, conn=None, size=SIZE) -> None:
        self.read_fd, self.write_fd = os.pipe()
        self.out = io.StringIO()
        self.ui = ui
        self.size = size
        self.screen = Screen(fd=self.read_fd, out=self.out)
        self.painted: list[tuple[list[str], bool]] = []
        paint = self.screen.paint

        def record(lines, *, full=False):
            self.painted.append((list(lines), full))
            paint(lines, full=full)

        self.screen.paint = record
        self.loop = Loop(
            ui, self.screen, client=client, conn=conn, size=lambda: self.size
        )
        self.task: asyncio.Task | None = None

    async def _run(self) -> int:
        with self.screen:
            return await self.loop.run()

    async def start(self) -> "Harness":
        self.task = asyncio.ensure_future(self._run())
        await self.settle(1)
        return self

    async def settle(self, extra: int = 1, timeout: float = 2.0) -> None:
        """Wait until `extra` more frames have been painted."""
        target = len(self.painted) + extra
        async with asyncio.timeout(timeout):
            while len(self.painted) < target:
                await asyncio.sleep(0.002)
                if self.task is not None and self.task.done():
                    self.task.result()  # re-raise whatever ended the loop
                    return

    async def press(self, data: bytes, frames: int = 1) -> None:
        os.write(self.write_fd, data)
        if frames:
            await self.settle(frames)

    @property
    def frame(self) -> list[str]:
        return self.painted[-1][0]

    async def stop(self) -> int:
        os.write(self.write_fd, b"\x03")  # ^C is the quit key in raw mode
        code = await asyncio.wait_for(self.task, 2.0)
        self.close()
        return code

    def close(self) -> None:
        for fd in (self.read_fd, self.write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.fixture
async def harness():
    made: list[Harness] = []

    async def build(ui=None, **kw):
        h = Harness(ui if ui is not None else FakeUI(), **kw)
        made.append(h)
        return await h.start()

    yield build
    for h in made:
        if h.task is not None and not h.task.done():
            h.task.cancel()
            with pytest.raises((asyncio.CancelledError, Exception)):
                await h.task
        h.close()


# ------------------------------------------------------------------ the frame


class TestPainting:
    async def test_the_first_frame_is_painted_whole(self, harness):
        h = await harness()
        assert h.painted[0][1] is True  # nothing to diff against

    async def test_and_only_the_first(self, harness):
        h = await harness()
        await h.press(b"x")
        assert h.painted[-1][1] is False

    async def test_it_draws_at_the_terminal_size(self, harness):
        h = await harness()
        assert h.frame == ["hello 80x24"]

    async def test_an_idle_loop_paints_once_and_stops(self, harness):
        # The point of the milestone: no 0.5s poll, so nothing happening costs
        # nothing at all.
        h = await harness()
        painted = len(h.painted)
        await asyncio.sleep(0.15)
        assert len(h.painted) == painted

    async def test_the_frame_time_is_reported_on_the_next_frame(self, harness):
        h = await harness()
        await h.press(b"x")
        assert h.ui.frame_ms > 0


# ------------------------------------------------------------------- the keys


class TestKeys:
    async def test_a_key_reaches_handle(self, harness):
        h = await harness()
        await h.press(b"x")
        assert h.ui.keys == [("x", 80, 24)]

    async def test_it_carries_the_geometry_the_frame_was_drawn_at(self, harness):
        h = await harness()
        await h.press(b"x")
        assert h.ui.keys[0][1:] == (80, 24)

    async def test_a_key_repaints(self, harness):
        h = await harness()
        painted = len(h.painted)
        h.ui.text = "changed"
        await h.press(b"x")
        assert len(h.painted) > painted
        assert h.frame == ["changed 80x24"]

    async def test_a_named_key_arrives_named(self, harness):
        h = await harness()
        await h.press(b"\x1b[A")
        assert [k for k, _, _ in h.ui.keys] == ["up"]

    async def test_a_multibyte_character_split_across_reads_survives(self, harness):
        # The incremental decoder's whole job: one plain decode per read would
        # turn the halves of a pasted ideograph into two replacement chars.
        h = await harness()
        payload = "漢".encode()
        await h.press(payload[:1], frames=0)
        await asyncio.sleep(0.02)
        await h.press(payload[1:])
        assert [k for k, _, _ in h.ui.keys] == ["漢"]

    async def test_a_paste_arrives_as_one_key(self, harness):
        h = await harness()
        await h.press(f"{PASTE_START}two\nlines{PASTE_END}".encode())
        assert [k for k, _, _ in h.ui.keys] == ["paste:two\nlines"]

    async def test_an_unfinished_paste_is_not_flushed_by_the_escape_timeout(
        self, harness, monkeypatch
    ):
        # Bracketing exists to stop a pasted newline being read as Return;
        # flushing half a paste as keystrokes would send half the message.
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 0.01)
        h = await harness()
        await h.press(f"{PASTE_START}half a paste".encode(), frames=0)
        await asyncio.sleep(0.05)
        assert h.ui.keys == []

    async def test_a_key_that_says_stop_ends_the_loop(self, harness):
        h = await harness()
        assert await h.stop() == 0


# --------------------------------------------------------- the escape timeout


class TestEscapeTimeout:
    async def test_a_lone_escape_becomes_the_escape_key(self, harness, monkeypatch):
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 0.01)
        h = await harness()
        await h.press(b"\x1b")
        assert [k for k, _, _ in h.ui.keys] == ["esc"]

    async def test_but_not_before_the_timeout_is_up(self, harness, monkeypatch):
        # Held, not named: what separates a pressed escape from an arrow key is
        # that the rest of the arrow is already in the terminal's buffer.
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 5.0)
        h = await harness()
        await h.press(b"\x1b", frames=0)
        await asyncio.sleep(0.05)
        assert h.ui.keys == []

    async def test_an_arrow_split_across_reads_is_still_an_arrow(
        self, harness, monkeypatch
    ):
        # The bug this exists to prevent: half a cursor movement decoding as
        # the escape key, which arms the stop gesture.
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 5.0)
        h = await harness()
        await h.press(b"\x1b", frames=0)
        await asyncio.sleep(0.02)
        await h.press(b"[A")
        assert [k for k, _, _ in h.ui.keys] == ["up"]

    async def test_the_timeout_does_not_repaint_when_it_names_nothing(
        self, harness, monkeypatch
    ):
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 0.01)
        h = await harness()
        painted = len(h.painted)
        await h.press(b"\x1b[", frames=0)  # a CSI that never finished
        await asyncio.sleep(0.05)
        assert h.ui.keys == []
        assert len(h.painted) == painted

    async def test_the_stop_gesture_survives_the_move_to_asyncio(
        self, harness, monkeypatch
    ):
        # End to end through the real UI: two escapes inside the window ask the
        # core to stop the turn, and half an arrow key does not.
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 0.01)
        ui = RowUI()
        ui.sessions = []
        h = await harness(ui)
        await h.press(b"\x1b")
        await h.press(b"\x1b")
        assert "stopped the turn" in "".join(h.frame)

    async def test_and_half_an_arrow_key_never_arms_it(self, harness, monkeypatch):
        monkeypatch.setattr("hpca.ui.run.ESC_TIMEOUT", 5.0)
        ui = RowUI()
        h = await harness(ui)
        await h.press(b"\x1b", frames=0)
        await asyncio.sleep(0.02)
        await h.press(b"[A")
        assert ui._esc_armed_at is None


class TestStaleFrames:
    async def test_a_frame_that_expires_books_its_own_repaint(self, harness):
        # The armed-escape hint is the one thing that changes with no input;
        # the loop asks the UI when, rather than polling in case.
        ui = FakeUI()
        ui.stale = 0.02
        h = await harness(ui)
        painted = len(h.painted)
        ui.stale = None
        await h.settle(1)
        assert len(h.painted) > painted

    async def test_the_armed_escape_hint_expires_on_its_own(
        self, harness, monkeypatch
    ):
        # The one thing on screen that goes stale with no input at all. Under
        # the old loop the 0.5s poll wiped it; here `next_wake` books the
        # repaint that does, and this is the test that says so.
        monkeypatch.setattr("hpca.ui.app.ESC_STOP_WINDOW", 0.05)
        h = await harness(RowUI())
        await h.press(b"\x1b")  # named "esc" once the escape timeout is up
        assert "esc again to stop" in "".join(h.frame)
        await h.settle(1)
        assert "esc again to stop" not in "".join(h.frame)


# ----------------------------------------------------------------- the events


class TestEvents:
    @pytest.fixture
    async def wired(self, harness):
        """A real client on one end of a real connection, a scripted core on
        the other — the same pair `test_ui_client.py` uses, with the loop in
        between instead of a hand-driven `flush`."""
        ui = RowUI()
        ui_end, core_end = InProcessConnection.pair()
        client = UIClient(ui, ui_end)
        peer = Peer(core_end)
        listening = asyncio.ensure_future(peer.listen())
        h = await harness(ui, client=client, conn=ui_end)
        try:
            yield h, client, peer
        finally:
            listening.cancel()
            await core_end.close()

    async def test_an_event_repaints(self, wired):
        h, _, peer = wired
        painted = len(h.painted)
        await peer.conn.send(protocol.Notify(text="the core said something"))
        await h.settle(1)
        assert len(h.painted) > painted
        assert "the core said something" in "".join(h.frame)

    async def test_the_handshake_reaches_the_client(self, wired):
        h, client, peer = wired
        await peer.conn.send(protocol.Hello(profile="hpc"))
        await h.settle(1)
        assert client.hello is not None

    async def test_and_what_it_asks_for_goes_back_on_the_wire(self, wired):
        # `hello` makes the client ask for the session list; the loop is what
        # puts it on the wire, before the frame that follows it.
        h, _, peer = wired
        await peer.conn.send(protocol.Hello(profile="hpc"))
        await h.settle(1)
        await _until(lambda: peer.took(protocol.SessionList))

    async def test_a_key_becomes_a_command(self, wired):
        h, _, peer = wired
        await peer.conn.send(
            protocol.SessionRows(
                rows=[protocol.SessionRow(session_id="s1", title="one")]
            )
        )
        await h.settle(1)
        await _until(lambda: peer.took(protocol.SessionOpen))
        peer.clear()
        # `i` from the chat row moves into the message box; then type and send.
        await h.press(b"ihi\r")
        await _until(lambda: peer.took(protocol.TurnSubmit))
        assert peer.last(protocol.TurnSubmit).text == "hi"

    async def test_a_burst_of_events_costs_one_frame(self, wired):
        # Reading and rendering are separated on purpose: a hundred appends
        # must not be a hundred frames.
        h, _, peer = wired
        painted = len(h.painted)
        for i in range(20):
            await peer.conn.send(protocol.Notify(text=f"toast {i}"))
        await h.settle(1)
        assert len(h.painted) - painted < 20

    async def test_the_core_hanging_up_ends_the_loop(self, wired):
        h, _, peer = wired
        await peer.conn.close()
        assert await asyncio.wait_for(h.task, 2.0) == 0


async def _until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.002)


# ----------------------------------------------------------------- the resize


class TestResize:
    async def test_a_sigwinch_forces_a_full_paint(self, harness):
        h = await harness()
        h.size = (100, 30)
        os.kill(os.getpid(), signal.SIGWINCH)
        await h.settle(1)
        assert h.painted[-1][1] is True
        assert h.frame == ["hello 100x30"]

    async def test_and_invalidates_what_was_measured_at_the_old_size(self, harness):
        h = await harness()
        invalidated = h.ui.invalidated
        os.kill(os.getpid(), signal.SIGWINCH)
        await h.settle(1)
        assert h.ui.invalidated > invalidated

    async def test_the_handler_is_given_back_at_the_end(self, harness):
        before = signal.getsignal(signal.SIGWINCH)
        h = await harness()
        await h.stop()
        assert signal.getsignal(signal.SIGWINCH) is before


# --------------------------------------------------------------- the teardown


class TestTeardown:
    async def test_a_traceback_still_restores_the_terminal(self):
        # The guarantee: a crash must never leave the user in the alternate
        # screen with no cursor and no echo.
        class Exploding(FakeUI):
            def render(self, width, height):
                raise RuntimeError("boom")

        read, write = os.pipe()
        out = io.StringIO()
        screen = Screen(fd=read, out=out)
        try:
            with pytest.raises(RuntimeError):
                await drive(Exploding(), screen=screen)
        finally:
            os.close(read), os.close(write)
        assert EXIT_MODES in out.getvalue()

    async def test_and_takes_its_reader_off_the_loop(self):
        # A callback left registered on a closed terminal fires for as long as
        # the process lives.
        class Exploding(FakeUI):
            def render(self, width, height):
                raise RuntimeError("boom")

        read, write = os.pipe()
        screen = Screen(fd=read, out=io.StringIO())
        with pytest.raises(RuntimeError):
            await drive(Exploding(), screen=screen)
        # Registering it again is only possible because the loop gave it back.
        asyncio.get_running_loop().add_reader(read, lambda: None)
        asyncio.get_running_loop().remove_reader(read)
        os.close(read), os.close(write)

    async def test_a_key_handler_that_raises_comes_out_of_run(self, harness):
        # Not into asyncio's default exception handler, which would print a
        # traceback onto a terminal still in raw mode and leave a UI that is
        # running and wrong.
        class Exploding(FakeUI):
            def handle(self, key, width, height):
                raise RuntimeError("boom")

        h = await harness(Exploding())
        os.write(h.write_fd, b"x")
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(h.task, 2.0)
        assert EXIT_MODES in h.out.getvalue()

    async def test_stdin_closing_ends_the_loop(self, harness):
        h = await harness()
        os.close(h.write_fd)
        assert await asyncio.wait_for(h.task, 2.0) == 0
        h.read_fd = -1  # closed by the fixture's `close`, not twice


class TestTerminalSize:
    def test_a_pipe_falls_back_to_a_usable_size(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "132")
        monkeypatch.setenv("LINES", "43")
        read, write = os.pipe()
        try:
            assert terminal_size(read) == (132, 43)
        finally:
            os.close(read), os.close(write)


def test_the_demo_still_only_knows_demo(capsys):
    from hpca.ui.run import main

    assert main([]) == 2
    assert "--demo" in capsys.readouterr().err
