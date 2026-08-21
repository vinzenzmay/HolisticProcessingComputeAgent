"""The event loop: stdin, the connection and SIGWINCH, on one asyncio loop.

    pixi run -e dev python -m hpca.ui.run --demo

The only module that touches a terminal, which is what keeps everything above
it testable without one (specs-ui-replacement.md §3.1). It knows about I/O and
about nothing else: what a key *means* is `app.py`'s business, what an event
means is `client.py`'s, and neither of them has to be async for this to work.

**One loop, three inputs.** `loop.add_reader(fd)` for stdin, an `async for`
over the `Connection` for events, and `loop.add_signal_handler(SIGWINCH)` for
the resize. No threads, and no polling: the synchronous core is called from
callbacks on this loop, so `RowUI.render()` and `RowUI.handle()` are as
untouched as they were under `select`.

**A repaint is caused, not polled.** The old loop woke twice a second whether
or not anything had happened. Here every input sets one `asyncio.Event` and
the loop paints once per wake, so an idle UI costs nothing at all — and the
two things that change with no input, the armed-escape hint and the spinner,
are booked by `RowUI.next_wake()` for exactly when each next differs. A
spinner therefore costs ten wakes a second while a turn is in flight and
nothing whatsoever when none is, which is the same bargain without a timer
belonging to a widget.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import contextlib
import os
import shutil
import signal
import sys
import time

from hpca.ui.keys import PASTE_START, decode
from hpca.ui.screen import Screen

# How long a held-back escape waits for the rest of its sequence before it is
# taken to be the escape key. A pressed escape and the first byte of an arrow
# key are the same byte; what tells them apart is that the terminal already has
# the rest of the arrow buffered and delivers it within microseconds, while a
# person's next keystroke does not. The classic vim/readline figure.
ESC_TIMEOUT = 0.05

# One read. Large enough that a pasted payload arrives in a handful of them and
# small enough to stay one allocation.
READ_SIZE = 65536


def terminal_size(fd: int) -> tuple[int, int]:
    """The geometry to draw at, from the terminal itself where there is one.

    `shutil.get_terminal_size` is the fallback rather than the primary because
    it asks stdout, and the UI's terminal is the one stdin is attached to —
    they differ the moment output is redirected. It does bring the useful half
    of the deal though: `$COLUMNS`/`$LINES`, then 80x24, so a headless run
    still has a size to render at instead of an `OSError`.
    """
    try:
        size = os.get_terminal_size(fd)
    except OSError:
        size = shutil.get_terminal_size()
    if size.columns <= 0 or size.lines <= 0:
        # A pty whose window size was never set answers 0x0, and a frame
        # rendered at that height is no rows at all — a UI that looks hung.
        size = shutil.get_terminal_size()
    return max(1, size.columns), max(1, size.lines)


class Loop:
    """One `RowUI`, one terminal, and — when there is one — one core.

    Constructed with the pieces rather than building them, so a test can hand
    it a pipe for stdin, a `Screen` over a `StringIO` and an
    `InProcessConnection` peer, and drive the whole thing with no terminal at
    all.
    """

    def __init__(
        self,
        ui,
        screen: Screen,
        *,
        client=None,
        conn=None,
        size=None,
    ) -> None:
        self.ui = ui
        self.screen = screen
        self.client = client
        self.conn = conn
        self._size = size or (lambda: terminal_size(screen.fd))
        self.width, self.height = self._size()
        # What `decode` could not name yet, carried to the front of the next
        # read: an escape sequence can straddle a read boundary, and a paste
        # routinely spans dozens of them.
        self._pending = ""
        # A *character* can straddle one too, and one plain decode() per read
        # would turn the halves of a pasted ideograph into two replacement
        # characters. The incremental decoder holds the partial bytes instead.
        self._chars = codecs.getincrementaldecoder("utf-8")("replace")
        # The first frame has no previous frame to diff against, so it is a
        # full paint — the same thing a resize makes of the frame after it.
        self._full = True
        self._wake = asyncio.Event()
        self._stop = False
        # What ended the loop, when what ended it was a bug. Kept rather than
        # raised where it happened: a callback that raises into asyncio's
        # default handler prints a traceback onto a terminal that is still in
        # raw mode and leaves the UI running but wrong.
        self._error: BaseException | None = None
        self.code = 0
        self._esc_timer: asyncio.TimerHandle | None = None
        self._stale_timer: asyncio.TimerHandle | None = None
        self._events_task: asyncio.Task | None = None
        # Whether SIGWINCH is on the loop, and what handler that displaced.
        self._winch = False
        self._winch_was: object = signal.SIG_DFL

    # ------------------------------------------------------------- the loop

    async def run(self) -> int:
        """Paint, wait for something to happen, paint again.

        Everything that can happen — a keypress, an event, a resize, an expiry
        — is a callback on this loop that mutates state and sets `_wake`. The
        body below is therefore the whole schedule, and the order in it is the
        one thing that matters: what the last batch of keys asked for goes on
        the wire *before* the frame that shows it being asked for.
        """
        loop = asyncio.get_running_loop()
        loop.add_reader(self.screen.fd, self._readable)
        self._watch_resize(loop)
        if self.conn is not None:
            self._events_task = asyncio.ensure_future(self._events())
        try:
            while not self._stop:
                self._wake.clear()
                await self._flush()
                self._paint()
                await self._wake.wait()
        finally:
            await self._close(loop)
        if self._error is not None:
            # Out through `drive`'s `with`, so the traceback lands on a
            # terminal that has its cursor and its echo back.
            raise self._error
        return self.code

    def _paint(self) -> None:
        started = time.perf_counter()
        full, self._full = self._full, False
        if full:
            # A resize invalidates the diff baseline outright — it was taken at
            # the old geometry — so the frame after one is painted whole, at a
            # size read again rather than remembered.
            self.width, self.height = self._size()
            self.ui.invalidate()
        self.screen.paint(self.ui.render(self.width, self.height), full=full)
        # Shown on the next frame rather than this one: measuring the frame
        # that reports the measurement would need two passes to say anything
        # true.
        self.ui.frame_ms = (time.perf_counter() - started) * 1000
        self._book_stale_repaint()

    def _book_stale_repaint(self) -> None:
        """Wake again when the frame just painted stops being true.

        The loop has no idle tick to notice on its own, and the alternative —
        painting on a timer in case something expired — is the poll this
        milestone removed.
        """
        if self._stale_timer is not None:
            self._stale_timer.cancel()
            self._stale_timer = None
        delay = self.ui.next_wake()
        if delay is not None:
            self._stale_timer = asyncio.get_running_loop().call_later(
                max(0.0, delay), self._wake.set
            )

    async def _flush(self) -> None:
        if self.client is not None:
            await self.client.flush()

    def stop(self, code: int = 0) -> None:
        self.code = code
        self._stop = True
        self._wake.set()

    def _fail(self, exc: BaseException) -> None:
        """A callback raised. End the loop and carry it out through `run`."""
        self._error = exc
        self.stop(1)

    # ------------------------------------------------------------ the inputs

    def _readable(self) -> None:
        """stdin has bytes. Name what can be named; hold back what cannot."""
        try:
            data = os.read(self.screen.fd, READ_SIZE)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b""
        if not data:
            self.stop()  # the terminal went away
            return
        try:
            keys, self._pending = decode(
                self._pending + self._chars.decode(data)
            )
            self._arm_escape()
            self._dispatch(keys)
        except Exception as e:  # a bug in handle(), not a bad key
            self._fail(e)

    def _arm_escape(self) -> None:
        """Start (or restart) the clock on a sequence that is not finished.

        A held-back escape is only ever resolved by *nothing* following it, so
        it needs a timer rather than a shorter wait: the read that would have
        completed the arrow key is the read that never comes. An unfinished
        paste is not on this clock — it is held until its end marker whatever
        happens, because flushing it as keystrokes is the bug bracketing exists
        to prevent, and one of those keystrokes is the newline that would send
        half the message.
        """
        self._cancel_escape()
        if self._pending and not self._pending.startswith(PASTE_START):
            self._esc_timer = asyncio.get_running_loop().call_later(
                ESC_TIMEOUT, self._escape_timeout
            )

    def _cancel_escape(self) -> None:
        if self._esc_timer is not None:
            self._esc_timer.cancel()
            self._esc_timer = None

    def _escape_timeout(self) -> None:
        """Nothing followed it inside the timeout, so it is all there will be:
        a lone ESC is the escape key, and an unfinished sequence is dropped
        rather than typed."""
        self._esc_timer = None
        try:
            keys, self._pending = decode(self._pending, final=True)
            self._dispatch(keys)
        except Exception as e:
            self._fail(e)

    def _dispatch(self, keys: list[str]) -> None:
        if not keys:
            return  # a partial sequence changed nothing worth a frame
        for key in keys:
            if not self.ui.handle(key, self.width, self.height):
                self.stop()
                return
        self._wake.set()

    async def _events(self) -> None:
        """Frames from the core, until it stops sending.

        Reading and rendering are deliberately separated: this applies as many
        events as have arrived and the loop paints once, so a burst of a
        hundred `chat.append`s costs one frame rather than a hundred.
        """
        try:
            async for env in self.conn:
                self.client.apply(env)
                self._wake.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A frame the client cannot *parse* is already counted and dropped
            # one layer down; getting here means a handler raised, which is a
            # bug and is worth ending the run over — loudly, and with the
            # terminal restored.
            self._fail(e)
            return
        self.stop()  # the core hung up; there is nothing left to draw

    # ------------------------------------------------------------ the resize

    def _watch_resize(self, loop: asyncio.AbstractEventLoop) -> None:
        """SIGWINCH, straight onto the loop.

        This is the self-pipe the old `Resizes` class built by hand: a Python
        signal handler runs between bytecodes and PEP 475 restarts the
        interrupted wait underneath it, so a bare flag would not be noticed
        until something else woke the loop. `add_signal_handler` writes to
        asyncio's own wakeup fd, which is exactly the same trick with the
        callback already bolted on.
        """
        if not hasattr(signal, "SIGWINCH"):  # not a thing on Windows
            return
        try:
            self._winch_was = signal.getsignal(signal.SIGWINCH)
            loop.add_signal_handler(signal.SIGWINCH, self._resized)
            self._winch = True
        except (NotImplementedError, RuntimeError, ValueError):
            # Not the main thread, or a loop that cannot take signals. The
            # frame then follows the terminal at the next repaint instead of
            # immediately, which is a worse resize and not a broken UI.
            self._winch = False

    def _unwatch_resize(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._winch:
            return
        self._winch = False
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(signal.SIGWINCH)
        # `remove_signal_handler` installs SIG_DFL rather than whatever was
        # there before, and this loop is a guest in someone else's process —
        # the test suite's, and one day a `--serve` parent's. A handler set
        # from outside Python reads back as None and cannot be put back.
        if self._winch_was is not None:
            with contextlib.suppress(ValueError, TypeError, OSError):
                signal.signal(signal.SIGWINCH, self._winch_was)

    def _resized(self) -> None:
        self._full = True
        self._wake.set()

    # ----------------------------------------------------------- the teardown

    async def _close(self, loop: asyncio.AbstractEventLoop) -> None:
        """Give back everything registered on the loop, however this ended.

        In a `finally`, because the alternative to a loop that unregisters
        itself after an exception is a reader callback firing on a closed
        terminal for as long as the process lives.
        """
        with contextlib.suppress(OSError, ValueError):
            loop.remove_reader(self.screen.fd)
        self._unwatch_resize(loop)
        self._cancel_escape()
        if self._stale_timer is not None:
            self._stale_timer.cancel()
            self._stale_timer = None
        if self._events_task is not None:
            self._events_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._events_task
            self._events_task = None


async def drive(ui, *, client=None, conn=None, screen: Screen | None = None) -> int:
    """Set the terminal up, run the loop, and put the terminal back.

    The `with` is the whole exception-safety story: a traceback out of the loop
    unwinds through `Screen.__exit__`, so it reaches the user on a terminal
    that has its cursor, its echo and its own screen back. An injected screen
    goes through the same `with` rather than around it — a test that skipped it
    would be testing a shutdown path nobody runs.
    """
    with (screen if screen is not None else Screen()) as scr:
        return await Loop(ui, scr, client=client, conn=conn).run()


def main(argv: list[str] | None = None) -> int:
    """`python -m hpca.ui.run --demo`: the UI over synthetic content.

    The demo's loopback core is the test double for everything the real entry
    point (`hpca --new-ui`, see `hpca/__main__.py`) does with an
    `AgentService`, and it stays that way: it needs no database, no backend and
    no event loop of its own, so it is the fastest way to look at a frame.
    """
    parser = argparse.ArgumentParser(
        description="Row-oriented HPCA UI (no Textual)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="draw the UI over synthetic content, with no backend",
    )
    parser.add_argument("--chat", type=int, default=400, help="chat entries")
    parser.add_argument("--sessions", type=int, default=14)
    parser.add_argument("--watchers", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.demo:
        # The real core is stood up by `python -m hpca --new-ui`, which owns
        # the databases and the shutdown order; saying so beats standing up a
        # second, subtly different startup path here.
        print(
            "hpca.ui.run only knows --demo; the real UI is `hpca --new-ui`.",
            file=sys.stderr,
        )
        return 2

    if not sys.stdin.isatty():
        print("the row UI needs a terminal.", file=sys.stderr)
        return 2

    from hpca.ui.demo import build

    ui = build(args.chat, args.sessions, args.watchers)
    return asyncio.run(drive(ui))


if __name__ == "__main__":
    raise SystemExit(main())
