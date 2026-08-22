"""The raw terminal: alternate screen, differential repaint, terminal modes.

The only module that writes to stdout, which is what keeps the rest of the
package testable by calling ``render`` and comparing strings.

Resizes used to live here too, as a self-pipe a ``select`` could watch. M3 put
the loop on asyncio and ``loop.add_signal_handler(SIGWINCH, ...)`` *is* that
pipe with a callback bolted on, so the class collapsed into the one line in
``run.py`` its own docstring predicted.
"""

from __future__ import annotations

import contextlib
import os
import sys
import termios
import tty

from hpca.ui.ansi import ESC

# ?1049h alternate screen, ?25l hide the cursor, ?7l autowrap off, ?2004h
# bracketed paste.
#
# Autowrap matters: every line is padded to the full width and painted by
# absolute cursor address, and a character landing in the last cell of the last
# row would otherwise leave a wrap pending that scrolls the whole frame as soon
# as the next one is written.
#
# Bracketed paste matters for the same class of reason: without it a pasted
# newline is indistinguishable from a pressed Return, so pasting three lines
# into the message box sends the first one.
PASTE_ON = f"{ESC}[?2004h"
PASTE_OFF = f"{ESC}[?2004l"
ENTER_MODES = f"{ESC}[?1049h{ESC}[?25l{ESC}[?7l{PASTE_ON}{ESC}[2J"
# Reset in the opposite order, and every mode that was set: bracketed paste
# left on outlives the process, and every shell prompt afterwards would be
# handed ESC[200~ around anything pasted into it.
EXIT_MODES = f"{PASTE_OFF}{ESC}[?7h{ESC}[?25h{ESC}[?1049l"


class Screen:
    """Raw terminal, alternate screen, differential repaint.

    ``fd`` and ``out`` are injectable so a test can drive a real ``Screen``
    over a pipe: the differential paint and the mode strings are the two things
    worth checking, and neither needs a terminal to be true. A file descriptor
    that is not a tty simply keeps its line discipline — there is none to put
    into raw mode — which is also what makes ``python -m hpca.ui.run`` fail
    with a message rather than a ``termios.error`` when stdin is a pipe.
    """

    def __init__(self, fd: int | None = None, out=None) -> None:
        self.fd = sys.stdin.fileno() if fd is None else fd
        self._out = out
        self._saved: list | None = None
        self._prev: list[str] = []

    def write(self, text: str) -> None:
        """Straight to the terminal, no diff, no frame.

        Used for what is said *outside* a frame — the wait at the end of
        ``ui/boot.py``'s shutdown — which is why it flushes every time.
        """
        stream = sys.stdout if self._out is None else self._out
        stream.write(text)
        stream.flush()

    def __enter__(self) -> "Screen":
        if os.isatty(self.fd):
            self._saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        self.write(ENTER_MODES)
        return self

    def __exit__(self, *exc) -> None:
        """Never conditional on how the loop ended.

        A traceback out of the loop must not leave the user in the alternate
        screen with no cursor and no echo, so this is a context manager rather
        than a pair of calls, and it puts the modes back before it restores the
        line discipline — in the opposite order to ``__enter__``.
        """
        self.write(EXIT_MODES)
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    @contextlib.contextmanager
    def suspended(self):
        """Hand the terminal back for the length of the block (§4.3 item 37).

        What Textual spelled `app.suspend()`: out of the alternate screen, out
        of raw mode, cursor and echo and autowrap restored — because the thing
        that runs in the block is a full-screen program of its own and a
        `$EDITOR` started in raw mode is an editor with no line discipline,
        no visible cursor and a screen it shares with a frame it cannot see.

        The same `try`/`finally` shape as `__enter__`/`__exit__`, and for the
        same reason: an editor that segfaults, a command that does not exist,
        a `KeyboardInterrupt` in between — none of them may leave the user in
        the alternate screen with no echo. Coming back also drops the diff
        baseline, since whatever ran in here owned the screen and the previous
        frame describes something that is no longer on it.
        """
        self.write(EXIT_MODES)
        saved, self._saved = self._saved, None
        try:
            if saved is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, saved)
            yield
        finally:
            if saved is not None and os.isatty(self.fd):
                tty.setraw(self.fd)
            self._saved = saved
            self.write(ENTER_MODES)
            self._prev = []  # nothing on this screen is ours any more

    def paint(self, lines: list[str], *, full: bool = False) -> None:
        out = [f"{ESC}[?2026h"]  # begin synchronised update
        if full:
            out.append(f"{ESC}[2J")
            self._prev = []
        for row, line in enumerate(lines):
            if not full and row < len(self._prev) and self._prev[row] == line:
                continue
            out.append(f"{ESC}[{row + 1};1H{ESC}[2K{line}")
        out.append(f"{ESC}[?2026l")
        self.write("".join(out))
        self._prev = list(lines)
