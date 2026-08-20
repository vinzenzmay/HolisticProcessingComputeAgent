"""The raw terminal: alternate screen, differential repaint, resize.

The only module that writes to stdout, which is what keeps the rest of the
package testable by calling ``render`` and comparing strings.
"""

from __future__ import annotations

import os
import signal
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


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class Resizes:
    """SIGWINCH turned into something a poll can wait on.

    The trap this avoids is the well-known one. A Python signal handler runs
    between bytecodes, and since PEP 475 the interrupted ``select`` is
    restarted underneath it, so setting a flag in the handler does not wake the
    loop — the resize is not noticed until the poll times out, and until then
    the frame is drawn at the old geometry.

    The fix in both worlds is a self-pipe the poll can watch: the C-level
    handler writes a byte, ``select`` returns, and the Python handler's flag is
    read. ``asyncio``'s ``loop.add_signal_handler`` is exactly this pipe with a
    callback bolted on, so when M2 moves run.py onto an asyncio loop, this
    class collapses into ``loop.add_signal_handler(SIGWINCH, ...)`` setting the
    same flag and nothing else about the loop changes.

    Starts already pending: the first frame has no previous frame to diff
    against, so it must be a full paint anyway.
    """

    def __init__(self) -> None:
        self._pending = True
        self._read = -1
        self._write = -1
        self._previous = None
        self._previous_fd = -1

    @property
    def fd(self) -> int:
        """The read end, for the loop's select list."""
        return self._read

    def __enter__(self) -> "Resizes":
        self._read, self._write = os.pipe()
        os.set_blocking(self._read, False)
        os.set_blocking(self._write, False)
        # Every signal writes its number here, which is all we want it for; the
        # flag below says which one it was.
        self._previous_fd = signal.set_wakeup_fd(
            self._write, warn_on_full_buffer=False
        )
        if hasattr(signal, "SIGWINCH"):  # not a thing on Windows
            self._previous = signal.getsignal(signal.SIGWINCH)
            signal.signal(signal.SIGWINCH, self._caught)
        return self

    def __exit__(self, *exc) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGWINCH, self._previous)
            self._previous = None
        signal.set_wakeup_fd(self._previous_fd)
        for fd in (self._read, self._write):
            if fd >= 0:
                os.close(fd)
        self._read = self._write = -1

    def _caught(self, *_) -> None:
        self._pending = True

    def taken(self) -> bool:
        """Whether the terminal changed size since this was last asked.

        Draining is part of asking: the pipe exists to wake the poll, and a
        byte left in it would wake every subsequent poll immediately.
        """
        if self._read >= 0:
            try:
                while os.read(self._read, 4096):
                    pass
            except BlockingIOError:
                pass
        was, self._pending = self._pending, False
        return was


class Screen:
    """Raw terminal, alternate screen, differential repaint."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._saved: list | None = None
        self._prev: list[str] = []

    def __enter__(self) -> "Screen":
        self._saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        _write(ENTER_MODES)
        return self

    def __exit__(self, *exc) -> None:
        _write(EXIT_MODES)
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

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
        _write("".join(out))
        self._prev = list(lines)
