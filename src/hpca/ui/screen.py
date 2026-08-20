"""The raw terminal: alternate screen, differential repaint.

The only module that writes to stdout, which is what keeps the rest of the
package testable by calling ``render`` and comparing strings.
"""

from __future__ import annotations

import sys
import termios
import tty

from hpca.ui.ansi import ESC


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class Screen:
    """Raw terminal, alternate screen, differential repaint."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._saved: list | None = None
        self._prev: list[str] = []

    def __enter__(self) -> "Screen":
        self._saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        # ?1049h alternate screen, ?25l hide the cursor, ?7l autowrap off.
        # Autowrap matters: every line is padded to the full width and painted
        # by absolute cursor address, and a character landing in the last cell
        # of the last row would otherwise leave a wrap pending that scrolls the
        # whole frame as soon as the next one is written.
        _write(f"{ESC}[?1049h{ESC}[?25l{ESC}[?7l{ESC}[2J")
        return self

    def __exit__(self, *exc) -> None:
        _write(f"{ESC}[?7h{ESC}[?25h{ESC}[?1049l")
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
