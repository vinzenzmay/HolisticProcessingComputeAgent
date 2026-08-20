#!/usr/bin/env python3
"""A runnable sketch of the row-oriented UI, drawn without Textual.

    pixi run -e dev python prototypes/row_ui.py
    pixi run -e dev python prototypes/row_ui.py --chat 2000 --sessions 40

Nothing here imports hpca and nothing here opens a database: the point is to
try the *shape* — a header, four stacked rows, a footer, and the three screens
that open over them — before any of it is wired to real data. The content is
synthetic and deliberately over-long, because the question this exists to
answer is what the UI feels like once a turn has produced hundreds of steps.

The rendering model is pi-tui's, which is the reason for the exercise. pi-tui
is TypeScript, so what is ported here is the architecture, not the library:

* a component renders to ``list[str]`` at a known width. There is no style
  cascade, no auto-height measurement and no arrange pass, so there is no
  O(conversation) traversal to accidentally trigger — which is the whole of
  what makes the Textual version stutter;
* only the *visible* slice of a pane is ever turned into lines, so frame time
  is flat in the number of entries. The header prints it: that number staying
  put while you scroll a 2000-entry chat is the claim being tested;
* the screen is repainted differentially — lines that did not change are not
  written — inside a synchronised-output pair so the terminal shows each frame
  atomically instead of tearing.

Keys follow the ones already bound in the Textual app (app.py BINDINGS) so
that muscle memory survives the move: m manage llms, a profiles & learnings,
c config editor, r/t/d on a session, d unwatch, q quit. The one addition is
→/← to open and close an entry, which the row design needs and the column
design had no equivalent of.

The message box edits the way an editor does: it wraps rather than overflowing,
ctrl+arrow moves by word, ctrl+backspace and ctrl+delete cut one, and shift with
any motion marks text. That last one also fixed a real annoyance — an escape
sequence the key table did not know used to be read as the esc key followed by
its letters, so reaching for shift+arrow threw you out of the box and into the
chat. Escapes are now measured by shape and unknown ones are dropped whole.

What is deliberately still missing: real data, mouse support, selecting text out
of the *chat*, and approval prompts. Those are arguments *against* leaving
Textual and this prototype is not the place to pretend they are solved.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import textwrap
import time
import tty
from dataclasses import dataclass, field

# Cosmetic only — kept literal so this file stands alone and can be copied out
# of the repo to try in another terminal.
VERSION = "0.25.0-proto"

# How long a first escape stays armed for a second one to complete the stop
# gesture. The same 1.0s the Textual app uses, and for the same reason a single
# escape must keep meaning nothing: ESC is the byte a terminal also sends as the
# prefix of every arrow key and of bracketed paste, so one arriving on its own
# is not evidence that the user wants the turn dead. Two in a second are.
ESC_STOP_WINDOW = 1.0

# --------------------------------------------------------------------- ansi

ESC = "\x1b"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
REVERSE = f"{ESC}[7m"
CYAN = f"{ESC}[38;5;44m"
GREEN = f"{ESC}[38;5;71m"
YELLOW = f"{ESC}[38;5;179m"
RED = f"{ESC}[38;5;167m"
BLUE = f"{ESC}[38;5;68m"


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _pad(text: str, width: int) -> str:
    """Exactly ``width`` visible characters — truncated with an ellipsis, or
    padded out.

    Every line is built plain and padded *before* any SGR is wrapped around it,
    so a highlight covers the full row and no escape sequence is ever cut in
    half by the truncation.
    """
    if width <= 0:
        return ""
    if len(text) > width:
        return text[: width - 1] + "…" if width > 1 else text[:width]
    return text + " " * (width - len(text))


def _rule(label: str, width: int, right: str = "") -> str:
    left = f"── {label} "
    tail = f"{right} ──" if right else "──"
    gap = max(1, width - len(left) - len(tail))
    return _pad(f"{left}{'─' * gap}{tail}", width)


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


# ---------------------------------------------------------------------- keys

KEYS = {
    f"{ESC}[A": "up",
    f"{ESC}OA": "up",
    f"{ESC}[B": "down",
    f"{ESC}OB": "down",
    f"{ESC}[C": "right",
    f"{ESC}OC": "right",
    f"{ESC}[D": "left",
    f"{ESC}OD": "left",
    f"{ESC}[1;5A": "ctrl-up",
    f"{ESC}[1;5B": "ctrl-down",
    f"{ESC}[1;5C": "ctrl-right",
    f"{ESC}[1;5D": "ctrl-left",
    f"{ESC}[1;3A": "alt-up",
    f"{ESC}[1;3B": "alt-down",
    f"{ESC}[1;3C": "ctrl-right",
    f"{ESC}[1;3D": "ctrl-left",
    f"{ESC}b": "ctrl-left",
    f"{ESC}f": "ctrl-right",
    # Shift is selection everywhere: plain, by word, and to the ends.
    f"{ESC}[1;2A": "shift-up",
    f"{ESC}[1;2B": "shift-down",
    f"{ESC}[1;2C": "shift-right",
    f"{ESC}[1;2D": "shift-left",
    f"{ESC}[1;6C": "shift-ctrl-right",
    f"{ESC}[1;6D": "shift-ctrl-left",
    f"{ESC}[1;4C": "shift-ctrl-right",
    f"{ESC}[1;4D": "shift-ctrl-left",
    f"{ESC}[1;2H": "shift-home",
    f"{ESC}[1;2F": "shift-end",
    f"{ESC}[1;2~": "shift-home",
    f"{ESC}[4;2~": "shift-end",
    f"{ESC}[5~": "pgup",
    f"{ESC}[6~": "pgdn",
    f"{ESC}[H": "home",
    f"{ESC}[F": "end",
    f"{ESC}OH": "home",
    f"{ESC}OF": "end",
    f"{ESC}[1~": "home",
    f"{ESC}[4~": "end",
    f"{ESC}[3~": "delete",
    f"{ESC}[3;5~": "ctrl-delete",
    f"{ESC}[3;3~": "ctrl-delete",
    f"{ESC}\x7f": "ctrl-backspace",
    f"{ESC}[Z": "shift-tab",
    f"{ESC}\r": "alt-enter",
    ESC: "esc",
    "\t": "tab",
    "\r": "enter",
    "\n": "enter",
    "\x7f": "backspace",
    "\x08": "ctrl-backspace",  # what most terminals send for ctrl+backspace
    "\x17": "ctrl-backspace",  # and ctrl-w, for the terminals that do not
    "\x13": "ctrl-s",
    "\x15": "ctrl-u",
    "\x03": "quit",
    "\x04": "quit",
}


def _escape_len(text: str, at: int) -> int:
    """How many characters the escape sequence starting at ``at`` occupies.

    Measured by shape rather than looked up, so a sequence this prototype does
    not know — a shift+alt+arrow from some other terminal, a bracketed paste
    marker — is still consumed whole. Matching by table alone left the leading
    ESC to be read as the esc key and the rest as typed letters, which is why
    shift+arrow used to throw you out of the message box and into the chat.
    """
    n = len(text)
    if at + 1 >= n:
        return 1
    nxt = text[at + 1]
    if nxt == "[":
        i = at + 2
        while i < n and (text[i].isdigit() or text[i] in ";?<"):
            i += 1
        return min(n, i + 1) - at
    if nxt == "O":
        return min(n, at + 3) - at
    if nxt == ESC:
        return 1
    return 2  # alt-<char>


def decode(data: bytes) -> list[str]:
    """One read into a list of key names. Unknown escapes are dropped."""
    text = data.decode("utf-8", "replace")
    keys: list[str] = []
    at = 0
    while at < len(text):
        if text[at] == ESC:
            size = _escape_len(text, at)
            chunk = text[at : at + size]
            if chunk in KEYS:
                keys.append(KEYS[chunk])
            elif size == 1:
                keys.append("esc")
            at += size
            continue
        keys.append(KEYS.get(text[at], text[at]))
        at += 1
    return keys


# -------------------------------------------------------------------- editor


def _wrap_spans(text: str, width: int) -> list[tuple[int, int]]:
    """Where one logical line breaks to fit ``width``, as ``(start, end)``.

    Broken at a space where there is one and mid-run where there is not. The
    spans partition the line exactly — nothing is dropped, not even the space
    that caused the break — because the cursor is addressed by column, and a
    swallowed character would leave a column with nowhere to stand.
    """
    if width < 1:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    at, n = 0, len(text)
    while at < n:
        if n - at <= width:
            spans.append((at, n))
            break
        cut = text.rfind(" ", at, at + width)
        spans.append((at, cut + 1 if cut > at else at + width))
        at = spans[-1][1]
    if not spans:
        return [(0, 0)]
    if spans[-1][1] - spans[-1][0] == width:
        spans.append((n, n))  # a full last line still needs a cursor slot
    return spans


def _reverse(text: str, ranges: list[tuple[int, int]]) -> str:
    """``text`` with those column ranges highlighted, and nothing else moved."""
    spans = sorted((a, b) for a, b in ranges if b > a)
    if not spans:
        return text
    out: list[str] = []
    at = 0
    for start, end in spans:
        start, end = max(start, at), min(end, len(text))
        if end <= start:
            continue
        out.append(text[at:start])
        out.append(REVERSE + text[start:end] + RESET)
        at = end
    out.append(text[at:])
    return "".join(out)


class Editor:
    """A minimal multi-line text buffer with a cursor and a selection.

    Shared by the three places that need real typing — the message row, the
    config editor and a profile's learnings — because they differ only in what
    Enter means, which is the caller's business, not the buffer's.

    ``wrap`` is the one place they genuinely differ: prose in the message row
    has to fold at the box edge or the row overflows, while a JSON file edited
    at column 90 must not reflow the twenty lines under it on every keystroke,
    so the config editor scrolls sideways instead.
    """

    def __init__(self, text: str = "", *, wrap: bool = False) -> None:
        self.lines = text.split("\n") or [""]
        self.row = 0
        self.col = 0
        self.offset = 0
        self.wrap = wrap
        self.anchor: tuple[int, int] | None = None
        self._rev = 0
        self._vis: list[tuple[int, int, int]] = []
        self._vis_key: tuple | None = None
        self._last_width = 60

    # ------------------------------------------------------------- content

    def text(self) -> str:
        return "\n".join(self.lines)

    def _touch(self) -> None:
        self._rev += 1

    def clear(self) -> None:
        self.lines = [""]
        self.row = self.col = self.offset = 0
        self.anchor = None
        self._touch()

    def set_text(self, text: str) -> None:
        """Replace the buffer, cursor left at the end of it."""
        self.lines = text.split("\n") or [""]
        self.row = len(self.lines) - 1
        self.col = len(self.lines[self.row])
        self.anchor = None
        self._touch()

    def insert(self, ch: str) -> None:
        self.delete_selection()
        line = self.lines[self.row]
        self.lines[self.row] = line[: self.col] + ch + line[self.col :]
        self.col += 1
        self._touch()

    def newline(self) -> None:
        self.delete_selection()
        line = self.lines[self.row]
        self.lines[self.row : self.row + 1] = [line[: self.col], line[self.col :]]
        self.row += 1
        self.col = 0
        self._touch()

    def backspace(self) -> None:
        if self.delete_selection():
            return
        if self.col:
            line = self.lines[self.row]
            self.lines[self.row] = line[: self.col - 1] + line[self.col :]
            self.col -= 1
        elif self.row:
            above = self.lines[self.row - 1]
            self.col = len(above)
            self.lines[self.row - 1] = above + self.lines[self.row]
            del self.lines[self.row]
            self.row -= 1
        self._touch()

    def delete(self) -> None:
        if self.delete_selection():
            return
        line = self.lines[self.row]
        if self.col < len(line):
            self.lines[self.row] = line[: self.col] + line[self.col + 1 :]
        elif self.row < len(self.lines) - 1:
            self.lines[self.row] += self.lines[self.row + 1]
            del self.lines[self.row + 1]
        self._touch()

    # ------------------------------------------------------------ selection

    def sel_range(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """The marked region as ordered ``(row, col)`` ends, or None."""
        if self.anchor is None:
            return None
        here = (self.row, self.col)
        if self.anchor == here:
            return None
        return (self.anchor, here) if self.anchor < here else (here, self.anchor)

    def selected(self) -> str:
        span = self.sel_range()
        if span is None:
            return ""
        (r1, c1), (r2, c2) = span
        if r1 == r2:
            return self.lines[r1][c1:c2]
        parts = [self.lines[r1][c1:]] + self.lines[r1 + 1 : r2] + [self.lines[r2][:c2]]
        return "\n".join(parts)

    def delete_selection(self) -> bool:
        """Remove what is marked, if anything. Every edit starts here, which is
        what makes typing over a selection replace it the way it should."""
        span = self.sel_range()
        self.anchor = None
        if span is None:
            return False
        (r1, c1), (r2, c2) = span
        self.lines[r1 : r2 + 1] = [self.lines[r1][:c1] + self.lines[r2][c2:]]
        self.row, self.col = r1, c1
        self._touch()
        return True

    def _mark(self, extend: bool) -> None:
        """Called before every motion: shift keeps an anchor, others drop it."""
        if not extend:
            self.anchor = None
        elif self.anchor is None:
            self.anchor = (self.row, self.col)

    # ---------------------------------------------------------- navigation

    def move(self, drow: int, dcol: int) -> None:
        if drow:
            self.row = max(0, min(len(self.lines) - 1, self.row + drow))
            self.col = min(self.col, len(self.lines[self.row]))
        if dcol:
            self.col += dcol
            if self.col < 0:
                if self.row:
                    self.row -= 1
                    self.col = len(self.lines[self.row])
                else:
                    self.col = 0
            elif self.col > len(self.lines[self.row]):
                if self.row < len(self.lines) - 1:
                    self.row += 1
                    self.col = 0
                else:
                    self.col = len(self.lines[self.row])

    def move_line(self, delta: int) -> None:
        """Up and down by what is on the screen, not by logical line.

        In a wrapped box the two differ: one typed sentence can be three rows,
        and stepping over all three at once is the thing that makes a wrapped
        box feel broken. Uses the width the last render used, which is the
        width the line the user is looking at was folded at.
        """
        if not self.wrap:
            self.move(delta, 0)
            return
        rows = self.visual(self._last_width)
        index, offset = self.cursor_visual(self._last_width)
        target = index + delta
        if not 0 <= target < len(rows):
            return
        row, start, end = rows[target]
        self.row = row
        self.col = min(start + offset, end, len(self.lines[row]))

    def home(self) -> None:
        self.col = 0

    def end(self) -> None:
        self.col = len(self.lines[self.row])

    def word_left(self) -> None:
        if self.col == 0:
            self.move(0, -1)
            return
        line = self.lines[self.row]
        at = self.col
        while at and line[at - 1] == " ":
            at -= 1
        while at and line[at - 1] != " ":
            at -= 1
        self.col = at

    def word_right(self) -> None:
        line = self.lines[self.row]
        if self.col >= len(line):
            self.move(0, 1)
            return
        at = self.col
        while at < len(line) and line[at] == " ":
            at += 1
        while at < len(line) and line[at] != " ":
            at += 1
        self.col = at

    def delete_word_left(self) -> None:
        if self.delete_selection():
            return
        self.anchor = (self.row, self.col)
        self.word_left()
        self.delete_selection()

    def delete_word_right(self) -> None:
        if self.delete_selection():
            return
        self.anchor = (self.row, self.col)
        self.word_right()
        self.delete_selection()

    # ------------------------------------------------------------- geometry

    def visual(self, width: int) -> list[tuple[int, int, int]]:
        """``(row, start, end)`` for every screen line. Cached per edit.

        Rebuilt only when the text or the width changes, so holding a key down
        costs one re-wrap per keystroke and nothing per line above it.
        """
        key = (width, self._rev, self.wrap)
        if self._vis_key == key:
            return self._vis
        rows: list[tuple[int, int, int]] = []
        for index, line in enumerate(self.lines):
            if self.wrap:
                rows += [(index, s, e) for s, e in _wrap_spans(line, width)]
            else:
                rows.append((index, 0, len(line)))
        self._vis, self._vis_key = rows, key
        return rows

    def cursor_visual(self, width: int) -> tuple[int, int]:
        """Which screen line the cursor sits on, and how far into it."""
        rows = self.visual(width)
        last = 0
        for index, (row, start, end) in enumerate(rows):
            if row != self.row:
                continue
            last = index
            if start <= self.col < end:
                return index, self.col - start
        return last, self.col - rows[last][1]

    def height(self, width: int) -> int:
        return len(self.visual(width))

    # ---------------------------------------------------------------- keys

    def handle(self, key: str) -> bool:
        """The keys every editor shares. False means "not mine"."""
        name, extend = key, False
        if name.startswith("shift-") and name != "shift-tab":
            name, extend = name[6:], True
        motions = {
            "left": lambda: self.move(0, -1),
            "right": lambda: self.move(0, 1),
            "up": lambda: self.move_line(-1),
            "down": lambda: self.move_line(1),
            "home": self.home,
            "end": self.end,
            "ctrl-left": self.word_left,
            "ctrl-right": self.word_right,
        }
        if name in motions:
            self._mark(extend)
            motions[name]()
        elif key == "backspace":
            self.backspace()
        elif key == "delete":
            self.delete()
        elif key == "ctrl-backspace":
            self.delete_word_left()
        elif key == "ctrl-delete":
            self.delete_word_right()
        elif key == "ctrl-u":
            self.clear()
        elif len(key) == 1 and key.isprintable():
            self.insert(key)
        else:
            return False
        return True

    # ------------------------------------------------------------- drawing

    def render(
        self, width: int, height: int, *, focused: bool, numbers: bool = False
    ) -> list[str]:
        gutter = len(str(len(self.lines))) + 2 if numbers else 0
        body = max(4, width - gutter)
        self._last_width = body
        rows = self.visual(body)
        cursor_row, cursor_col = self.cursor_visual(body)
        if cursor_row < self.offset:
            self.offset = cursor_row
        elif cursor_row >= self.offset + height:
            self.offset = cursor_row - height + 1
        self.offset = max(0, min(self.offset, max(0, len(rows) - height)))
        # A wrapped row already fits the box; only the sideways editors need an
        # offset into the line.
        hoff = 0 if self.wrap else max(0, cursor_col - body + 2)
        span = self.sel_range() if focused else None
        out = []
        for index in range(self.offset, self.offset + height):
            if index >= len(rows):
                out.append(" " * width)
                continue
            row, start, end = rows[index]
            if not numbers:
                prefix = ""
            elif index == 0 or rows[index - 1][0] != row:
                prefix = f"{DIM}{row + 1:>{gutter - 1}} {RESET}"
            else:
                prefix = " " * gutter
            text = _pad(self.lines[row][start:end][hoff : hoff + body], body)
            marks: list[tuple[int, int]] = []
            if span is not None:
                (r1, c1), (r2, c2) = span
                if r1 <= row <= r2:
                    left = start + hoff
                    lo = (c1 if row == r1 else 0) - left
                    hi = (c2 if row == r2 else len(self.lines[row]) + 1) - left
                    marks.append((lo, hi))
            at = cursor_col - hoff
            if (
                focused
                and index == cursor_row
                and not any(a <= at < b for a, b in marks)
            ):
                marks.append((at, at + 1))
            out.append(prefix + _reverse(text, marks))
        return out


# --------------------------------------------------------------------- panes


@dataclass
class Item:
    """One entry: a single line, plus the body it opens into.

    ``kind`` and ``text`` are what Enter needs: the log has to be able to say
    which rows are the user's own words, and to hand back the words themselves
    rather than the decorated line they are drawn as (app.py keeps the same two
    on Entry, see OWN_MESSAGE_KINDS).
    """

    head: str
    body: list[str] = field(default_factory=list)
    accent: str = ""
    kind: str = ""
    text: str = ""


class Pane:
    """One navigable list of entries.

    Holds its own cursor line, scroll offset and set of open entries, which is
    what makes "each row remembers where you were" fall out rather than need
    arranging: leaving a pane changes nothing about it.
    """

    def __init__(self, name: str, items: list[Item]) -> None:
        self.name = name
        self.items = items
        self.expanded: set[int] = set()
        self.cursor = 0  # index into the flattened line list
        self.offset = 0  # first visible flattened line
        self._flat: list[tuple[int, str, bool]] | None = None
        self._flat_width = -1

    # ------------------------------------------------------------- content

    def flat(self, width: int) -> list[tuple[int, str, bool]]:
        """``(item index, text, is head line)`` for every line the pane shows.

        Cached, and rebuilt only when an entry opens or closes or the width
        changes — never per keystroke. Scrolling slices this list, which is why
        moving the cursor costs the same at twenty entries as at two thousand.
        """
        if self._flat is not None and self._flat_width == width:
            return self._flat
        lines: list[tuple[int, str, bool]] = []
        for index, item in enumerate(self.items):
            marker = ("▾" if index in self.expanded else "▸") if item.body else " "
            lines.append((index, f"{marker} {item.head}", True))
            if index in self.expanded:
                for raw in item.body:
                    for piece in textwrap.wrap(raw, max(8, width - 4)) or [""]:
                        lines.append((index, f"    {piece}", False))
        self._flat = lines
        self._flat_width = width
        return lines

    def invalidate(self) -> None:
        self._flat = None

    def current(self, width: int) -> int:
        lines = self.flat(width)
        if not lines:
            return -1
        return lines[min(self.cursor, len(lines) - 1)][0]

    # ----------------------------------------------------------- navigation

    def _go_to(self, item: int, width: int) -> None:
        for row, (owner, _, _) in enumerate(self.flat(width)):
            if owner == item:
                self.cursor = row
                return

    def _scroll_into_view(self, view_h: int, total: int) -> None:
        view_h = max(1, view_h)
        self.cursor = max(0, min(self.cursor, max(0, total - 1)))
        self.offset = max(0, min(self.offset, max(0, total - view_h)))
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + view_h:
            self.offset = self.cursor - view_h + 1

    def move(self, delta: int, view_h: int, width: int) -> None:
        total = len(self.flat(width))
        if not total:
            return
        self.cursor = max(0, min(total - 1, self.cursor + delta))
        self._scroll_into_view(view_h, total)

    def expand(self, width: int) -> bool:
        """Open the entry under the cursor. False if there was nothing to open."""
        item = self.current(width)
        if item < 0 or not self.items[item].body or item in self.expanded:
            return False
        self.expanded.add(item)
        self.invalidate()
        # Land back on the entry's own first line: opening one twelve lines
        # long and being left in the middle of it reads as a jump.
        self._go_to(item, width)
        return True

    def collapse(self, width: int) -> bool:
        """Close the entry the cursor is anywhere inside. False if it was shut."""
        item = self.current(width)
        if item < 0 or item not in self.expanded:
            return False
        self.expanded.discard(item)
        self.invalidate()
        self._go_to(item, width)
        return True

    def expand_all(self, width: int) -> None:
        item = self.current(width)
        self.expanded = {i for i, entry in enumerate(self.items) if entry.body}
        self.invalidate()
        if item >= 0:
            self._go_to(item, width)

    def collapse_all(self, width: int) -> None:
        item = self.current(width)
        self.expanded.clear()
        self.invalidate()
        if item >= 0:
            self._go_to(item, width)

    def reorder(self, delta: int, view_h: int, width: int) -> bool:
        """Move the entry under the cursor up or down past its neighbour.

        The cursor travels with the entry rather than staying on the line,
        which is what makes holding alt+↓ walk one watcher down the list
        instead of shuffling a different one each press. ``expanded`` is keyed
        by position, so the two entries trade that flag along with their slot.
        """
        item = self.current(width)
        target = item + delta
        if item < 0 or not 0 <= target < len(self.items):
            return False
        self.items[item], self.items[target] = self.items[target], self.items[item]
        was = (item in self.expanded, target in self.expanded)
        self.expanded.difference_update({item, target})
        if was[1]:
            self.expanded.add(item)
        if was[0]:
            self.expanded.add(target)
        self.invalidate()
        self._go_to(target, width)
        self._scroll_into_view(view_h, len(self.flat(width)))
        return True

    # -------------------------------------------------------------- drawing

    def render(self, width: int, height: int, *, focused: bool) -> list[str]:
        """The pane as exactly ``height`` lines: a title, then the body."""
        inner = max(8, width - 2)  # two columns go to the gutter
        lines = self.flat(inner)
        body_h = max(1, height - 1)
        self._scroll_into_view(body_h, len(lines))
        current = lines[self.cursor][0] if lines else -1
        right = f"line {min(self.cursor + 1, len(lines))}/{len(lines)}"
        if self.expanded:
            right += f" · {len(self.expanded)} open"
        title = _rule(self.name, width, right)
        out = [(BOLD + CYAN if focused else DIM) + title + RESET]
        for row in range(self.offset, self.offset + body_h):
            if row >= len(lines):
                out.append(" " * width)
                continue
            owner, text, is_head = lines[row]
            gutter = "▌ " if owner == current else "  "
            painted = _pad(gutter + text, width)
            if row == self.cursor:
                # The unfocused pane still shows where it was left, dimmed —
                # that is the "memory" being visible rather than merely kept.
                painted = (REVERSE if focused else DIM + REVERSE) + painted + RESET
            elif not is_head:
                painted = DIM + painted + RESET
            elif self.items[owner].accent:
                painted = self.items[owner].accent + painted + RESET
            out.append(painted)
        return out


# ------------------------------------------------------------------- footers


def footer_line(
    pairs: list[tuple[str, str]], width: int, note: str = "", style: str = YELLOW
) -> str:
    """As many ``key label`` pairs as fit, keys bright and labels dim.

    Truncation is by whole pairs rather than by characters: half a hint is
    worse than one hint fewer, and ``?`` opens the full list anyway — which is
    the honest answer to "show *all* the hotkeys" on an 80-column terminal.
    """
    plain: list[str] = []
    styled: list[str] = []
    used = 1
    if note:
        used += len(note) + 2
    for key, label in pairs:
        piece = f"{key} {label}"
        extra = len(piece) + (2 if plain else 0)
        if used + extra > width - 1:
            break
        plain.append(piece)
        styled.append(f"{CYAN}{key}{RESET} {DIM}{label}{RESET}")
        used += extra
    head = f"{style}{note}{RESET}  " if note else ""
    return " " + head + "  ".join(styled) + " " * max(0, width - used)


# ------------------------------------------------------------------ overlays


class Overlay:
    """A screen drawn over the rows. ``handle`` returning False closes it."""

    title = ""

    def render(self, width: int, height: int) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def footer(self) -> list[tuple[str, str]]:
        return [("esc", "back")]


FORK, ROLLBACK, COPY = "fork", "rollback", "copy"

# The dialog quotes the message being rewound from so you can check you grabbed
# the right one — a preview, not the transcript; the log has the rest. Same
# figure as hpca.tui.rewind_screen.
PREVIEW_CHARS = 300
PREVIEW_LINES = 6  # what the Textual dialog's max-height comes to


class RewindOverlay(Overlay):
    """Enter on one of your own messages in the log (§ chat rewind).

    Three ways to pick the conversation up from it, the same three
    RewindScreen offers and on the same keys: fork the session from just
    before it (the original stays whole), roll this conversation back to just
    before it (everything after is dropped), or copy the text into the message
    box — the old behaviour, still on Enter, so the reflex of activating a
    message twice keeps doing what it always did.

    The choice is read back by the caller rather than acted on here, for the
    reason app.py captures the session before pushing the screen: what happens
    belongs to the conversation the choice was made in.
    """

    title = "this message again"

    OPTIONS = [
        ("f", "fork the session from here — the original stays whole"),
        ("r", "roll this conversation back to here"),
        ("c / enter", "copy it into the message box"),
        ("esc", "cancel"),
    ]

    def __init__(self, message: str, index: int) -> None:
        self.message = message
        self.index = index
        self.choice = ""

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + _rule(self.title, width) + RESET]
        preview = self.message[:PREVIEW_CHARS]
        if preview != self.message:
            preview += " …"
        quoted: list[str] = []
        for paragraph in preview.split("\n"):
            quoted += textwrap.wrap(paragraph, max(8, width - 6)) or [""]
        for line in quoted[:PREVIEW_LINES]:
            out.append(DIM + _pad(f"    {line}", width) + RESET)
        if len(quoted) > PREVIEW_LINES:
            out.append(DIM + _pad("    …", width) + RESET)
        out.append(" " * width)
        for key, label in self.OPTIONS:
            row = _pad(f"      {key:<12}{label}", width)
            out.append(row[:6] + CYAN + row[6:18] + RESET + row[18:])
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        picked = {"f": FORK, "r": ROLLBACK, "c": COPY, "enter": COPY}
        if key in picked:
            self.choice = picked[key]
            return False
        if key in ("esc", "quit"):
            return False
        return True  # a modal ignores what it has no answer for

    def footer(self) -> list[tuple[str, str]]:
        return [
            ("f", "fork"),
            ("r", "roll back"),
            ("c / enter", "copy"),
            ("esc", "cancel"),
        ]


class HelpOverlay(Overlay):
    """Every key, since the footer can only ever show the ones that fit."""

    title = "keys"

    SECTIONS = [
        (
            "anywhere",
            [
                ("^↑ ^↓", "move between rows (tab / shift-tab also)"),
                ("esc esc", "stop the agent, from any row — one esc does nothing"),
                ("m", "manage llms"),
                ("a", "profiles & learnings"),
                ("c", "config editor"),
                ("^l", "switch llm for this session"),
                ("shift-tab", "cycle agent mode"),
                ("?", "this list"),
                ("q", "quit"),
            ],
        ),
        (
            "in any row",
            [
                ("↑ ↓", "move one line"),
                ("pgup pgdn", "move one screen"),
                ("home end", "first / last line"),
                ("→", "open the entry under the cursor, or step into it"),
                ("←", "close it again"),
                ("shift-→ ←", "open or close every entry in the row"),
            ],
        ),
        (
            "sessions row",
            [
                ("enter", "open it and go straight to the message box"),
                ("r", "rename"),
                ("t", "ask the llm for a title"),
                ("d", "delete"),
            ],
        ),
        (
            "chat row",
            [
                ("i", "go to the message box"),
                ("enter", "on one of your own messages: fork, roll back, copy"),
                ("enter", "on anything else: go to the message box"),
            ],
        ),
        (
            "message box",
            [
                ("enter", "send"),
                ("alt-enter", "new line"),
                ("↑ ↓", "move one screen line (long text wraps)"),
                ("^← ^→", "jump a word (alt-← alt-→ also)"),
                ("shift-← →", "mark text; shift-^← ^→ by the word"),
                ("shift-↑ ↓", "mark whole lines; shift-home end to an end"),
                ("^⌫", "delete the word before the cursor"),
                ("^del", "delete the word after it"),
                ("⌫ del", "delete the marked text, if any"),
                ("^u", "clear"),
                ("^↑", "back to the chat — escape is the stop gesture, not an exit"),
            ],
        ),
        (
            "watchers row",
            [
                ("enter", "peek at the log"),
                ("d", "unwatch"),
                ("alt-↑ alt-↓", "reorder"),
            ],
        ),
    ]

    def __init__(self) -> None:
        self.offset = 0

    def _body(self, width: int) -> list[str]:
        out = []
        for name, keys in self.SECTIONS:
            out.append(DIM + _pad(f"  {name}", width) + RESET)
            for key, label in keys:
                # Padded plain and coloured afterwards by column, never by
                # adding the escape lengths to the width — that arithmetic is
                # exactly the kind that leaves a row one cell short.
                row = _pad(f"      {key:<12}{label}", width)
                out.append(row[:6] + CYAN + row[6:18] + RESET + row[18:])
            out.append(" " * width)
        return out

    def render(self, width: int, height: int) -> list[str]:
        body = self._body(width)
        rows = max(1, height - 1)
        self.offset = max(0, min(self.offset, max(0, len(body) - rows)))
        more = len(body) > rows
        tail = f"{self.offset + rows}/{len(body)}" if more else ""
        out = [BOLD + CYAN + _rule("keys", width, tail) + RESET]
        out += body[self.offset : self.offset + rows]
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        # Scrolls, because the list outgrew a short terminal. Anything else
        # closes it: a list you opened by accident should not need the one key
        # you were looking it up to find.
        step = {"up": -1, "down": 1, "pgup": -(height - 2), "pgdn": height - 2}
        if key in step:
            self.offset += step[key]
            return True
        if key in ("home", "end"):
            self.offset = 0 if key == "home" else 10**9
            return True
        return False

    def footer(self) -> list[tuple[str, str]]:
        return [("↑↓", "scroll"), ("any key", "back")]


class LlmOverlay(Overlay):
    """Manage LLMs (m): discovered endpoints above, configured catalog below.

    Stacked rather than side by side, like the main view. Columns cost this
    screen more than they cost anywhere else: an endpoint line is a URL and a
    model name and a context size, which is most of eighty characters before
    the catalog gets any, and halving the width truncated all of it. Rows give
    each list the whole width and cost only vertical space, which is the one
    thing a list can scroll.

    ^↑/^↓ move between the two — the same keys as the main view, so the habit
    transfers — and the footer offers "add" only on discovered and
    "remove"/"make default" only on configured, which is what the Textual
    version does by hanging bindings off each panel.
    """

    title = "manage llms"

    def __init__(self, discovered: list[Item], configured: list[Item]) -> None:
        self.panes = [Pane("discovered", discovered), Pane("configured", configured)]
        self.side = 0
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        keys = [("^↑^↓", "row"), ("↑↓", "move"), ("→←", "open")]
        if self.side == 0:
            keys.append(("enter", "add to catalog"))
            keys.append(("s", "rescan"))
        else:
            keys.append(("enter", "make default"))
            keys.append(("d", "remove"))
        return keys + [("esc", "back")]

    def _heights(self, height: int, width: int) -> list[int]:
        """Half each, but a short list only takes what it has.

        Same rule as the main view: whatever the top does not need goes to the
        bottom, so three discovered endpoints do not hold half the screen open
        above a catalog that has to scroll.
        """
        avail = max(4, height - 1)
        inner = max(8, width - 2)
        top = max(2, min(1 + len(self.panes[0].flat(inner)), avail // 2))
        bottom = avail - top
        if bottom < 2:
            top, bottom = avail - 2, 2
        return [top, bottom]

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + _rule(self.title, width, self.note) + RESET]
        for index, pane_h in enumerate(self._heights(height, width)):
            out += self.panes[index].render(width, pane_h, focused=self.side == index)
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        pane = self.panes[self.side]
        inner = max(8, width - 2)
        view = max(1, self._heights(height, width)[self.side] - 1)
        if key == "esc":
            return False
        if key in ("ctrl-up", "ctrl-down", "tab", "shift-tab", "left", "right"):
            self.side = 1 - self.side
        elif key == "up":
            pane.move(-1, view, inner)
        elif key == "down":
            pane.move(1, view, inner)
        elif key == "pgup":
            pane.move(-view, view, inner)
        elif key == "pgdn":
            pane.move(view, view, inner)
        elif key == "right":
            if not pane.expand(inner) and pane.current(inner) in pane.expanded:
                pane.move(1, view, inner)
        elif key == "left":
            pane.collapse(inner)
        elif key == "shift-right":
            pane.expand_all(inner)
        elif key == "shift-left":
            pane.collapse_all(inner)
        elif key == "enter":
            self.note = (
                "added to the catalog" if self.side == 0 else "made the default"
            )
        elif key == "d" and self.side == 1:
            self.note = "removed (confirm in the real screen)"
        elif key == "s" and self.side == 0:
            self.note = "rescanned: 3 endpoints"
        return True


class ProfilesOverlay(Overlay):
    """Profiles & learnings (a): the list, and one profile's memories open in
    a plain editor — the two states the Textual screen has."""

    title = "profiles & learnings"

    def __init__(self, profiles: list[Item], learnings: dict[str, str]) -> None:
        self.pane = Pane("profiles", profiles)
        self.learnings = learnings
        self.editor: Editor | None = None
        self.editing = ""
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        if self.editor is not None:
            return [("^s", "keep"), ("esc", "discard"), ("↑↓←→", "move")]
        return [
            ("↑↓", "move"),
            ("→←", "open"),
            ("enter", "edit learnings"),
            ("c", "copy"),
            ("d", "delete"),
            ("esc", "back"),
        ]

    def render(self, width: int, height: int) -> list[str]:
        if self.editor is not None:
            head = _rule(f"learnings · {self.editing}", width, self.note)
            out = [BOLD + CYAN + head + RESET]
            out += self.editor.render(width, height - 1, focused=True, numbers=True)
            return out[:height]
        out = [BOLD + CYAN + _rule(self.title, width, self.note) + RESET]
        out += self.pane.render(width, height - 1, focused=True)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        inner = max(8, width - 2)
        if self.editor is not None:
            if key == "esc":
                self.editor = None
                self.note = "discarded"
            elif key == "ctrl-s":
                self.learnings[self.editing] = self.editor.text()
                self.editor = None
                self.note = "kept"
            elif key == "enter":
                self.editor.newline()
            else:
                self.editor.handle(key)
            return True
        if key == "esc":
            return False
        view = max(1, height - 2)
        if key == "up":
            self.pane.move(-1, view, inner)
        elif key == "down":
            self.pane.move(1, view, inner)
        elif key == "right":
            if (
                not self.pane.expand(inner)
                and self.pane.current(inner) in self.pane.expanded
            ):
                self.pane.move(1, view, inner)
        elif key == "left":
            self.pane.collapse(inner)
        elif key == "shift-right":
            self.pane.expand_all(inner)
        elif key == "shift-left":
            self.pane.collapse_all(inner)
        elif key == "enter":
            item = self.pane.current(inner)
            name = self.pane.items[item].head.split("  ")[0].strip("▸▾ ")
            self.editing = name
            self.editor = Editor(self.learnings.get(name, "(nothing learned yet)\n"))
            self.note = ""
        elif key == "c":
            self.note = "copied under a new name"
        elif key == "d":
            self.note = "deleted (never the default, never one in use)"
        return True


class ConfigOverlay(Overlay):
    """Config editor (c): raw JSON over the whole settings file, validated on
    save — deliberately not a friendly settings menu, which is what the real
    screen's docstring insists on."""

    title = "config editor"

    def __init__(self, text: str) -> None:
        self.editor = Editor(text)
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        return [
            ("^s", "validate & save"),
            ("↑↓←→", "move"),
            ("^u", "clear"),
            ("esc", "back"),
        ]

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + _rule(self.title, width, self.note) + RESET]
        out += self.editor.render(width, height - 1, focused=True, numbers=True)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        if key == "esc":
            return False
        if key == "ctrl-s":
            try:
                json.loads(self.editor.text())
            except json.JSONDecodeError as e:
                self.note = f"invalid: line {e.lineno} — {e.msg}"
            else:
                self.note = "saved"
            return True
        if key == "enter":
            self.editor.newline()
        else:
            self.editor.handle(key)
        return True


# ----------------------------------------------------------------------- app

SESSIONS, CHAT, INPUT, WATCHERS = range(4)


class SessionState:
    """Everything that belongs to one conversation rather than to the app.

    Switching sessions swaps this and nothing else, which is why the chat's
    cursor line, its open entries and a half-typed message all survive going
    away and coming back — the same property each row has, one level up. The
    Textual app parks drafts per session for the same reason; here it falls out
    of where the Editor lives instead of needing a store.

    Watchers belong to the session that registered them (WatchStore.list takes
    a session_id), so they swap too. A session with none simply shows an empty
    row, which the layout already charges nothing for.

    The chat and the watchers are built on first visit. The real app reads a
    thread from the checkpointer on switch, and this mirrors that: fourteen
    sessions of four hundred steps should not all exist because one is open.
    """

    def __init__(
        self,
        *,
        title: str,
        profile: str,
        model: str,
        started: str,
        size: int,
        watch_count: int,
        index: int,
    ) -> None:
        self.title = title
        self.profile = profile
        self.model = model
        self.started = started
        self.size = size
        self.watch_count = watch_count
        self.index = index
        self.draft = Editor(wrap=True)
        self._chat: Pane | None = None
        self._watchers: Pane | None = None

    @property
    def loaded(self) -> bool:
        return self._chat is not None

    @property
    def chat(self) -> Pane:
        if self._chat is None:
            self._chat = Pane("chat", sample_chat(self.size, self.index, self.title))
            self._chat.cursor = 10**9  # open at the newest, as the app does
        return self._chat

    @property
    def watchers(self) -> Pane:
        if self._watchers is None:
            self._watchers = Pane(
                "watchers", sample_watchers(self.watch_count, self.index)
            )
        return self._watchers

    def invalidate(self) -> None:
        for pane in (self._chat, self._watchers):
            if pane is not None:
                pane.invalidate()


class RowUI:
    MIN_CHAT = 4
    MAX_INPUT = 6

    def __init__(
        self,
        sessions: list[SessionState],
        *,
        learnings: dict[str, str],
        settings_json: str,
        llms: tuple[list[Item], list[Item]],
    ) -> None:
        self.sessions = sessions
        self.active = 0
        self.session_pane = Pane("sessions", [])
        self._refresh_sessions()
        self.focus = CHAT
        self.mode = "agent"
        self.frame_ms = 0.0
        self.note = ""
        self.overlay: Overlay | None = None
        # When the last escape landed, so the next one can tell whether it is
        # the second half of a stop. Injectable so the headless check can drive
        # the clock instead of sleeping through the window.
        self.clock = time.monotonic
        self._esc_armed_at: float | None = None
        self._learnings = learnings
        self._settings_json = settings_json
        self._llms = llms

    # ------------------------------------------------------ the open session

    @property
    def session(self) -> SessionState:
        return self.sessions[self.active]

    @property
    def chat(self) -> Pane:
        return self.session.chat

    @property
    def watchers(self) -> Pane:
        return self.session.watchers

    @property
    def input(self) -> Editor:
        return self.session.draft

    @property
    def profile(self) -> str:
        return self.session.profile

    @property
    def model(self) -> str:
        return self.session.model

    @property
    def panes(self) -> list[Pane]:
        """The three list rows, top to bottom, for the session on screen."""
        return [self.session_pane, self.chat, self.watchers]

    def _refresh_sessions(self) -> None:
        """Redraw the session list so the open one is marked.

        Rebuilt rather than patched because it is fourteen rows, not fourteen
        hundred — the cost that matters is the chat's, and that one is never
        rebuilt at all. The cursor is kept: which session you are *looking at*
        is not the same as which one is open, and moving the highlight must not
        follow the switch.
        """
        cursor = self.session_pane.cursor
        self.session_pane.items = [
            Item(
                head=(
                    f"{'●' if i == self.active else '○'} "
                    f"{state.title[:40]:<42}{state.started:>9}   {state.model}"
                ),
                body=[
                    f"session 9f3c{i:04x} · profile {state.profile} · mode agent",
                    f"{state.size} entries · {(i * 13) % 90}% of context used",
                    f"{state.watch_count} watches"
                    + (" · open" if i == self.active else ""),
                ],
                accent=GREEN if i == self.active else "",
            )
            for i, state in enumerate(self.sessions)
        ]
        self.session_pane.invalidate()
        self.session_pane.cursor = cursor

    def _switch(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        if index == self.active:
            self.note = "already open"
            return
        self.active = index
        self._refresh_sessions()
        self.note = f"opened “{self.session.title}”"

    def invalidate(self) -> None:
        """Every pane that has been built — a session never visited has none."""
        self.session_pane.invalidate()
        for state in self.sessions:
            state.invalidate()

    # ------------------------------------------------------------- geometry

    def _input_h(self, width: int) -> int:
        """Screen rows the message box wants, title included.

        Counted after wrapping, not from the number of typed lines: one long
        sentence is several rows, and sizing the box from the logical count is
        what let the text run off the edge of the row.
        """
        rows = self.input.height(self._input_body(width))
        return 1 + max(1, min(self.MAX_INPUT, rows))

    def _input_body(self, width: int) -> int:
        """The width the message text is folded at — the marker column costs
        two, and the editor is handed the rest."""
        return max(4, width - 2)

    def _heights(self, height: int, width: int) -> list[int]:
        """How the rows split the screen.

        A quarter each for sessions and watchers and the rest to the chat — but
        only as much of a quarter as the pane actually has to show, so a short
        session list or an empty watcher row costs nothing instead of holding a
        quarter of the screen open. The message box takes what it needs up to
        six lines. Everything left over goes to the chat, which is the row that
        can use it.
        """
        avail = max(8, height - 2)  # header and footer
        inp = self._input_h(width)
        quarter = max(2, avail // 4)
        inner = max(8, width - 2)
        top = max(2, min(1 + len(self.panes[0].flat(inner)), quarter))
        bottom = max(2, min(1 + len(self.panes[2].flat(inner)), quarter))
        while avail - top - bottom - inp < self.MIN_CHAT and (top > 2 or bottom > 2):
            if top >= bottom and top > 2:
                top -= 1
            elif bottom > 2:
                bottom -= 1
            else:
                break
        middle = avail - top - bottom - inp
        if middle < 1:  # a terminal too short for the design at all
            top = bottom = 2
            inp = 2
            middle = max(1, avail - 6)
        return [top, middle, inp, bottom]

    # -------------------------------------------------------------- drawing

    def render(self, width: int, height: int) -> list[str]:
        out = [self._header(width)]
        if self.overlay is not None:
            out += self.overlay.render(width, height - 2)
            out.append(footer_line(self.overlay.footer(), width))
            while len(out) < height:
                out.insert(len(out) - 1, " " * width)
            return out[:height]
        heights = self._heights(height, width)
        order = [
            (SESSIONS, self.panes[0], heights[0]),
            (CHAT, self.panes[1], heights[1]),
            (INPUT, None, heights[2]),
            (WATCHERS, self.panes[2], heights[3]),
        ]
        for slot, pane, pane_h in order:
            if slot == INPUT:
                out += self._render_input(width, pane_h)
            else:
                out += pane.render(width, pane_h, focused=self.focus == slot)
        note, style = self.note, YELLOW
        if self._esc_armed():
            note, style = "esc again to stop", RED
        out.append(footer_line(self._keys(), width, note, style))
        while len(out) < height:
            out.insert(len(out) - 1, " " * width)
        return out[:height]

    def _render_input(self, width: int, height: int) -> list[str]:
        focused = self.focus == INPUT
        right = f"{self.mode} · {self.model} · 31% ctx"
        title = _rule("message", width, right)
        out = [(BOLD + CYAN if focused else DIM) + title + RESET]
        rows = max(1, height - 1)
        body = self.input.render(self._input_body(width), rows, focused=focused)
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((CYAN if focused else DIM) + marker + RESET + line)
        return out[:height]

    def _header(self, width: int) -> str:
        left = f" HPCA {VERSION}  ·  {self.profile}  ·  {self.mode}"
        right = f"{self.frame_ms:5.2f}ms  ·  ? keys  "
        gap = width - len(left) - len(right)
        text = left + " " * gap + right if gap > 0 else left
        return REVERSE + _pad(text, width) + RESET

    def _keys(self) -> list[tuple[str, str]]:
        """Only what applies where the cursor is — the footer's whole job.

        The Textual app gets this from ``check_action`` per binding; here it is
        one function, which is easier to read and impossible to get out of step
        with what the keys actually do.
        """
        # esc esc works from every row, so it belongs in the part every row
        # shows — and ahead of "? keys" and "q quit", because a footer drops
        # whole pairs off its end and this is the one that must not be the pair
        # that goes. Leaving the message box is ^↑, not escape: escape has a
        # job now.
        common = [("^↑^↓", "row"), ("esc esc", "stop"), ("?", "keys"), ("q", "quit")]
        if self.focus == INPUT:
            # Spelled out rather than built from ``common`` so that send and
            # stop come first — the message box is where you sit while a turn
            # runs, and those are the two keys that matter there.
            return [
                ("enter", "send"),
                ("esc esc", "stop"),
                ("^↑^↓", "row"),
                ("alt-enter", "new line"),
                ("^←→", "word"),
                ("⇧←→", "select"),
                ("^⌫ ^del", "cut word"),
                ("^u", "clear"),
                ("?", "keys"),
                ("q", "quit"),
            ]
        rows = [("↑↓", "line"), ("→←", "open"), ("⇧→←", "open all")]
        if self.focus == SESSIONS:
            rows += [("enter", "switch"), ("r", "rename"), ("t", "retitle"), ("d", "delete")]
        elif self.focus == CHAT:
            rows += [("i", "write"), ("enter", "reuse")]
        else:
            rows += [("enter", "peek"), ("d", "unwatch"), ("alt-↑↓", "move")]
        return rows + [("m", "llms"), ("a", "profiles"), ("c", "config")] + common

    # --------------------------------------------------------------- input

    def handle(self, key: str, width: int, height: int) -> bool:
        if self.overlay is not None:
            overlay = self.overlay
            if not overlay.handle(key, width, height - 2):
                self.overlay = None
                self._closed(overlay)
            return True
        if self.focus == INPUT:
            return self._handle_input(key)
        return self._handle_row(key, width, height)

    def _closed(self, overlay: Overlay) -> None:
        """A screen that answered with something the rows have to act on."""
        if isinstance(overlay, RewindOverlay) and overlay.choice:
            self._rewind(overlay.choice, overlay.index, overlay.message)

    # -------------------------------------------------------- the chat rewind

    def _activate_chat(self, width: int) -> None:
        """Enter in the chat log.

        On one of your own messages it opens the rewind. On anything else it
        moves to the message box, which is what app.py answers an Enter it has
        nothing better to do with.
        """
        index = self.chat.current(width)
        item = self.chat.items[index] if index >= 0 else None
        if item is not None and item.kind == "user":
            self.overlay = RewindOverlay(item.text, index)
        else:
            self.focus = INPUT

    def _rewind(self, choice: str, index: int, message: str) -> None:
        if choice == COPY:
            self.reuse_message(message)
        elif choice == FORK:
            self._fork_at(index)
        elif choice == ROLLBACK:
            self._rollback_to(index)

    def reuse_message(self, text: str) -> None:
        """Put one of your own past messages back in the box, to send again or
        edit into the next one — usually a command that needs a word changed,
        which is otherwise retyped off the screen.

        Added to whatever is already being written rather than replacing it, so
        activating a message can never lose a draft. It starts its own line,
        except after a draft left ending in whitespace — that space is how you
        say "continue here" (``rerun this: `` + the old command).
        """
        draft = self.input.text()
        if draft and not draft[-1].isspace():
            draft += "\n"
        self.input.set_text(draft + text)
        self.focus = INPUT  # cursor behind the reused text, ready to send
        self.note = "copied into the message box"

    def _fork_at(self, index: int) -> None:
        """A copy of this conversation that stops just before that message.

        The original stays whole — that is the difference from a rollback, and
        the reason both are offered instead of one being the safe version of
        the other.
        """
        source = self.session
        fork = SessionState(
            title=f"{source.title} (fork)",
            profile=source.profile,
            model=source.model,
            started="just now",
            size=0,
            watch_count=0,
            index=source.index,
        )
        fork._chat = Pane("chat", list(self.chat.items[:index]))
        fork._chat.cursor = 10**9
        self.sessions.insert(0, fork)
        self.active = 0
        self._refresh_sessions()
        self.focus = INPUT
        self.note = f"forked “{source.title}” — this copy stops before that message"

    def _rollback_to(self, index: int) -> None:
        """Drop this conversation back to just before that message."""
        chat = self.chat
        dropped = len(chat.items) - index
        del chat.items[index:]
        chat.expanded = {i for i in chat.expanded if i < index}
        chat.invalidate()
        chat.cursor = 10**9
        self.focus = INPUT
        self.note = f"rolled back to just before that message ({dropped} entries gone)"

    def _esc_armed(self) -> bool:
        """Whether a first escape is still waiting for its second.

        Read by the footer, which says so in red. app.py deliberately stays
        silent here, but its objection is to a *toast*: a notification for
        every stray escape sequence would be noise on top of the screen. A
        word in the footer costs nothing, cannot cover anything, and goes away
        on its own — the poll timeout repaints twice a second, so the hint
        expires with the window rather than sitting there until the next key.
        """
        return (
            self._esc_armed_at is not None
            and self.clock() - self._esc_armed_at <= ESC_STOP_WINDOW
        )

    def _escape(self) -> bool:
        """Two escapes in quick succession stop the turn. One does nothing.

        Deliberately nothing, which is what app.py's ``action_stop_turn``
        settled on: the doubling *is* the confirmation, and a lone escape is
        too easy to arrive by accident — it is also the first byte of every
        arrow key — to end a turn on.

        It works from wherever the cursor is, the message box included, and
        that is the whole point of the gesture: a turn calling tool after tool
        is writing rows into the very list you would otherwise have to aim at.
        Escape needs no target.
        """
        now = self.clock()
        first, self._esc_armed_at = self._esc_armed_at, now
        if first is None or now - first > ESC_STOP_WINDOW:
            return False
        self._esc_armed_at = None  # spent: a third press opens a fresh pair
        self.note = "stopped the turn"
        return True

    def _handle_input(self, key: str) -> bool:
        if key == "esc":
            self._escape()
        elif key == "enter":
            self._send()
        elif key == "alt-enter":
            self.input.newline()
        elif key in ("ctrl-up", "shift-tab"):
            self.focus = CHAT
        elif key == "ctrl-down":
            self.focus = WATCHERS
        elif key == "quit":
            return False
        else:
            self.input.handle(key)
        return True

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        chat = self.chat
        chat.items.append(
            Item(head=f"you   {text}", accent=BLUE, kind="user", text=text)
        )
        chat.items.append(
            Item(
                head="hpca  looking at that now…",
                body=["(the prototype has no backend; this is where a turn would start)"],
                accent=YELLOW,
            )
        )
        chat.invalidate()
        chat.cursor = 10**9
        self.input.clear()
        self.note = "sent"

    def _handle_row(self, key: str, width: int, height: int) -> bool:
        if key in ("q", "quit"):
            return False
        inner = max(8, width - 2)
        slots = [SESSIONS, CHAT, INPUT, WATCHERS]
        view = max(1, self._heights(height, width)[slots.index(self.focus)] - 1)
        pane = {
            SESSIONS: self.session_pane,
            CHAT: self.chat,
            WATCHERS: self.watchers,
        }[self.focus]
        self.note = ""
        if key == "esc":
            self._escape()
        elif key == "?":
            self.overlay = HelpOverlay()
        elif key == "m":
            self.overlay = LlmOverlay(*self._llms)
        elif key == "a":
            self.overlay = ProfilesOverlay(sample_profiles(), self._learnings)
        elif key == "c":
            self.overlay = ConfigOverlay(self._settings_json)
        elif key in ("ctrl-down", "tab"):
            self.focus = slots[(slots.index(self.focus) + 1) % len(slots)]
        elif key in ("ctrl-up", "shift-tab"):
            self.focus = slots[(slots.index(self.focus) - 1) % len(slots)]
        elif key == "i" and self.focus == CHAT:
            self.focus = INPUT
        elif key == "up":
            pane.move(-1, view, inner)
        elif key == "down":
            pane.move(1, view, inner)
        elif key == "pgup":
            pane.move(-view, view, inner)
        elif key == "pgdn":
            pane.move(view, view, inner)
        elif key == "home":
            pane.move(-(10**9), view, inner)
        elif key == "end":
            pane.move(10**9, view, inner)
        elif key == "right":
            # Open it; on one already open, step into what it opened, the way
            # a file tree does. On an entry with no body, nothing.
            if not pane.expand(inner) and pane.current(inner) in pane.expanded:
                pane.move(1, view, inner)
        elif key == "left":
            pane.collapse(inner)
        elif key == "shift-right":
            pane.expand_all(inner)
        elif key == "shift-left":
            pane.collapse_all(inner)
        elif key in ("alt-up", "alt-down") and self.focus == WATCHERS:
            moved = pane.reorder(-1 if key == "alt-up" else 1, view, inner)
            self.note = "moved" if moved else ""
        elif key == "enter":
            if self.focus == SESSIONS:
                self._switch(self.session_pane.current(inner))
                # Straight to the box: opening a session is something you do
                # in order to say something in it.
                self.focus = INPUT
            elif self.focus == CHAT:
                self._activate_chat(inner)
            elif self.focus == WATCHERS:
                self.note = "peeking at the log"
        elif key == "r" and self.focus == SESSIONS:
            self.note = "rename: a modal in the real app"
        elif key == "t" and self.focus == SESSIONS:
            self.note = "asking the llm for a title"
        elif key == "d":
            self.note = (
                "delete session (confirm)"
                if self.focus == SESSIONS
                else "unwatched"
            )
        return True


# ------------------------------------------------------------- sample content

TASKS = [
    "annotate the cohort BAMs with sniffles",
    "why did the snakemake run stall at merge_vcf",
    "write the methods section for the SV paper",
    "check GPU utilisation on the last training job",
    "rebuild the reference index on scratch",
    "compare coverage between the two batches",
]

LONG_ASK = (
    "Use the reference on scratch rather than the one in my home directory, "
    "keep the intermediate BAMs so I can check them afterwards, and if the "
    "merge step stalls again do not retry it silently — stop and tell me which "
    "shards were still open, because last time two of them wrote to the same "
    "temp path and I only found out from the log three hours later."
)

SNIPPETS = [
    "/scratch/proj/cohort/run3/annotation.tsv (412 lines)",
    "wrote 38 lines, backup kept in trash",
    "exit 0 in 2.4s",
    "no match — falling back to a wider search",
    "/home/mayv_c/projects/sv-paper/methods.md",
    "12 files, 3.2 GB",
]

REPLIES = [
    "The merge rule stalled because two shards wrote to the same temp path.",
    "Coverage is 31x in batch A and 28x in batch B; the gap is one flowcell.",
    "I have written the methods section — it is 42 lines, have a look.",
    "The job is still queued behind a reservation; nothing is wrong with it.",
]

SETTINGS_JSON = """{
  "llm": {
    "base_url": "http://localhost:20001/v1",
    "model": "qwen3-27b-fp8",
    "context_window": 112000,
    "temperature": 0.2
  },
  "database": {
    "local_cache": true,
    "sync_interval_s": 60
  },
  "logging": {
    "enabled": true,
    "dir": ""
  },
  "agent": {
    "mode": "agent",
    "max_steps": 40
  }
}
"""

LEARNINGS = {
    "hpc": (
        "The cluster's scratch is /scratch/proj, and $HOME is NFS — never write\n"
        "large intermediates to $HOME.\n"
        "Slurm partitions: gpu (a100), cpu-long, cpu-short.\n"
        "Prefers sniffles over cuteSV for long-read SV calling.\n"
    ),
    "default": "(nothing learned yet)\n",
    "writing": "Writes in British spelling. Dislikes bullet lists in prose.\n",
}


def sample_chat(count: int, seed: int = 0, task: str = "") -> list[Item]:
    """One session's conversation. ``seed`` shifts the content so that two
    sessions never look alike, and ``task`` is what the user keeps asking
    about — a session is about one thing, and that is what makes a switch
    visible at a glance."""
    tools = ["read_file", "edit_file", "create_file", "run_bash", "list_dir"]
    task = task or TASKS[seed % len(TASKS)]
    items: list[Item] = []
    for n in range(count):
        i = n + seed * 7
        slot = n % 3
        if slot == 0:
            # Every fourth one is long, so that the rewind's preview has
            # something to truncate and the wrapping is exercised by running
            # the thing rather than only by the headless check.
            said = task if n % 12 else f"{task}. {LONG_ASK}"
            items.append(
                Item(head=f"you   {said}", accent=BLUE, kind="user", text=said)
            )
        elif slot == 1:
            steps = 3 + (i * 5) % 18
            names = [tools[(i + k) % len(tools)] for k in range(steps)]
            items.append(
                Item(
                    head=f"      {steps} steps · " + " → ".join(names[:3]) + " …",
                    body=[
                        f"{name:<14}{SNIPPETS[(i + k) % len(SNIPPETS)]}"
                        for k, name in enumerate(names)
                    ],
                )
            )
        else:
            reply = REPLIES[i % len(REPLIES)]
            items.append(
                Item(
                    head=f"hpca  {reply}",
                    body=[
                        reply,
                        "The detail is in the log at "
                        "/scratch/proj/cohort/run3/logs/merge_vcf.log, and the "
                        "two shards are listed at the bottom of it.",
                    ],
                    accent=YELLOW,
                )
            )
    return items


def sample_watchers(count: int, seed: int = 0) -> list[Item]:
    states = [("RUNNING", GREEN), ("PENDING", ""), ("COMPLETED", ""), ("FAILED", RED)]
    out = []
    for n in range(count):
        i = n + seed * 3
        state, accent = states[i % len(states)]
        out.append(
            Item(
                head=(
                    f"{'job ' + str(4821000 + i):<16}{state:<11}"
                    f"last write {3 + i * 11}s ago"
                ),
                body=[
                    f"/scratch/proj/cohort/run{seed}/logs/step{n}.log",
                    "[12:41:07] merging shard 3 of 8",
                    "[12:41:44] merging shard 4 of 8",
                ],
                accent=accent,
            )
        )
    return out


def sample_profiles() -> list[Item]:
    return [
        Item(
            head=f"{'hpc':<18}12 memories · 4 skills · 3 sessions",
            body=["created 2026-04-02", "used by the open session"],
            accent=GREEN,
        ),
        Item(
            head=f"{'default':<18}0 memories · 0 skills · 1 session",
            body=["created 2026-01-11", "the fallback; cannot be deleted"],
        ),
        Item(
            head=f"{'writing':<18}5 memories · 1 skill · 0 sessions",
            body=["created 2026-06-20", "copied from hpc"],
        ),
        Item(head="(new profile)"),
    ]


def sample_llms() -> tuple[list[Item], list[Item]]:
    discovered = [
        Item(
            head="● 10.12.4.31:20001    qwen3-27b-fp8      112k ctx",
            body=["found in the shared endpoints manifest", "node gpu014, 42s ago"],
            accent=GREEN,
        ),
        Item(
            head="● localhost:20001     qwen3-27b-fp8      112k ctx",
            body=["found by localhost scan (ssh tunnel)"],
            accent=GREEN,
        ),
        Item(
            head="○ 10.12.4.55:20001    llama-3.3-70b       128k ctx",
            body=["in the manifest, did not answer the probe"],
        ),
    ]
    # The whole width, which is the point of stacking these rather than
    # putting them in a column: url, model and context all fit on one line.
    configured = [
        Item(
            head="★ ● cluster-qwen   http://10.12.4.31:20001/v1   qwen3-27b-fp8   112k",
            body=["the active default", "answered the last probe 42s ago"],
            accent=GREEN,
        ),
        Item(
            head="  ● tunnel-qwen    http://localhost:20001/v1     qwen3-27b-fp8   112k",
            body=["the same server through an ssh tunnel"],
            accent=GREEN,
        ),
        Item(
            head="  ○ big-llama      http://10.12.4.55:20001/v1    llama-3.3-70b   128k",
            body=["not answering; last seen 3d ago"],
        ),
    ]
    return discovered, configured


# ---------------------------------------------------------------------- main


def build(chat: int = 400, sessions: int = 14, watchers: int = 5) -> RowUI:
    states = [
        SessionState(
            title=TASKS[i % len(TASKS)],
            profile=("hpc", "writing", "default")[i % 3],
            model="qwen3-27b-fp8" if i % 2 == 0 else "llama-3.3-70b",
            started=f"{2 + i * 7}m ago",
            # Varied on purpose: a session with a handful of entries next to
            # one with hundreds is what shows the chat row taking up the slack.
            size=max(6, chat // (1 + i % 4)),
            # And some with no watches at all, which is the case the layout
            # charges nothing for.
            watch_count=watchers if i == 0 else (i * 3) % 4,
            index=i,
        )
        for i in range(sessions)
    ]
    return RowUI(
        states,
        learnings=dict(LEARNINGS),
        settings_json=SETTINGS_JSON,
        llms=sample_llms(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Row-oriented HPCA UI prototype (no Textual)."
    )
    parser.add_argument("--chat", type=int, default=400, help="chat entries")
    parser.add_argument("--sessions", type=int, default=14)
    parser.add_argument("--watchers", type=int, default=5)
    args = parser.parse_args(argv)

    if not sys.stdin.isatty():
        print("row_ui needs a terminal.", file=sys.stderr)
        return 2

    ui = build(args.chat, args.sessions, args.watchers)
    with Screen() as screen:
        size = (0, 0)
        while True:
            width, height = os.get_terminal_size()
            resized = (width, height) != size
            if resized:
                size = (width, height)
                ui.invalidate()
            started = time.perf_counter()
            screen.paint(ui.render(width, height), full=resized)
            # Shown on the next frame rather than this one: measuring the frame
            # that reports the measurement would need two passes to say
            # anything true.
            ui.frame_ms = (time.perf_counter() - started) * 1000
            # The timeout is also how fast a resize is noticed; SIGWINCH would
            # be tidier and is not worth a handler in a prototype.
            if not select.select([screen.fd], [], [], 0.5)[0]:
                continue
            data = os.read(screen.fd, 4096)
            if not data:
                return 0
            for key in decode(data):
                if not ui.handle(key, width, height):
                    return 0


if __name__ == "__main__":
    raise SystemExit(main())
