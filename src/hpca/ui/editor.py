"""The shared multi-line text buffer, and the line-folding it renders through."""

from __future__ import annotations

from hpca.ui.ansi import (
    DIM,
    RESET,
    cell_width,
    char_width,
    fit_index,
    pad,
    reverse,
    wrap_spans,
)

# Re-exported: the folding lives in ``ansi`` with the rest of the cell-width
# arithmetic, but it is the message box's geometry and this is where it is
# looked for.
__all__ = ["Editor", "wrap_spans"]


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

    def insert_text(self, text: str) -> None:
        """Insert a whole block at the cursor — what a paste is.

        Newlines in it become newlines in the buffer, never a send: the caller
        already knows this arrived as one bracketed unit rather than as typing,
        which is the entire reason the terminal is asked to bracket pastes.
        """
        self.delete_selection()
        parts = text.split("\n")
        line = self.lines[self.row]
        head, tail = line[: self.col], line[self.col :]
        if len(parts) == 1:
            self.lines[self.row] = head + parts[0] + tail
            self.col += len(parts[0])
        else:
            self.lines[self.row : self.row + 1] = [
                head + parts[0],
                *parts[1:-1],
                parts[-1] + tail,
            ]
            self.row += len(parts) - 1
            self.col = len(parts[-1])
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
        # The column is kept in *cells*, not characters: stepping off a row of
        # ideographs onto one of latin text has to land under the cursor, and
        # those two rows hold a different number of characters in the same
        # number of columns.
        here = rows[min(index, len(rows) - 1)]
        column = cell_width(self.lines[here[0]][here[1] : here[1] + offset])
        row, start, end = rows[target]
        self.row = row
        line = self.lines[row]
        self.col = min(fit_index(line, start, column), end, len(line))

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
                rows += [(index, s, e) for s, e in wrap_spans(line, width)]
            else:
                rows.append((index, 0, len(line)))
        self._vis, self._vis_key = rows, key
        return rows

    def cursor_visual(self, width: int) -> tuple[int, int]:
        """Which screen line the cursor sits on, and how far into it."""
        rows = self.visual(width)
        if not rows:  # only reachable if wrap_spans ever stops answering
            return 0, 0
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
        # offset into the line. Measured in cells and answered in characters:
        # the smallest number of characters to drop from the front that leaves
        # the cursor inside the box.
        hoff = 0
        if not self.wrap and rows:
            here = rows[min(cursor_row, len(rows) - 1)]
            line, start = self.lines[here[0]], here[1]
            # Walked backwards from the cursor rather than forwards from the
            # start of the line: the answer is at most ``body`` cells away, and
            # a JSON file edited at column 90 must not cost its whole line on
            # every keystroke.
            used, at = 0, start + cursor_col
            while at > start and used + char_width(line[at - 1]) <= body - 2:
                used += char_width(line[at - 1])
                at -= 1
            hoff = at - start
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
            seg = self.lines[row][start:end]
            text = pad(seg[hoff : fit_index(seg, min(hoff, len(seg)), body)], body)
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
            out.append(prefix + reverse(text, marks))
        return out
