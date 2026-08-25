"""The shared multi-line text buffer, and the line-folding it renders through."""

from __future__ import annotations

import contextlib

from hpca.ui import theme
from hpca.ui.ansi import (
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

# How many undo steps a buffer keeps. A step is a whole copy of the text
# rather than a recorded diff: the buffers here are a message someone is
# typing and a settings file, both small, and an inverse-diff that is wrong in
# one case corrupts the file it was meant to protect. Two hundred is deep
# enough that the config editor — the one place a long editing session
# actually happens — never runs out in practice.
UNDO_DEPTH = 200


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
        # Undo, as two stacks of ``(lines, row, col)``. The cursor is part of
        # a snapshot because landing back at the site of the edit is what
        # makes a second ctrl+z read as continuous; the scroll ``offset`` is
        # not, because ``render`` recomputes it to keep the cursor on screen
        # and a restored one would only fight it.
        self._undo: list[tuple[list[str], int, int]] = []
        self._redo: list[tuple[list[str], int, int]] = []
        # Which run of same-kind edits is open, so that typing a word is one
        # step rather than five. None means the next edit starts its own.
        self._run: str | None = None
        # Depth of the compound edit in progress: an edit built out of other
        # edits (typing over a selection, deleting a word) opens exactly one
        # step, and the parts it is made of must not open more.
        self._depth = 0

    # ------------------------------------------------------------- content

    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def revision(self) -> int:
        """Bumped by every change to the text. What a caller watches to learn
        that a keypress *edited* rather than merely moved the cursor."""
        return self._rev

    def _touch(self) -> None:
        self._rev += 1

    def clear(self) -> None:
        with self._compound(None):
            self.lines = [""]
            self.row = self.col = self.offset = 0
            self.anchor = None
            self._touch()

    def set_text(self, text: str) -> None:
        """Replace the buffer, cursor left at the end of it."""
        with self._compound(None):
            self.replace(text)

    def replace(self, text: str) -> None:
        """``set_text`` with the undo stack left alone.

        The one buffer swap that is not an edit: walking the message history
        with ↑/↓ replaces the draft over and over, and each step is already
        reversible by pressing the other arrow. Putting those on the undo
        stack would give ctrl+z a second, subtly different way to walk
        backwards — retracing the browse rather than stepping to an older
        message — which is two mechanisms for one job.
        """
        self.lines = text.split("\n") or [""]
        self.row = len(self.lines) - 1
        self.col = len(self.lines[self.row])
        self.anchor = None
        self._run = None
        self._touch()

    def insert(self, ch: str) -> None:
        # A run of typing is one step, broken after whitespace so that one
        # ctrl+z takes back one word rather than the whole paragraph — and
        # typing *over* a selection is always its own step, because what it
        # undoes is a replacement rather than another letter.
        kind = None if self.sel_range() is not None else "insert"
        with self._compound(kind):
            self.delete_selection()
            line = self.lines[self.row]
            self.lines[self.row] = line[: self.col] + ch + line[self.col :]
            self.col += 1
            self._touch()
        if ch.isspace():
            self._run = None

    def insert_text(self, text: str) -> None:
        """Insert a whole block at the cursor — what a paste is.

        Newlines in it become newlines in the buffer, never a send: the caller
        already knows this arrived as one bracketed unit rather than as typing,
        which is the entire reason the terminal is asked to bracket pastes.
        """
        # One step, always: a paste arrived as one unit and it comes back off
        # the undo stack as one, however many lines it turned out to be.
        with self._compound(None):
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
        with self._compound(None):
            self.delete_selection()
            line = self.lines[self.row]
            self.lines[self.row : self.row + 1] = [line[: self.col], line[self.col :]]
            self.row += 1
            self.col = 0
            self._touch()

    def backspace(self) -> None:
        kind = None if self.sel_range() is not None else "backspace"
        with self._compound(kind):
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
        kind = None if self.sel_range() is not None else "delete"
        with self._compound(kind):
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
        with self._compound(None):
            (r1, c1), (r2, c2) = span
            self.lines[r1 : r2 + 1] = [self.lines[r1][:c1] + self.lines[r2][c2:]]
            self.row, self.col = r1, c1
            self._touch()
        return True

    # ----------------------------------------------------------------- undo

    def _snapshot(self) -> tuple[list[str], int, int]:
        return (self.lines[:], self.row, self.col)

    def _step(self, kind: str | None) -> None:
        """Open an undo step in front of an edit, if this edit starts one.

        ``kind`` names the run the edit belongs to; None is an edit that is
        always a step of its own — a paste, a newline, a whole-buffer swap, a
        word deletion, anything that replaces a selection.
        """
        if self._depth:
            return  # a compound edit already opened the one step it gets
        if kind is None or kind != self._run:
            self._undo.append(self._snapshot())
            del self._undo[: max(0, len(self._undo) - UNDO_DEPTH)]
        self._run = kind
        # Unconditional, including mid-run: anything typed after an undo is a
        # new branch, and the future it replaced is not coming back.
        self._redo.clear()

    @contextlib.contextmanager
    def _compound(self, kind: str | None):
        """One undo step around an edit that is built out of other edits."""
        self._step(kind)
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def _restore(self, snap: tuple[list[str], int, int]) -> None:
        self.lines, self.row, self.col = snap[0][:], snap[1], snap[2]
        self.row = max(0, min(self.row, len(self.lines) - 1))
        self.col = max(0, min(self.col, len(self.lines[self.row])))
        # The selection is not restored with the text. Storing the anchor
        # would bring back a highlight that the next keystroke silently wipes
        # again, which is a worse place to be than an unmarked buffer.
        self.anchor = None
        self._run = None
        self._touch()

    def undo(self) -> bool:
        """Back one step. False when there is nothing to go back to."""
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        """Forward one step, as far as the last undo went."""
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def reset_undo(self) -> None:
        """Forget both stacks: what is in the buffer now is the beginning.

        What a send does to the message box. The stack is about the message
        being written, and once it has gone to the core the way back to it is
        the history the ↑ key walks — not an undo that would stage a second
        copy of something already sent.
        """
        self._undo.clear()
        self._redo.clear()
        self._run = None

    def _mark(self, extend: bool) -> None:
        """Called before every motion: shift keeps an anchor, others drop it.

        Also where a run of edits ends. Moving the cursor is the boundary
        between "still typing that word" and "typing somewhere else", and it
        is the one hook every key-driven motion passes through — the motions a
        compound edit makes internally (``delete_word_left``) do not, which is
        what keeps them one step.
        """
        self._run = None
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
        with self._compound(None):
            if self.delete_selection():
                return
            self.anchor = (self.row, self.col)
            self.word_left()
            self.delete_selection()

    def delete_word_right(self) -> None:
        with self._compound(None):
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

    def at_first_row(self) -> bool:
        """Is the cursor on the top screen line of the buffer?

        Measured at the width the last render used, like ``move_line``: the
        keys are answered before the next frame is drawn, so that is the width
        the line the user is looking at was folded at. It is also the only
        width available where these are asked from — a keypress carries no
        geometry, and threading one down would change the signature of every
        caller that drives keys.
        """
        return self.cursor_visual(self._last_width)[0] == 0

    def at_last_row(self) -> bool:
        """Is the cursor on the bottom screen line of the buffer?"""
        width = self._last_width
        return self.cursor_visual(width)[0] == self.height(width) - 1

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
        elif key == "ctrl-z":
            self.undo()
        elif key == "ctrl-y":
            self.redo()
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
                prefix = f"{theme.faint}{row + 1:>{gutter - 1}} {RESET}"
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
