"""One navigable list of entries: the row that sessions, chat and watchers are."""

from __future__ import annotations

from dataclasses import dataclass, field

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, REVERSE, fold, pad, rule


@dataclass
class Item:
    """One entry: a single line, plus the body it opens into.

    ``kind`` and ``text`` are what Enter needs: the log has to be able to say
    which rows are the user's own words, and to hand back the words themselves
    rather than the decorated line they are drawn as (app.py keeps the same two
    on Entry, see OWN_MESSAGE_KINDS).

    ``key`` is what the row *is*, for the lists whose rows have an identity
    that outlives their position — a chat entry's core-assigned ``seq``, a
    sidebar row's session id, a watch box's ``PanelRow.key``. Everything a pane
    remembers per row is remembered against it, so a row arriving above another
    cannot silently take over what was open.
    """

    head: str
    body: list[str] = field(default_factory=list)
    accent: str = ""
    kind: str = ""
    text: str = ""
    key: str = ""


class Pane:
    """One navigable list of entries.

    Holds its own cursor line, scroll offset and set of open entries, which is
    what makes "each row remembers where you were" fall out rather than need
    arranging: leaving a pane changes nothing about it.
    """

    def __init__(self, name: str, items: list[Item]) -> None:
        self.name = name
        self.items = items
        # Keys, not positions: see `key_at`.
        self.expanded: set[str] = set()
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
            opened = self.key_at(index) in self.expanded
            marker = ("▾" if opened else "▸") if item.body else " "
            lines.append((index, f"{marker} {item.head}", True))
            if opened:
                for raw in item.body:
                    # Folded by cells rather than by characters: a body line of
                    # CJK holds half as many characters in the same row, and
                    # counting them would leave the row over the width and the
                    # padding to truncate what did not fit.
                    for piece in fold(raw, max(8, width - 4)):
                        lines.append((index, f"    {piece}", False))
        self._flat = lines
        self._flat_width = width
        return lines

    def invalidate(self) -> None:
        self._flat = None

    def key_at(self, item: int) -> str:
        """What the row at that position is called.

        A row that carries a ``key`` is addressed by it; one that does not
        falls back to its position, which is what this pane used to do for
        every row and is still right for a list that is only ever appended to
        or reordered whole. The ``#`` keeps the two namespaces apart, since a
        real key is a session id or a number the core assigned.
        """
        if 0 <= item < len(self.items):
            return self.items[item].key or f"#{item}"
        return ""

    def is_open(self, item: int) -> bool:
        """Whether the entry at that position is showing its body."""
        return self.key_at(item) in self.expanded

    def replace(self, items: list[Item], width: int | None = None) -> None:
        """Take a new list of rows, keeping what the user had done to the old.

        The cursor stays on the row it was on — by key, so a row inserted above
        it does not move the selection — and rows that are still here stay
        open. Rows that are gone take their state with them rather than leaving
        a key behind for a later row to inherit.

        Only for the panes the core repaints whole (the sidebar, the watchers).
        The chat is never rebuilt; it appends (specs-ui-replacement.md §3.2).
        """
        # The client repaints a column without knowing the terminal size; the
        # width only decides which flattened line the cursor lands on, and the
        # last one this pane was drawn at is the right answer for that.
        width = self._flat_width if width is None else width
        was = self.key_at(self.current(width))
        self.items = items
        keys = [self.key_at(i) for i in range(len(items))]
        self.expanded &= set(keys)
        self.invalidate()
        if was in keys:
            self._go_to(keys.index(was), width)

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
        if item < 0 or not self.items[item].body or self.is_open(item):
            return False
        self.expanded.add(self.key_at(item))
        self.invalidate()
        # Land back on the entry's own first line: opening one twelve lines
        # long and being left in the middle of it reads as a jump.
        self._go_to(item, width)
        return True

    def collapse(self, width: int) -> bool:
        """Close the entry the cursor is anywhere inside. False if it was shut."""
        item = self.current(width)
        if item < 0 or not self.is_open(item):
            return False
        self.expanded.discard(self.key_at(item))
        self.invalidate()
        self._go_to(item, width)
        return True

    def expand_all(self, width: int) -> None:
        item = self.current(width)
        self.expanded = {
            self.key_at(i) for i, entry in enumerate(self.items) if entry.body
        }
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
        instead of shuffling a different one each press.

        A row that carries a key takes what was open with it for free. One that
        does not is keyed by position, so the two slots have just traded flags
        and the flags are put back by hand — the same patch this always needed,
        written once against ``key_at`` rather than against bare ints.
        """
        item = self.current(width)
        target = item + delta
        if item < 0 or not 0 <= target < len(self.items):
            return False
        was = (self.is_open(item), self.is_open(target))
        self.expanded.difference_update({self.key_at(item), self.key_at(target)})
        self.items[item], self.items[target] = self.items[target], self.items[item]
        self.expanded.difference_update({self.key_at(item), self.key_at(target)})
        if was[0]:
            self.expanded.add(self.key_at(target))
        if was[1]:
            self.expanded.add(self.key_at(item))
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
        title = rule(self.name, width, right)
        out = [(BOLD + CYAN if focused else DIM) + title + RESET]
        for row in range(self.offset, self.offset + body_h):
            if row >= len(lines):
                out.append(" " * width)
                continue
            owner, text, is_head = lines[row]
            gutter = "▌ " if owner == current else "  "
            painted = pad(gutter + text, width)
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
