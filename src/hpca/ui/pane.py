"""One navigable list of entries: the row that sessions, chat and watchers are."""

from __future__ import annotations

from dataclasses import dataclass, field

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, REVERSE, fold, pad, rule


@dataclass
class Fold:
    """One part of an entry, foldable on its own inside the entry's own fold.

    A turn's working is one row when it is over — "14 steps · read_file →
    run_bash …" — and that row opens into the steps, each of which opens into
    what the tool actually returned. Two levels rather than one because the
    two questions are different sizes: "what did it do" is a line per step,
    and "what did that print" is four hundred lines of log that must not be in
    the way of the step after it (`tui/app.py`'s StepBox made the same split).
    """

    head: str
    body: list[str] = field(default_factory=list)


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

    ``folds`` are sub-rows with the same property one level down: each is
    addressed as ``<key>/<n>``, so the third step of a turn stays open across
    the `chat.update` that revises the row, and cannot be inherited by
    whatever step ends up third next time.
    """

    head: str
    body: list[str] = field(default_factory=list)
    accent: str = ""
    kind: str = ""
    text: str = ""
    key: str = ""
    folds: list[Fold] = field(default_factory=list)

    @property
    def openable(self) -> bool:
        return bool(self.body or self.folds)


class Pane:
    """One navigable list of entries.

    Holds its own cursor line, scroll offset and set of open entries, which is
    what makes "each row remembers where you were" fall out rather than need
    arranging: leaving a pane changes nothing about it.

    Two things are addressed by *key* rather than by position, for the same
    reason — a list the core repaints under the user must not move what the
    user had open. Entries are keyed by `Item.key`, and each entry's folds by
    ``<key>/<n>`` beneath it.
    """

    def __init__(self, name: str, items: list[Item]) -> None:
        self.name = name
        self.items = items
        # Keys, not positions: see `key_at`.
        self.expanded: set[str] = set()
        self.cursor = 0  # index into the flattened line list
        self.offset = 0  # first visible flattened line
        # A row pinned after the last entry, redrawn from the clock rather
        # than from an event: the working indicator (§4.3 item 16). Kept out
        # of `items` so the chat's rows stay one-for-one with the entries the
        # core sent — the spinner is not a transcript row and must never be
        # numbered as one — and so a frame of it costs one cached line rather
        # than a rebuild of the cache (`set_tail`).
        self.tail: Item | None = None
        self._flat: list[tuple[int, str, bool]] | None = None
        # Parallel to `_flat`: what each line belongs to, for the two levels
        # of fold. `_openable` is every key on this pane with something behind
        # it, so "is there anything to open here" is a set lookup rather than
        # a parse of a key back into a row and a part.
        self._keys: list[str] = []
        self._openable: set[str] = set()
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
        keys: list[str] = []
        openable: set[str] = set()
        for index, item in enumerate(self.items):
            key = self.key_at(index)
            if item.openable:
                openable.add(key)
            opened = key in self.expanded
            marker = ("▾" if opened else "▸") if item.openable else " "
            lines.append((index, f"{marker} {item.head}", True))
            keys.append(key)
            if not opened:
                continue
            for raw in item.body:
                # Folded by cells rather than by characters: a body line of
                # CJK holds half as many characters in the same row, and
                # counting them would leave the row over the width and the
                # padding to truncate what did not fit.
                for piece in fold(raw, max(8, width - 4)):
                    lines.append((index, f"    {piece}", False))
                    keys.append(key)
            for n, part in enumerate(item.folds):
                sub = f"{key}/{n}"
                if part.body:
                    openable.add(sub)
                sub_open = sub in self.expanded
                mark = ("▾" if sub_open else "▸") if part.body else " "
                lines.append((index, f"  {mark} {part.head}", True))
                keys.append(sub)
                if not sub_open:
                    continue
                for raw in part.body:
                    for piece in fold(raw, max(8, width - 6)):
                        lines.append((index, f"      {piece}", False))
                        keys.append(sub)
        if self.tail is not None:
            lines.append(self._tail_line())
            keys.append(self.key_at(len(self.items)))
        self._flat, self._keys, self._openable = lines, keys, openable
        self._flat_width = width
        return lines

    def _tail_line(self) -> tuple[int, str, bool]:
        return (len(self.items), f"  {self.tail.head}", True)

    def set_tail(self, item: Item | None) -> None:
        """Pin (or take away) the live row after the last entry.

        A spinner frame changes one line and must not cost a relayout of the
        log — the Textual widget this replaces carried the same rule, and paid
        44ms of event-loop lag at 300 messages for getting it wrong. So a tail
        that is merely *redrawn* patches the cached line in place; only its
        arrival or departure changes how many lines there are, and only that
        invalidates.
        """
        was, self.tail = self.tail, item
        if (item is None) != (was is None):
            self.invalidate()
        elif item is not None and self._flat is not None:
            self._flat[-1] = self._tail_line()
            self._keys[-1] = self.key_at(len(self.items))

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
        row = self.item_at(item)
        if row is None:
            return ""
        return row.key or f"#{item}"

    def item_at(self, index: int) -> Item | None:
        """The row at that position, the live tail included — it is one too."""
        if 0 <= index < len(self.items):
            return self.items[index]
        if index == len(self.items) and self.tail is not None:
            return self.tail
        return None

    def is_tail(self, index: int) -> bool:
        """Whether that position is the live row rather than an entry."""
        return self.tail is not None and index == len(self.items)

    def row_key(self, width: int) -> str:
        """The key of the *line* the cursor is on — an entry, or one of its
        folds. What expand and collapse act on, since the cursor addresses
        lines and a fold is not an entry."""
        lines = self.flat(width)
        if not lines:
            return ""
        return self._keys[min(self.cursor, len(lines) - 1)]

    def is_open(self, item: int) -> bool:
        """Whether the entry at that position is showing its body."""
        return self.key_at(item) in self.expanded

    def _fold_keys(self, index: int) -> set[str]:
        """Everything the pane may remember about one entry."""
        item = self.item_at(index)
        if item is None:
            return set()
        key = self.key_at(index)
        return {key} | {f"{key}/{n}" for n in range(len(item.folds))}

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
        self.invalidate()
        keys = [self.key_at(i) for i in range(len(items))]
        kept: set[str] = set()
        for index in range(len(items)):
            kept |= self._fold_keys(index)
        self.expanded &= kept
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

    def _go_to_key(self, key: str, width: int) -> None:
        self.flat(width)
        for row, name in enumerate(self._keys):
            if name == key:
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
        """Open the row under the cursor. False if there was nothing to open.

        The row, not the entry: on a step of an opened turn this opens that
        step, so the second level is reached with the same key as the first.
        """
        key = self.row_key(width)
        if not key or key in self.expanded or key not in self._openable:
            return False
        self.expanded.add(key)
        self.invalidate()
        # Land back on the row's own first line: opening one twelve lines
        # long and being left in the middle of it reads as a jump.
        self._go_to_key(key, width)
        return True

    def collapse(self, width: int) -> bool:
        """Close what the cursor is inside. False if nothing was open.

        Inside an open step it closes the step; on a closed step it closes the
        entry that holds it, which is how ← walks back out of a tree instead
        of having to be aimed at the head line first.
        """
        key = self.row_key(width)
        if key in self.expanded:
            self.expanded.discard(key)
            self.invalidate()
            self._go_to_key(key, width)
            return True
        owner = self.key_at(self.current(width))
        if owner and owner in self.expanded:
            self.expanded.discard(owner)
            self.invalidate()
            self._go_to_key(owner, width)
            return True
        return False

    def expand_all(self, width: int) -> None:
        item = self.current(width)
        # Walked over the items rather than read off `_openable`, which only
        # ever knows about the lines that were on screen: the steps of a
        # closed turn have never been flattened, and "open everything" has to
        # reach them too.
        keys: set[str] = set()
        for index, row in enumerate(self.items):
            key = self.key_at(index)
            if row.openable:
                keys.add(key)
            keys |= {f"{key}/{n}" for n, part in enumerate(row.folds) if part.body}
        self.expanded = keys
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
        if item < 0 or item >= len(self.items):
            return False  # the live row is not an entry, and does not move
        if not 0 <= target < len(self.items):
            return False
        was = (self.is_open(item), self.is_open(target))
        self.expanded -= self._fold_keys(item) | self._fold_keys(target)
        self.items[item], self.items[target] = self.items[target], self.items[item]
        self.expanded -= self._fold_keys(item) | self._fold_keys(target)
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
            item = self.item_at(owner)
            if row == self.cursor:
                # The unfocused pane still shows where it was left, dimmed —
                # that is the "memory" being visible rather than merely kept.
                painted = (REVERSE if focused else DIM + REVERSE) + painted + RESET
            elif not is_head:
                painted = DIM + painted + RESET
            elif item is not None and item.accent:
                painted = item.accent + painted + RESET
            out.append(painted)
        return out
