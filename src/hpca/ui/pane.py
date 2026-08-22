"""One navigable list of entries: the row that sessions, chat and watchers are."""

from __future__ import annotations

from dataclasses import dataclass, field

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, REVERSE, clip, fold, pad, rule

# How far a row's head sits from its fold marker, on the two kinds of pane.
#
# One space is all a marker needs to stand off the word after it, and on the
# lists that is the whole job: their bodies are indented four columns under
# the head, so which lines are the pane talking *about* a row and which are
# the row itself is never in question there. The chat has no such indent — a
# message's words start at column 0 so that a drag-select picks up prose and
# nothing else, see `Pane.flush` — and there the head line's only claim to
# being furniture was the marker, which a row with nothing to open does not
# even carry. So on that pane the gap is four spaces: the column of `you` /
# `hpca` / `12 steps` heads stands off the prose under it at a glance. It is
# conditional rather than global because on a list it would buy nothing and
# cost something — the heads would sit four columns right of the bodies that
# currently line up under them.
FLUSH_HEAD_GAP = "    "
LIST_HEAD_GAP = " "


class Wrapped:
    """A row that remembers its ``body`` wrapped, so nothing wraps it twice.

    The flattened line list is rebuilt whenever a row is revised, and every
    row on the pane is measured again when it is. Without a memo that means
    re-wrapping every open body on each rebuild — work that grows with how
    much *text* the conversation holds rather than with how many rows it has,
    and 60% of a rebuild on a 5000-entry chat with the messages open.
    `Item.clipped` is the same bargain for the closed ones, which is a third
    of a rebuild when they are.

    The memo is per row, so revising one row costs that row. It is keyed by
    width, because a resize is the one thing that legitimately invalidates it,
    and it assumes ``body`` is replaced rather than mutated in place — which
    is what every builder here does: `state.entry_item` returns a new row for
    a revised entry rather than editing the old one.
    """

    body: list[str]

    def folded(self, width: int) -> list[str]:
        memo = getattr(self, "_folded", None)
        if memo is None or memo[0] != width:
            memo = (width, [piece for raw in self.body for piece in fold(raw, width)])
            self._folded = memo
        return memo[1]

@dataclass
class Fold(Wrapped):
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
class Item(Wrapped):
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

    ``label`` says the head is a *nameplate* rather than content — `you
    22-08-2026 13:04:47`, `hpca 22-08-2026 13:05:02` — and is what gets it
    drawn bold. It is a flag rather than an escape sequence baked into `head`
    because every line is padded to an exact number of cells before any SGR is
    wrapped around it (`ansi.pad`), and a head that arrived already styled
    would be measured with its escapes in it. The rows whose head is a summary
    of what is under it (a turn's "8 steps · …") are not labels and are not
    bold: the weight is there to say where one turn ends and the next begins,
    which is a thing only the speaker lines can say.

    ``preview`` is the one line a *closed* row shows of what it is hiding —
    the whole of `body` on one line, clipped to the terminal with `ansi.clip`.
    It is what makes a folded conversation still readable as one: a column of
    `you` and `hpca` labels with nothing between them says who spoke and not a
    word of what was said. A row with no preview closes to its head alone,
    which is right for the rows whose head already summarises them (a turn's
    working: "8 steps · edit_file → run_bash …").
    """

    head: str
    body: list[str] = field(default_factory=list)
    preview: str = ""
    accent: str = ""
    label: bool = False
    kind: str = ""
    text: str = ""
    key: str = ""
    folds: list[Fold] = field(default_factory=list)

    def clipped(self, cells: int) -> str:
        """``preview``, cut to one line — the closed row's half of `folded`.

        On `Item` rather than on `Wrapped` because `preview` is: a step has a
        body and no preview, and reaching for one through a defaulted
        ``getattr`` would make a row that quietly previews nothing look the
        same as a row that has nothing to preview.
        """
        memo = getattr(self, "_clipped", None)
        if memo is None or memo[0] != cells:
            memo = (cells, clip(self.preview, cells))
            self._clipped = memo
        return memo[1]

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

    def __init__(
        self, name: str, items: list[Item], *, flush: bool = False
    ) -> None:
        self.name = name
        self.items = items
        # Whether the rows on this pane are *text somebody will select*.
        #
        # A pane spends four columns before a character of its content is
        # drawn — two on the gutter that bands the current entry, two more on
        # the fold marker, and four again on the indent under it — and a
        # terminal's own drag-to-select takes every one of them along with the
        # words. Selecting out of the chat is the terminal's job here
        # (specs-ui-replacement.md §4.1 item 6), so on the one pane whose rows
        # are somebody's prose the columns come off: content lines start at
        # column 0 with nothing in front of them, and the current entry is
        # marked by weight instead of by a character. The lists — sessions,
        # watchers, the overlays — keep the gutter, because a title in a list
        # is not something anyone pastes into a shell.
        self.flush = flush
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
            rows, names, opens = self._item_lines(index, item, width)
            lines += rows
            keys += names
            openable |= opens
        if self.tail is not None:
            lines.append(self._tail_line())
            keys.append(self.key_at(len(self.items)))
        self._flat, self._keys, self._openable = lines, keys, openable
        self._flat_width = width
        return lines

    def _item_lines(
        self, index: int, item: Item, width: int
    ) -> tuple[list[tuple[int, str, bool]], list[str], set[str]]:
        """The lines one row draws as, what each belongs to, and what opens.

        Split out of `flat` so `extend` can ask for one row's worth without
        rebuilding the pane — see there for why that matters.

        The indent is where `flush` shows up. Content lines lose it entirely;
        the head lines that carry a marker keep theirs, because those are the
        pane talking *about* the row rather than the row itself, and the two
        are meant to be told apart at a glance. Anything at column 0 is
        verbatim and selects as-is; anything indented is furniture — which is
        also why the gap between marker and head is wider there
        (`FLUSH_HEAD_GAP`): with the body flush, the head's indent is the only
        thing left saying it is not part of the text.
        """
        body_pad = "" if self.flush else "    "
        step_pad = "" if self.flush else "      "
        gap = self.head_gap
        lines: list[tuple[int, str, bool]] = []
        keys: list[str] = []
        openable: set[str] = set()
        key = self.key_at(index)
        if item.openable:
            openable.add(key)
        opened = key in self.expanded
        # The gap goes on whatever occupies the marker slot — the open
        # marker, the closed one, or the blank a row with nothing behind it
        # gets. The heads are a column, and a column that shifted as its rows
        # opened and closed would read as the pane twitching rather than as a
        # fold.
        marker = ("▾" if opened else "▸") if item.openable else " "
        lines.append((index, f"{marker}{gap}{item.head}", True))
        keys.append(key)
        if not opened:
            # Closed: neither its words nor its steps — both hang off the same
            # marker, which is what "open the row" means — but one line of
            # what it is hiding, if it has one to give.
            if item.preview:
                room = max(8, width - len(body_pad))
                lines.append((index, body_pad + item.clipped(room), False))
                keys.append(key)
            return lines, keys, openable
        # Folded by cells rather than by characters: a body line of CJK holds
        # half as many characters in the same row, and counting them would
        # leave the row over the width and the padding to truncate what did
        # not fit.
        for piece in item.folded(max(8, width - len(body_pad))):
            lines.append((index, f"{body_pad}{piece}", False))
            keys.append(key)
        for n, part in enumerate(item.folds):
            sub = f"{key}/{n}"
            if part.body:
                openable.add(sub)
            sub_open = sub in self.expanded
            mark = ("▾" if sub_open else "▸") if part.body else " "
            lines.append((index, f"  {mark}{gap}{part.head}", True))
            keys.append(sub)
            if not sub_open:
                continue
            for piece in part.folded(max(8, width - len(step_pad))):
                lines.append((index, f"{step_pad}{piece}", False))
                keys.append(sub)
        return lines, keys, openable

    def extend(self, item: Item) -> None:
        """Add a row without throwing the flattened line list away.

        The counterpart of the chat's append-only invariant
        (specs-ui-replacement.md §3.2), and what that invariant is *for*: a
        message arriving costs the lines that message draws, not a re-flatten
        of the conversation behind it. `invalidate` would be correct and
        O(conversation), which is the shape this UI exists to not have.

        A pane whose cache is already cold simply takes the row: the next
        `flat` was going to build the whole list anyway.
        """
        self.items.append(item)
        if self._flat is None:
            return
        # The live row sits after the last entry, so it moves down one.
        if self.tail is not None:
            self._flat.pop()
            self._keys.pop()
        index = len(self.items) - 1
        lines, keys, openable = self._item_lines(index, item, self._flat_width)
        self._flat += lines
        self._keys += keys
        self._openable |= openable
        if self.tail is not None:
            self._flat.append(self._tail_line())
            self._keys.append(self.key_at(len(self.items)))

    def reflow_last(self) -> None:
        """Redraw the newest row in the cache, leaving everything above it.

        `extend`'s twin, and it exists for the same reason: the chat opens the
        last row and closes it again the moment a newer one arrives
        (`state.SessionState._open_last`), and a row that opened and closed
        through `invalidate` would re-flatten the whole conversation twice per
        message — the O(conversation) event `extend` was written to abolish,
        put back on the same path.

        It is only ever the *last* row because that is the only one whose
        lines are a suffix of the cache: everything it draws sits after every
        other row's lines and before the live tail, so replacing them is a pop
        and an append rather than a splice. A pane whose cache is already cold
        has nothing to patch and says so by doing nothing — the next `flat`
        builds the list from what `expanded` says now.
        """
        if self._flat is None or not self.items:
            return
        index = len(self.items) - 1
        if self.tail is not None:
            self._flat.pop()
            self._keys.pop()
        while self._flat and self._flat[-1][0] == index:
            self._flat.pop()
            self._keys.pop()
        lines, keys, openable = self._item_lines(
            index, self.items[index], self._flat_width
        )
        self._flat += lines
        self._keys += keys
        self._openable |= openable
        if self.tail is not None:
            self._flat.append(self._tail_line())
            self._keys.append(self.key_at(len(self.items)))

    @property
    def head_gap(self) -> str:
        """The space between a row's fold marker and its head on this pane.

        A property rather than a literal at each place that draws one, because
        the live row below is a head line too and has to land in the same
        column as the rows above it — one of them reading the gap off `flush`
        and the other keeping a hardcoded space is how a spinner ends up three
        columns left of the conversation it belongs to.
        """
        return FLUSH_HEAD_GAP if self.flush else LIST_HEAD_GAP

    def _tail_line(self) -> tuple[int, str, bool]:
        # A blank where the marker would be: there is nothing to open on a
        # spinner, and it still belongs in the head column.
        return (len(self.items), f" {self.head_gap}{self.tail.head}", True)

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

    def here(self) -> str:
        """The key of the row the cursor is on, at the width last drawn.

        For the callers that have a cursor and no terminal size — the footer,
        which lists the keys that apply to whatever the sidebar is pointing
        at. A key rather than an index, because the answer outlives a repaint.
        """
        return self.key_at(self.current(max(8, self._flat_width)))

    def show(self, key: str) -> None:
        """Put the cursor on the row with that key, if the row is here.

        The counterpart of `replace`, and it takes no width for the same
        reason: whoever calls it — the sidebar, when a session is opened from
        somewhere other than this pane — knows a session id and not a terminal
        size, and the width only decides which flattened line the cursor lands
        on. The last one this pane was drawn at is the right answer for that.
        """
        self._go_to_key(key, max(8, self._flat_width))

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
        # Two columns are reserved whether or not this pane spends them on a
        # gutter. A flush pane draws its content at column 0 and leaves the
        # slack on the right, which is worth two columns of nothing: every
        # caller that asks a pane where its cursor is — `app.py` has a
        # terminal width and no opinion about gutters — would otherwise have
        # to know which panes are flush, and one that guessed wrong would ask
        # `flat` for a width it is not cached at and rebuild it every
        # keystroke.
        inner = max(8, width - 2)
        lines = self.flat(inner)
        body_h = max(1, height - 1)
        self._scroll_into_view(body_h, len(lines))
        current = lines[self.cursor][0] if lines else -1
        right = f"line {min(self.cursor + 1, len(lines))}/{len(lines)}"
        if self.expanded:
            right += f" · {len(self.expanded)} open"
        title = rule(self.name, width, right)
        out = [(BOLD + CYAN if focused else DIM) + title + RESET]
        # A conversation shorter than the pane hangs from the bottom, with the
        # empty screen above it rather than below: the newest line is the one
        # being read, and a log that grew downwards from the title rule would
        # put it a different distance from the message box on every frame —
        # the eye has to find it again after each reply. Only where the rows
        # are somebody's words. A list of sessions is a list, and a list that
        # started half way down the column would read as scrolled rather than
        # as short.
        blank = max(0, body_h - len(lines)) if self.flush else 0
        out += [" " * width] * blank
        for row in range(self.offset, self.offset + body_h - blank):
            if row >= len(lines):
                out.append(" " * width)
                continue
            owner, text, is_head = lines[row]
            here = owner == current
            gutter = "" if self.flush else ("▌ " if here else "  ")
            painted = pad(gutter + text, width)
            item = self.item_at(owner)
            # A label line is the row's nameplate, and it is drawn heavier
            # than what it names. With the words themselves flush at column 0
            # there is no gutter and no rule left to separate one turn from
            # the next, so the weight of the `you …` / `hpca …` line is the
            # whole of what does it.
            bold = is_head and item is not None and item.label
            if row == self.cursor:
                # The unfocused pane still shows where it was left, dimmed —
                # that is the "memory" being visible rather than merely kept.
                style = REVERSE if focused else DIM + REVERSE
            elif item is not None and item.accent:
                # Head *and* body, where it used to be the head alone: a
                # message is drawn in its speaker's colour down to its last
                # line rather than as a coloured first line over a grey wall.
                # Dim is what a row with no colour of its own gets, and a
                # turn's tool output is the whole of that — the one body here
                # that really is secondary to what is around it.
                style = item.accent
                bold = bold or (here and self.flush and is_head)
            elif not is_head:
                style = DIM
            else:
                # No gutter column to band the current entry with, so it is
                # marked by weight instead — and on the head line only,
                # because a paragraph in bold is not an indication, it is a
                # shout.
                style = ""
                bold = bold or (here and self.flush)
            if bold:
                style += BOLD
            out.append(style + painted + RESET if style else painted)
        return out
