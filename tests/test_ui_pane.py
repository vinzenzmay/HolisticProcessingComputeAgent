"""Tests for hpca.ui.pane: opening entries, closing them, and reordering.

Driven through `RowUI.handle` where the harness did, because the arrow keys are
the behaviour: what `→` means depends on whether the entry is already open and
on whether it has a body at all.
"""

from hpca.ui.app import CHAT, SESSIONS, WATCHERS
from hpca.ui.demo import build
from hpca.ui.pane import Fold, Item, Pane
from hpca.ui.state import ChatEntry
from tests.ui_harness import frame, plain, widths

# 120 columns of terminal, minus the two the gutter takes.
INNER = 118


def on_a_closed_entry(ui):
    """Walk the chat cursor down to the first entry that is *closed* and has
    something to open.

    Closed is the part that has to be looked for rather than assumed: a row
    can be openable and already open, and a turn's steps open themselves while
    they arrive, so "the first one with a body" is not the same question.
    """
    pane = ui.chat
    pane.cursor = 0
    for _ in range(len(pane.flat(INNER))):
        at = pane.current(INNER)
        if pane.items[at].openable and not pane.is_open(at):
            return pane, at
        pane.move(1, 20, INNER)
    raise AssertionError("the demo has no closed entry to open")


def chat_ui():
    ui = build()
    ui.focus = CHAT
    return ui


class TestArrowsOpenAndCloseEntries:
    def test_right_opens_the_entry(self):
        ui = chat_ui()
        pane, item = on_a_closed_entry(ui)
        ui.handle("right", 120, 40)
        assert pane.is_open(item)

    def test_the_row_got_longer(self):
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        before = len(pane.flat(INNER))
        ui.handle("right", 120, 40)
        assert len(pane.flat(INNER)) > before

    def test_and_the_cursor_sits_on_its_head_line(self):
        ui = chat_ui()
        pane, item = on_a_closed_entry(ui)
        ui.handle("right", 120, 40)
        assert pane.current(INNER) == item

    def test_right_again_steps_into_the_open_entry(self):
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        ui.handle("right", 120, 40)
        head = pane.cursor
        ui.handle("right", 120, 40)
        assert pane.cursor == head + 1

    def test_without_closing_it(self):
        ui = chat_ui()
        pane, item = on_a_closed_entry(ui)
        ui.handle("right", 120, 40)
        ui.handle("right", 120, 40)
        assert pane.is_open(item)

    def test_left_closes_it_from_inside_the_body(self):
        ui = chat_ui()
        pane, item = on_a_closed_entry(ui)
        ui.handle("right", 120, 40)
        ui.handle("right", 120, 40)
        ui.handle("left", 120, 40)
        assert not pane.is_open(item)

    def test_and_puts_you_back_on_its_head_line(self):
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        ui.handle("right", 120, 40)
        head = pane.cursor
        ui.handle("right", 120, 40)
        ui.handle("left", 120, 40)
        assert pane.cursor == head

    def test_left_on_a_closed_entry_does_nothing(self):
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        # Not "nothing is open": the demo's session is mid-turn, and a turn's
        # steps open themselves while they arrive.
        was = set(pane.expanded)
        ui.handle("right", 120, 40)
        head = pane.cursor
        ui.handle("right", 120, 40)
        ui.handle("left", 120, 40)
        ui.handle("left", 120, 40)
        assert pane.cursor == head
        assert pane.expanded == was

    def test_shift_right_opens_every_entry(self):
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        ui.handle("shift-right", 120, 40)
        entries = {
            pane.key_at(i) for i, x in enumerate(pane.items) if x.openable
        }
        assert entries and entries <= pane.expanded

    def test_and_every_step_inside_one(self):
        # "Open everything" means the second level too — otherwise a turn's
        # steps would be open and what each of them returned would not.
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        ui.handle("shift-right", 120, 40)
        steps = {
            f"{pane.key_at(i)}/{n}"
            for i, x in enumerate(pane.items)
            for n, part in enumerate(x.folds)
            if part.body
        }
        assert steps and steps <= pane.expanded

    def test_shift_left_closes_them_all(self):
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        ui.handle("shift-right", 120, 40)
        ui.handle("shift-left", 120, 40)
        assert pane.expanded == set()

    def test_e_no_longer_opens_anything(self):
        # `e` was an earlier expand key; it is a free letter now.
        ui = chat_ui()
        pane, _ = on_a_closed_entry(ui)
        # Not "nothing is open": the demo's session is mid-turn, and a turn's
        # steps open themselves while they arrive.
        was = set(pane.expanded)
        ui.handle("e", 120, 40)
        assert pane.expanded == was
        assert ui.overlay is None


class TestReorderingWatchers:
    def test_the_entry_moved_down(self):
        ui = build()
        ui.focus = WATCHERS
        names = [x.head for x in ui.watchers.items]
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        assert [x.head for x in ui.watchers.items][:2] == [names[1], names[0]]

    def test_the_cursor_followed_it(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        assert ui.watchers.current(INNER) == 1

    def test_and_it_says_so(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        assert ui.note == "moved"

    def test_alt_up_puts_it_back(self):
        ui = build()
        ui.focus = WATCHERS
        names = [x.head for x in ui.watchers.items]
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        ui.handle("alt-up", 120, 40)
        assert [x.head for x in ui.watchers.items] == names

    def test_cursor_followed_back(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        ui.handle("alt-up", 120, 40)
        assert ui.watchers.current(INNER) == 0

    def test_alt_up_at_the_top_is_a_no_op(self):
        ui = build()
        ui.focus = WATCHERS
        names = [x.head for x in ui.watchers.items]
        ui.watchers.cursor = 0
        ui.handle("alt-up", 120, 40)
        assert [x.head for x in ui.watchers.items] == names
        assert ui.note == ""

    def test_an_open_entry_stays_open_when_it_moves(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("right", 120, 40)
        ui.handle("alt-down", 120, 40)
        assert ui.watchers.is_open(1)
        assert not ui.watchers.is_open(0)

    def test_and_it_is_still_the_same_entry(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("right", 120, 40)
        opened_head = ui.watchers.items[0].head
        ui.handle("alt-down", 120, 40)
        assert ui.watchers.items[1].head == opened_head

    def test_frame_still_exact_after_a_reorder(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        drawn = ui.render(120, 40)
        assert len(drawn) == 40
        assert widths(drawn) == {120}

    def test_sessions_do_not_reorder(self):
        ui = build()
        ui.focus = SESSIONS
        first = ui.session_pane.items[0].head
        ui.handle("alt-down", 120, 40)
        assert ui.session_pane.items[0].head == first


class TestFlattening:
    def test_a_closed_entry_is_one_line(self):
        pane = Pane("x", [Item(head="one", body=["a", "b"])])
        assert len(pane.flat(40)) == 1

    def test_an_open_entry_carries_its_body_under_it(self):
        pane = Pane("x", [Item(head="one", body=["a", "b"])])
        pane.expand(40)
        assert [text.strip() for _, text, _ in pane.flat(40)] == ["▾ one", "a", "b"]

    def test_only_the_head_line_is_marked_as_one(self):
        pane = Pane("x", [Item(head="one", body=["a"])])
        pane.expand(40)
        assert [is_head for _, _, is_head in pane.flat(40)] == [True, False]

    def test_an_entry_with_no_body_will_not_open(self):
        pane = Pane("x", [Item(head="one")])
        assert pane.expand(40) is False
        assert pane.expanded == set()

    def test_an_empty_pane_has_no_current_entry(self):
        assert Pane("x", []).current(40) == -1

    def test_the_title_counts_the_open_entries(self):
        pane = Pane("x", [Item(head="one", body=["a"])])
        pane.expand(40)
        assert "1 open" in pane.render(40, 4, focused=True)[0]


class TestRowsAreKeyedByIdentity:
    """`expanded` used to be keyed by position, which is only right while
    entries are appended at the end. The chat's rows are named by the core
    (`Entry.seq`) and the sidebar's by session id, so both are keyed by that
    instead — and a row arriving above another cannot take over what was open.
    """

    def keyed(self) -> Pane:
        return Pane(
            "x",
            [
                Item(head="first", body=["a"], key="k1"),
                Item(head="second", body=["b"], key="k2"),
            ],
        )

    def test_a_row_with_no_key_of_its_own_falls_back_to_its_position(self):
        pane = Pane("x", [Item(head="one"), Item(head="two")])
        assert [pane.key_at(0), pane.key_at(1)] == ["#0", "#1"]

    def test_a_row_with_one_is_addressed_by_it(self):
        assert self.keyed().key_at(1) == "k2"

    def test_off_the_end_is_nothing_rather_than_an_error(self):
        assert self.keyed().key_at(9) == ""

    def test_opening_a_row_remembers_the_row_and_not_the_line(self):
        pane = self.keyed()
        pane.cursor = 1
        pane.expand(40)
        assert pane.expanded == {"k2"}

    def test_a_row_inserted_above_does_not_steal_what_was_open(self):
        pane = self.keyed()
        pane.cursor = 1
        pane.expand(40)
        pane.replace([Item(head="new", body=["c"], key="k0"), *pane.items], 40)
        assert pane.is_open(2) and not pane.is_open(0)

    def test_and_the_cursor_stays_on_the_row_it_was_on(self):
        pane = self.keyed()
        pane.cursor = 1
        pane.replace([Item(head="new", key="k0"), *pane.items], 40)
        assert pane.items[pane.current(40)].key == "k2"

    def test_a_row_that_left_takes_its_open_state_with_it(self):
        pane = self.keyed()
        pane.cursor = 1
        pane.expand(40)
        pane.replace([pane.items[0]], 40)
        assert pane.expanded == set()

    def test_a_reorder_carries_it_along(self):
        pane = self.keyed()
        pane.cursor = 0
        pane.expand(40)
        pane.reorder(1, 10, 40)
        assert pane.expanded == {"k1"}
        assert pane.items[1].key == "k1"

    def test_and_still_does_for_rows_with_no_key(self):
        # The old hand-patch, which is still what a keyless list needs.
        pane = Pane("x", [Item(head="one", body=["a"]), Item(head="two", body=["b"])])
        pane.cursor = 0
        pane.expand(40)
        pane.reorder(1, 10, 40)
        assert pane.expanded == {"#1"}
        assert pane.items[1].head == "one"


class TestTheChatSelectsClean:
    """The chat is what people copy out of, so nothing may precede its words.

    Selecting text out of the chat is the terminal's own drag-to-select in
    this UI — the mouse is deliberately left released so it keeps working
    (specs-ui-replacement.md §4.1 items 5 and 6) — and a terminal selects
    whole screen columns. So every column the pane spends in front of a line
    of the conversation is a character that lands in the paste buffer, and
    these tests are that promise: a message's own lines start at column 0 and
    are the text and nothing else.

    The rows that are *not* somebody's words keep their furniture, and that is
    the other half of the rule this asserts: indented means the UI talking,
    flush means verbatim.
    """

    SAID = ["the first line of it", "and the second, which is different"]

    def a_chat_with(self, *entries, opened=True):
        ui = build(chat=0)
        ui.session.reset(list(entries))
        ui.focus = CHAT
        if opened:
            ui.chat.expand_all(118)
        return ui

    def said(self, seq=1, kind="user"):
        return ChatEntry(kind=kind, text="\n".join(self.SAID), seq=seq)

    def test_a_message_line_is_drawn_at_column_zero(self):
        ui = self.a_chat_with(self.said())
        drawn = [x.rstrip() for x in frame(ui, 120, 40)]
        assert self.SAID[0] in drawn
        assert self.SAID[1] in drawn

    def test_and_that_is_true_of_the_agents_words_too(self):
        ui = self.a_chat_with(self.said(kind="assistant"))
        assert self.SAID[0] in [x.rstrip() for x in frame(ui, 120, 40)]

    def test_and_of_a_line_that_had_to_be_wrapped(self):
        # The wrap is where an indent would come back if it came back
        # anywhere: a continuation line is drawn by the pane, not by the core.
        long = "word " * 60
        ui = self.a_chat_with(ChatEntry(kind="assistant", text=long.strip(), seq=1))
        wrapped = [x for x in frame(ui, 120, 40) if x.startswith("word")]
        assert len(wrapped) > 1, "the text has to have wrapped to mean anything"

    def test_a_closed_message_shows_one_line_of_itself(self):
        # Two lines a message when folded: who said it, and as much of what
        # they said as the terminal holds. A column of bare labels would say
        # who spoke and not a word of what was said.
        ui = self.a_chat_with(self.said(), opened=False)
        drawn = [x.rstrip() for x in frame(ui, 120, 40)]
        assert "▸ you" in drawn
        assert " ".join(self.SAID) in drawn  # short enough to survive whole

    def test_and_says_so_when_there_is_more(self):
        ui = self.a_chat_with(
            ChatEntry(kind="assistant", text="a long reply. " * 40, seq=1),
            opened=False,
        )
        clipped = [x.rstrip() for x in frame(ui, 120, 40) if x.startswith("a long")]
        assert len(clipped) == 1, "a closed row draws one line and no more"
        assert clipped[0].endswith(" [...]")

    def test_and_that_line_is_flush_too(self):
        # It is still a line of the conversation, so it still starts at
        # column 0 — the mark at the end is the only thing on it that is not
        # what was said.
        ui = self.a_chat_with(self.said(), opened=False)
        assert " ".join(self.SAID) in [x.rstrip() for x in frame(ui, 120, 40)]

    def test_a_turn_closes_to_its_head_alone(self):
        # Its head is already the summary, so a preview would repeat it.
        ui = self.a_chat_with(ChatEntry(kind="thinking", seq=1, steps=3), opened=False)
        drawn = [x.rstrip() for x in frame(ui, 120, 40)]
        assert "  3 steps" in drawn
        assert sum(1 for x in drawn if "steps" in x) == 1

    def test_the_label_is_a_line_of_its_own(self):
        # …which is what buys the line below it: there is nothing left on the
        # message's own lines to have to select around.
        ui = self.a_chat_with(self.said())
        drawn = [x.rstrip() for x in frame(ui, 120, 40)]
        assert "  you" in drawn or "▾ you" in drawn or "▸ you" in drawn
        assert not any(x.endswith("you " + self.SAID[0]) for x in drawn)

    def test_a_row_the_ui_wrote_keeps_its_furniture(self):
        # A turn's working is the UI talking about the conversation. It is
        # indented, it is not verbatim, and nobody pastes it.
        ui = self.a_chat_with(ChatEntry(kind="thinking", seq=1, steps=3))
        assert any(x.startswith("  ") and "3 steps" in x for x in frame(ui, 120, 40))

    def test_the_lists_still_have_their_gutter(self):
        # Only the chat is flush. A sidebar row is a title, not a paste, and
        # the band that says which one the cursor is in is worth two columns
        # there.
        ui = build()
        drawn = [plain(x) for x in ui.session_pane.render(120, 8, focused=True)]
        assert any(x.startswith("▌ ") for x in drawn)

    def test_and_the_chat_has_none_at_all(self):
        # Not one line of it, wherever the cursor happens to be — the band is
        # drawn on every line of the entry the cursor is in, so a chat that
        # kept it would put a character in front of a message being read.
        ui = build()
        ui.chat.cursor = 0
        top = [plain(x) for x in ui.chat.render(120, 30, focused=True)]
        ui.chat.cursor = 10**9
        bottom = [plain(x) for x in ui.chat.render(120, 30, focused=True)]
        assert not any(x.startswith("▌") for x in top + bottom)


class TestAppendingWithoutReflattening:
    """`Pane.extend`: a row arriving costs that row.

    A chat row is a label and at least one line of what it holds, so the
    flattened line list is longer than the conversation is deep, and
    rebuilding it on each arriving row would be the O(conversation) event this
    whole UI exists to not have. These are the two halves of that: the cache
    is added to rather than dropped, and what it ends up holding is exactly
    what a rebuild would have produced.
    """

    def two_rows(self):
        pane = Pane("p", [Item(head="first", body=["a", "b"])])
        pane.expanded = {"#0", "#1"}
        return pane

    def test_extending_agrees_with_rebuilding(self):
        grown = self.two_rows()
        grown.flat(40)
        grown.extend(Item(head="second", body=["c", "d"]))
        built = Pane("p", list(grown.items))
        built.expanded = set(grown.expanded)
        assert grown.flat(40) == built.flat(40)

    def test_and_does_not_drop_the_cache_to_do_it(self):
        pane = self.two_rows()
        was = pane.flat(40)
        pane.extend(Item(head="second"))
        assert pane.flat(40) is was, "the list was rebuilt rather than added to"

    def test_a_cold_pane_just_takes_the_row(self):
        pane = Pane("p", [])
        pane.extend(Item(head="only"))
        assert [x[1] for x in pane.flat(40)] == ["  only"]

    def test_the_live_row_stays_last(self):
        # The spinner is pinned after the last entry, so a row arriving goes
        # in front of it rather than under it.
        pane = self.two_rows()
        pane.set_tail(Item(head="working"))
        pane.flat(40)
        pane.extend(Item(head="second"))
        assert pane.flat(40)[-1][1].strip() == "working"

    def test_and_what_opens_is_known_about_the_new_row(self):
        pane = self.two_rows()
        pane.flat(40)
        pane.extend(Item(head="second", body=["c"], key="k"))
        pane.cursor = len(pane.flat(40)) - 1
        assert pane.expand(40), "the row it just took cannot be opened"


class TestABodyIsWrappedOnce:
    def test_the_same_lines_come_back(self):
        item = Item(head="h", body=["a rather long line that will have to wrap"])
        assert item.folded(12) == item.folded(12)

    def test_and_the_second_time_is_the_same_list(self):
        # Identity, not equality: this is the memo the rebuild leans on.
        item = Item(head="h", body=["a rather long line that will have to wrap"])
        assert item.folded(12) is item.folded(12)

    def test_a_different_width_wraps_again(self):
        item = Item(head="h", body=["a rather long line that will have to wrap"])
        assert item.folded(12) != item.folded(80)

    def test_a_step_body_is_memoised_too(self):
        # Tool output is the biggest body on the pane — four hundred lines of
        # log is routine — so it is the one that most needs not to be re-folded.
        part = Fold(head="run_bash", body=["a line of output"] * 40)
        assert part.folded(20) is part.folded(20)

    def test_a_closed_row_remembers_its_clipped_line(self):
        # The other half, and the one that costs on an ordinary screen: most
        # rows are closed, so most of a rebuild is clipping rather than
        # folding. Measured on a 5000-entry chat, it was a third of it.
        item = Item(head="you", preview="a line long enough to want cutting")
        assert item.clipped(12) is item.clipped(12)

    def test_and_clips_again_at_a_new_width(self):
        item = Item(head="you", preview="a line long enough to want cutting")
        assert item.clipped(12) != item.clipped(80)
