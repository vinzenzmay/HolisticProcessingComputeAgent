"""Tests for hpca.ui.pane: opening entries, closing them, and reordering.

Driven through `RowUI.handle` where the harness did, because the arrow keys are
the behaviour: what `→` means depends on whether the entry is already open and
on whether it has a body at all.
"""

from hpca.ui import theme
from hpca.ui.ansi import BOLD, REVERSE, cell_width
from hpca.ui.app import CHAT, SESSIONS, WATCHERS
from hpca.ui.demo import build
from hpca.ui.pane import SPACER_KEY, Fold, Item, Pane
from hpca.ui.state import ChatEntry, ChatPart, SessionState
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
    """alt+↑/↓ on the watchers column, through the demo core and back.

    Nothing here is done locally any more: the key sends `watch.move` and the
    order on screen is the `panel.update` that answers it (`app._move_watch`,
    and `TestReorderingSurvivesTheNextFrame` in test_ui_client.py for the
    round trip in slow motion). The demo's loopback is synchronous, so the
    answer has landed by the time `handle` returns and these read the same as
    they did when the swap was made here.
    """

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

    def test_holding_it_down_walks_the_box_past_two(self):
        # The reason the core swaps with a neighbour rather than assigning a
        # slot: the second press is aimed at the same box, one row further on.
        ui = build()
        ui.focus = WATCHERS
        names = [x.head for x in ui.watchers.items]
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        ui.handle("alt-down", 120, 40)
        assert [x.head for x in ui.watchers.items][:3] == [
            names[1],
            names[2],
            names[0],
        ]

    def test_frame_still_exact_after_a_reorder(self):
        ui = build()
        ui.focus = WATCHERS
        ui.watchers.cursor = 0
        ui.handle("alt-down", 120, 40)
        drawn = ui.render(120, 40)
        assert len(drawn) == 40
        assert widths(drawn) == {120}

class TestReorderingSessions:
    """The sidebar's own `alt+↑/↓`, alongside the watchers' (§ above): asked
    for so the two reorderable rows answer to the same keys. Cursor row 0 is
    always "(new session)"; row 1 is the first real one, `self.sessions[0]`.

    Sent rather than done, like the watchers': `session.move` goes out and the
    `session.rows` that answers it is the order — the arrangement is a fact
    about the store, and the sidebar is rebuilt from `self.sessions` on almost
    every keystroke, so a swap made only here lasted until the next one.
    """

    def test_holding_it_down_walks_the_session_past_two(self):
        ui = build()
        ui.focus = SESSIONS
        names = [x.head for x in ui.session_pane.items]
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        ui.handle("alt-down", 120, 40)
        assert [x.head for x in ui.session_pane.items][1:4] == [
            names[2],
            names[3],
            names[1],
        ]

    def test_the_entry_moved_down(self):
        ui = build()
        ui.focus = SESSIONS
        names = [x.head for x in ui.session_pane.items]
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        assert [x.head for x in ui.session_pane.items][1:3] == [names[2], names[1]]

    def test_the_cursor_followed_it(self):
        ui = build()
        ui.focus = SESSIONS
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        assert ui.session_pane.current(INNER) == 2

    def test_and_it_says_so(self):
        ui = build()
        ui.focus = SESSIONS
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        assert ui.note == "moved"

    def test_alt_up_puts_it_back(self):
        ui = build()
        ui.focus = SESSIONS
        names = [x.head for x in ui.session_pane.items]
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        ui.handle("alt-up", 120, 40)
        assert [x.head for x in ui.session_pane.items] == names

    def test_cursor_followed_back(self):
        ui = build()
        ui.focus = SESSIONS
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        ui.handle("alt-up", 120, 40)
        assert ui.session_pane.current(INNER) == 1

    def test_alt_up_at_the_top_is_a_no_op(self):
        ui = build()
        ui.focus = SESSIONS
        names = [x.head for x in ui.session_pane.items]
        ui.session_pane.cursor = 1
        ui.handle("alt-up", 120, 40)
        assert [x.head for x in ui.session_pane.items] == names
        assert ui.note == ""

    def test_the_new_session_row_never_moves(self):
        ui = build()
        ui.focus = SESSIONS
        first = ui.session_pane.items[0].head
        ui.session_pane.cursor = 0
        ui.handle("alt-down", 120, 40)
        assert ui.session_pane.items[0].head == first
        assert ui.note == ""

    def test_the_active_marker_follows_the_active_session(self):
        # `active` is a position in `self.sessions`; the sidebar the core
        # sends back moves the session out from under it, and `sync_sessions`
        # recomputes it from the id — or the ● lands on whatever session
        # happens to sit at the old index instead of the one actually open.
        ui = build()
        ui.focus = SESSIONS
        assert ui.active == 0
        ui.session_pane.cursor = 1
        ui.handle("alt-down", 120, 40)
        assert ui.active == 1
        assert ui.session_pane.items[1].head.startswith("○")
        assert ui.session_pane.items[2].head.startswith("●")


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
    (specs/specs-ui-replacement.md §4.1 items 5 and 6) — and a terminal selects
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
        else:
            # A transcript opens its newest row by itself, and these are about
            # what a *closed* row draws — so this is the user having folded it
            # away again, which is a thing they are allowed to do.
            ui.chat.collapse_all(118)
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
        assert "▸    you" in drawn
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
        assert "     3 steps" in drawn  # nothing to open, so a blank marker
        assert sum(1 for x in drawn if "steps" in x) == 1

    def test_the_label_is_a_line_of_its_own(self):
        # …which is what buys the line below it: there is nothing left on the
        # message's own lines to have to select around.
        ui = self.a_chat_with(self.said())
        drawn = [x.rstrip() for x in frame(ui, 120, 40)]
        assert any(x in drawn for x in ("     you", "▾    you", "▸    you"))
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
        ui.chat.to_end()
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


# ------------------------------------------------- what the chat looks like


def chat_of(*entries: ChatEntry) -> Pane:
    """One chat pane holding exactly these rows, filled the way the real one
    is — through `reset`, so what opens itself opens itself."""
    session = SessionState("s1")
    session.reset(list(entries))
    return session.chat


def styled(pane: Pane, text: str, width: int = 60, height: int = 12) -> str:
    """The drawn line whose visible text is ``text``, escapes and all."""
    for line in pane.render(width, height, focused=True):
        if plain(line).rstrip() == text:
            return line
    raise AssertionError(f"no line reading {text!r}")


SAID = ChatEntry(kind="user", text="run it again", seq=1)
REPLIED = ChatEntry(kind="assistant", text="done", seq=2)
WORKED = ChatEntry(
    kind="thinking",
    seq=3,
    steps=2,
    parts=[
        ChatPart(
            kind="call",
            tool="read_file",
            target="/scratch/run.log",
            result="412 lines",
            done=True,
        ),
        ChatPart(kind="reasoning", text="two shards, one temp path", done=True),
    ],
)


class TestTheChatFillsFromTheBottom:
    """A conversation shorter than the pane hangs from the foot of it.

    The newest line is the one being read, and it has to be the same distance
    from the message box on every frame: a log that grew downwards from the
    title rule moves the thing the eye is looking for every time a row lands.
    Only here — a list of sessions is a list, and one that started half way
    down its column would read as scrolled rather than as short.
    """

    def test_a_short_conversation_sits_at_the_foot_of_the_pane(self):
        # The blank at the very bottom is the last row's own spacer, and it is
        # what stands between the conversation and the message box under it
        # (`pane.SPACER_LINES`).
        drawn = [plain(x).rstrip() for x in chat_of(SAID, REPLIED).render(
            60, 12, focused=True
        )]
        assert drawn[-3:] == ["▾    hpca", "done", ""]

    def test_and_the_room_it_does_not_need_is_above_it(self):
        drawn = [plain(x).rstrip() for x in chat_of(SAID, REPLIED).render(
            60, 12, focused=True
        )]
        assert drawn[1:5] == ["", "", "", ""]
        assert drawn[0].startswith("── chat")  # the title stays where it is

    def test_the_pane_is_still_exactly_as_tall_as_it_was_asked_for(self):
        drawn = chat_of(SAID).render(60, 12, focused=True)
        assert len(drawn) == 12
        assert widths(drawn) == {60}

    def test_a_conversation_too_long_to_fit_is_untouched(self):
        # The bottom is where a scrolled pane already ends; there is no room
        # to give back and nothing to move. Asked with the spacing off, so
        # that "no blank rows" still means "nothing was given back" — with it
        # on, every row carries a blank of its own and the question could not
        # be put this way.
        pane = chat_of(*[ChatEntry(kind="user", text=f"n{n}", seq=n) for n in
                         range(1, 30)])
        pane.spacer = 0
        drawn = [plain(x).rstrip() for x in pane.render(60, 12, focused=True)]
        assert "" not in drawn[1:]

    def test_a_list_still_fills_from_the_top(self):
        # The sessions pane is not flush, and a gap under its title would read
        # as a list that had been scrolled away from.
        pane = Pane("sessions", [Item(head="one"), Item(head="two")])
        drawn = [plain(x).rstrip() for x in pane.render(60, 12, focused=True)]
        assert drawn[1:3] == ["▌   one", "    two"]

    def test_the_highlight_lands_on_the_row_the_keys_act_on(self):
        # The blank lines are drawn above the rows, so every flattened line
        # moves down by however many there are. If the cursor line and the row
        # the pane reports as current came apart here, → would open a
        # different entry from the one under the highlight.
        pane = chat_of(SAID, REPLIED)
        for cursor in range(len(pane.flat(58))):
            pane.cursor = cursor
            drawn = pane.render(60, 12, focused=True)
            # Where the cursor ended up, which is not always where it was
            # aimed: a spacer is not a line anything may sit on, so the pane
            # puts it back on the row the blank belongs to.
            at = pane.cursor
            marked = [plain(x).rstrip() for x in drawn if "\x1b[7m" in x]
            owner = pane.flat(58)[at][0]
            assert marked == [pane.flat(58)[at][1].strip()]
            assert pane.current(58) == owner


def head_lines(pane: Pane, width: int = 58) -> list[str]:
    """The lines a pane draws as heads, marker and all."""
    return [text for _, text, is_head in pane.flat(width) if is_head]


def head_column(line: str) -> int:
    """Which column a head line's words start in, whatever is in front."""
    return len(line) - len(line.lstrip(" ▸▾"))


class TestTheHeadRowsStandOffTheProse:
    """`▸    you`, not `▸ you`: four spaces between the marker and the head.

    On the chat and nowhere else. A message's own lines are drawn at column 0
    so that a drag-select picks up prose and nothing else, which leaves the
    head rows with a single marker to say they are the UI talking about the
    conversation rather than more of it — and a row with nothing to open has
    not even that. The wider gap is what says it now.
    """

    def test_the_head_sits_four_spaces_off_its_marker(self):
        drawn = [plain(x).rstrip() for x in chat_of(SAID, REPLIED).render(
            60, 12, focused=True
        )]
        assert "▸    you" in drawn
        assert "▾    hpca" in drawn

    def test_and_the_words_under_it_are_still_flush(self):
        # The gap is the head's, not the row's: widening it must not push the
        # message itself off column 0, which is the whole point of this pane.
        drawn = [plain(x).rstrip() for x in chat_of(SAID, REPLIED).render(
            60, 12, focused=True
        )]
        assert "run it again" in drawn

    def test_opening_a_row_does_not_move_its_head(self):
        # ▸ and ▾ are one cell each and the gap is on both, so the column
        # cannot shift as rows open and close — one that did would read as the
        # pane twitching rather than as a fold.
        pane = chat_of(SAID, WORKED, REPLIED)
        pane.collapse_all(58)
        closed = head_lines(pane)
        pane.expand_all(58)
        assert {head_column(x) for x in closed} == {5}
        tops = [x for x in head_lines(pane) if not x.startswith("  ")]
        assert {head_column(x) for x in tops} == {5}

    def test_a_row_with_nothing_to_open_lines_up_with_one_that_has(self):
        # It carries a blank where the marker would be, and the gap goes on
        # the blank too, or the column goes ragged down the log.
        pane = chat_of(SAID, ChatEntry(kind="event", text="session resumed", seq=9))
        assert {head_column(x) for x in head_lines(pane)} == {5}

    def test_the_steps_of_a_turn_get_it_as_well(self):
        # A step's head sits over its output, and that output is flush too —
        # the same ambiguity one level in, so the same answer. Indented from
        # the turn that holds it, which is what says which it belongs to.
        pane = chat_of(SAID, WORKED)
        steps = [x for x in head_lines(pane) if x.startswith("  ")]
        assert steps, "the turn drew no steps to check"
        assert {head_column(x) for x in steps} == {7}

    def test_the_live_row_lands_in_the_head_column(self):
        # The spinner is a head line with nothing to open, and it is read
        # together with the rows above it: half a gutter to the left of them
        # is exactly how it would look wrong.
        pane = chat_of(SAID, REPLIED)
        pane.set_tail(Item(head="working"))
        # Its own blanks are under it, the way an entry's are (`_tail_height`),
        # so the head line is the first of the lines it draws rather than the
        # last line of the pane.
        assert pane.flat(58)[-1 - pane.spacer][1] == "     working"

    def test_a_list_keeps_its_single_space(self):
        # Nothing to fix there: a sidebar indents its bodies four columns
        # under the head already, so the wider gap would only push the heads
        # out of line with them.
        pane = Pane("sessions", [Item(head="one", body=["a"])])
        pane.expand_all(58)
        assert [x[1] for x in pane.flat(58)] == ["▾ one", "    a"]

    def test_and_the_rows_are_still_exactly_the_width(self):
        # Three cells came off the head's text budget, so the narrow case is
        # the one that matters: `pad` has to cut the head rather than let the
        # row run over, or the differential repaint leaves the screen wrong.
        pane = chat_of(SAID, WORKED, REPLIED)
        pane.expand_all(28)
        assert widths(pane.render(30, 16, focused=True)) == {30}
        pane.collapse_all(28)
        assert widths(pane.render(30, 16, focused=True)) == {30}


class TestTheLabelLinesAreDrawnHeavier:
    """`you 22-08-2026 13:04:47` and `hpca …` are bold; the words are not.

    With the conversation itself flush at column 0 there is no gutter and no
    rule left between one turn and the next, so the weight of the label is the
    whole of what separates them.
    """

    def test_your_own_label_is_bold(self):
        pane = chat_of(SAID, REPLIED)
        pane.cursor = 0  # off the reply, so the highlight is not what shows
        assert BOLD in styled(pane, "▸    you")

    def test_and_so_is_the_agents(self):
        pane = chat_of(SAID, REPLIED)
        pane.cursor = 0
        assert BOLD in styled(pane, "▾    hpca")

    def test_but_the_words_under_it_are_not(self):
        # A paragraph in bold is not an indication, it is a shout.
        pane = chat_of(SAID, REPLIED)
        pane.cursor = 0
        assert BOLD not in styled(pane, "done")

    def test_and_neither_is_a_line_that_names_no_speaker(self):
        pane = chat_of(SAID, WORKED)
        pane.cursor = 0
        assert BOLD not in styled(pane, "▾    2 steps · read_file → reasoning")


class TestATurnIsGreyThroughout:
    """The head and every line it opens into.

    A turn's working is the machinery behind the answer rather than the
    answer, and it is the bulkiest thing in the log — an opened one is rows of
    tool names between two paragraphs of prose. Uncoloured was not the same
    thing: the renderer dims a body it has no colour for and leaves the head
    at full weight, so a closed turn stood out exactly as loudly as a reply.
    """

    def test_the_head_of_a_closed_turn_is_grey(self):
        pane = chat_of(SAID, WORKED)
        pane.collapse_all(58)
        pane.cursor = 0
        assert theme.faint in styled(pane, "▸    2 steps · read_file → reasoning")

    def test_and_so_is_every_step_it_opens_into(self):
        # The head and both of the rows under it: the whole of what an opened
        # turn puts on the screen, and not only the lines the renderer would
        # have dimmed by default. Both open — a step's head is one line, and
        # what it holds (a script, a block of reasoning) is behind it.
        pane = chat_of(SAID, WORKED)
        pane.cursor = 0
        for text in (
            "▾    2 steps · read_file → reasoning",
            "  ▸    read_file     /scratch/run.log",
            "  ▸    reasoning     two shards, one temp path",
        ):
            assert theme.faint in styled(pane, text), text

    def test_including_what_a_step_returned(self):
        pane = chat_of(SAID, WORKED)
        pane.expanded.add("3/0")
        pane.invalidate()
        pane.cursor = 0
        assert theme.faint in styled(pane, "412 lines")


class TestTheLiveRowIsNotHighlighted:
    """The working row refuses the cursor's highlight.

    `follow` pins the cursor to the end of the chat, so a running turn parks it
    on the live row and leaves it there — and REVERSE across a status line is a
    grey band the width of the terminal under the one line being read. The row
    is a report on the turn rather than a row anything can be done to, so the
    cursor resting there has nothing to say, and it stops saying it.
    """

    @staticmethod
    def working() -> Pane:
        pane = chat_of(SAID)
        pane.set_tail(Item(head="ｸｪﾍﾕ reading…"))
        pane.cursor = len(pane.flat(58)) - 1
        return pane

    @staticmethod
    def row(pane: Pane, needle: str, *, focused: bool = True) -> str:
        drawn = [x for x in pane.render(60, 12, focused=focused) if needle in plain(x)]
        assert drawn, f"no line containing {needle!r}"
        return drawn[0]

    def test_no_reverse_on_it(self):
        assert REVERSE not in self.row(self.working(), "reading…")

    def test_and_none_unfocused_either(self):
        # Where the band was worst: the chat is not focused while a turn runs
        # — the keys are in the message box — so it was drawn faint-reversed,
        # which is a grey wash rather than a highlight.
        assert REVERSE not in self.row(self.working(), "reading…", focused=False)

    def test_a_real_row_under_the_cursor_still_is(self):
        pane = chat_of(SAID)
        pane.set_tail(Item(head="ｸｪﾍﾕ reading…"))
        pane.cursor = 0
        assert REVERSE in self.row(pane, "you")


class TestTheNewestLineStaysOnScreen:
    """The chat follows its own end — through every way it grows.

    A chat grows in three ways and only one of them is a new row: `chat.append`
    adds one, `chat.update` fills the row already there as the tokens land, and
    the live working row comes and goes underneath. Pinning the cursor at the
    end on *append* alone followed a third of that, so a turn with several
    steps in it wrote its newest lines below the fold and they had to be
    scrolled down to by hand. So it is a state (`Pane.follow`) rather than an
    assertion repeated at each of the places a line can appear.
    """

    @staticmethod
    def last_visible(pane: Pane, width: int = 60, height: int = 8) -> str:
        drawn = [plain(x).rstrip() for x in pane.render(width, height, focused=True)]
        return next(x for x in reversed(drawn) if x)

    def growing(self, pane: Pane) -> str:
        """The last line on a screen too short to hold what the pane holds."""
        return self.last_visible(pane)

    def test_a_row_arriving_scrolls_to_it(self):
        session = SessionState("s1")
        session.reset([SAID])
        session.append(REPLIED)
        assert "done" in self.growing(session.chat)

    def test_and_so_does_a_row_being_filled_in(self):
        # The bug this class exists for: an assistant row is appended empty
        # and then written into, and every token after the first arrived on a
        # line below the one the screen ended at.
        session = SessionState("s1")
        session.reset([SAID])
        session.append(ChatEntry(kind="assistant", text="", seq=2))
        for text in ("one", "one two", "one two three\nand a second line"):
            session.update(ChatEntry(kind="assistant", text=text, seq=2))
        assert "and a second line" in self.growing(session.chat)

    def test_and_a_step_landing_mid_turn(self):
        session = SessionState("s1")
        session.reset([SAID, REPLIED])
        session.append(WORKED)
        assert "reasoning" in self.growing(session.chat)

    def test_but_not_once_somebody_has_scrolled_up(self):
        # Following is a thing the user is doing, and moving off the last line
        # is how they stop: a reply landing must not yank the screen away from
        # what is being read.
        session = SessionState("s1")
        session.reset([SAID, WORKED])
        session.chat.render(60, 8, focused=True)
        session.chat.move(-3, 7, 58)
        parked = session.chat.cursor
        session.append(REPLIED)
        session.chat.render(60, 8, focused=True)
        assert session.chat.cursor == parked
        assert not session.chat.follow

    def test_and_scrolling_back_down_to_it_resumes(self):
        session = SessionState("s1")
        session.reset([SAID, WORKED])
        session.chat.render(60, 8, focused=True)
        # A row at a time, which is what the arrow keys send: a spacer is
        # stepped over without being landed on, so three presses up is three
        # rows up and three back down is the row it started on. A single move
        # of several *lines* is not the same distance in both directions once
        # the blanks are in the way, and never was meant to be — page up and
        # page down move a screenful, not a remembered place.
        for _ in range(3):
            session.chat.move(-1, 7, 58)
        for _ in range(3):
            session.chat.move(1, 7, 58)
        assert session.chat.follow
        session.append(REPLIED)
        assert "done" in self.growing(session.chat)

    def test_and_aiming_at_a_row_to_read_it_parks(self):
        pane = chat_of(SAID, WORKED)
        pane.render(60, 8, focused=True)
        pane.cursor = 0
        assert not pane.follow
        assert self.last_visible(pane) != ""


# ------------------------------------------------ the blank under every row


class TestTheRowsAreHeldApart:
    """One blank line under each chat row (`display.spacer_lines`).

    A chat is a column of paragraphs, and the only thing saying where one
    stopped was the weight of the next `you` / `hpca` nameplate — enough on a
    two-line exchange, and not enough once a reply runs to twenty. A blank
    line is what prose has always used for that.

    The blank belongs to the row *above* it rather than to the row below,
    which is what keeps `extend` an append: a spacer owned by the arriving row
    would mean patching the one before it every time a message landed, which
    is the O(conversation) event that method exists to abolish.
    """

    @staticmethod
    def texts(pane: Pane, width: int = 58) -> list[str]:
        return [text for _, text, _ in pane.flat(width)]

    def test_a_blank_follows_each_row(self):
        pane = chat_of(SAID, REPLIED)
        assert self.texts(pane) == [
            "▸    you",
            "run it again",
            "",
            "▾    hpca",
            "done",
            "",
        ]

    def test_a_closed_row_gets_one_too(self):
        # Both exits of `_item_lines` append it. A gap that came and went as
        # rows were folded would read as the conversation jumping.
        pane = chat_of(SAID, REPLIED)
        pane.expanded.clear()
        pane.invalidate()
        assert self.texts(pane)[-1] == ""
        assert self.texts(pane)[-2].startswith("done")

    def test_and_the_count_is_the_setting(self):
        pane = chat_of(SAID, REPLIED)
        pane.spacer = 3
        assert self.texts(pane)[-3:] == ["", "", ""]

    def test_zero_is_the_pane_as_it_was(self):
        pane = chat_of(SAID, REPLIED)
        pane.spacer = 0
        assert self.texts(pane) == ["▸    you", "run it again", "▾    hpca", "done"]

    def test_the_lists_are_not_spaced(self):
        # A sessions column is one-line titles that were never hard to tell
        # apart, and doubling its height would cost the chat the rows it is
        # given.
        pane = Pane("sessions", [Item(head="one"), Item(head="two")])
        assert pane.spacer == 0
        assert self.texts(pane) == ["  one", "  two"]

    def test_a_row_arriving_brings_its_own(self):
        # Through `extend`, which patches the flattened cache rather than
        # rebuilding it: the spacing must not be a property of how a row got
        # onto the pane.
        session = SessionState("s1")
        session.reset([SAID])
        session.append(REPLIED)
        assert self.texts(session.chat)[-1] == ""

    def test_and_so_does_the_live_row(self):
        # While a turn runs the spinner is the last row on the pane, so the
        # gap under the conversation has to survive one starting.
        pane = chat_of(SAID, REPLIED)
        pane.set_tail(Item(head="working"))
        assert self.texts(pane)[-2:] == ["     working", ""]

    def test_and_a_spinner_frame_does_not_relayout(self):
        # `set_tail` patches the head line in place; the blanks under it are
        # already right, and rewriting them is the relayout it exists to
        # avoid.
        pane = chat_of(SAID, REPLIED)
        pane.set_tail(Item(head="working"))
        was = len(pane.flat(58))
        pane.set_tail(Item(head="working ."))
        assert pane._flat is not None  # the cache survived the frame
        assert len(pane.flat(58)) == was
        assert self.texts(pane)[-2:] == ["     working .", ""]

    def test_changing_it_redraws_the_pane(self):
        pane = chat_of(SAID, REPLIED)
        pane.flat(58)
        pane.spacer = 2
        assert pane._flat is None

    def test_but_restating_it_does_not(self):
        # `set_display` restates every display key on every save, and an
        # unconditional invalidate here would throw the chat's flattened cache
        # away each time somebody changed a colour.
        pane = chat_of(SAID, REPLIED)
        pane.flat(58)
        pane.spacer = pane.spacer
        assert pane._flat is not None


class TestNothingLandsOnABlank:
    """The cursor steps over the spacers rather than onto them.

    `_keys` is what the cursor reads itself off, and a blank that answered
    with a real key would let the highlight sit on an empty line — which on
    this pane is a reversed band the width of the terminal under the reply
    being read.
    """

    def test_the_blanks_are_keyed_apart_from_the_rows(self):
        pane = chat_of(SAID, REPLIED)
        pane.flat(58)
        assert pane._keys == ["1", "1", SPACER_KEY, "2", "2", SPACER_KEY]

    def test_following_parks_on_the_last_row_not_its_blank(self):
        session = SessionState("s1")
        session.reset([SAID, REPLIED])
        session.chat.render(60, 12, focused=True)
        assert session.chat.follow
        assert session.chat._keys[session.chat.cursor] != SPACER_KEY

    def test_and_so_the_highlight_is_never_an_empty_band(self):
        session = SessionState("s1")
        session.reset([SAID, REPLIED])
        drawn = session.chat.render(60, 12, focused=True)
        banded = [plain(x).rstrip() for x in drawn if REVERSE in x]
        assert banded == ["done"]

    def test_an_arrow_moves_one_row(self):
        pane = chat_of(SAID, REPLIED)
        pane.render(60, 12, focused=True)
        pane.cursor = 4  # "done"
        pane.move(-1, 11, 58)
        assert pane.cursor == 3  # the label above it, not the blank between
        pane.move(1, 11, 58)
        assert pane.cursor == 4

    def test_aiming_at_a_blank_lands_on_the_row_it_belongs_to(self):
        pane = chat_of(SAID, REPLIED)
        pane.cursor = 2  # the blank under the user's message
        pane.render(60, 12, focused=True)
        assert pane.cursor == 1
        assert pane.current(58) == 0

    def test_following_shows_the_end_of_the_list_not_just_the_cursor(self):
        # The two used to be the same line. With the cursor parked on the last
        # *row* and its blanks below it, a scroll that followed the cursor
        # would push those blanks off the foot of the pane — and the gap
        # between the conversation and the message box is what they are for.
        pane = chat_of(*[ChatEntry(kind="user", text=f"n{n}", seq=n)
                         for n in range(1, 30)])
        drawn = [plain(x).rstrip() for x in pane.render(60, 12, focused=True)]
        assert drawn[-1] == ""
        assert drawn[-2] == "n29"

    def test_and_the_bottom_of_the_pane_still_resumes_following(self):
        # Page-down at the foot clamps onto the last line, which is a blank —
        # and the answer to "are we at the end" has to be yes anyway.
        pane = chat_of(SAID, REPLIED)
        pane.render(60, 12, focused=True)
        pane.move(-10, 11, 58)
        assert not pane.follow
        pane.move(10**9, 11, 58)
        assert pane.follow


# --------------------------------------- one inverted row on the screen, max


class TestOnlyTheFocusedPaneInvertsARow:
    """The highlight belongs to the pane the keys are in, and to no other.

    Every pane used to draw one — the unfocused ones faint-reversed — on the
    reasoning that a pane should show where it was left. It still shows it: a
    list bands its current entry with `▌` and the chat draws its head line
    bold, and neither of those depends on focus. What the dimmed reverse added
    on top of them was a second and a third grey band across a screen that
    already says where the keys are, so picking a session and starting to type
    left three rows all claiming to be the one being pointed at.
    """

    @staticmethod
    def inverted(pane: Pane, *, focused: bool) -> list[str]:
        drawn = pane.render(60, 12, focused=focused)
        return [plain(x).rstrip() for x in drawn if REVERSE in x]

    def test_the_focused_pane_still_has_one(self):
        pane = chat_of(SAID, REPLIED)
        assert self.inverted(pane, focused=True) == ["done"]

    def test_and_the_unfocused_pane_has_none(self):
        pane = chat_of(SAID, REPLIED)
        assert self.inverted(pane, focused=False) == []

    def test_a_list_still_says_where_it_was_left(self):
        # The `▌` gutter is what carries it there, and it is drawn whether or
        # not the pane has the keys — so nothing is actually lost.
        pane = Pane("sessions", [Item(head="one"), Item(head="two")])
        drawn = [plain(x).rstrip() for x in pane.render(60, 12, focused=False)]
        assert drawn[1] == "▌   one"
        assert REVERSE not in "".join(pane.render(60, 12, focused=False))

    def test_and_the_chat_says_it_with_weight(self):
        pane = chat_of(SAID, REPLIED)
        pane.cursor = 0
        drawn = pane.render(60, 12, focused=False)
        head = next(x for x in drawn if plain(x).rstrip() == "▸    you")
        assert BOLD in head
        assert REVERSE not in head


class TestAFilledHeadIsADivider:
    """`Item.fill`: a head that runs out to the pane's width.

    The chat's compaction boundary is the row this exists for. What is tested
    here is the mechanism, not that row: a head with words in it, a rule
    behind them, and the same right edge every other line of the pane has.
    """

    def pane(self, **kw):
        return Pane("chat", [Item(head="compacted ", fill="─", **kw)], flush=True)

    def head(self, pane) -> str:
        return pane.flat(INNER)[0][1]

    def test_the_rule_reaches_the_edge(self):
        head = self.head(self.pane())
        assert cell_width(head) == INNER
        assert head.endswith("─")

    def test_and_starts_after_the_words(self):
        assert "compacted ─" in self.head(self.pane())

    def test_a_head_without_fill_is_left_alone(self):
        pane = Pane("chat", [Item(head="compacted ")], flush=True)
        assert self.head(pane).rstrip() == self.head(pane).rstrip("─").rstrip()

    def test_a_head_too_long_to_fill_is_not_stretched(self):
        # No room left over: the rule is what is dropped, not the words.
        pane = Pane("chat", [Item(head="x" * (INNER + 20), fill="─")], flush=True)
        assert "─" not in self.head(pane)

    def test_a_pattern_repeats_and_is_cut_to_the_edge(self):
        # The dashed case: the rule has to stop at the same column a solid one
        # would, whether or not the pattern divides into the room left.
        for pattern in ("── ", "─ ", "-"):
            pane = Pane("chat", [Item(head="compacted ", fill=pattern)], flush=True)
            head = pane.flat(INNER)[0][1]
            assert cell_width(head) == INNER, pattern
            assert head.startswith(f"     compacted {pattern}"), pattern

    def test_the_fill_lands_behind_the_fold_marker_not_over_it(self):
        # A row that opens keeps its marker column; the rule starts after the
        # head, so the two never compete for the same cells.
        pane = self.pane(body=["the summary"])
        assert self.head(pane).startswith("▸")
