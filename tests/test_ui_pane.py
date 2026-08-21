"""Tests for hpca.ui.pane: opening entries, closing them, and reordering.

Driven through `RowUI.handle` where the harness did, because the arrow keys are
the behaviour: what `→` means depends on whether the entry is already open and
on whether it has a body at all.
"""

from hpca.ui.app import CHAT, SESSIONS, WATCHERS
from hpca.ui.demo import build
from hpca.ui.pane import Item, Pane
from tests.ui_harness import widths

# 120 columns of terminal, minus the two the gutter takes.
INNER = 118


def on_an_entry_with_a_body(ui):
    """Walk the chat cursor down to the first entry that has something to open."""
    pane = ui.chat
    pane.cursor = 0
    while not pane.items[pane.current(INNER)].body:
        pane.move(1, 20, INNER)
    return pane, pane.current(INNER)


def chat_ui():
    ui = build()
    ui.focus = CHAT
    return ui


class TestArrowsOpenAndCloseEntries:
    def test_right_opens_the_entry(self):
        ui = chat_ui()
        pane, item = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        assert pane.is_open(item)

    def test_the_row_got_longer(self):
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        before = len(pane.flat(INNER))
        ui.handle("right", 120, 40)
        assert len(pane.flat(INNER)) > before

    def test_and_the_cursor_sits_on_its_head_line(self):
        ui = chat_ui()
        pane, item = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        assert pane.current(INNER) == item

    def test_right_again_steps_into_the_open_entry(self):
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        head = pane.cursor
        ui.handle("right", 120, 40)
        assert pane.cursor == head + 1

    def test_without_closing_it(self):
        ui = chat_ui()
        pane, item = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        ui.handle("right", 120, 40)
        assert pane.is_open(item)

    def test_left_closes_it_from_inside_the_body(self):
        ui = chat_ui()
        pane, item = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        ui.handle("right", 120, 40)
        ui.handle("left", 120, 40)
        assert not pane.is_open(item)

    def test_and_puts_you_back_on_its_head_line(self):
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        head = pane.cursor
        ui.handle("right", 120, 40)
        ui.handle("left", 120, 40)
        assert pane.cursor == head

    def test_left_on_a_closed_entry_does_nothing(self):
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        ui.handle("right", 120, 40)
        head = pane.cursor
        ui.handle("right", 120, 40)
        ui.handle("left", 120, 40)
        ui.handle("left", 120, 40)
        assert pane.cursor == head
        assert not pane.expanded

    def test_shift_right_opens_every_entry(self):
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        ui.handle("shift-right", 120, 40)
        assert len(pane.expanded) == sum(1 for x in pane.items if x.body)

    def test_shift_left_closes_them_all(self):
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        ui.handle("shift-right", 120, 40)
        ui.handle("shift-left", 120, 40)
        assert pane.expanded == set()

    def test_e_no_longer_opens_anything(self):
        # `e` was an earlier expand key; it is a free letter now.
        ui = chat_ui()
        pane, _ = on_an_entry_with_a_body(ui)
        ui.handle("e", 120, 40)
        assert pane.expanded == set()
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
