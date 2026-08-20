"""Tests for hpca.ui.ansi: padding, rules and the footer.

Everything drawn goes through these, so the property under test is always the
same one — a line is exactly `width` visible cells, whatever styling was
wrapped around it afterwards.
"""

from hpca.ui.ansi import CYAN, DIM, RESET, footer_line, pad, reverse, rule
from hpca.ui.app import INPUT
from hpca.ui.demo import build
from tests.ui_harness import plain

PAIRS = [("enter", "send"), ("esc esc", "stop"), ("?", "keys"), ("q", "quit")]


class TestPad:
    def test_short_text_is_padded_out_to_the_width(self):
        assert pad("abc", 8) == "abc     "

    def test_long_text_is_truncated_with_an_ellipsis(self):
        assert pad("abcdefghij", 5) == "abcd…"

    def test_a_width_of_one_has_no_room_for_an_ellipsis(self):
        assert pad("abcdef", 1) == "a"

    def test_no_width_at_all_draws_nothing(self):
        assert pad("abcdef", 0) == ""


class TestRule:
    def test_a_rule_is_exactly_the_width(self):
        assert len(rule("chat", 40)) == 40

    def test_a_rule_carries_its_label_and_its_right_hand_note(self):
        drawn = rule("chat", 40, "line 3/9")
        assert drawn.startswith("── chat ")
        assert drawn.endswith("line 3/9 ──")


class TestReverse:
    def test_marking_nothing_leaves_the_text_alone(self):
        assert reverse("alpha beta", []) == "alpha beta"

    def test_a_marked_range_is_wrapped_and_nothing_else_moves(self):
        assert plain(reverse("alpha beta", [(0, 5)])) == "alpha beta"

    def test_an_empty_range_is_not_a_mark(self):
        assert reverse("alpha", [(2, 2)]) == "alpha"


class TestFooterLine:
    def test_narrow_footer_still_exactly_width(self):
        ui = build()
        ui.focus = INPUT
        assert len(plain(ui.render(60, 40)[-1])) == 60

    def test_narrow_footer_drops_whole_pairs(self):
        ui = build()
        ui.focus = INPUT
        narrow = plain(ui.render(60, 40)[-1])
        assert not narrow.rstrip().endswith("…")

    def test_a_pair_that_does_not_fit_is_left_out_entirely(self):
        wide = plain(footer_line(PAIRS, 200))
        narrow = plain(footer_line(PAIRS, 24))
        assert "q quit" in wide
        assert "q quit" not in narrow
        assert "enter send" in narrow

    def test_keys_are_bright_and_labels_dim(self):
        drawn = footer_line([("q", "quit")], 40)
        assert f"{CYAN}q{RESET} {DIM}quit{RESET}" in drawn

    def test_a_note_is_drawn_in_the_style_it_was_given(self):
        drawn = footer_line(PAIRS, 120, "sent", CYAN)
        assert f"{CYAN}sent{RESET}" in drawn
        assert len(plain(drawn)) == 120
