"""Tests for hpca.ui.ansi: padding, rules and the footer.

Everything drawn goes through these, so the property under test is always the
same one — a line is exactly `width` visible cells, whatever styling was
wrapped around it afterwards.
"""

from hpca.ui.ansi import (
    CLIP,
    CYAN,
    DIM,
    ONE_CELL_GLYPHS,
    RESET,
    REVERSE,
    cell_width,
    char_width,
    clip,
    fold,
    footer_line,
    pad,
    reverse,
    rule,
)
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


# ------------------------------------------------------------ cell widths

WIDE = "中文"  # two CJK ideographs: two characters, four cells
EMOJI = "🚀"  # one character, two cells
COMBINED = "é"  # e + combining acute: two characters, one cell
FAMILY = "👩‍💻"  # emoji ZWJ sequence: three characters


class TestCellWidth:
    def test_ascii_is_one_cell_each(self):
        assert cell_width("abc") == 3

    def test_a_cjk_ideograph_is_two(self):
        assert cell_width(WIDE) == 4

    def test_an_emoji_is_two(self):
        assert cell_width(EMOJI) == 2

    def test_a_combining_mark_is_none(self):
        assert cell_width(COMBINED) == 1

    def test_a_zero_width_joiner_is_none(self):
        assert cell_width("‍") == 0

    def test_the_box_drawing_the_ui_is_made_of_stays_one_cell(self):
        # East-asian "ambiguous" must not be counted as two: every rule, marker
        # and arrow in this UI is ambiguous-width, so widening them would
        # break every frame the tests above assert.
        assert cell_width("──▌▸▾●○…↑⇧") == 10


class TestTheGlyphTable:
    """`ONE_CELL_GLYPHS` is an optimisation, never a second opinion.

    It lets a row of the UI's own furniture take the arithmetic path instead
    of being measured character by character. That is only sound while every
    character in it really is one cell wide, and `char_width` is the authority
    on that — so the table is checked against it rather than against a list
    someone typed. A glyph the UI draws that is missing from the table is
    merely slower; a glyph in the table that is not one cell wide would
    silently shift every row it appears in, and fails here instead.
    """

    def test_every_glyph_in_it_is_one_cell_by_the_authority(self):
        wrong = {ch: char_width(ch) for ch in ONE_CELL_GLYPHS if char_width(ch) != 1}
        assert not wrong, wrong

    def test_it_holds_no_duplicates(self):
        # Not correctness, but a duplicate means two people added the same
        # glyph and neither noticed the other's line.
        assert len(set(ONE_CELL_GLYPHS)) == len(ONE_CELL_GLYPHS)

    def test_the_fast_path_agrees_with_the_slow_one(self):
        # The property that matters: taking the shortcut never changes the
        # answer. Checked against a real row rather than a synthetic string.
        row = "── chat ─────────── line 12/40 ──"
        assert cell_width(row) == sum(char_width(c) for c in row)

    def test_a_wide_character_still_defeats_it(self):
        # The table must not be reachable for a row containing anything it
        # does not name — that is what keeps CJK and emoji correct.
        assert cell_width("▸ " + WIDE) == 2 + 4


class TestPadCountsCells:
    def test_a_wide_string_is_padded_by_cells_not_characters(self):
        assert cell_width(pad(WIDE, 10)) == 10

    def test_a_wide_string_that_exactly_fills_is_not_truncated(self):
        assert pad(WIDE, 4) == WIDE

    def test_truncation_never_splits_a_wide_character(self):
        # Four cells of "中文" plus the ellipsis does not fit in four, so the
        # cut lands before 文 and the row is filled out with a space instead —
        # a half-drawn glyph would shift every cell after it.
        drawn = pad(WIDE + "x", 4)
        assert drawn == "中… "
        assert cell_width(drawn) == 4

    def test_a_truncated_row_is_still_exactly_the_width(self):
        for width in range(1, 12):
            assert cell_width(pad("a中b文c🚀d", width)) == width

    def test_a_single_cell_of_room_cannot_hold_a_wide_character(self):
        assert pad(EMOJI, 1) == " "

    def test_a_combining_mark_stays_with_the_character_it_marks(self):
        assert pad(COMBINED, 1) == COMBINED

    def test_a_zero_width_joiner_is_not_left_dangling(self):
        # Cutting between 👩 and the joiner would leave the terminal waiting
        # for a glyph that never comes.
        assert not pad(FAMILY + "xx", 3).startswith("👩‍")


class TestRuleCountsCells:
    def test_a_rule_with_a_wide_label_is_exactly_the_width(self):
        assert cell_width(rule("会話", 40)) == 40

    def test_a_rule_with_a_wide_right_hand_note_is_too(self):
        assert cell_width(rule("chat", 40, "行 3/9")) == 40


class TestFooterCountsCells:
    def test_a_wide_note_still_leaves_the_footer_exact(self):
        assert cell_width(plain(footer_line(PAIRS, 120, "送信しました"))) == 120


class TestFold:
    def test_folding_loses_nothing(self):
        assert "".join(fold("aaa bbb ccc ddd", 8)) == "aaa bbb ccc ddd"

    def test_every_folded_line_fits_the_cell_budget(self):
        for line in fold(WIDE * 10, 7):
            assert cell_width(line) <= 7

    def test_and_a_wide_character_is_never_split_across_two(self):
        assert all(x in ("中", "文") for line in fold(WIDE * 5, 3) for x in line)


class TestClip:
    """One line of a message, and whether it says it is only one line.

    The mark is `" [...]"` and deliberately not the `"…"` `pad` truncates
    with: that one means the terminal is this wide, and this one means there
    is more of this message and → will show it. Only the second has a gesture
    attached, so the two are not allowed to look the same.
    """

    def test_what_fits_is_left_alone(self):
        assert clip("short enough", 40) == "short enough"

    def test_and_exactly_the_budget_still_fits(self):
        assert clip("abcde", 5) == "abcde"

    def test_what_does_not_fit_says_so(self):
        assert clip("a rather longer line than that", 20).endswith(CLIP)

    def test_and_stays_inside_the_budget(self):
        for width in range(7, 40):
            assert cell_width(clip("a rather longer line than that", width)) <= width

    def test_the_mark_reads_as_one_space_after_the_last_word(self):
        # The cut lands on a space often enough that not stripping it would
        # show " [...]" behind a gap half the time and not the other half.
        assert not clip("aaa bbbbbbbbbbbb", 12).endswith("  [...]")

    def test_a_pane_too_narrow_to_say_both_keeps_the_words(self):
        # Six cells are the mark alone. Below that the mark would be the whole
        # line, which tells the reader nothing they can act on.
        assert clip("abcdefgh", 4) == "abcd"

    def test_a_wide_character_is_never_split_by_it(self):
        cut = clip(WIDE * 20, 15)
        assert all(x in ("中", "文") for x in cut.removesuffix(CLIP).rstrip())

    def test_it_is_measured_in_cells_and_not_characters(self):
        assert cell_width(clip(WIDE * 20, 21)) <= 21


class TestReverseKeepsGlyphsWhole:
    def test_a_mark_that_starts_on_a_combining_mark_takes_its_base_too(self):
        # Highlighting the accent alone would paint it over the character
        # before it, which is not the one the cursor is on.
        drawn = reverse(COMBINED + "x", [(1, 2)])
        assert plain(drawn) == COMBINED + "x"
        assert f"{REVERSE}{COMBINED}{RESET}" in drawn

    def test_a_mark_that_ends_before_a_combining_mark_takes_it_along(self):
        drawn = reverse(COMBINED, [(0, 1)])
        assert f"{REVERSE}{COMBINED}{RESET}" in drawn
