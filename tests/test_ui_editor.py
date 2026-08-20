"""Tests for hpca.ui.editor: folding, word motion, selection, drawing."""

from hpca.ui.editor import Editor, wrap_spans
from tests.ui_harness import plain

SPACED = "aaa bbb ccc ddd"
RUN = "abcdefghijkl"


def pieces(text: str, width: int) -> list[str]:
    return [text[a:b] for a, b in wrap_spans(text, width)]


def marked(text: str, presses: int, key: str = "shift-right") -> Editor:
    editor = Editor(text, wrap=True)
    editor.col = 0
    for _ in range(presses):
        editor.handle(key)
    return editor


class TestWrapping:
    def test_wraps_at_a_space(self):
        assert pieces(SPACED, 8) == ["aaa bbb ", "ccc ddd"]

    def test_spans_lose_nothing(self):
        assert "".join(pieces(SPACED, 8)) == SPACED

    def test_hard_breaks_a_word_with_no_space(self):
        assert pieces(RUN, 5) == ["abcde", "fghij", "kl"]

    def test_a_full_last_line_keeps_a_cursor_slot(self):
        assert wrap_spans("abcde", 5)[-1] == (5, 5)

    def test_an_empty_line_is_still_one_span(self):
        assert wrap_spans("", 8) == [(0, 0)]

    def test_a_width_below_one_refuses_to_fold(self):
        assert wrap_spans(SPACED, 0) == [(0, len(SPACED))]

    def test_sideways_editors_are_unwrapped(self):
        # The config editor scrolls sideways rather than reflowing the twenty
        # lines under the cursor on every keystroke.
        assert Editor("x" * 200).height(40) == 1

    def test_a_wrapping_editor_folds_the_same_text_into_several_rows(self):
        assert Editor("x " * 60, wrap=True).height(40) > 1


class TestWordMotion:
    def test_ctrl_left_goes_to_the_start_of_the_word(self):
        editor = Editor("alpha beta gamma", wrap=True)
        editor.col = len("alpha beta gamma")
        editor.handle("ctrl-left")
        assert editor.col == 11

    def test_ctrl_left_again_reaches_the_word_before(self):
        editor = Editor("alpha beta gamma", wrap=True)
        editor.col = len("alpha beta gamma")
        editor.handle("ctrl-left")
        editor.handle("ctrl-left")
        assert editor.col == 6

    def test_ctrl_right_goes_to_the_end_of_the_word(self):
        editor = Editor("alpha beta gamma", wrap=True)
        editor.col = 6
        editor.handle("ctrl-right")
        assert editor.col == 10

    def test_ctrl_right_again_reaches_the_next_words_end(self):
        editor = Editor("alpha beta gamma", wrap=True)
        editor.col = 6
        editor.handle("ctrl-right")
        editor.handle("ctrl-right")
        assert editor.col == 16


class TestWordDelete:
    def test_ctrl_del_eats_the_next_word(self):
        editor = Editor("alpha beta gamma", wrap=True)
        editor.col = 6
        editor.handle("ctrl-delete")
        assert editor.text() == "alpha  gamma"

    def test_ctrl_backspace_eats_the_word_before(self):
        editor = Editor("alpha  gamma", wrap=True)
        editor.col = 6
        editor.handle("ctrl-backspace")
        assert editor.text() == " gamma"


class TestSelection:
    def test_shift_right_marks(self):
        assert marked("alpha beta gamma", 5).selected() == "alpha"

    def test_shift_ctrl_right_marks_a_whole_word(self):
        editor = marked("alpha beta gamma", 5)
        editor.handle("shift-ctrl-right")
        assert editor.selected() == "alpha beta"

    def test_typing_replaces_the_marked_text(self):
        editor = marked("alpha beta gamma", 5)
        editor.handle("shift-ctrl-right")
        editor.handle("x")
        assert editor.text() == "x gamma"

    def test_and_the_mark_is_gone(self):
        editor = marked("alpha beta gamma", 5)
        editor.handle("shift-ctrl-right")
        editor.handle("x")
        assert editor.sel_range() is None

    def test_shift_left_marks_backwards(self):
        editor = Editor("alpha beta", wrap=True)
        editor.end()
        for _ in range(4):
            editor.handle("shift-left")
        assert editor.selected() == "beta"

    def test_backspace_deletes_the_mark_not_one_char(self):
        editor = Editor("alpha beta", wrap=True)
        editor.end()
        for _ in range(4):
            editor.handle("shift-left")
        editor.handle("backspace")
        assert editor.text() == "alpha "

    def test_shift_end_marks_to_the_end(self):
        editor = Editor("alpha beta", wrap=True)
        editor.handle("shift-end")
        assert editor.selected() == "alpha beta"

    def test_a_plain_motion_drops_the_mark(self):
        editor = Editor("alpha beta", wrap=True)
        editor.handle("shift-end")
        editor.handle("left")
        assert editor.sel_range() is None

    def test_marks_across_lines(self):
        editor = Editor("one\ntwo\nthree", wrap=True)
        editor.handle("shift-down")
        editor.handle("shift-end")
        assert editor.selected() == "one\ntwo"

    def test_delete_removes_it_and_joins(self):
        editor = Editor("one\ntwo\nthree", wrap=True)
        editor.handle("shift-down")
        editor.handle("shift-end")
        editor.handle("delete")
        assert editor.text() == "\nthree"


class TestDrawing:
    def test_selection_renders_without_shifting_the_row(self):
        editor = marked("alpha beta", 5)
        assert len(plain(editor.render(40, 1, focused=True)[0])) == 40

    def test_selection_is_highlighted(self):
        editor = marked("alpha beta", 5)
        assert "\x1b[7m" in editor.render(40, 1, focused=True)[0]

    def test_an_unfocused_editor_shows_neither_cursor_nor_selection(self):
        editor = marked("alpha beta", 5)
        assert "\x1b[7m" not in editor.render(40, 1, focused=False)[0]

    def test_numbered_editors_gutter_every_logical_line_once(self):
        editor = Editor("one\ntwo")
        drawn = [plain(x) for x in editor.render(40, 2, focused=False, numbers=True)]
        assert drawn[0].strip() == "1 one"
        assert drawn[1].strip() == "2 two"
