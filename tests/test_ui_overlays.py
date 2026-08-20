"""Tests for hpca.ui.overlays: the screens that draw over the rows.

Each overlay is an independent render/handle pair, so most of these construct
the app and press the key that opens it — which is also the only thing that
proves the key is wired.
"""

import pytest

from hpca.ui.app import CHAT
from hpca.ui.demo import build
from hpca.ui.overlays import (
    PREVIEW_CHARS,
    ConfigOverlay,
    HelpOverlay,
    LlmOverlay,
    ProfilesOverlay,
    RewindOverlay,
)
from tests.ui_harness import frame, on_own_message, plain, widths

HELP_MENTIONS = ["m", "a", "c", "→", "←", "^u", "^← ^→", "shift-← →", "^del"]


def opened(key: str, width: int = 120, height: int = 40):
    ui = build()
    ui.handle(key, width, height)
    return ui


class TestHelpOverlay:
    def test_question_mark_opens_help(self):
        assert isinstance(opened("?").overlay, HelpOverlay)

    @pytest.mark.parametrize("key", HELP_MENTIONS)
    def test_help_mentions_the_key(self, key: str):
        # The list is longer than a 40-row terminal, so both ends are read.
        ui = opened("?")
        seen = "\n".join(frame(ui, 120, 40))
        ui.handle("end", 120, 40)
        seen += "\n".join(frame(ui, 120, 40))
        assert key in seen

    def test_help_scrolls_back_to_the_top(self):
        ui = opened("?")
        ui.handle("end", 120, 40)
        ui.handle("home", 120, 40)
        assert ui.overlay.offset == 0

    def test_any_other_key_closes_help(self):
        ui = opened("?")
        ui.handle("x", 120, 40)
        assert ui.overlay is None

    def test_the_arrows_scroll_rather_than_close(self):
        ui = opened("?")
        ui.handle("down", 120, 40)
        assert isinstance(ui.overlay, HelpOverlay)
        assert ui.overlay.offset == 1


class TestRewindOverlay:
    def test_enter_opens_the_rewind(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        assert isinstance(ui.overlay, RewindOverlay)

    def test_it_quotes_the_message(self):
        ui = build()
        index = on_own_message(ui)
        ui.handle("enter", 120, 40)
        shown = "\n".join(frame(ui, 120, 40))
        said = ui.chat.items[index].text
        assert (
            said[:30] in shown.replace("\n", " ")
            or said.split(".")[0][:30] in shown
        )

    def test_it_offers_fork(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        assert "fork the session from here" in "\n".join(frame(ui, 120, 40))

    def test_it_offers_roll_back(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        assert "roll this conversation back to here" in "\n".join(frame(ui, 120, 40))

    def test_it_offers_copy(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        assert "copy it into the message box" in "\n".join(frame(ui, 120, 40))

    def test_the_footer_names_all_three(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        foot = plain(ui.render(160, 40)[-1])
        for pair in ("f fork", "r roll back", "c / enter copy", "esc cancel"):
            assert pair in foot

    def test_a_key_it_has_no_answer_for_leaves_it_open(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        ui.handle("z", 120, 40)
        assert isinstance(ui.overlay, RewindOverlay)

    def test_esc_cancels(self):
        ui = build()
        index = on_own_message(ui)
        ui.handle("enter", 120, 40)
        ui.handle("esc", 120, 40)
        assert ui.overlay is None
        assert ui.chat.items[index].kind == "user"

    def test_and_changes_nothing(self):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", 120, 40)
        ui.handle("esc", 120, 40)
        assert ui.input.text() == ""

    def test_long_messages_are_cut_to_a_preview(self):
        shown = "\n".join(plain(x) for x in RewindOverlay("x" * 900, 0).render(80, 20))
        assert shown.count("x") <= PREVIEW_CHARS + 10

    def test_and_say_they_were_cut(self):
        shown = "\n".join(plain(x) for x in RewindOverlay("x" * 900, 0).render(80, 20))
        assert "…" in shown

    @pytest.mark.parametrize("width,height", [(80, 24), (120, 40), (60, 14)])
    def test_the_rewind_frame_is_exactly_the_terminal_size(self, width, height):
        ui = build()
        on_own_message(ui)
        ui.handle("enter", width, height)
        drawn = ui.render(width, height)
        assert len(drawn) == height
        assert widths(drawn) == {width}


class TestManageLlms:
    def test_m_opens_llms(self):
        assert isinstance(opened("m").overlay, LlmOverlay)

    def test_two_stacked_rows(self):
        body = "\n".join(frame(opened("m"), 120, 40))
        assert "── discovered " in body
        assert "── configured " in body

    def test_connected_markers(self):
        body = "\n".join(frame(opened("m"), 120, 40))
        assert "●" in body and "○" in body and "★" in body

    def test_discovered_offers_add(self):
        ui = opened("m")
        assert "enter add to catalog" in plain(ui.render(160, 40)[-1])

    def test_configured_offers_remove(self):
        ui = opened("m")
        ui.handle("ctrl-down", 160, 40)
        foot = plain(ui.render(160, 40)[-1])
        assert "d remove" in foot
        assert "add to catalog" not in foot

    def test_esc_closes(self):
        ui = opened("m")
        ui.handle("esc", 120, 40)
        assert ui.overlay is None


class TestProfilesAndLearnings:
    def test_a_opens_profiles(self):
        assert isinstance(opened("a").overlay, ProfilesOverlay)

    def test_it_lists_a_new_profile_row(self):
        assert "(new profile)" in "\n".join(frame(opened("a"), 120, 40))

    def test_enter_opens_the_editor(self):
        ui = opened("a")
        ui.handle("enter", 120, 40)
        assert ui.overlay.editor is not None

    def test_it_shows_that_profiles_learnings(self):
        ui = opened("a")
        ui.handle("enter", 120, 40)
        assert "scratch/proj" in "\n".join(frame(ui, 120, 40))

    def test_typing_edits(self):
        ui = opened("a")
        ui.handle("enter", 120, 40)
        ui.handle("X", 120, 40)
        assert "X" in ui.overlay.editor.text()

    def test_ctrl_s_keeps_and_closes_the_editor(self):
        ui = opened("a")
        ui.handle("enter", 120, 40)
        ui.handle("X", 120, 40)
        ui.handle("ctrl-s", 120, 40)
        assert ui.overlay.editor is None

    def test_learnings_kept(self):
        ui = opened("a")
        ui.handle("enter", 120, 40)
        ui.handle("X", 120, 40)
        ui.handle("ctrl-s", 120, 40)
        assert "X" in ui.overlay.learnings["hpc"]

    def test_esc_closes_the_screen(self):
        ui = opened("a")
        ui.handle("esc", 120, 40)
        assert ui.overlay is None


class TestConfigEditor:
    def test_c_opens_config(self):
        assert isinstance(opened("c").overlay, ConfigOverlay)

    def test_it_shows_json_with_line_numbers(self):
        shown = "\n".join(frame(opened("c"), 120, 40))
        assert "local_cache" in shown
        assert "  1 {" in shown

    def test_valid_json_saves(self):
        ui = opened("c")
        ui.handle("ctrl-s", 120, 40)
        assert ui.overlay.note == "saved"

    def test_invalid_json_is_reported_not_saved(self):
        ui = opened("c")
        ui.overlay.editor.row = 0
        ui.overlay.editor.col = 0
        ui.overlay.handle("}", 120, 38)
        ui.handle("ctrl-s", 120, 40)
        assert ui.overlay.note.startswith("invalid")

    def test_esc_closes(self):
        ui = opened("c")
        ui.handle("esc", 120, 40)
        assert ui.overlay is None


@pytest.mark.parametrize("width,height", [(80, 24), (120, 40), (60, 14)])
@pytest.mark.parametrize("key", ["?", "m", "a", "c"])
def test_an_overlay_frame_is_exactly_the_terminal_size(key, width, height):
    ui = build()
    ui.handle(key, width, height)
    drawn = ui.render(width, height)
    assert len(drawn) == height
    assert widths(drawn) == {width}


def test_an_overlay_covers_the_rows_it_is_drawn_over():
    ui = build()
    ui.focus = CHAT
    assert "── chat " in "\n".join(frame(ui, 120, 40))
    ui.handle("?", 120, 40)
    assert "── chat " not in "\n".join(frame(ui, 120, 40))
