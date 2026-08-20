"""Tests for hpca.ui.app: layout, focus, key dispatch and the chat rewind.

`RowUI` is synchronous and does no I/O, so every one of these is "construct it,
feed keys, read the frame back and look at it".
"""

import pytest

from hpca.ui.app import CHAT, INPUT, SESSIONS, WATCHERS, RowUI
from hpca.ui.demo import build
from hpca.ui.keys import decode
from tests.ui_harness import clocked, frame, on_own_message, plain, widths

SIZES = [(80, 24), (120, 40), (200, 60), (60, 14), (40, 10), (100, 8)]
ROW_NAMES = ["sessions", "chat", "message", "watchers"]
LONG = (
    "the quick brown fox jumps over the lazy dog and keeps going well past "
    "the edge of the box"
)


def typed(ui: RowUI, text: str, width: int = 120, height: int = 40) -> RowUI:
    for ch in text:
        ui.handle(ch, width, height)
    return ui


def in_the_box() -> RowUI:
    ui = build()
    ui.focus = INPUT
    return ui


def footer_of(focus: int, width: int = 160) -> str:
    ui = build()
    ui.focus = focus
    return plain(ui.render(width, 40)[-1])


# ---------------------------------------------------------------- geometry


@pytest.mark.parametrize("width,height", SIZES)
def test_the_frame_is_exactly_the_terminal_size(width, height):
    drawn = build().render(width, height)
    assert len(drawn) == height
    assert widths(drawn) == {width}


@pytest.mark.parametrize("name", ROW_NAMES)
def test_the_row_is_on_screen(name):
    assert f"── {name} " in "\n".join(frame(build(), 120, 40))


def test_the_rows_are_drawn_top_to_bottom():
    text = "\n".join(frame(build(), 120, 40))
    order = [text.index(f"── {name} ") for name in ROW_NAMES]
    assert order == sorted(order)


# ------------------------------------------------------- the context footer


def test_chat_offers_write():
    assert "i write" in footer_of(CHAT)


def test_sessions_offers_rename():
    assert "r rename" in footer_of(SESSIONS)


def test_sessions_does_not_offer_unwatch():
    assert "unwatch" not in footer_of(SESSIONS)


def test_watchers_offers_unwatch():
    assert "d unwatch" in footer_of(WATCHERS)


def test_watchers_does_not_offer_rename():
    assert "r rename" not in footer_of(WATCHERS)


def test_input_offers_send():
    assert "enter send" in footer_of(INPUT)


# ------------------------------------------------------------ the message row


def test_i_focuses_the_message_row():
    ui = build()
    ui.focus = CHAT
    ui.handle("i", 120, 40)
    assert ui.focus == INPUT


def test_typing_lands_in_the_buffer():
    assert typed(in_the_box(), "hello there").input.text() == "hello there"


def test_backspace_removes_the_last_character():
    ui = typed(in_the_box(), "hello there")
    ui.handle("backspace", 120, 40)
    assert ui.input.text() == "hello ther"


def test_alt_enter_grows_the_row():
    ui = typed(in_the_box(), "hello")
    before = ui._input_h(120)
    ui.handle("alt-enter", 120, 40)
    assert ui._input_h(120) == before + 1


def test_enter_sends():
    ui = typed(in_the_box(), "hello there")
    before = len(ui.panes[1].items)
    ui.handle("enter", 120, 40)
    assert len(ui.panes[1].items) == before + 2
    assert ui.input.text() == ""


def test_ctrl_up_returns_to_chat():
    ui = in_the_box()
    ui.handle("ctrl-up", 120, 40)
    assert ui.focus == CHAT


def test_the_input_row_caps_at_max_input():
    ui = in_the_box()
    for _ in range(20):
        ui.handle("alt-enter", 120, 40)
    assert ui._input_h(120) == 1 + ui.MAX_INPUT


def test_the_frame_is_still_exact_with_a_tall_input():
    ui = in_the_box()
    for _ in range(20):
        ui.handle("alt-enter", 120, 40)
    drawn = ui.render(120, 40)
    assert len(drawn) == 40
    assert widths(drawn) == {120}


# ------------------------------- the message box wraps instead of overflowing


def long_draft() -> RowUI:
    return typed(in_the_box(), LONG, width=80)


def test_a_long_message_is_still_one_logical_line():
    assert len(long_draft().input.lines) == 1


def test_but_several_screen_lines():
    assert long_draft().input.height(78) > 1


def test_the_box_grew_with_it():
    ui = long_draft()
    assert ui._input_h(80) == 1 + ui.input.height(78)


def test_the_frame_is_still_exactly_eighty_wide():
    assert widths(long_draft().render(80, 40)) == {80}


def test_nothing_is_lost_to_the_fold():
    drawn = "\n".join(frame(long_draft(), 80, 40))
    assert LONG.replace(" ", "") in drawn.replace(" ", "").replace("\n", "")


def test_no_screen_line_is_over_the_width():
    assert all(len(x) == 80 for x in frame(long_draft(), 80, 40))


def test_up_moves_one_screen_line():
    ui = long_draft()
    ui.input.row, ui.input.col = 0, len(LONG)
    before = ui.input.cursor_visual(78)[0]
    ui.handle("up", 80, 40)
    assert ui.input.cursor_visual(78)[0] == before - 1


def test_and_stays_on_the_same_logical_line():
    ui = long_draft()
    ui.input.row, ui.input.col = 0, len(LONG)
    ui.handle("up", 80, 40)
    assert ui.input.row == 0


# ------------------------------------------------------------- key dispatch


def test_shift_arrow_does_not_throw_you_into_the_chat():
    # The old failure: an escape the table did not know arrived as "esc" plus
    # its letters, and the esc left the box.
    ui = in_the_box()
    for key in decode(b"\x1b[1;2D\x1b[1;9Z"):
        ui.handle(key, 120, 40)
    assert ui.focus == INPUT


def test_letters_type_rather_than_trigger_screens():
    ui = typed(in_the_box(), "maceq")
    assert ui.overlay is None
    assert ui.input.text() == "maceq"


# --------------------------------------------------------- switching sessions


def switched_to_the_second() -> RowUI:
    ui = build()
    ui.chat  # noqa: B018 — looking at the first session is what loads it
    ui.focus = SESSIONS
    ui.handle("home", 120, 40)
    ui.handle("down", 120, 40)
    ui.handle("enter", 120, 40)
    ui.render(120, 40)  # drawing the new session is what loads it
    return ui


def test_nothing_is_loaded_until_it_is_looked_at():
    assert sum(x.loaded for x in build().sessions) == 0


def test_looking_at_one_loads_exactly_it():
    ui = build()
    ui.chat  # noqa: B018
    assert sum(x.loaded for x in ui.sessions) == 1


def test_enter_opens_the_session_under_the_cursor():
    assert switched_to_the_second().active == 1


def test_the_chat_row_is_a_different_conversation():
    ui = build()
    first_chat = ui.chat
    ui.focus = SESSIONS
    ui.handle("home", 120, 40)
    ui.handle("down", 120, 40)
    ui.handle("enter", 120, 40)
    assert ui.chat is not first_chat


def test_what_is_drawn_actually_changed():
    ui = build()
    first_text = frame(ui, 120, 40)
    ui.focus = SESSIONS
    ui.handle("home", 120, 40)
    ui.handle("down", 120, 40)
    ui.handle("enter", 120, 40)
    assert frame(ui, 120, 40) != first_text


def test_the_chat_shows_the_new_sessions_topic():
    ui = switched_to_the_second()
    assert ui.sessions[1].title.split()[1] in "\n".join(frame(ui, 120, 40))


def test_the_watchers_swapped_too():
    ui = switched_to_the_second()
    assert ui.watchers is not ui.sessions[0].watchers


def test_the_model_line_follows_the_session():
    ui = switched_to_the_second()
    assert ui.model == ui.sessions[1].model


def test_the_profile_follows_the_session():
    ui = switched_to_the_second()
    assert ui.profile == ui.sessions[1].profile


def test_exactly_one_session_is_marked_open():
    ui = switched_to_the_second()
    marks = [x for x in frame(ui, 120, 40) if "●" in x and "m ago" in x]
    assert len(marks) == 1


def test_the_mark_is_on_the_open_one():
    ui = switched_to_the_second()
    marks = [x for x in frame(ui, 120, 40) if "●" in x and "m ago" in x]
    assert ui.sessions[1].title[:20] in marks[0]


def test_switching_is_still_lazy():
    ui = switched_to_the_second()
    assert sum(x.loaded for x in ui.sessions) == 2


def remembering_second_session() -> tuple[RowUI, int, set[int]]:
    """The second session, left with a moved cursor, an open entry and a
    half-typed message — the three things a switch has to bring back."""
    ui = switched_to_the_second()
    ui.focus = CHAT
    ui.handle("home", 120, 40)
    ui.handle("down", 120, 40)
    ui.handle("right", 120, 40)
    where, opened = ui.chat.cursor, set(ui.chat.expanded)
    ui.focus = INPUT
    typed(ui, "half typed")
    return ui, where, opened


def test_you_can_go_back_to_the_first_session():
    ui, _, _ = remembering_second_session()
    ui.focus = SESSIONS
    ui.handle("up", 120, 40)
    ui.handle("enter", 120, 40)
    assert ui.active == 0


def test_enter_also_lands_you_in_the_message_box():
    ui, _, _ = remembering_second_session()
    ui.focus = SESSIONS
    ui.handle("up", 120, 40)
    ui.handle("enter", 120, 40)
    assert ui.focus == INPUT


def test_the_first_sessions_own_draft_is_empty():
    ui, _, _ = remembering_second_session()
    ui.focus = SESSIONS
    ui.handle("up", 120, 40)
    ui.handle("enter", 120, 40)
    assert ui.input.text() == ""


def back_and_forth() -> tuple[RowUI, int, set[int]]:
    ui, where, opened = remembering_second_session()
    ui.focus = SESSIONS
    ui.handle("up", 120, 40)
    ui.handle("enter", 120, 40)
    ui.focus = SESSIONS
    ui.handle("down", 120, 40)
    ui.handle("enter", 120, 40)
    return ui, where, opened


def test_the_cursor_line_is_remembered():
    ui, where, _ = back_and_forth()
    assert ui.chat.cursor == where


def test_the_open_entries_are_remembered():
    ui, _, opened = back_and_forth()
    assert ui.chat.expanded == opened


def test_the_draft_is_remembered():
    ui, _, _ = back_and_forth()
    assert ui.input.text() == "half typed"


def test_opening_does_not_move_the_session_highlight():
    ui, _, _ = back_and_forth()
    ui.focus = SESSIONS
    ui.handle("end", 120, 40)
    last = ui.session_pane.cursor
    ui.handle("enter", 120, 40)
    assert ui.session_pane.cursor == last


def with_an_empty_watcher_row() -> RowUI:
    ui = build()
    ui.active = next(i for i, x in enumerate(ui.sessions) if x.watch_count == 0)
    ui._refresh_sessions()
    return ui


def test_an_empty_watcher_row_shrinks_to_its_minimum():
    assert with_an_empty_watcher_row()._heights(40, 120)[3] == 2


def test_the_frame_is_still_exact_with_an_empty_watcher_row():
    drawn = with_an_empty_watcher_row().render(120, 40)
    assert len(drawn) == 40
    assert widths(drawn) == {120}


# ----------------------------------------- escape is the stop gesture, not an exit


def half_a_message() -> RowUI:
    ui = clocked(build())
    ui.focus = INPUT
    return typed(ui, "half a message")


def test_one_esc_does_not_leave_the_box():
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    assert ui.focus == INPUT


def test_one_esc_sets_no_note_of_its_own():
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    assert ui.note == ""


def test_one_esc_keeps_the_draft():
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    assert ui.input.text() == "half a message"


def test_the_footer_says_esc_again_to_stop():
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    assert "esc again to stop" in plain(ui.render(160, 40)[-1])


def test_the_hint_is_in_red():
    from hpca.ui.ansi import RED

    ui = half_a_message()
    ui.handle("esc", 120, 40)
    assert RED + "esc again to stop" in ui.render(160, 40)[-1]


def test_the_armed_footer_is_still_exactly_the_width():
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    assert len(plain(ui.render(160, 40)[-1])) == 160


def test_the_hint_expires_on_its_own():
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    ui._now += 2.0
    assert "esc again to stop" not in plain(ui.render(160, 40)[-1])


def armed_then_stopped() -> RowUI:
    ui = half_a_message()
    ui.handle("esc", 120, 40)
    ui._now += 0.2
    ui.handle("esc", 120, 40)
    return ui


def test_the_second_esc_stops_the_turn():
    assert armed_then_stopped().note == "stopped the turn"


def test_and_the_hint_is_gone():
    ui = armed_then_stopped()
    assert "esc again to stop" not in plain(ui.render(160, 40)[-1])


def test_still_in_the_box_still_typing():
    ui = armed_then_stopped()
    assert ui.focus == INPUT
    assert ui.input.text() == "half a message"


def test_a_third_esc_opens_a_fresh_pair_and_does_not_re_stop():
    ui = armed_then_stopped()
    ui._now += 0.2
    ui.note = ""
    ui.handle("esc", 120, 40)
    assert ui.note == ""


def test_but_it_is_armed_for_a_fourth():
    ui = armed_then_stopped()
    ui._now += 0.2
    ui.note = ""
    ui.handle("esc", 120, 40)
    assert ui._esc_armed_at is not None


def test_the_fourth_stops_again():
    ui = armed_then_stopped()
    ui._now += 0.2
    ui.note = ""
    ui.handle("esc", 120, 40)
    ui._now += 0.1
    ui.handle("esc", 120, 40)
    assert ui.note == "stopped the turn"


def test_two_escapes_too_far_apart_do_nothing():
    ui = clocked(build())
    ui.focus = INPUT
    ui.handle("esc", 120, 40)
    ui._now += 2.0
    ui.handle("esc", 120, 40)
    assert ui.note == ""


@pytest.mark.parametrize("row", [CHAT, SESSIONS, WATCHERS])
def test_esc_esc_stops_from_the_row_too(row):
    ui = clocked(build())
    ui.focus = row
    ui.handle("esc", 120, 40)
    ui._now += 0.1
    ui.handle("esc", 120, 40)
    assert ui.note == "stopped the turn"


def test_a_lone_esc_in_a_row_moves_nothing():
    ui = clocked(build())
    ui.focus = CHAT
    ui.handle("esc", 120, 40)
    assert ui.focus == CHAT
    assert ui.note == ""


def test_arrow_keys_do_not_arm_the_stop():
    # They are esc-prefixed on the wire, but the decoder resolves them before
    # handle() ever sees an escape.
    ui = clocked(build())
    ui.focus = CHAT
    for key in decode(b"\x1b[A\x1b[B"):
        ui.handle(key, 120, 40)
    assert ui._esc_armed_at is None


@pytest.mark.parametrize("row", [INPUT, CHAT, SESSIONS, WATCHERS])
def test_the_footer_offers_esc_esc_stop_in_every_row(row):
    assert "esc esc stop" in footer_of(row)


@pytest.mark.parametrize("row", [INPUT, CHAT, SESSIONS, WATCHERS])
def test_and_never_plain_esc_chat(row):
    assert "esc chat" not in footer_of(row)


# ------------------------------------- enter on a session goes straight to typing


def test_the_session_opened():
    assert switched_to_the_second().active == 1


def test_focus_is_the_message_box():
    assert switched_to_the_second().focus == INPUT


def test_so_typing_goes_straight_in():
    assert typed(switched_to_the_second(), "hi").input.text() == "hi"


def test_the_footer_says_send_after_opening_a_session():
    ui = switched_to_the_second()
    assert "enter send" in plain(ui.render(160, 40)[-1])


def test_it_still_says_which_session_opened():
    assert "opened" in switched_to_the_second().note


# --------------------------- enter on your own message: fork, roll back, copy


def at_the_rewind() -> tuple[RowUI, int, str]:
    ui = build()
    index = on_own_message(ui)
    said = ui.chat.items[index].text
    ui.handle("enter", 120, 40)
    return ui, index, said


def test_enter_elsewhere_in_the_log_goes_to_the_box():
    ui = build()
    ui.focus = CHAT
    ui.chat.cursor = 0
    while ui.chat.items[ui.chat.current(118)].kind == "user":
        ui.chat.move(1, 20, 118)
    ui.handle("enter", 120, 40)
    assert ui.focus == INPUT
    assert ui.overlay is None


def test_c_copies_it_in():
    ui, _, said = at_the_rewind()
    ui.handle("c", 120, 40)
    assert ui.input.text() == said


def test_and_focuses_the_box():
    ui, _, _ = at_the_rewind()
    ui.handle("c", 120, 40)
    assert ui.focus == INPUT


def test_the_cursor_is_behind_the_reused_text():
    ui, _, _ = at_the_rewind()
    ui.handle("c", 120, 40)
    assert (ui.input.row, ui.input.col) == (
        len(ui.input.lines) - 1,
        len(ui.input.lines[-1]),
    )


def test_enter_is_copy_too():
    ui = build()
    on_own_message(ui)
    ui.focus = INPUT
    typed(ui, "already typing")
    ui.focus = CHAT
    ui.handle("enter", 120, 40)
    ui.handle("enter", 120, 40)
    assert ui.overlay is None


def test_a_draft_is_never_lost():
    ui = build()
    index = on_own_message(ui)
    said = ui.chat.items[index].text
    ui.focus = INPUT
    typed(ui, "already typing")
    ui.focus = CHAT
    ui.handle("enter", 120, 40)
    ui.handle("enter", 120, 40)
    assert ui.input.text() == f"already typing\n{said}"


def test_a_draft_ending_in_a_space_continues_on_that_line():
    ui = build()
    index = on_own_message(ui)
    said = ui.chat.items[index].text
    ui.input.set_text("rerun this: ")
    ui.handle("enter", 120, 40)
    ui.handle("c", 120, 40)
    assert ui.input.text() == f"rerun this: {said}"


def forked() -> tuple[RowUI, int, list, str, int]:
    ui = build()
    index = on_own_message(ui)
    before = list(ui.chat.items)
    title = ui.session.title
    sessions = len(ui.sessions)
    ui.handle("enter", 120, 40)
    ui.handle("f", 120, 40)
    return ui, index, before, title, sessions


def test_f_makes_a_new_session():
    ui, _, _, _, sessions = forked()
    assert len(ui.sessions) == sessions + 1


def test_it_is_named_as_a_fork():
    ui, _, _, title, _ = forked()
    assert ui.session.title == f"{title} (fork)"


def test_that_the_fork_stops_before_the_message():
    ui, index, _, _, _ = forked()
    assert len(ui.chat.items) == index


def test_the_original_is_untouched():
    ui, _, before, _, _ = forked()
    assert ui.sessions[1].chat.items == before


def test_and_it_opens_ready_to_type():
    ui, _, _, _, _ = forked()
    assert ui.focus == INPUT


def test_the_fork_is_the_one_marked_open():
    ui, _, _, _, _ = forked()
    assert "●" in "\n".join(frame(ui, 120, 40))


def test_the_frame_is_still_exact_after_a_fork():
    ui, _, _, _, _ = forked()
    drawn = ui.render(120, 40)
    assert len(drawn) == 40
    assert widths(drawn) == {120}


def rolled_back() -> tuple[RowUI, int, int, int]:
    ui = build()
    index = on_own_message(ui)
    total = len(ui.chat.items)
    sessions = len(ui.sessions)
    ui.chat.expanded = {1, index + 1}
    ui.handle("enter", 120, 40)
    ui.handle("r", 120, 40)
    return ui, index, total, sessions


def test_r_drops_everything_from_there():
    ui, index, _, _ = rolled_back()
    assert len(ui.chat.items) == index


def test_but_keeps_what_came_before():
    ui, index, total, _ = rolled_back()
    assert index > 0
    assert len(ui.chat.items) < total


def test_a_rollback_makes_no_new_session():
    ui, _, _, sessions = rolled_back()
    assert len(ui.sessions) == sessions


def test_it_says_how_much_went():
    ui, index, total, _ = rolled_back()
    assert f"({total - index} entries gone)" in ui.note


def test_open_entries_past_the_cut_are_forgotten():
    ui, _, _, _ = rolled_back()
    assert ui.chat.expanded == {1}


def test_the_frame_is_still_exact_after_a_rollback():
    ui, _, _, _ = rolled_back()
    drawn = ui.render(120, 40)
    assert len(drawn) == 40
    assert widths(drawn) == {120}
