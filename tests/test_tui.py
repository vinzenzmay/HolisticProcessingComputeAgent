"""Tests for the skeleton TUI (§3): layout, focus model, settings modal."""

import json

import pytest
from textual.widgets import Footer, ListView, TextArea

from hpca.config import Settings
from hpca.tui.app import ChatInput, ColumnPanel, HpcaApp, TopBar
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.settings_screen import SettingsScreen


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    """Isolated app dir so tests never touch the real ~/.HolisticProcessingComputeAgent."""
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def test_app_boots_with_three_columns(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        columns = list(app.query(ColumnPanel))
        assert [c.id for c in columns] == ["sessions", "chat", "processes"]


async def test_top_bar_shows_model_from_settings(hpca_home):
    settings = Settings()
    settings.llm.model = "Qwen/Qwen3.6-35B-A3B-FP8"
    settings.save()
    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        top = app.query_one(TopBar)
        assert "Qwen/Qwen3.6-35B-A3B-FP8" in top.render_text()


async def test_top_bar_shows_version(hpca_home):
    from hpca import __version__

    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        assert f"v{__version__}" in app.query_one(TopBar).render_text()


async def test_footer_present(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        assert app.query_one(Footer)


async def test_initial_focus_is_sessions(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        assert app.focused_column_id == "sessions"


async def test_arrow_keys_cycle_columns(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("right")
        assert app.focused_column_id == "chat"
        await pilot.press("right")
        assert app.focused_column_id == "processes"
        await pilot.press("right")  # wraps
        assert app.focused_column_id == "sessions"
        await pilot.press("left")  # wraps back
        assert app.focused_column_id == "processes"
        await pilot.press("left")
        assert app.focused_column_id == "chat"


async def test_focused_widget_is_the_columns_list(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("right")
        assert isinstance(app.focused, ListView)
        assert app.focused.id == "chat-list"


async def new_session_via_picker(app, pilot):
    """Enter on "(new session)" opens the profile picker; enter again takes
    the highlighted (current) profile."""
    await pilot.press("enter")
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


class TestNewSession:
    async def test_enter_asks_for_a_profile_then_opens_the_session(
        self, hpca_home
    ):
        from hpca.tui.profiles_screen import ProfilePickerScreen

        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one("#sessions-list", ListView).index == 0
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ProfilePickerScreen)
            await pilot.press("enter")  # the current profile is highlighted
            await pilot.pause()
            await pilot.pause()
            assert app.active_session is not None
            assert app.active_session.profile == "default"
            assert app.session_store.list(profile="default") != []
            assert app.focused.id == "chat-input"

    async def test_escaping_the_picker_opens_nothing(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")
            await pilot.press("escape")
            await pilot.pause()
            assert app.active_session is None
            assert app.session_store.list_all() == []

    async def test_chat_entry_only_exists_inside_a_session(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat_input = app.query_one("#chat-input", ChatInput)
            assert not chat_input.display
            await pilot.press("right")  # chat column without a session
            assert app.focused.id == "chat-list"
            await pilot.press("left")
            await new_session_via_picker(app, pilot)
            assert chat_input.display

    async def test_repeated_new_session_reuses_the_empty_one(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            first = app.active_session
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 0  # "(new session)" again
            await new_session_via_picker(app, pilot)
            assert app.active_session.session_id == first.session_id
            assert len(app.session_store.list(profile="default")) == 1


class TestCommandPalette:
    """The palette is rebound from ctrl+p to bare 'p', and only from the
    sessions column — elsewhere 'p' is a letter (e.g. typed into chat)."""

    def test_palette_binding_is_p(self):
        assert HpcaApp.COMMAND_PALETTE_BINDING == "p"

    async def test_palette_only_from_the_sessions_column(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.focused_column_id == "sessions"
            assert app.check_action("command_palette", ()) is True
            await new_session_via_picker(app, pilot)  # -> chat entry focused
            assert app.focused_column_id == "chat"
            assert app.check_action("command_palette", ()) is False

    async def test_p_types_into_chat_instead_of_opening_the_palette(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            await pilot.press("p")
            assert app.query_one("#chat-input", ChatInput).text == "p"


class TestAgentModeSwitching:
    """shift+tab cycles the agent mode, but only from the chat column —
    the sessions and processes columns leave the mode alone."""

    async def test_shift_tab_cycles_mode_from_the_chat_column(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await new_session_via_picker(app, pilot)  # -> chat entry focused
            assert app.focused_column_id == "chat"
            assert app._mode_of(app.active_session) == "manual"
            assert app.check_action("cycle_mode", ()) is True
            await pilot.press("shift+tab")
            assert app._mode_of(app.active_session) == "auto"

    async def test_mode_switch_is_unavailable_off_the_chat_column(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await new_session_via_picker(app, pilot)  # a session now exists
            start = app._mode_of(app.active_session)

            app._focus_column("sessions")
            await pilot.pause()
            assert app.focused_column_id == "sessions"
            assert app.check_action("cycle_mode", ()) is False
            await pilot.press("shift+tab")
            assert app._mode_of(app.active_session) == start

            app._focus_column("processes")
            await pilot.pause()
            assert app.focused_column_id == "processes"
            assert app.check_action("cycle_mode", ()) is False
            await pilot.press("shift+tab")
            assert app._mode_of(app.active_session) == start


class TestChatEntryNavigation:
    async def test_arrow_keys_leave_the_entry_only_at_its_edges(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await new_session_via_picker(app, pilot)  # -> entry focused
            await pilot.press("h", "i")
            chat_input = app.query_one("#chat-input", ChatInput)
            assert chat_input.text == "hi"

            await pilot.press("left")  # inside the text: cursor only
            assert app.focused.id == "chat-input"
            assert chat_input.cursor_location == (0, 1)
            await pilot.press("left")
            assert chat_input.cursor_location == (0, 0)
            await pilot.press("left")  # at the left edge: leave the column
            assert app.focused_column_id == "sessions"

    async def test_returning_to_chat_resumes_typing_where_it_stopped(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            await pilot.press("h", "i")
            await pilot.press("right")  # cursor at the end: leave the column
            assert app.focused_column_id == "processes"

            await pilot.press("left")  # back into the chat column
            chat_input = app.query_one("#chat-input", ChatInput)
            assert app.focused is chat_input
            assert chat_input.text == "hi"
            assert chat_input.cursor_location == (0, 2)
            await pilot.press("!")
            assert chat_input.text == "hi!"

    async def test_config_editor_not_offered_in_the_chat_column(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.check_action("open_settings", ()) is True
            await new_session_via_picker(app, pilot)  # -> chat entry
            assert app.check_action("open_settings", ()) is False
            await pilot.press("c")  # typed, not a hotkey
            assert app.query_one("#chat-input", ChatInput).text == "c"
            assert not isinstance(app.screen, SettingsScreen)


class TestConfigEditorModal:
    async def test_c_opens_the_config_editor(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            assert isinstance(app.screen, SettingsScreen)

    async def test_escape_closes_without_saving(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            await pilot.press("escape")
            assert not isinstance(app.screen, SettingsScreen)
            assert not (hpca_home / "settings.json").exists()

    async def test_editor_prefilled_with_current_settings_json(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            assert data["llm"]["model"] == "qwen3-6b"

    async def test_escape_with_changes_asks_then_saves_on_yes(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            data["llm"]["model"] = "new-model"
            editor.text = json.dumps(data)
            await pilot.press("escape")  # unsaved changes -> "Keep changes?"
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            # modal closed, file written, app state updated
            assert not isinstance(app.screen, SettingsScreen)
            on_disk = json.loads((hpca_home / "settings.json").read_text())
            assert on_disk["llm"]["model"] == "new-model"
            assert app.settings.llm.model == "new-model"
            assert "new-model" in app.query_one(TopBar).render_text()

    async def test_escape_with_changes_discards_on_no(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            data["llm"]["model"] = "unwanted"
            editor.text = json.dumps(data)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            assert not isinstance(app.screen, SettingsScreen)
            assert not (hpca_home / "settings.json").exists()
            assert app.settings.llm.model == "qwen3-6b"

    async def test_invalid_json_shows_error_and_stays_open(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            editor = app.screen.query_one(TextArea)
            editor.text = "{broken"
            await pilot.press("escape")  # cannot save invalid; stays open
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            error = app.screen.query_one("#settings-error")
            assert "JSON" in str(error.render())
            assert not (hpca_home / "settings.json").exists()

    async def test_invalid_value_shows_error_and_stays_open(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("c")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            data["clipboard"]["mode"] = "telepathy"
            editor.text = json.dumps(data)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            assert not (hpca_home / "settings.json").exists()


async def test_copy_text_uses_clipboard_manager(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        result = app.copy_text("hello from hpca")
        assert result.ok


class TestMultiLineEntry:
    async def test_draft_wraps_and_the_box_grows_with_it(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(80, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            chat_input = app.query_one("#chat-input", ChatInput)
            assert chat_input.soft_wrap
            one_line = chat_input.size.height

            chat_input.text = "x" * 400  # far wider than the chat column
            await pilot.pause()
            assert chat_input.size.height > one_line
            # the whole draft is laid out, not clipped to a single strip
            assert chat_input.wrapped_document.height > 1

    async def test_growth_is_capped_so_the_log_stays_visible(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(80, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "line\n" * 100
            for _ in range(3):
                await pilot.pause()
            assert chat_input.size.height <= 10
            assert app.query_one("#chat-list", ListView).size.height > 0

    async def test_enter_sends_and_shift_enter_starts_a_line(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(80, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            chat_input = app.query_one("#chat-input", ChatInput)
            await pilot.press("a")
            await pilot.press("shift+enter")
            await pilot.press("b")
            assert chat_input.text == "a\nb"  # newline, not a submit

    async def test_alt_enter_and_ctrl_j_also_start_a_line(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(80, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            chat_input = app.query_one("#chat-input", ChatInput)
            await pilot.press("a", "alt+enter", "b", "ctrl+j", "c")
            assert chat_input.text == "a\nb\nc"

    async def test_up_moves_between_draft_lines_before_leaving(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(80, 40)) as pilot:
            await new_session_via_picker(app, pilot)
            chat_input = app.query_one("#chat-input", ChatInput)
            await pilot.press("a", "shift+enter", "b")
            assert chat_input.cursor_location == (1, 1)
            await pilot.press("up")  # within the draft
            assert app.focused is chat_input
            assert chat_input.cursor_location[0] == 0
            await pilot.press("up")  # first line: leave for the log (empty here)
            assert app.focused is chat_input
