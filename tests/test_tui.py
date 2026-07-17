"""Tests for the skeleton TUI (§3): layout, focus model, settings modal."""

import json

import pytest
from textual.widgets import Footer, Input, ListView, TextArea

from hpca.config import Settings
from hpca.tui.app import ColumnPanel, HpcaApp, TopBar
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


class TestNewSession:
    async def test_enter_on_new_session_opens_one_and_focuses_the_entry(
        self, hpca_home
    ):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one("#sessions-list", ListView).index == 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.active_session is not None
            assert app.session_store.list(profile="default") != []
            assert app.focused.id == "chat-input"

    async def test_chat_entry_only_exists_inside_a_session(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat_input = app.query_one("#chat-input", Input)
            assert not chat_input.display
            await pilot.press("right")  # chat column without a session
            assert app.focused.id == "chat-list"
            await pilot.press("left")
            await pilot.press("enter")  # (new session)
            await pilot.pause()
            assert chat_input.display

    async def test_repeated_new_session_reuses_the_empty_one(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            first = app.active_session
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 0  # "(new session)" again
            await pilot.press("enter")
            await pilot.pause()
            assert app.active_session.session_id == first.session_id
            assert len(app.session_store.list(profile="default")) == 1


class TestChatEntryNavigation:
    async def test_arrow_keys_leave_the_entry_only_at_its_edges(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")  # (new session) -> entry focused
            await pilot.pause()
            await pilot.press("h", "i")
            chat_input = app.query_one("#chat-input", Input)
            assert chat_input.value == "hi"

            await pilot.press("left")  # inside the text: cursor only
            assert app.focused.id == "chat-input"
            assert chat_input.cursor_position == 1
            await pilot.press("left")
            assert chat_input.cursor_position == 0
            await pilot.press("left")  # at the left edge: leave the column
            assert app.focused_column_id == "sessions"

    async def test_returning_to_chat_resumes_typing_where_it_stopped(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("h", "i")
            await pilot.press("right")  # cursor at the end: leave the column
            assert app.focused_column_id == "processes"

            await pilot.press("left")  # back into the chat column
            chat_input = app.query_one("#chat-input", Input)
            assert app.focused is chat_input
            assert chat_input.value == "hi"
            assert chat_input.cursor_position == 2
            await pilot.press("!")
            assert chat_input.value == "hi!"

    async def test_settings_not_offered_in_the_chat_column(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.check_action("open_settings", ()) is True
            await pilot.press("enter")  # (new session) -> chat entry
            await pilot.pause()
            assert app.check_action("open_settings", ()) is False
            await pilot.press("s")  # typed, not a hotkey
            assert app.query_one("#chat-input", Input).value == "s"
            assert not isinstance(app.screen, SettingsScreen)


class TestSettingsModal:
    async def test_s_opens_settings_modal(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("s")
            assert isinstance(app.screen, SettingsScreen)

    async def test_escape_closes_without_saving(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("s")
            await pilot.press("escape")
            assert not isinstance(app.screen, SettingsScreen)
            assert not (hpca_home / "settings.json").exists()

    async def test_editor_prefilled_with_current_settings_json(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("s")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            assert data["llm"]["model"] == "qwen3-6b"

    async def test_save_persists_and_updates_app(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("s")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            data["llm"]["model"] = "new-model"
            editor.text = json.dumps(data)
            await pilot.press("ctrl+s")
            await pilot.pause()
            # modal closed, file written, app state updated
            assert not isinstance(app.screen, SettingsScreen)
            on_disk = json.loads((hpca_home / "settings.json").read_text())
            assert on_disk["llm"]["model"] == "new-model"
            assert app.settings.llm.model == "new-model"
            assert "new-model" in app.query_one(TopBar).render_text()

    async def test_invalid_json_shows_error_and_stays_open(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("s")
            editor = app.screen.query_one(TextArea)
            editor.text = "{broken"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            error = app.screen.query_one("#settings-error")
            assert "JSON" in str(error.render())
            assert not (hpca_home / "settings.json").exists()

    async def test_invalid_value_shows_error_and_stays_open(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("s")
            editor = app.screen.query_one(TextArea)
            data = json.loads(editor.text)
            data["clipboard"]["mode"] = "telepathy"
            editor.text = json.dumps(data)
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            assert not (hpca_home / "settings.json").exists()


async def test_copy_text_uses_clipboard_manager(hpca_home):
    app = HpcaApp()
    async with app.run_test(size=(120, 40)):
        result = app.copy_text("hello from hpca")
        assert result.ok
