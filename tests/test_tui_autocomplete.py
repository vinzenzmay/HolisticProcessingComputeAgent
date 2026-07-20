"""Slash-command autocomplete: substring match, frequency sort, ↑/↓ select."""

import pytest

from hpca.db import command_use_counts, connect, init_db, record_command_use
from hpca.tui.app import COMMANDS, ChatInput, HpcaApp


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def menu_names(app):
    return [name for name, _ in app._command_matches]


class TestCommandUsageStore:
    def test_record_and_read_counts(self, tmp_path):
        conn = connect(tmp_path / "t.db")
        init_db(conn)
        assert command_use_counts(conn) == {}
        record_command_use(conn, "skills-list")
        record_command_use(conn, "skills-list")
        record_command_use(conn, "memorize")
        assert command_use_counts(conn) == {"skills-list": 2, "memorize": 1}
        conn.close()


class TestMatching:
    async def test_slash_lists_every_command(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/")
            assert app.command_menu_active()
            assert set(menu_names(app)) == {name for name, _ in COMMANDS}

    async def test_substring_not_only_prefix(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            # "list" is not a prefix of any command, but is inside "skills-list"
            app._update_command_menu("/list")
            assert menu_names(app) == ["skills-list"]

    async def test_skill_narrows_to_skill_commands(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/skill")
            assert set(menu_names(app)) == {
                "skill-creator",
                "skills-list",
                "skill-remove",
            }

    async def test_no_match_hides_the_menu(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/zzz")
            assert not app.command_menu_active()
            assert not app.query_one("#command-menu").display

    async def test_frequency_sorts_most_used_first(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            record_command_use(app._conn, "skill-remove")
            record_command_use(app._conn, "skill-remove")
            record_command_use(app._conn, "skills-list")
            app._update_command_menu("/skill")
            # skill-remove (2) before skills-list (1) before skill-creator (0)
            assert menu_names(app)[:2] == ["skill-remove", "skills-list"]


class TestNavigation:
    async def test_down_and_up_move_the_selection(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app._update_command_menu("/skill")
            assert app._command_index == 0
            await pilot.press("down")
            assert app._command_index == 1
            await pilot.press("down")
            assert app._command_index == 2
            await pilot.press("up")
            assert app._command_index == 1

    async def test_enter_fills_a_partial_command(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "/skill-c"  # matches only skill-creator
            app._update_command_menu(chat_input.text)
            await pilot.press("enter")
            await pilot.pause()
            assert chat_input.text == "/skill-creator "  # filled, not submitted

    async def test_tab_also_fills(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "/concl"
            app._update_command_menu(chat_input.text)
            await pilot.press("tab")
            await pilot.pause()
            assert chat_input.text == "/conclude "

    async def test_enter_on_a_complete_command_runs_it(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "/skills-list"  # fully typed
            app._update_command_menu(chat_input.text)
            await pilot.press("enter")
            await pilot.pause()
            # it ran: the entry cleared and the command was counted
            assert chat_input.text == ""
            assert command_use_counts(app._conn).get("skills-list") == 1
