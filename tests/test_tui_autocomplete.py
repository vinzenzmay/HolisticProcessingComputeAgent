"""Slash-command autocomplete: substring match, frequency sort, ↑/↓ select."""

import pytest

from hpca.db import command_use_counts, connect, init_db, record_command_use
from hpca.skills import load_builtin_skills
from textual.widgets import Static

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
            # Every built-in command, plus the skills HPCA ships — those are
            # invocable as "/<skill>" too, so the menu offers them.
            assert set(menu_names(app)) == {name for name, _ in COMMANDS} | {
                s.name for s in load_builtin_skills()
            }

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

    async def test_the_menu_closes_once_the_command_is_chosen(self, hpca_home):
        """A space after the name means the command is settled and what
        follows is its arguments. The menu has nothing left to offer, and
        keeping it open would hold on to ↑/↓ while the user writes the body."""
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/conclude")
            assert app.command_menu_active()
            for draft in (
                "/conclude ",  # the space alone settles it
                "/conclude what I learned today",
                "/conclude first line\nsecond line",  # shift+enter body
            ):
                app._update_command_menu(draft)
                assert not app.command_menu_active(), draft
                assert not app.query_one("#command-menu").display

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


class TestMistyped:
    """A "/" word naming neither a command nor a skill is nearly always a
    typo, so the draft stays put and can be corrected in place."""

    async def test_unknown_command_keeps_the_draft(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "/skils-list"  # mistyped /skills-list
            app._update_command_menu(chat_input.text)
            await pilot.press("enter")
            await pilot.pause()
            assert chat_input.text == "/skils-list"
            # …and it was not sent to the model as an ordinary message either
            assert not app._pending_work

    async def test_correcting_it_then_runs(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "/skils-list"
            app._update_command_menu(chat_input.text)
            await pilot.press("enter")
            await pilot.pause()
            chat_input.text = "/skills-list"  # the retained draft, fixed up
            app._update_command_menu(chat_input.text)
            await pilot.press("enter")
            await pilot.pause()
            assert chat_input.text == ""
            assert command_use_counts(app._conn).get("skills-list") == 1


class TestBuiltinHighlight:
    """The menu mixes HPCA's own commands with the profile's skills, which
    look identical otherwise; a built-in's name is bold, skills are left
    alone, and the border title says what the bold means."""

    def styled_usages(self, app):
        """{usage text: style} read back off the rendered Content — the spans
        are what actually reaches the screen."""
        content = app.query_one("#command-menu", Static).render()
        return {
            content.plain[span.start : span.end]: span.style
            for span in content.spans
        }

    async def test_only_the_builtin_name_is_bold(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/")
            styled = self.styled_usages(app)
            builtin = {name for name, _ in COMMANDS}
            for name, usage in app._command_matches:
                if name in builtin:
                    # the name carries the mark, the description does not
                    assert styled[f"/{name}"] == "bold", name
                    assert usage not in styled, name
                else:  # a skill needs no mark of its own
                    assert usage not in styled, name

    async def test_skills_carry_no_styling_at_all(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/")
            builtin = {name for name, _ in COMMANDS}
            # every span belongs to a built-in name; nothing else is touched
            assert set(self.styled_usages(app)) == {
                f"/{name}" for name, _ in app._command_matches if name in builtin
            }

    async def test_the_description_is_left_unstyled(self, hpca_home):
        # "/memorize <note> — form memories…": only "/memorize" is bold, so
        # the argument hint and the prose read as ordinary text.
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/memorize")
            bolded = [
                text for text, style in self.styled_usages(app).items()
                if style == "bold"
            ]
            assert bolded == ["/memorize"]
            assert "<note>" in app.query_one("#command-menu", Static).render().plain

    async def test_the_styles_survive_being_painted(self, hpca_home):
        # A span style is parsed at paint time, and a theme variable there
        # (`bold $text`) raises UnresolvedVariableError — which inspecting the
        # spans alone would never catch. Parse them the way rendering does.
        from textual.style import Style

        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/")
            for style in set(self.styled_usages(app).values()):
                Style.parse(style)  # raises if it could never be painted

    async def test_the_title_says_what_bold_means(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._update_command_menu("/")
            title = str(app.query_one("#command-menu", Static).border_title)
            assert "built-in" in title

    async def test_skill_descriptions_are_not_parsed_as_markup(self, hpca_home):
        # A skill description is user-written; markup would eat the brackets.
        app = HpcaApp()
        async with app.run_test(size=(120, 40)):
            app._command_matches = [("bracketed", "/bracketed — keep [these] intact")]
            app._command_index = 0
            app._render_command_menu()
            rendered = app.query_one("#command-menu", Static).render().plain
            assert "[these]" in rendered
