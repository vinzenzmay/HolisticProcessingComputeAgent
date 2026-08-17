"""Tests for the TUI side of the thinking dial (hpca.thinking).

Three things, all of them per session and none of them global: the ``/thinking``
chooser, the level shown in the top row next to the fill and the speed, and the
level a turn actually puts on the wire.
"""

import json

import pytest
from textual.widgets import ListView

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.context_bar import ContextBar
from hpca.tui.thinking_screen import ThinkingScreen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class FakeLLM:
    """Records the thinking arguments of every non-titling call."""

    def __init__(self, outputs=()):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self.calls.append(kwargs)
        return ChatResponse(
            content=self._outputs.pop(0)
            if self._outputs
            else json.dumps({"action": "respond", "response": "done"})
        )

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def open_chooser(app, pilot):
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "/thinking"
    await pilot.press("enter")
    await pilot.pause()
    return app.screen


async def pick(app, pilot, effort):
    """Move the chooser's cursor onto ``effort`` and select it."""
    from hpca.thinking import EFFORTS

    listing = app.screen.query_one("#thinking-list", ListView)
    listing.index = EFFORTS.index(effort)
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


class TestChooser:
    async def test_the_command_opens_it_with_all_four_levels(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            screen = await open_chooser(app, pilot)
            assert isinstance(screen, ThinkingScreen)
            labels = [
                str(item.query_one("Label").content)
                for item in screen.query_one("#thinking-list", ListView).children
            ]
            assert len(labels) == 4
            assert [line.split()[0] for line in labels] == [
                "off", "low", "medium", "xhigh"
            ]

    async def test_xhigh_is_flagged_unusable_in_the_list(self, hpca_home):
        # It has to read as unusable at the moment of choosing, not only in the
        # toast afterwards — and "NOT USABLE" sits on the name line so it
        # survives a skim of the list.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            screen = await open_chooser(app, pilot)
            labels = [
                str(item.query_one("Label").content)
                for item in screen.query_one("#thinking-list", ListView).children
            ]
            assert "NOT USABLE" in labels[3]
            # on the name line, above the explanation, not buried after it
            assert "NOT USABLE" in labels[3].splitlines()[0]
            assert not any("NOT USABLE" in line for line in labels[:3])

    async def test_the_current_level_is_starred_and_preselected(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await open_chooser(app, pilot)
            await pick(app, pilot, "medium")
            await open_chooser(app, pilot)
            listing = app.screen.query_one("#thinking-list", ListView)
            assert listing.index == 2
            label = str(listing.children[2].query_one("Label").content)
            assert "★" in label

    async def test_escape_leaves_the_level_alone(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await open_chooser(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert app.active_session.thinking == "off"

    async def test_without_a_session_it_says_so_instead(self, hpca_home):
        # The level belongs to a conversation; with none open there is nothing
        # to set it on, and silently setting the app default would be wrong.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app._handle_slash_command("/thinking")
            await pilot.pause()
            assert len(app.screen_stack) == 1


class TestLifecycle:
    async def test_a_choice_is_stored_on_the_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await open_chooser(app, pilot)
            await pick(app, pilot, "xhigh")
            assert app.active_session.thinking == "xhigh"
            stored = app.session_store.get(app.active_session.session_id)
            assert stored.thinking == "xhigh"

    async def test_new_sessions_start_at_the_configured_default(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        app.settings.agent.default_thinking = "low"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert app.active_session.thinking == "low"
            assert app._thinking_for_turn(app.active_session.session_id) == "low"

    async def test_two_sessions_keep_their_own_levels(self, hpca_home):
        # The point of moving the dial off the backend: two conversations on
        # one model must be able to disagree about how hard to think.
        app = HpcaApp(llm=FakeLLM())
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            first = app.active_session
            await open_chooser(app, pilot)
            await pick(app, pilot, "medium")
            # An untouched session is reused rather than replaced, so the first
            # one has to have been used before a second exists at all.
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "hello"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await app.start_new_session()
            await pilot.pause()
            second = app.active_session
            assert second.session_id != first.session_id
            assert app._thinking_for_turn(second.session_id) == "off"
            assert app._thinking_for_turn(first.session_id) == "medium"

    async def test_an_empty_stored_level_follows_the_setting(self, hpca_home):
        # "" means "whatever the default is", so changing the setting reaches
        # sessions that were never explicitly switched — as it does for mode.
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.active_session.thinking = ""
            app.session_store.set_thinking(app.active_session.session_id, "")
            app.settings.agent.default_thinking = "xhigh"
            assert app._thinking_for_turn(app.active_session.session_id) == "xhigh"


class TestContextBarShowsIt:
    async def test_shown_for_the_open_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert "think off" in app.query_one("#context-bar", ContextBar).text

    async def test_it_follows_a_choice_immediately(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await open_chooser(app, pilot)
            await pick(app, pilot, "low")
            assert "think low" in app.query_one("#context-bar", ContextBar).text

    async def test_hidden_with_no_session_open(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await app.close_session()
            await pilot.pause()
            assert "think" not in app.query_one("#context-bar", ContextBar).text


class TestTheTurnCarriesIt:
    async def test_off_puts_no_level_on_the_wire(self, hpca_home):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "hello"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert llm.calls
            assert llm.calls[0]["enable_thinking"] is False
            assert "reasoning_effort" not in llm.calls[0]

    async def test_a_chosen_level_reaches_the_next_decision(self, hpca_home):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await open_chooser(app, pilot)
            await pick(app, pilot, "xhigh")
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "hello"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert llm.calls[0]["enable_thinking"] is True
            assert llm.calls[0]["reasoning_effort"] == "xhigh"


class TestTheCommandIsDiscoverable:
    def test_it_is_offered_in_the_slash_menu(self):
        from hpca.tui.app import COMMANDS

        assert "thinking" in {name for name, _ in COMMANDS}
        usage = next(text for name, text in COMMANDS if name == "thinking")
        assert "/thinking" in usage
        # The four levels are named in the help line, so the menu answers
        # "what will this ask me?" without opening the chooser.
        for level in ("off", "low", "medium", "xhigh"):
            assert level in usage

    async def test_typing_slash_lists_it(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "/think"
            await pilot.pause()
            assert "thinking" in {name for name, _ in app._command_matches}
