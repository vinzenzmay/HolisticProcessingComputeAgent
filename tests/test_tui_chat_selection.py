"""Marking and copying text in the chat window.

Textual turns a press and release on one widget into a Click no matter how
far the mouse travelled between them (App.on_event), so marking text inside a
message used to activate the message. Pilot's own drag helpers bypass that
synthesis, so these tests drive App.on_event the way the terminal driver does
— otherwise they would pass without exercising the bug at all.
"""

import json

import pytest
from textual.events import MouseDown, MouseMove, MouseUp
from textual.pilot import _get_mouse_message_arguments
from textual.widgets import ListView

from hpca.clipboard import CopyResult
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, ChatItem, HpcaApp, ThinkingBox


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    def __init__(self, outputs, reasoning=None):
        self._outputs = list(outputs)
        self._reasoning = list(reasoning or [])

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=json.dumps({"title": "a test session"}))
        return ChatResponse(
            content=self._outputs.pop(0),
            reasoning=self._reasoning.pop(0) if self._reasoning else None,
        )

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def drag(app, widget, start, end):
    """Mark text: press, move, release — through the driver's own code path."""
    await app.on_event(MouseDown(**_get_mouse_message_arguments(widget, start, button=1)))
    await app.on_event(MouseMove(**_get_mouse_message_arguments(widget, end, button=1)))
    await app.on_event(MouseUp(**_get_mouse_message_arguments(widget, end, button=1)))


async def plain_click(app, widget, at=(2, 0)):
    await app.on_event(MouseDown(**_get_mouse_message_arguments(widget, at, button=1)))
    await app.on_event(MouseUp(**_get_mouse_message_arguments(widget, at, button=1)))


async def chat_app(pilot, app, message="which BAMs are in the cohort dir?"):
    await app.start_new_session()
    await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = message
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def make_app():
    return HpcaApp(
        llm=FakeLLM(
            [respond_json("Four BAMs match: /data/cohort/s1.bam and s2.bam")],
            reasoning=["I should list the cohort directory first."],
        )
    )


class TestMarkingText:
    async def test_dragging_across_a_message_marks_text(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            agent_message = app.query_one("#chat-list", ListView).children[-1]
            await drag(app, agent_message, (2, 0), (20, 0))
            await pilot.pause()
            assert app.screen.get_selected_text()

    async def test_marking_a_message_does_not_jump_to_the_entry(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            user_message = app.query_one("#chat-list", ListView).children[0]
            app.query_one("#chat-list", ListView).focus()
            await pilot.pause()
            await drag(app, user_message, (2, 0), (18, 0))
            await pilot.pause()
            assert app.focused.id == "chat-list"  # not yanked to the entry

    async def test_marking_the_thinking_box_does_not_collapse_it(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            box = app.query_one(ThinkingBox)
            box.toggle()  # expanded: the reasoning is on screen to be marked
            await pilot.pause()
            await drag(app, box, (2, 0), (18, 0))
            await pilot.pause()
            assert not box.collapsed
            assert app.screen.get_selected_text()

    async def test_a_marked_selection_copies_via_the_clipboard_manager(
        self, hpca_home
    ):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            copied = []

            def fake_copy(text):
                copied.append(text)
                return CopyResult(methods=["test"], message="Copied")

            app.clipboard_manager.copy = fake_copy
            agent_message = app.query_one("#chat-list", ListView).children[-1]
            await drag(app, agent_message, (2, 0), (20, 0))
            await pilot.pause()
            selected = app.screen.get_selected_text()
            await pilot.press("ctrl+c")
            await pilot.pause()
            # OSC 52 alone would be swallowed by tmux/screen over ssh
            assert copied == [selected]


class TestClicking:
    async def test_a_plain_click_on_a_message_still_focuses_the_entry(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            user_message = app.query_one("#chat-list", ListView).children[0]
            await plain_click(app, user_message)
            await pilot.pause()
            assert app.focused.id == "chat-input"

    async def test_a_plain_click_toggles_the_thinking_box_once(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            box = app.query_one(ThinkingBox)
            assert box.collapsed
            await plain_click(app, box)
            await pilot.pause()
            assert not box.collapsed  # exactly one toggle, not two
            await plain_click(app, box)
            await pilot.pause()
            assert box.collapsed

    async def test_chat_entries_are_chat_items(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            await chat_app(pilot, app)
            rows = app.query_one("#chat-list", ListView).children
            assert rows and all(isinstance(row, ChatItem) for row in rows)
