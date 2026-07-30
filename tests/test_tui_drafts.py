"""Per-session drafts: the chat entry belongs to the session on screen.

One entry widget serves every session, so without help the half-written message
in it follows the user into the next session — and would be sent to the wrong
thread. Each session parks its unsent text on the way out and gets it back on
the way in.
"""

import json

import pytest
from textual.widgets import Static

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=json.dumps({"title": "a test session"}))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def entry(app) -> ChatInput:
    return app.query_one("#chat-input", ChatInput)


async def type_draft(app, pilot, text):
    """Type into the entry the way the user does — through the widget, so the
    Changed handling (command menu) runs too."""
    chat_input = entry(app)
    chat_input.focus()
    chat_input.text = text
    chat_input.move_cursor(chat_input.document.end)
    await pilot.pause()


def two_sessions(app):
    return (
        app.session_store.create(profile="default", title="first"),
        app.session_store.create(profile="default", title="second"),
    )


async def test_draft_does_not_follow_the_user_into_the_next_session(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "half a thought for the first session")
        await app.open_session(b)
        await pilot.pause()
        assert entry(app).text == ""


async def test_returning_to_a_session_restores_its_draft(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "half a thought")
        await app.open_session(b)
        await app.open_session(a)
        await pilot.pause()
        chat_input = entry(app)
        assert chat_input.text == "half a thought"
        # Behind what was typed, so typing continues where it left off.
        assert chat_input.cursor_location == (0, len("half a thought"))


async def test_each_session_keeps_its_own_draft(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "for A")
        await app.open_session(b)
        await type_draft(app, pilot, "for B")
        await app.open_session(a)
        await pilot.pause()
        assert entry(app).text == "for A"
        await app.open_session(b)
        await pilot.pause()
        assert entry(app).text == "for B"


async def test_a_multi_line_draft_comes_back_whole(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "line one\nline two")
        await app.open_session(b)
        await app.open_session(a)
        await pilot.pause()
        chat_input = entry(app)
        assert chat_input.text == "line one\nline two"
        assert chat_input.cursor_location == (1, len("line two"))


async def test_a_sent_message_leaves_no_draft_behind(hpca_home):
    app = HpcaApp(llm=FakeLLM([respond_json("hi")]))
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "an actual message")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.open_session(b)
        await app.open_session(a)
        await pilot.pause()
        assert entry(app).text == ""


async def test_a_new_session_starts_with_an_empty_entry(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        session = app.session_store.create(profile="default", title="first")
        await app.open_session(session)
        await type_draft(app, pilot, "not for the new one")
        await app.start_new_session()
        await pilot.pause()
        assert entry(app).text == ""


async def test_closing_and_reopening_a_session_keeps_the_draft(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        session = app.session_store.create(profile="default", title="first")
        await app.open_session(session)
        await type_draft(app, pilot, "back in a moment")
        await app.close_session()
        await pilot.pause()
        assert entry(app).text == ""  # nothing to type into, nothing shown
        await app.open_session(session)
        await pilot.pause()
        assert entry(app).text == "back in a moment"


async def test_deleting_a_session_drops_its_draft(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "goes with the session")
        await app.open_session(b)
        await app._delete_session(a)
        await pilot.pause()
        assert a.session_id not in app._drafts


async def test_the_command_menu_follows_the_restored_draft(hpca_home):
    """A parked "/…" draft brings its autocomplete menu back with it, and a
    session with no draft does not inherit the previous session's menu."""
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = two_sessions(app)
        await app.open_session(a)
        await type_draft(app, pilot, "/comp")
        menu = app.query_one("#command-menu", Static)
        assert menu.display is True

        await app.open_session(b)
        await pilot.pause()
        assert menu.display is False

        await app.open_session(a)
        await pilot.pause()
        assert entry(app).text == "/comp"
        assert menu.display is True
