"""Vertical navigation in the chat entry.

↑/↓ have to do double duty: step the text cursor through a multi-line draft,
but hand focus to the message log once the cursor is on the *top* row. The
catch is soft wrap — a long draft with no explicit newlines is a single logical
line spread over several visual rows, so the handoff must key off the visual
row, not the logical one, or ↑ escapes the moment the draft wraps.
"""

import json

import pytest
from textual.widgets import ListView

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    def __init__(self, outputs):
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


def make_app():
    return HpcaApp(llm=FakeLLM([respond_json("ok")]))


async def seed_chat(pilot, app):
    """A session with one exchange, so the message log has rows to browse."""
    await app.start_new_session()
    await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "hello"
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    return chat_input


# A single logical line long enough to soft-wrap onto several visual rows in
# the narrow middle column.
WRAPPING_DRAFT = "word " * 60


class TestVerticalNavigation:
    async def test_up_from_a_lower_wrapped_row_stays_in_the_draft(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            chat_input = await seed_chat(pilot, app)
            chat_input.focus()
            chat_input.text = WRAPPING_DRAFT
            await pilot.pause()
            # Draft really wraps and the cursor (end of text) is below the top row.
            assert chat_input.wrapped_document.height > 1
            chat_input.move_cursor(chat_input.document.end)
            assert not chat_input.navigator.is_first_wrapped_line(chat_input.selection.end)

            before = chat_input.cursor_location
            await pilot.press("up")
            await pilot.pause()

            assert app.focused is chat_input  # did not escape to the log
            assert chat_input.cursor_location != before  # moved up a visual row

    async def test_up_from_the_top_row_browses_the_log(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            chat_input = await seed_chat(pilot, app)
            chat_input.focus()
            chat_input.text = WRAPPING_DRAFT
            chat_input.move_cursor((0, 0))  # first visual row
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()

            assert app.focused is app.query_one("#chat-list", ListView)
            assert chat_input.text == WRAPPING_DRAFT  # draft kept

    async def test_arrows_traverse_the_body_of_a_slash_command(self, hpca_home):
        """The command menu takes ↑/↓ to move its selection. It must give them
        back the moment the command is chosen — otherwise every draft that
        opens with "/" (or "\\") is untraversable for as long as it is being
        written, which is exactly when the user needs to move around in it."""
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            chat_input = await seed_chat(pilot, app)
            chat_input.focus()
            chat_input.text = "/plan write me a\nmulti-line request"
            await pilot.pause()
            assert not app.command_menu_active()  # past the name, into the body

            chat_input.move_cursor(chat_input.document.end)
            before = chat_input.cursor_location
            await pilot.press("up")
            await pilot.pause()
            assert app.focused is chat_input
            assert chat_input.cursor_location != before

            await pilot.press("down")
            await pilot.pause()
            assert chat_input.cursor_location == before

    async def test_arrows_still_pick_a_command_while_the_name_is_typed(
        self, hpca_home
    ):
        # The other half of the same rule: while the name is still being typed
        # the menu owns ↑/↓, and the draft must not move under them.
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            chat_input = await seed_chat(pilot, app)
            chat_input.focus()
            chat_input.text = "/skill"
            await pilot.pause()
            assert app.command_menu_active()
            chat_input.move_cursor(chat_input.document.end)
            before = chat_input.cursor_location

            await pilot.press("down")
            await pilot.pause()
            assert app._command_index == 1  # the menu moved
            assert chat_input.cursor_location == before  # the cursor did not

    async def test_down_then_up_round_trips_within_the_draft(self, hpca_home):
        app = make_app()
        async with app.run_test(size=(110, 30)) as pilot:
            chat_input = await seed_chat(pilot, app)
            chat_input.focus()
            chat_input.text = WRAPPING_DRAFT
            chat_input.move_cursor((0, 0))
            await pilot.pause()

            await pilot.press("down")  # into the wrapped body, not the log
            await pilot.pause()
            assert app.focused is chat_input
            moved = chat_input.cursor_location
            assert moved != (0, 0)

            await pilot.press("up")  # back toward the top, still in the draft
            await pilot.pause()
            assert app.focused is chat_input
