"""Reusing one of your own past messages.

Enter on a message you sent opens the rewind dialog (see test_tui_rewind);
copy sits on Enter there, so the old reflex — activate the message, hit Enter
again — still puts its text back in the entry, to send again or edit into the
next one. A queued message opens its own dialog (see test_tui_queue), with
copy on Enter for the same reason. Messages that are not yours (the agent's replies,
background events, recalled memory) are not text you would re-send, so Enter
on those does what it always did: hand focus to the entry.
"""

import json

import pytest
from textual.widgets import ListView

from hpca.llm import ChatResponse
from hpca.transcript import Entry
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.rewind_screen import QueuedScreen, RewindScreen


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


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = entry(app)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def press_enter_on(app, pilot, index):
    """Highlight one row of the message log and activate it."""
    chat_list = app.query_one("#chat-list", ListView)
    chat_list.focus()
    chat_list.index = index
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    if isinstance(app.screen, (RewindScreen, QueuedScreen)):
        # Your own messages offer their dialog first; copy is on Enter in both.
        await pilot.press("enter")
        await pilot.pause()


async def test_enter_on_your_own_message_puts_it_back_in_the_entry(hpca_home):
    app = HpcaApp(llm=FakeLLM([respond_json("on it")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "sbatch the pipeline")
        await press_enter_on(app, pilot, 0)  # the message the user sent
        chat_input = entry(app)
        assert chat_input.text == "sbatch the pipeline"
        assert app.focused is chat_input
        # Behind the text, ready to edit the one word that needs changing.
        assert chat_input.cursor_location == (0, len("sbatch the pipeline"))


async def test_it_appends_on_its_own_line_to_what_is_already_typed(hpca_home):
    app = HpcaApp(llm=FakeLLM([respond_json("on it")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "sbatch the pipeline")
        chat_input = entry(app)
        chat_input.text = "same thing but with 8 cores:"
        await press_enter_on(app, pilot, 0)
        assert chat_input.text == "same thing but with 8 cores:\nsbatch the pipeline"
        assert chat_input.cursor_location == (1, len("sbatch the pipeline"))


async def test_a_draft_left_open_for_it_continues_on_the_same_line(hpca_home):
    """Ending the draft in a space is how you say "keep going here"."""
    app = HpcaApp(llm=FakeLLM([respond_json("on it")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "sbatch the pipeline")
        chat_input = entry(app)
        chat_input.text = "again: "
        await press_enter_on(app, pilot, 0)
        assert chat_input.text == "again: sbatch the pipeline"


async def test_the_agents_reply_is_not_copied(hpca_home):
    app = HpcaApp(llm=FakeLLM([respond_json("running it now")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "sbatch the pipeline")
        chat_list = app.query_one("#chat-list", ListView)
        await press_enter_on(app, pilot, len(chat_list) - 1)  # the agent's reply
        chat_input = entry(app)
        assert chat_input.text == ""  # nothing pasted
        assert app.focused is chat_input  # Enter still hands over focus


async def test_a_queued_message_is_yours_too(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.start_new_session()
        # What typing ahead of a running turn leaves in the log.
        app._add_chat_entry(Entry(kind="queued", text="and then squeue"))
        await pilot.pause()
        await press_enter_on(app, pilot, 0)
        assert entry(app).text == "and then squeue"


async def test_a_background_event_is_not_yours(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.start_new_session()
        app._add_chat_entry(Entry(kind="event", text="job 27744534 finished"))
        await pilot.pause()
        await press_enter_on(app, pilot, 0)
        assert entry(app).text == ""


async def test_reusing_twice_stacks_both_messages(hpca_home):
    app = HpcaApp(llm=FakeLLM([respond_json("ok"), respond_json("ok")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "first message")
        await submit_chat(app, pilot, "second message")
        await press_enter_on(app, pilot, 0)
        await press_enter_on(app, pilot, 2)  # user, agent, user, agent
        assert entry(app).text == "first message\nsecond message"


async def test_a_reused_message_belongs_to_the_session_it_was_taken_from(hpca_home):
    """It lands in the entry, so it is a draft like any other: parked on the
    way out and waiting on the way back (see test_tui_drafts)."""
    app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "sbatch the pipeline")
        first = app.active_session
        await press_enter_on(app, pilot, 0)
        elsewhere = app.session_store.create(profile="default", title="second")
        await app.open_session(elsewhere)
        await pilot.pause()
        assert entry(app).text == ""
        await app.open_session(first)
        await pilot.pause()
        assert entry(app).text == "sbatch the pipeline"
