"""Rewinding a conversation from the chat log (§ chat rewind).

Enter on a message you sent opens a small dialog: fork the session from just
before that message, roll the conversation back to just before it, or copy the
text into the entry. The goal is trimming a conversation once the agent goes
into an unwanted direction — the fork keeps the original whole, the rollback
does not.
"""

import asyncio
import json

import pytest
from textual.widgets import ListView

from hpca.agent.graph import thread_message_count
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.rewind_screen import RewindScreen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


class AnswerThenBlockLLM:
    """First turn answers; the second parks until released — a session with a
    turn genuinely in flight, which the rollback must refuse to cut under."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._calls = 0

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self._calls += 1
        if self._calls == 1:
            return ChatResponse(content=respond_json("a1"))
        self.entered.set()
        await self.release.wait()
        return ChatResponse(content=respond_json("a2"))

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def entry(app) -> ChatInput:
    return app.query_one("#chat-input", ChatInput)


def chat_rows(app) -> int:
    return len(app.query_one("#chat-list", ListView))


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = entry(app)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def two_exchanges(app, pilot):
    """user q1, agent a1, user q2, agent a2 — rows 0..3, messages 0..3."""
    await submit_chat(app, pilot, "q1")
    await submit_chat(app, pilot, "q2")


async def open_rewind_on(app, pilot, row):
    chat_list = app.query_one("#chat-list", ListView)
    chat_list.focus()
    chat_list.index = row
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def choose(app, pilot, key):
    await pilot.press(key)
    await app.workers.wait_for_complete()
    await pilot.pause()


class TestDialog:
    async def test_enter_on_a_sent_message_opens_the_rewind_dialog(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            await open_rewind_on(app, pilot, 2)
            assert isinstance(app.screen, RewindScreen)

    async def test_escape_changes_nothing(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            sid = app.active_session.session_id
            await open_rewind_on(app, pilot, 2)
            await choose(app, pilot, "escape")
            assert not isinstance(app.screen, RewindScreen)
            assert entry(app).text == ""
            assert chat_rows(app) == 4
            assert await thread_message_count(app.graph, session_id=sid) == 4

    async def test_enter_again_copies_like_it_always_did(self, hpca_home):
        """The old reflex — activate a message twice — still lands the text in
        the entry; copy sits on Enter inside the dialog."""
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            await open_rewind_on(app, pilot, 2)
            await choose(app, pilot, "enter")
            assert entry(app).text == "q2"
            assert chat_rows(app) == 4  # nothing trimmed


class TestRollback:
    async def test_rollback_trims_to_before_the_message(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            sid = app.active_session.session_id
            await open_rewind_on(app, pilot, 2)  # the second user message
            await choose(app, pilot, "r")
            assert await thread_message_count(app.graph, session_id=sid) == 2
            assert chat_rows(app) == 2  # q1 and a1
            # the trimmed message is back in the entry, ready to re-edit
            assert entry(app).text == "q2"
            assert app.focused is entry(app)

    async def test_rollback_to_the_first_message_empties_the_thread(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            sid = app.active_session.session_id
            await open_rewind_on(app, pilot, 0)
            await choose(app, pilot, "r")
            assert await thread_message_count(app.graph, session_id=sid) == 0
            assert chat_rows(app) == 0
            assert entry(app).text == "q1"

    async def test_the_next_turn_runs_on_the_trimmed_thread(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [respond_json("a1"), respond_json("a2"), respond_json("a3")]
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            sid = app.active_session.session_id
            await open_rewind_on(app, pilot, 2)
            await choose(app, pilot, "r")
            entry(app).text = "q2 but better"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert await thread_message_count(app.graph, session_id=sid) == 4
            # q1/a1 survived, the old q2/a2 did not
            texts = [e.text for e in app._chat_entries]
            assert "q1" in texts and "a1" in texts
            assert "q2 but better" in texts and "a3" in texts
            assert "q2" not in texts and "a2" not in texts

    async def test_rollback_refused_while_a_turn_is_running(self, hpca_home):
        llm = AnswerThenBlockLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "q1")
            sid = app.active_session.session_id
            chat_input = entry(app)
            chat_input.focus()
            chat_input.text = "q2"
            await pilot.press("enter")
            await asyncio.wait_for(llm.entered.wait(), timeout=5)
            await pilot.pause()
            # q2 is already in the thread; the turn is parked on the model.
            assert await thread_message_count(app.graph, session_id=sid) == 3
            await open_rewind_on(app, pilot, 0)  # q1: a real thread message
            assert isinstance(app.screen, RewindScreen)
            await pilot.press("r")
            await pilot.pause()
            # refused: nothing was cut from under the running turn
            assert await thread_message_count(app.graph, session_id=sid) == 3
            llm.release.set()
            await app.workers.wait_for_complete()


class TestFork:
    async def test_fork_opens_a_copy_trimmed_to_before_the_message(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            source = app.active_session
            await open_rewind_on(app, pilot, 2)
            await choose(app, pilot, "f")
            fork = app.active_session
            assert fork.session_id != source.session_id
            assert fork.title.endswith("(fork)")
            assert fork.profile == source.profile
            assert fork.backend == source.backend
            assert chat_rows(app) == 2  # q1 and a1 only
            assert entry(app).text == "q2"
            assert (
                await thread_message_count(app.graph, session_id=fork.session_id)
                == 2
            )

    async def test_the_source_session_is_untouched(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            source = app.active_session
            await open_rewind_on(app, pilot, 2)
            await choose(app, pilot, "f")
            assert (
                await thread_message_count(app.graph, session_id=source.session_id)
                == 4
            )
            await app.open_session(source)
            await pilot.pause()
            assert chat_rows(app) == 4

    async def test_both_sessions_are_in_the_sidebar(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a1"), respond_json("a2")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await two_exchanges(app, pilot)
            await open_rewind_on(app, pilot, 2)
            await choose(app, pilot, "f")
            assert len(app.session_store.list_all()) == 2
