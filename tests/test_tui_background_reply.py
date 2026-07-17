"""A reply landing in a session the user has left must not force it open.

The turn is the session's, not the screen's: its chat update only applies if
that session is still open; otherwise its row gets a colored frame in the
sessions list. Everything the turn touches — transcript log, tool context,
resume thread — is captured when the turn starts, not read from whichever
session happens to be open when the reply lands.
"""

import asyncio
import json

import pytest
from textual.widgets import ListView

from hpca.agent.tools import Tool, ToolRegistry
from hpca.config import Settings
from hpca.llm import ChatResponse
from hpca.logs import log_path
from hpca.tui.app import ChatInput, HpcaApp, WorkingIndicator
from pydantic import BaseModel, Field


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class GatedLLM:
    """Holds every decision until released, so the user can switch away."""

    def __init__(self, outputs, titles=("a test session",)):
        self._outputs = list(outputs)
        self._titles = list(titles)
        self.released = asyncio.Event()

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            title = self._titles.pop(0) if len(self._titles) > 1 else self._titles[0]
            return ChatResponse(content=json.dumps({"title": title}))
        await self.released.wait()
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def submit(app, pilot, text):
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


async def settle(app, pilot):
    await app.workers.wait_for_complete()
    for _ in range(3):
        await pilot.pause()


def row_of(app, session):
    for item in app.query_one("#sessions-list", ListView).children:
        row_session = getattr(item, "data_session", None)
        if row_session is not None and row_session.session_id == session.session_id:
            return item
    raise AssertionError("session row not found")


class TestBackgroundReply:
    async def test_reply_stays_put_and_frames_the_row(self, hpca_home):
        llm = GatedLLM([respond_json("the answer for A")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "question in A")

            await app.start_new_session()  # user moves on to B while A waits
            await pilot.pause()
            session_b = app.active_session
            assert session_b.session_id != session_a.session_id

            llm.released.set()
            await settle(app, pilot)

            # still in B, B's chat untouched — no force-open
            assert app.active_session.session_id == session_b.session_id
            assert not any(
                "the answer for A" in t for t in app.chat_log_texts()
            )
            assert row_of(app, session_a).has_class("session-updated")

    async def test_opening_the_framed_session_shows_the_reply_and_clears_it(
        self, hpca_home
    ):
        llm = GatedLLM([respond_json("the answer for A")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "question in A")
            await app.start_new_session()
            await pilot.pause()
            llm.released.set()
            await settle(app, pilot)

            await app.open_session(session_a)
            await pilot.pause()
            assert any("the answer for A" in t for t in app.chat_log_texts())
            assert not row_of(app, session_a).has_class("session-updated")

    async def test_the_turn_logs_into_its_own_session_transcript(
        self, hpca_home, tmp_path
    ):
        directory = tmp_path / "chatlogs"
        settings = Settings()
        settings.logging.dir = str(directory)
        settings.save()
        llm = GatedLLM([respond_json("the answer for A")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "question in A")
            await app.start_new_session()
            await pilot.pause()
            session_b = app.active_session
            llm.released.set()
            await settle(app, pilot)

            log_a = log_path(directory, session_a).read_text()
            assert "the answer for A" in log_a
            log_b = log_path(directory, session_b).read_text()
            assert "the answer for A" not in log_b

    async def test_the_backgrounded_session_still_gets_its_title(self, hpca_home):
        llm = GatedLLM([respond_json("a")], titles=["Cohort BAM inventory"])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "which BAMs?")
            await app.start_new_session()
            await pilot.pause()
            llm.released.set()
            await settle(app, pilot)
            assert app.session_store.get(session_a.session_id).title == (
                "Cohort BAM inventory"
            )

    async def test_a_tool_runs_with_its_own_sessions_context(self, hpca_home):
        """Switching away must not hand the running turn another session's
        registry, runner and session id."""

        class Probe(BaseModel):
            what: str = Field(description="anything")

        seen = {}
        release = asyncio.Event()

        async def probe_handler(args, ctx):
            await release.wait()
            seen["session_id"] = ctx.session_id
            return "probed"

        registry = ToolRegistry()
        registry.register(
            Tool(name="probe", description="Probe", params=Probe, handler=probe_handler)
        )
        llm = GatedLLM([tool_json("probe", what="x"), respond_json("done")])
        app = HpcaApp(llm=llm, tools=registry)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "probe it")
            llm.released.set()  # decision made; tool now blocks on `release`
            await pilot.pause()

            await app.start_new_session()  # switch while the tool is running
            await pilot.pause()
            release.set()
            await settle(app, pilot)
            assert seen["session_id"] == session_a.session_id


class TestBusyGuard:
    async def test_sending_elsewhere_while_busy_is_refused_and_keeps_the_draft(
        self, hpca_home
    ):
        llm = GatedLLM([respond_json("the answer for A")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await submit(app, pilot, "question in A")
            await app.start_new_session()
            await pilot.pause()

            await submit(app, pilot, "question in B")
            chat_input = app.query_one("#chat-input", ChatInput)
            assert chat_input.text == "question in B"  # kept, not swallowed
            llm.released.set()
            await settle(app, pilot)
            # the refused message never became a turn
            assert not any("question in B" in t for t in app.chat_log_texts())

    async def test_the_first_turn_survives_the_refused_second(self, hpca_home):
        llm = GatedLLM([respond_json("the answer for A")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "question in A")
            await app.start_new_session()
            await pilot.pause()
            await submit(app, pilot, "question in B")  # refused, must not cancel A
            llm.released.set()
            await settle(app, pilot)
            await app.open_session(session_a)
            await pilot.pause()
            assert any("the answer for A" in t for t in app.chat_log_texts())

    async def test_spinner_reappears_when_returning_to_the_busy_session(
        self, hpca_home
    ):
        llm = GatedLLM([respond_json("a")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_a = app.active_session
            await submit(app, pilot, "question in A")
            await app.start_new_session()
            await pilot.pause()
            assert not list(app.query(WorkingIndicator))  # B is idle

            await app.open_session(session_a)
            await pilot.pause()
            assert list(app.query(WorkingIndicator))  # A is still waiting
            llm.released.set()
            await settle(app, pilot)
            assert not list(app.query(WorkingIndicator))
