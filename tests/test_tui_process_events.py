"""Background completions reaching the agent through the TUI (§5.4).

The unit half lives in test_process_events.py; these cover the routing the app
does with a change once the watcher has produced one.
"""

import json

import pytest

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp

TITLE_REPLY = json.dumps({"title": "a test session"})


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    """Records what the agent was asked, so we can assert the event arrived."""

    def __init__(self):
        self.seen: list[list[dict]] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self.seen.append(list(messages))
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "noted"})
        )

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def open_session(app, pilot):
    await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "hello"
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


class TestEventDelivery:
    async def test_failed_background_script_reaches_the_agent(self, hpca_home):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await open_session(app, pilot)
            record = await app._tool_ctx.runner.start(
                ["bash", "-c", "echo 'boom: not found' >&2; exit 7"],
                name="night_job",
                background=True,
            )
            await app._tool_ctx.runner.wait(record.pid)

            await app.watch_processes()
            await app.workers.wait_for_complete()
            await pilot.pause()

            prompts = [
                m["content"]
                for messages in llm.seen
                for m in messages
                if m["role"] == "user"
            ]
            assert any("[process failed]" in p and "night_job" in p for p in prompts)
            assert any("boom: not found" in p for p in prompts)

    async def test_event_shows_in_the_transcript(self, hpca_home):
        """The user must see why the agent suddenly spoke."""
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test() as pilot:
            await open_session(app, pilot)
            record = await app._tool_ctx.runner.start(
                ["bash", "-c", "exit 1"], name="night_job", background=True
            )
            await app._tool_ctx.runner.wait(record.pid)
            await app.watch_processes()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(
                entry.kind == "event" and "night_job" in entry.text
                for entry in app._chat_entries
            )

    async def test_foreground_run_produces_no_event(self, hpca_home):
        """run_bash already hands its failure back as the tool result."""
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await open_session(app, pilot)
            record = await app._tool_ctx.runner.start(
                ["bash", "-c", "exit 1"], name="probe"
            )
            await app._tool_ctx.runner.wait(record.pid)
            before = len(llm.seen)
            await app.watch_processes()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(llm.seen) == before

    async def test_event_waits_while_a_turn_is_in_flight(self, hpca_home):
        """Concurrent ainvoke on one thread_id would interleave checkpoint
        writes, so a completion mid-turn must be held, not delivered."""
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test() as pilot:
            await open_session(app, pilot)
            record = await app._tool_ctx.runner.start(
                ["bash", "-c", "exit 1"], name="night_job", background=True
            )
            await app._tool_ctx.runner.wait(record.pid)

            # pretend this session has a turn running: its event must buffer
            from hpca.tui.app import TurnState

            app._turns[app.active_session.session_id] = TurnState(
                session=app.active_session
            )
            await app.watch_processes()
            await pilot.pause()
            assert len(app._pending_work) == 1  # buffered, not delivered

            app._turns.clear()
            await app.drain_work()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._pending_work == []
