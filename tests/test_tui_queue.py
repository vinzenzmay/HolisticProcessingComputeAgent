"""Typing while a turn runs (§3). Messages queue instead of being refused.

Exactly one turn still runs at a time — two on one thread_id would interleave
checkpoint writes — so this is about never blocking the user's input, not
about running turns in parallel.
"""

import asyncio
import json

import pytest

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp

TITLE_REPLY = json.dumps({"title": "a test session"})


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class SlowLLM:
    """Holds a turn open until released, so 'busy' is a real state."""

    def __init__(self):
        self.gate = asyncio.Event()
        self.seen_user_texts: list[str] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        for message in messages:
            if message["role"] == "user" and not str(
                message["content"]
            ).startswith(("[tool", "[process", "[job")):
                self.seen_user_texts.append(message["content"])
        await self.gate.wait()
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "ok"})
        )

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def send(app, pilot, text):
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


class TestQueueing:
    async def test_a_message_sent_while_busy_is_accepted(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            assert app._busy_turn is not None  # the turn is held open

            await send(app, pilot, "second")
            assert app.queued_texts_for(app.active_session.session_id) == ["second"]

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_the_input_is_cleared_so_typing_continues(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            assert app.query_one("#chat-input", ChatInput).text == ""
            llm.gate.set()
            await app.workers.wait_for_complete()

    async def test_queued_message_is_visible_in_the_transcript(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            assert any(
                entry.kind == "queued" and entry.text == "second"
                for entry in app._chat_entries
            )
            llm.gate.set()
            await app.workers.wait_for_complete()

    async def test_only_one_turn_runs_at_a_time(self, hpca_home):
        """The queue exists because two turns on one thread would interleave
        checkpoint writes."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            await send(app, pilot, "third")
            # one running, two waiting — never three threads in flight
            assert app._busy_turn is not None
            assert len(app._pending_work) == 2
            assert llm.seen_user_texts == ["first"]
            llm.gate.set()
            await app.workers.wait_for_complete()

    async def test_queue_drains_in_order_after_the_turn_ends(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            llm.gate.set()
            for _ in range(40):
                await pilot.pause()
                if not app._pending_work and app._busy_turn is None:
                    break
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._pending_work == []
            assert llm.seen_user_texts[:1] == ["first"]
            assert "second" in llm.seen_user_texts

    async def test_a_queued_message_survives_the_transcript_rebuild(self, hpca_home):
        """The turn's reply rebuilds the chat from graph messages; a message
        typed meanwhile is not in the graph yet and must not vanish."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            await send(app, pilot, "third")
            llm.gate.set()
            await pilot.pause()
            await pilot.pause()
            # while "second" runs, "third" is still queued and still shown
            texts = [e.text for e in app._chat_entries]
            assert "third" in texts
            await app.workers.wait_for_complete()

    async def test_slash_commands_are_refused_not_queued(self, hpca_home):
        """They act on the UI and run their own exclusive workers."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "/conclude")
            assert app._pending_work == []
            llm.gate.set()
            await app.workers.wait_for_complete()


class TestApprovalInteraction:
    async def test_a_blocked_session_does_not_stall_the_queue(self, hpca_home):
        """A session parked on an approval must wait, but a message for a
        different session should still start."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            blocked = app.active_session
            app._awaiting_approval.add(blocked.session_id)
            from hpca.tui.app import PendingWork

            app._pending_work = [
                PendingWork(session_id=blocked.session_id, text="waits", kind="user"),
                PendingWork(session_id="other-session", text="runs", kind="event"),
            ]
            await app.drain_work()
            # the blocked session's message is still queued, the other is gone
            assert [w.text for w in app._pending_work] == ["waits"]
