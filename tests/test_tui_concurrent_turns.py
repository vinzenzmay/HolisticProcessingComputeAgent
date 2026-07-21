"""Concurrent per-session turns (stage 1).

Two sessions on two different backends can run their turns at the same time:
the orchestrator no longer serialises globally. Same-session turns still
serialise (the real checkpoint-write hazard), and no turn ever reads another
session's client.

Harness: a gated ``SlowLLM`` (holds a turn open until released) driven via
Textual's ``Pilot`` — the pattern from ``test_tui_queue.py`` and
``test_tui_session_llm.py``.
"""

import asyncio
import json

import pytest

from hpca.config import LLMBackend
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class SlowLLM:
    """Holds a turn open until released, recording the user text it sees so a
    cross-contamination leak (session B's turn hitting session A's client) is
    detectable."""

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


def backend_blob(model, url):
    return LLMBackend(model=model, base_url=url).model_dump_json()


class TestConcurrentTurns:
    async def test_two_sessions_turns_overlap(self, hpca_home):
        """A turn held open in session A does not block a turn starting in B."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session_a = app.active_session
            await send(app, pilot, "in A")
            assert session_a.session_id in app._turns  # A is running, held open

            await app.start_new_session()  # switch to a fresh session B
            session_b = app.active_session
            assert session_b.session_id != session_a.session_id
            await send(app, pilot, "in B")

            # B starts immediately — not queued behind A — and both turns live
            assert app.queued_texts_for(session_b.session_id) == []
            assert session_b.session_id in app._turns
            assert session_a.session_id in app._turns

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_no_cross_contamination_between_backends(self, hpca_home):
        """A and B on different backends: each turn resolves its own client, so
        each SlowLLM only ever sees its own session's user text."""
        boot = SlowLLM()  # bootstrap: only serves title requests here
        slow_a = SlowLLM()
        slow_b = SlowLLM()
        app = HpcaApp(llm=boot)

        def client_for(session):
            if session is not None and "qwen-a" in (session.backend or ""):
                return slow_a
            if session is not None and "qwen-b" in (session.backend or ""):
                return slow_b
            return boot

        async with app.run_test(size=(120, 40)) as pilot:
            app._client_for = client_for  # route each session to its own fake
            await app.start_new_session(backend=backend_blob("qwen-a", "http://a/v1"))
            session_a = app.active_session
            await send(app, pilot, "question A")
            assert session_a.session_id in app._turns

            await app.start_new_session(backend=backend_blob("qwen-b", "http://b/v1"))
            session_b = app.active_session
            await send(app, pilot, "question B")
            assert session_b.session_id in app._turns

            slow_a.gate.set()
            slow_b.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Each client saw only its own session's message — no leak either way
            assert slow_a.seen_user_texts == ["question A"]
            assert slow_b.seen_user_texts == ["question B"]

    async def test_same_session_still_serializes(self, hpca_home):
        """A second message in a session whose turn is running is queued for
        that session and does not start a second turn on the same thread."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session_a = app.active_session
            await send(app, pilot, "first")
            assert session_a.session_id in app._turns

            await send(app, pilot, "second")
            # queued for A, not started; still exactly one turn for A
            assert app.queued_texts_for(session_a.session_id) == ["second"]
            assert list(app._turns) == [session_a.session_id]

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
