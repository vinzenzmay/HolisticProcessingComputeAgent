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

from textual.widgets import ListView

from hpca.config import LLMBackend
from hpca.llm import ChatResponse
from hpca.tui.app import WORKING_MARK, ChatInput, HpcaApp


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class SlowLLM:
    """Holds a turn open until released, recording the user text it sees so a
    cross-contamination leak (session B's turn hitting session A's client) is
    detectable.

    ``prompt_tokens`` is the usage the meter reads: give each backend a
    distinct number so a leak onto the wrong session's meter is visible.
    """

    def __init__(self, prompt_tokens: int | None = None):
        self.gate = asyncio.Event()
        self.seen_user_texts: list[str] = []
        self._prompt_tokens = prompt_tokens

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        for message in messages:
            if message["role"] == "user" and not str(
                message["content"]
            ).startswith(("[tool", "[process", "[job")):
                self.seen_user_texts.append(message["content"])
        await self.gate.wait()
        usage = (
            {"prompt_tokens": self._prompt_tokens}
            if self._prompt_tokens is not None
            else {}
        )
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "ok"}),
            usage=usage,
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


async def _wait_turn_done(app, pilot, session_id, tries=100):
    """Pump the UI until ``session_id``'s background turn has cleared."""
    for _ in range(tries):
        if session_id not in app._turns:
            return
        await pilot.pause()
    raise AssertionError(f"turn for {session_id} did not finish")


def _row_for(app, session_id):
    """The sidebar ``ListItem`` whose session matches ``session_id``."""
    for item in app.query_one("#sessions-list", ListView).children:
        row_session = getattr(item, "data_session", None)
        if row_session is not None and row_session.session_id == session_id:
            return item
    raise AssertionError(f"no sidebar row for {session_id}")


class TestSidebarWorkingIndicator:
    """Decision 6: any session with a live ``TurnState`` shows an in-flight
    marker on its sidebar row (glyph + ``session-working`` class), so the user
    can see an OFF-SCREEN session is still working. The marker clears when the
    turn ends."""

    async def test_live_turn_lights_its_row(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session_a = app.active_session
            await send(app, pilot, "in A")
            assert session_a.session_id in app._turns

            # Both the glyph (via _session_row_text) and the CSS class light up.
            assert WORKING_MARK in app._session_row_text(session_a).plain
            assert _row_for(app, session_a.session_id).has_class("session-working")

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_marker_clears_when_turn_completes(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session_a = app.active_session
            await send(app, pilot, "in A")
            assert _row_for(app, session_a.session_id).has_class("session-working")

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert session_a.session_id not in app._turns
            assert WORKING_MARK not in app._session_row_text(session_a).plain
            assert not _row_for(app, session_a.session_id).has_class(
                "session-working"
            )

    async def test_marker_shows_for_off_screen_session(self, hpca_home):
        """Start A's turn, switch to B, and A's row still shows working while
        A is held open in the background."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session_a = app.active_session
            await send(app, pilot, "in A")
            assert session_a.session_id in app._turns

            await app.start_new_session()  # switch to a fresh session B
            session_b = app.active_session
            assert session_b.session_id != session_a.session_id
            await pilot.pause()

            # A is off screen but its row still carries the working marker.
            assert session_a.session_id in app._turns
            assert _row_for(app, session_a.session_id).has_class("session-working")
            assert WORKING_MARK in app._session_row_text(session_a).plain
            # B, idle, does not.
            assert not _row_for(app, session_b.session_id).has_class(
                "session-working"
            )

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()


class TestPerSessionContextMeter:
    """Decision 9: the meter always reflects the session on screen — a live
    foreground turn ticks it, a background turn silently updates its OWN stored
    number, and switching to that session later shows its current value (not a
    stale zero, not another session's count)."""

    def _route(self, boot, slow_a, slow_b):
        def client_for(session):
            if session is not None and "qwen-a" in (session.backend or ""):
                return slow_a
            if session is not None and "qwen-b" in (session.backend or ""):
                return slow_b
            return boot

        return client_for

    async def test_background_usage_updates_its_own_stored_number(self, hpca_home):
        """A finishes off-screen while B is visible: A's count is stored, and
        the visible (B) meter never shows A's number."""
        boot = SlowLLM()
        slow_a = SlowLLM(prompt_tokens=5000)
        slow_b = SlowLLM(prompt_tokens=9000)
        app = HpcaApp(llm=boot)
        async with app.run_test(size=(120, 40)) as pilot:
            app._client_for = self._route(boot, slow_a, slow_b)
            await app.start_new_session(backend=backend_blob("qwen-a", "http://a/v1"))
            session_a = app.active_session
            await send(app, pilot, "in A")

            await app.start_new_session(backend=backend_blob("qwen-b", "http://b/v1"))
            session_b = app.active_session
            await send(app, pilot, "in B")

            # Let A finish in the background; B stays held open and on screen.
            slow_a.gate.set()
            await _wait_turn_done(app, pilot, session_a.session_id)

            # A's usage landed on A's stored number, not B's, and not the meter.
            assert app._context_used[session_a.session_id] == 5000
            assert app._context_used.get(session_b.session_id) is None
            assert "5,000" not in app._context_bar().text

            slow_b.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_switching_shows_the_sessions_measured_number(self, hpca_home):
        """After A ran off-screen, switching back to A shows A's measured count
        — not zero, and not B's number."""
        boot = SlowLLM()
        slow_a = SlowLLM(prompt_tokens=5000)
        slow_b = SlowLLM(prompt_tokens=9000)
        app = HpcaApp(llm=boot)
        async with app.run_test(size=(120, 40)) as pilot:
            app._client_for = self._route(boot, slow_a, slow_b)
            await app.start_new_session(backend=backend_blob("qwen-a", "http://a/v1"))
            session_a = app.active_session
            await send(app, pilot, "in A")

            await app.start_new_session(backend=backend_blob("qwen-b", "http://b/v1"))
            session_b = app.active_session
            await send(app, pilot, "in B")

            slow_a.gate.set()
            slow_b.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

            # On screen is B: its meter reads B's number.
            assert app.active_session.session_id == session_b.session_id
            assert "9,000" in app._context_bar().text

            # Switch back to A: the meter now reads A's measured number.
            await app.open_session(session_a)
            await pilot.pause()
            text = app._context_bar().text
            assert "5,000" in text
            assert "9,000" not in text

    async def test_visible_turn_ticks_the_meter_live(self, hpca_home):
        """The on-screen session's own turn drives the meter to its number."""
        llm = SlowLLM(prompt_tokens=7000)
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session_a = app.active_session
            await send(app, pilot, "hello")
            assert session_a.session_id in app._turns

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert app._context_used[session_a.session_id] == 7000
            assert "7,000" in app._context_bar().text
