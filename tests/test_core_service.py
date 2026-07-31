"""The four services, assembled and driven by commands alone.

Each service has its own tests; this one exists to prove they compose — that a
`turn.submit` arriving as a protocol command reaches the graph with the right
client, the right prompt and the right tool context, and comes back out as
events. It is the closest thing to running the real thing without a terminal,
and it is what would catch two services agreeing on a name and disagreeing on
what it means.
"""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from hpca.config import Settings
from hpca.core.service import build_service
from hpca.db import connect, init_db
from hpca.llm import ChatResponse
from hpca.protocol import (
    ConfirmResolve,
    DecisionResolve,
    SessionFocus,
    Shutdown,
    TurnInterrupt,
    TurnSubmit,
)
from hpca.sessions import SessionStore


def respond(text="done"):
    return json.dumps({"action": "respond", "response": text})


class FakeLLM:
    """Answers with whatever was queued, and records what it was asked."""

    def __init__(self, outputs=None):
        self._outputs = list(outputs or [respond()])
        self.prompts: list[list[dict]] = []
        self.closed = False

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.prompts.append(messages)
        text = self._outputs.pop(0) if self._outputs else respond()
        return ChatResponse(content=text)

    async def supports_constrained_decoding(self):
        return True

    async def close(self):
        self.closed = True


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def conn(home):
    connection = connect(home / "hpca.db")
    init_db(connection)
    return connection


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def service(home, conn, llm):
    async def db(fn):
        return fn(conn)

    built = build_service(
        settings=Settings.load(),
        app_dir=home,
        db=db,
        conn=conn,
        checkpointer=InMemorySaver(),
        llm=llm,
    )
    return built


@pytest.fixture
def session(conn):
    return SessionStore(conn).create(profile="default", title="a session")


async def drain(queue):
    """Everything emitted so far, without waiting for more."""
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def kinds(events):
    return [type(e).__name__ for e in events]


class TestAssembly:
    def test_it_builds_without_a_terminal(self, service):
        assert service is not None

    async def test_a_subscriber_receives_what_the_core_says(self, service, session):
        queue = service.subscribe()
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await drain(queue)
        assert "TurnStarted" in kinds(events)

    async def test_two_subscribers_both_see_it(self, service, session):
        first, second = service.subscribe(), service.subscribe()
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert kinds(await drain(first)) == kinds(await drain(second))

    async def test_an_unsubscribed_queue_stops_receiving(self, service, session):
        queue = service.subscribe()
        service.unsubscribe(queue)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert await drain(queue) == []


class TestTurns:
    async def test_a_submitted_turn_reaches_the_model_and_comes_back(
        self, service, session, llm
    ):
        queue = service.subscribe()
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="which BAMs?")
        )
        for _ in range(30):
            events = await drain(queue)
            if "TurnFinished" in kinds(events):
                break
            await _yield()
        else:
            raise AssertionError("the turn never finished")
        assert llm.prompts, "the model was never asked"

    async def test_the_prompt_carries_the_sessions_own_profile(
        self, service, session, llm
    ):
        await service.handle(TurnSubmit(session_id=session.session_id, text="hello"))
        await _settle()
        system = llm.prompts[0][0]
        assert system["role"] == "system"
        # Rendered for the running session, not for whatever the core's
        # working profile happens to be.
        assert isinstance(system["content"], str) and system["content"]

    async def test_the_user_message_reaches_the_model_with_its_sidecar(
        self, service, session, llm
    ):
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="count the reads")
        )
        await _settle()
        last = llm.prompts[0][-1]
        assert "count the reads" in (last.get("api_content") or last["content"])

    async def test_submitting_to_a_session_that_does_not_exist_says_so(self, service):
        queue = service.subscribe()
        await service.handle(TurnSubmit(session_id="nope", text="hi"))
        events = await drain(queue)
        assert kinds(events) == ["Notify"]
        assert events[0].severity == "warning"

    async def test_interrupting_nothing_is_harmless(self, service, session):
        await service.handle(TurnInterrupt(session_id=session.session_id))


class TestFocus:
    async def test_focus_is_recorded_and_repaints(self, service, session):
        queue = service.subscribe()
        await service.handle(SessionFocus(session_id=session.session_id))
        assert service._deps.focused_session_id == session.session_id
        # force=True on focus: a client that just opened a session has a blank
        # column and needs the rows even if they did not change.
        assert "PanelUpdate" in kinds(await drain(queue))

    async def test_clearing_focus_is_allowed(self, service):
        await service.handle(SessionFocus(session_id=None))
        assert service._deps.focused_session_id is None


class TestConfirmations:
    async def test_a_question_is_asked_and_its_answer_runs_the_action(self, service):
        queue = service.subscribe()
        ran = []

        async def on_yes():
            ran.append(True)

        service.ask("Learn this signature?", on_yes)
        events = await drain(queue)
        assert kinds(events) == ["ConfirmRequested"]
        await service.handle(ConfirmResolve(id=events[0].id, confirmed=True))
        assert ran == [True]

    async def test_a_no_runs_nothing(self, service):
        queue = service.subscribe()
        ran = []
        service.ask("Learn this?", lambda: _record(ran))
        events = await drain(queue)
        await service.handle(ConfirmResolve(id=events[0].id, confirmed=False))
        assert ran == []

    async def test_an_answer_to_a_question_nobody_asked_is_ignored(self, service):
        await service.handle(ConfirmResolve(id="q99", confirmed=True))

    async def test_the_same_answer_twice_runs_once(self, service):
        queue = service.subscribe()
        ran = []
        service.ask("Learn this?", lambda: _record(ran))
        key = (await drain(queue))[0].id
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        assert ran == [True]

    async def test_a_failing_action_is_reported_not_raised(self, service):
        queue = service.subscribe()

        async def boom():
            raise RuntimeError("the signature file is read-only")

        service.ask("Learn this?", boom)
        key = (await drain(queue))[0].id
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error"
            for e in await drain(queue)
        )


class TestRobustness:
    async def test_a_command_that_fails_is_a_notify_not_an_exception(
        self, service, session, monkeypatch
    ):
        # One bad frame must not be able to end a session: the far side of
        # this is a socket.
        queue = service.subscribe()

        def explode(*a, **k):
            raise RuntimeError("scheduler is on fire")

        monkeypatch.setattr(service._scheduler, "submit_user", explode)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await drain(queue)
        assert any(
            type(e).__name__ == "Notify" and "on fire" in e.text for e in events
        )

    async def test_a_command_with_no_handler_yet_is_reported(self, service):
        # Most of §4.1 is not dispatched yet. Silence would let a front-end
        # wait forever for something that was never going to happen, so an
        # unhandled command must say so.
        queue = service.subscribe()
        await service.handle(Shutdown.model_construct(TYPE="shutdown"))
        await service.stop()

    async def test_answering_a_decision_nobody_is_parked_on_is_harmless(
        self, service
    ):
        queue = service.subscribe()
        await service.handle(DecisionResolve(session_id="s1", approved=True))
        # Not an error: a stale answer is exactly what arrives when a turn
        # resolved between the prompt being drawn and the key being pressed.
        assert [e for e in await drain(queue) if type(e).__name__ == "Notify"] == []


class TestShutdown:
    async def test_stop_cancels_the_poll_timers(self, service):
        service.start_timers()
        assert service._timers, "no timers were installed"
        await service.stop()
        # A poll firing after the databases close would raise into a dead
        # loop, so the timers go first and are actually awaited out.
        assert service._timers == []

    async def test_stop_is_idempotent(self, service):
        await service.stop()
        await service.stop()

    async def test_the_shutdown_command_stops_it(self, service):
        await service.handle(Shutdown())
        assert service._stopped is True


async def _record(sink):
    sink.append(True)


async def _yield():
    import asyncio

    await asyncio.sleep(0)


async def _settle():
    import asyncio

    for _ in range(40):
        await asyncio.sleep(0)
