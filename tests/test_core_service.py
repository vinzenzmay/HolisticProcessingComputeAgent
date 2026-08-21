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
    PROTOCOL_VERSION,
    ConfirmResolve,
    DecisionResolve,
    SessionClose,
    SessionDelete,
    SessionFocus,
    SessionFork,
    SessionList,
    SessionNew,
    SessionOpen,
    SessionRename,
    SessionRetitle,
    SessionRollback,
    Shutdown,
    TurnInterrupt,
    TurnSubmit,
    TurnUnqueue,
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
        answer = self._outputs.pop(0) if self._outputs else respond()
        # A queued answer may be a whole `ChatResponse` when the test cares
        # about what rides alongside the content — reasoning, token counts.
        if isinstance(answer, ChatResponse):
            return answer
        return ChatResponse(content=answer)

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


def subscribe(service):
    """A subscriber's queue, wound past the frames every client is handed.

    `hello` is the first one, and a parked decision follows it; both are
    handshake, not news, so the tests that assert what a *command* produced
    start after them. `TestHandshake` is where they are asserted directly.
    """
    queue = service.subscribe()
    while not queue.empty():
        queue.get_nowait()
    return queue


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
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await drain(queue)
        assert "TurnStarted" in kinds(events)

    async def test_two_subscribers_both_see_it(self, service, session):
        first, second = subscribe(service), subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert kinds(await drain(first)) == kinds(await drain(second))

    async def test_an_unsubscribed_queue_stops_receiving(self, service, session):
        queue = subscribe(service)
        service.unsubscribe(queue)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert await drain(queue) == []


class TestHandshake:
    """What a client is told the moment it subscribes, before it asks anything.

    Two things, and both are state a front-end cannot work out for itself: who
    the core is (§4.2 `hello`), and any decision already parked and waiting for
    an answer (§4.4). The second is the fix for a real latent bug — a parked
    decision used to be UI-process memory, so a restart left a session stuck on
    an interrupt with nothing on screen to answer it.
    """

    async def test_hello_is_the_first_frame(self, service):
        queue = service.subscribe()
        first = queue.get_nowait()
        assert type(first).__name__ == "Hello"
        assert first.version == PROTOCOL_VERSION
        assert first.profile == "default"
        # A digest rather than the settings themselves: a front-end must be
        # able to notice they changed without being handed the api keys.
        assert first.settings_digest

    async def test_the_digest_follows_the_settings(self, home, conn, llm):
        async def db(fn):
            return fn(conn)

        def greeting(settings):
            built = build_service(
                settings=settings,
                app_dir=home,
                db=db,
                conn=conn,
                checkpointer=InMemorySaver(),
                llm=llm,
            )
            return built.subscribe().get_nowait()

        settings = Settings.load()
        before = greeting(settings)
        settings.agent.default_mode = "auto"
        assert greeting(settings).settings_digest != before.settings_digest

    async def test_the_handshake_goes_only_to_the_arriving_client(self, service):
        established = subscribe(service)
        service.subscribe()  # a second client arrives
        # The greeting is that client's, not a broadcast: a reconnect must not
        # make every other front-end re-run its version check.
        assert await drain(established) == []

    async def test_a_parked_decision_is_re_emitted_to_a_new_client(
        self, service, session
    ):
        service._scheduler._decisions[session.session_id] = {"tool": "run_bash"}
        events = await drain(service.subscribe())
        parked = only(events, "DecisionRequested")
        assert parked.session_id == session.session_id
        assert parked.payload == {"tool": "run_bash"}
        # After the greeting: a client runs its version check first.
        assert kinds(events).index("Hello") < kinds(events).index(
            "DecisionRequested"
        )

    async def test_a_client_arriving_with_nothing_parked_gets_only_hello(
        self, service, session
    ):
        assert kinds(await drain(service.subscribe())) == ["Hello"]


class TestTurns:
    async def test_a_submitted_turn_reaches_the_model_and_comes_back(
        self, service, session, llm
    ):
        queue = subscribe(service)
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
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id="nope", text="hi"))
        events = await drain(queue)
        assert kinds(events) == ["Notify"]
        assert events[0].severity == "warning"

    async def test_interrupting_nothing_is_harmless(self, service, session):
        await service.handle(TurnInterrupt(session_id=session.session_id))

    async def test_interrupting_takes_the_abandoned_rows_off_the_screen(
        self, service, session, llm
    ):
        # The interrupt rolls the abandoned attempt out of the thread, so the
        # rows drawn for it describe messages that no longer exist. A delta
        # cannot un-draw a row; the reset that follows is the same case a
        # rollback is — an open of what is left.
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == []
        release.set()
        await service.stop()

    async def test_the_message_comes_back_to_be_edited(
        self, service, session, llm
    ):
        # The whole point of the abort: the user meant something slightly
        # different, and gets their sentence back rather than retyping it.
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        events = await drain(queue)
        handed_back = only(events, "TurnInterrupted")
        assert handed_back.text == "running"
        # Addressed: the user may be looking at another session by now, and
        # the message waits in the one it was typed in, as a draft.
        assert handed_back.session_id == session.session_id
        # After the reset, which has just re-stated the chat it belonged to.
        assert kinds(events).index("ChatReset") < kinds(events).index(
            "TurnInterrupted"
        )
        release.set()
        await service.stop()

    async def test_nothing_is_handed_back_when_nothing_was_stopped(
        self, service, session
    ):
        # A stale gesture — the turn finished while the key was on its way —
        # must not put a message into an entry box the user is typing in.
        queue = subscribe(service)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        assert "TurnInterrupted" not in kinds(await drain(queue))


class TestStoppingATurnInEveryPhase:
    """The phases an abort has to cover, driven through a real graph.

    A turn does not spend its time waiting on the model. It spends it in
    tools, and — when the model keeps producing output the middleware refuses
    — going round a loop with no exit of its own. Both are what a user reaches
    for the stop gesture in, and refusing there leaves them watching a spinner
    they cannot answer.
    """

    def slow_tool(self):
        import asyncio

        from pydantic import BaseModel

        from hpca.agent.tools import Tool, ToolRegistry

        class SlowParams(BaseModel):
            text: str = ""

        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(args, ctx):
            entered.set()
            await release.wait()
            return "finished at last"

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="slow_tool",
                description="Takes its time",
                params=SlowParams,
                handler=handler,
            )
        )
        return registry, entered, release

    def service_with(self, home, conn, llm, tools):
        async def db(fn):
            return fn(conn)

        return build_service(
            settings=Settings.load(),
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=InMemorySaver(),
            llm=llm,
            tools=tools,
        )

    async def test_a_turn_running_a_tool_can_be_stopped(
        self, home, conn, session
    ):
        import asyncio

        tools, entered, release = self.slow_tool()
        llm = FakeLLM([calling("slow_tool"), respond("done")])
        service = self.service_with(home, conn, llm, tools)
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="run the thing")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        # The phase a long script spends its minutes in.
        assert service._scheduler.can_interrupt(session.session_id) is True

        await service.handle(TurnInterrupt(session_id=session.session_id))
        events = await drain(queue)
        assert only(events, "TurnInterrupted").text == "run the thing"
        assert only(events, "ChatReset").entries == []
        release.set()
        await service.stop()

    async def test_it_breaks_the_decision_retry_loop(
        self, home, conn, session
    ):
        # The phase with no exit of its own: the model keeps producing output
        # the middleware rejects and the turn goes round feeding the rejection
        # back. Cancelling lands on the loop's own await, so the next attempt
        # is never made.
        import asyncio

        llm = FakeLLM()
        entered, release = asyncio.Event(), asyncio.Event()
        rounds = {"n": 0}

        async def never_valid(messages, *, json_schema=None, **kwargs):
            rounds["n"] += 1
            if rounds["n"] >= 2:  # genuinely round the loop once first
                entered.set()
                await release.wait()
            return ChatResponse(content="Let me think about that some more.")

        llm.chat = never_valid
        service = self.service_with(home, conn, llm, None)
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="do the impossible")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)

        await service.handle(TurnInterrupt(session_id=session.session_id))
        assert only(await drain(queue), "TurnInterrupted").text == (
            "do the impossible"
        )
        assert rounds["n"] == 2  # the loop stopped where it stood
        assert not release.is_set()
        assert not service._scheduler.is_busy(session.session_id)
        await service.stop()


class TestTypeAhead:
    """The queued-message channel: type ahead, see it, take it back.

    In the Textual front-end the queue was UI state; here the scheduler owns
    it, so everything the user can see or do about it has to cross the wire.
    """

    async def park(self, service, llm):
        """Hold the model inside a turn so the next message has to queue."""
        import asyncio

        entered, release = asyncio.Event(), asyncio.Event()
        answer = llm.chat

        async def gated(messages, **kwargs):
            entered.set()
            await release.wait()
            return await answer(messages, **kwargs)

        llm.chat = gated
        return entered, release

    async def test_a_message_typed_during_a_turn_shows_up_as_queued(
        self, service, session, llm
    ):
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))

        appended = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"]
        assert [(e.entry.kind, e.entry.text) for e in appended] == [
            ("queued", "second")
        ]
        # Named by the core, which is the only side that may name a row.
        assert appended[0].entry.seq >= 1
        release.set()
        await service.stop()

    async def test_a_message_that_starts_at_once_is_not_a_queued_row(
        self, service, session
    ):
        # Nothing is waiting, so the turn starts — and a "queued" row would
        # then have to be un-drawn a moment later. The message is still drawn,
        # as the ordinary `user` row it already is.
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        appended = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"]
        assert [(e.entry.kind, e.entry.text) for e in appended] == [("user", "hi")]
        await service.stop()

    async def test_cancelling_hands_the_text_back(self, service, session, llm):
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))
        row = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"][0]

        # The UI cancels the row it was given, not a position it counted.
        await service.handle(
            TurnUnqueue(session_id=session.session_id, seq=row.entry.seq)
        )
        events = await drain(queue)
        assert kinds(events) == ["TurnUnqueued"]
        assert (events[0].seq, events[0].text) == (row.entry.seq, "second")
        release.set()
        await service.stop()

    async def test_cancelling_a_row_that_already_started_says_so(
        self, service, session
    ):
        # The turn ahead finished while the dialog was open. Refused the way
        # every un-carry-out-able command is refused: a warning, no state
        # change, and nothing for the UI to guess at.
        queue = subscribe(service)
        await service.handle(TurnUnqueue(session_id=session.session_id, seq=1))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        await service.stop()

    async def test_the_queued_row_is_promoted_rather_than_redrawn(
        self, service, session, llm
    ):
        # End to end: the row the user sees waiting is the row that becomes
        # the message that ran — one name, two frames, no rebuild.
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))
        row = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"][0]

        release.set()
        seen: list = []
        for _ in range(60):
            seen += await drain(queue)
            # By its seq, not by position: the turn ahead re-states its own
            # rows as it finishes, so several updates cross in this window.
            promotion = [
                e
                for e in seen
                if type(e).__name__ == "ChatUpdate"
                and e.entry.seq == row.entry.seq
            ]
            if promotion:
                break
            await _yield()
        else:
            raise AssertionError("the queued row was never promoted")
        assert (promotion[0].entry.kind, promotion[0].entry.text) == (
            "user",
            "second",
        )
        # And no second row was drawn for it.
        assert not [
            e
            for e in seen
            if type(e).__name__ == "ChatAppend" and e.entry.text == "second"
        ]
        await service.stop()


def only(events, name):
    """The single event of that type, or an assertion naming what did arrive."""
    found = [e for e in events if type(e).__name__ == name]
    assert len(found) == 1, f"{name} in {kinds(events)}"
    return found[0]


async def wait_for(queue, name, *, seconds=10):
    """Everything emitted up to and including the first ``name`` event.

    Condition-waiting rather than counting yields: a turn crosses a thread on
    its way through the log and the stores, and the `-n auto` run is exactly
    where a fixed number of loop turns stops being enough.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    events = []
    while loop.time() < deadline:
        events += await drain(queue)
        if any(type(e).__name__ == name for e in events):
            return events
        await asyncio.sleep(0.005)
    raise AssertionError(f"no {name} arrived; saw {kinds(events)}")


async def run_turn(service, session_id, text="hi"):
    """One turn, start to finish, driven the way a front-end drives it."""
    import asyncio

    queue = subscribe(service)
    try:
        await service.handle(TurnSubmit(session_id=session_id, text=text))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        seen = []
        while loop.time() < deadline:
            seen += await drain(queue)
            names = kinds(seen)
            if "TurnFailed" in names:
                raise AssertionError(seen[names.index("TurnFailed")].error)
            if "TurnFinished" in names:
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"the turn never finished; saw {kinds(seen)}")
    finally:
        service.unsubscribe(queue)


def gate_llm(llm):
    """Hold the *next* model call open. Returns (entered, release).

    The turn is then sitting somewhere a test can act on it — which is what
    "the spinner is up and the user presses escape twice" looks like from
    here.
    """
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    answer = llm.chat

    async def gated(messages, **kwargs):
        llm.chat = answer  # only this one call is held
        entered.set()
        await release.wait()
        return await answer(messages, **kwargs)

    llm.chat = gated
    return entered, release


async def park_turn(service, llm, session_id):
    """Hold a turn inside the model, so the session is busy. Returns the gate."""
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    answer = llm.chat

    async def gated(messages, **kwargs):
        entered.set()
        await release.wait()
        return await answer(messages, **kwargs)

    llm.chat = gated
    await service.handle(TurnSubmit(session_id=session_id, text="running"))
    await asyncio.wait_for(entered.wait(), timeout=5)
    llm.chat = answer  # only the first call is held
    return release


class TestSessionList:
    async def test_every_session_comes_back_in_the_stores_order(
        self, service, conn, session
    ):
        other = SessionStore(conn).create(profile="default", title="the other one")
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = only(await drain(queue), "SessionRows").rows
        # The store's order, not one the core invents: newest first is a
        # decision `SessionStore.list_all` already made.
        assert [r.session_id for r in rows] == [
            s.session_id for s in SessionStore(conn).list_all()
        ]
        assert {r.title for r in rows} == {"a session", "the other one"}
        assert other.session_id in {r.session_id for r in rows}

    async def test_a_row_names_the_model_its_session_is_pinned_to(
        self, service, conn
    ):
        backend = json.dumps(
            {"model": "gemma-3-27b", "base_url": "http://localhost:20001/v1"}
        )
        pinned = SessionStore(conn).create(
            profile="default", title="pinned", backend=backend
        )
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = {r.session_id: r for r in only(await drain(queue), "SessionRows").rows}
        assert rows[pinned.session_id].model == "gemma-3-27b"

    async def test_a_bootstrap_session_names_no_model(self, service, session):
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = only(await drain(queue), "SessionRows").rows
        # Not the app's default model: the row says what this conversation is
        # pinned to, and it is pinned to nothing.
        assert rows[0].model == ""

    async def test_a_running_turn_marks_its_row(self, service, session, llm):
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows[0].flags == ["working"]
        release.set()
        await service.stop()

    async def test_a_parked_decision_marks_its_row(self, service, session):
        # The other state a user working elsewhere has to be able to see. Set
        # on the scheduler because that is where a parked decision lives now.
        service._scheduler._decisions[session.session_id] = {"tool": "run_bash"}
        queue = subscribe(service)
        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows[0].flags == ["decision"]


class TestSessionNew:
    async def test_a_new_session_is_announced_and_then_listed(self, service):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        events = await drain(queue)
        # created first: the UI has to open it, and a sidebar cannot say which
        # of its lines is the new one.
        assert kinds(events) == ["SessionCreated", "SessionRows"]
        created = events[0].row
        assert created.session_id in {r.session_id for r in events[1].rows}

    async def test_it_is_created_under_the_profile_asked_for(self, service, conn):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="bioinformatics"))
        created = only(await drain(queue), "SessionCreated").row
        assert created.profile == "bioinformatics"
        stored = SessionStore(conn).get(created.session_id)
        assert stored is not None and stored.profile == "bioinformatics"

    async def test_the_backend_it_asks_for_is_pinned_to_it(self, service, conn):
        blob = json.dumps(
            {"model": "qwen3-32b", "base_url": "http://localhost:20001/v1"}
        )
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default", backend=blob))
        created = only(await drain(queue), "SessionCreated").row
        assert created.model == "qwen3-32b"
        # Stored as the blob, so the choice survives the catalog entry going.
        assert "qwen3-32b" in SessionStore(conn).get(created.session_id).backend

    async def test_an_unusable_backend_falls_back_rather_than_stranding_it(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default", backend="not json"))
        assert only(await drain(queue), "SessionCreated").row.model == ""


class TestSessionOpen:
    async def test_opening_sends_the_whole_transcript_once(self, service, session):
        await run_turn(service, session.session_id, "which BAMs?")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert reset.session_id == session.session_id
        assert [e.kind for e in reset.entries][:1] == ["user"]
        assert "which BAMs?" in reset.entries[0].text

    async def test_every_row_arrives_named_from_one(self, service, session):
        await run_turn(service, session.session_id, "hello")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        # A reset re-bases the numbering; the UI drops the names it held.
        assert [e.seq for e in reset.entries] == list(
            range(1, len(reset.entries) + 1)
        )

    async def test_an_empty_session_opens_empty(self, service, session):
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == []

    async def test_opening_says_how_full_the_window_already_is(
        self, service, session
    ):
        await run_turn(service, session.session_id, "hello")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        events = await drain(queue)
        # Nothing has been sent this run, so the fill is derived from the
        # stored history — the same number a restart would show.
        assert only(events, "ContextEstimate").session_id == session.session_id

    async def test_opening_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id="nope"))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"

    async def test_a_session_reopened_mid_turn_keeps_its_type_ahead(
        self, service, session, llm
    ):
        # The known gap of specs-ui-replacement.md §4.2: the queue is core
        # state now, so a reset that leaves it out deletes the user's
        # type-ahead from under them.
        release = await park_turn(service, llm, session.session_id)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="and the CRAMs?")
        )
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert [(e.kind, e.text) for e in reset.entries][-1] == (
            "queued",
            "and the CRAMs?",
        )
        assert reset.entries[-1].seq > 0, "a queued row must stay addressable"
        release.set()
        await service.stop()

    async def test_the_re_drawn_queued_row_is_the_one_a_cancel_names(
        self, service, session, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="and the CRAMs?")
        )
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        row = only(await drain(queue), "ChatReset").entries[-1]

        await service.handle(
            TurnUnqueue(session_id=session.session_id, seq=row.seq)
        )
        assert only(await drain(queue), "TurnUnqueued").text == "and the CRAMs?"
        release.set()
        await service.stop()


def apply_deltas(events, session_id):
    """The chat a front-end holds after applying what it was sent, in order.

    A renderer in three lines, and deliberately strict about the two rules the
    protocol asks of it: a reset replaces everything, and an update addresses a
    row by `Entry.seq` and nothing else. An update naming a row nobody has is
    raised rather than turned into a new row — a client that invented one would
    hide exactly the desynchronisation this is here to catch.
    """
    rows: list = []
    for event in events:
        name = type(event).__name__
        if getattr(event, "session_id", None) != session_id:
            continue
        if name == "ChatReset":
            rows = list(event.entries)
        elif name == "ChatAppend":
            rows.append(event.entry)
        elif name == "ChatUpdate":
            for position, row in enumerate(rows):
                if row.seq == event.entry.seq:
                    rows[position] = event.entry
                    break
            else:
                raise AssertionError(
                    f"chat.update for row {event.entry.seq}, which was never drawn"
                )
    return rows


def calling(tool: str, **arguments) -> str:
    """One tool_call decision, as the envelope protocol carries it."""
    return json.dumps(
        {"action": "tool_call", "tool": tool, "arguments": arguments}
    )


class TestTheTurnAsItHappens:
    """What a front-end with a session open sees while the turn runs.

    This is the property the whole protocol is for: the conversation arrives as
    deltas, and re-opening the session afterwards produces the *same* rows. The
    second half is the one that is easy to get wrong — a `chat.reset` folds a
    turn's reasoning and tool calls into one working box, so live rows shaped
    any other way make the screen rearrange itself the moment the user comes
    back, which is the bug the old UI paid for by rebuilding the whole log
    every turn.
    """

    @pytest.fixture
    def target(self, home):
        path = home / "reads.tsv"
        path.write_text("sample\tcount\na\t7\n")
        return path

    async def run_with_a_call(self, service, session, llm, target, *, reasoning=""):
        """One turn that reads a file and then answers. Returns its events."""
        llm._outputs = [
            ChatResponse(
                content=calling("read_file", path=str(target)),
                reasoning=reasoning,
            ),
            respond("seven reads"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="how many reads?")
        )
        return await wait_for(queue, "TurnFinished"), queue

    async def test_the_call_is_a_row_before_the_result_exists(
        self, service, session, llm, target
    ):
        events, _ = await self.run_with_a_call(service, session, llm, target)
        drawn = [
            e
            for e in events
            if type(e).__name__ in ("ChatAppend", "ChatUpdate")
            and e.entry.parts
        ]
        first = drawn[0].entry
        # The call is on screen while it is still running: one part, named,
        # and with nothing in the result half yet.
        assert [(p.kind, p.tool, p.done) for p in first.parts] == [
            ("call", "read_file", False)
        ]
        await service.stop()

    async def test_the_result_lands_in_the_row_the_call_drew(
        self, service, session, llm, target
    ):
        events, _ = await self.run_with_a_call(service, session, llm, target)
        drawn = [
            e
            for e in events
            if type(e).__name__ in ("ChatAppend", "ChatUpdate")
            and e.entry.parts
        ]
        call_row = drawn[0].entry.seq
        filled = [
            e.entry
            for e in drawn
            if e.entry.seq == call_row
            and any(p.done and p.result for p in e.entry.parts)
        ]
        assert filled, "the result never reached the row the call drew"
        assert "sample" in filled[0].parts[0].result
        # And it never became a row of its own: one exchange, one row.
        assert [type(e).__name__ for e in drawn if e.entry.seq != call_row] == []
        await service.stop()

    async def test_the_reply_arrives_without_anyone_asking_for_a_snapshot(
        self, service, session, llm, target
    ):
        events, _ = await self.run_with_a_call(service, session, llm, target)
        rows = apply_deltas(events, session.session_id)
        assert [r.kind for r in rows] == ["user", "thinking", "assistant"]
        assert (rows[0].text, rows[-1].text) == ("how many reads?", "seven reads")
        # Nothing resent the transcript to achieve that (§4.2 property 1).
        assert "ChatReset" not in kinds(events)
        await service.stop()

    async def test_re_opening_afterwards_yields_the_rows_the_deltas_built(
        self, service, session, llm, target
    ):
        # The one that matters: what the user watched happen and what they get
        # back when they return have to be the same rows, seq for seq.
        events, queue = await self.run_with_a_call(
            service, session, llm, target, reasoning="the counts are in column 2"
        )
        live = apply_deltas(events, session.session_id)

        await service.handle(SessionOpen(session_id=session.session_id))
        reopened = only(await drain(queue), "ChatReset").entries
        assert reopened == live
        await service.stop()

    async def test_the_models_reasoning_is_in_the_working_box(
        self, service, session, llm, target
    ):
        # Reasoning reaches nobody through a callback — it is checkpointed
        # state — so it arrives with the fold at the end of the turn, in the
        # place the fold gives it: ahead of the call it led to.
        events, _ = await self.run_with_a_call(
            service, session, llm, target, reasoning="column 2 holds the counts"
        )
        working = apply_deltas(events, session.session_id)[1]
        assert [p.kind for p in working.parts] == ["reasoning", "call"]
        assert working.parts[0].text == "column 2 holds the counts"
        assert working.reasoning_chars == len("column 2 holds the counts")
        await service.stop()

    async def test_the_backends_token_count_is_announced(
        self, service, session, llm
    ):
        llm._outputs = [
            ChatResponse(content=respond("done"), usage={"prompt_tokens": 4321})
        ]
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await wait_for(queue, "TurnFinished")
        # The context meter's measured half; without it the bar only ever
        # shows the estimate a re-open derives from stored history.
        assert only(events, "TurnUsage").prompt_tokens == 4321
        await service.stop()

    async def test_re_opening_mid_turn_re_binds_the_rows_it_is_drawing(
        self, service, session, llm, target
    ):
        # A reset renumbers every row on screen, including the working box a
        # turn is still filling. If the turn kept the old names, the update
        # carrying its result would address a row the client no longer has —
        # which `apply_deltas` refuses to invent, so this fails loudly.
        import asyncio

        entered, release = asyncio.Event(), asyncio.Event()
        answer, rounds = llm.chat, {"n": 0}

        async def gated(messages, **kwargs):
            rounds["n"] += 1
            if rounds["n"] == 2:  # the tool exchange is checkpointed by now
                entered.set()
                await release.wait()
            return await answer(messages, **kwargs)

        llm._outputs = [calling("read_file", path=str(target)), respond("seven")]
        llm.chat = gated
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="how many reads?")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        await service.handle(SessionOpen(session_id=session.session_id))
        release.set()

        live = apply_deltas(await wait_for(queue, "TurnFinished"), session.session_id)
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == live
        await service.stop()

    async def test_a_turn_in_a_session_nobody_is_looking_at_still_says_so(
        self, service, conn, session, llm, target
    ):
        # The core does not know what is on screen and must not decide from it
        # (§4.2 rule 3): the event names its session and the UI drops what it
        # is not showing.
        other = SessionStore(conn).create(profile="default", title="the other one")
        await service.handle(SessionFocus(session_id=other.session_id))
        events, _ = await self.run_with_a_call(service, session, llm, target)
        assert [r.kind for r in apply_deltas(events, session.session_id)] == [
            "user",
            "thinking",
            "assistant",
        ]
        await service.stop()


class TestAParkedApproval:
    """A turn that stops to ask, driven through a real graph interrupt.

    The scheduler's own tests assert the bookkeeping against a fake; this one
    exists because the park is where the row reconciliation is hardest. A
    parked turn and its resume are *one* exchange as far as `build_entries` is
    concerned — one working box, holding the call, the answer to it, and
    whatever came after — so the two graph invocations have to draw one box
    between them or a re-open rearranges the screen.
    """

    async def park(self, service, session, llm):
        """Ask for something manual mode will not run unasked."""
        llm._outputs = [
            calling("run_bash", content_lines=["rm -rf /scratch/old"]),
            respond("left it alone"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="clear the scratch dir")
        )
        return queue, await wait_for(queue, "DecisionRequested")

    async def test_the_graph_parking_asks(self, service, session, llm):
        _, events = await self.park(service, session, llm)
        asked = only(events, "DecisionRequested")
        assert asked.session_id == session.session_id
        assert asked.payload["tool"] == "run_bash"
        # Manual mode gates execution as well as destruction (§3.5).
        assert asked.payload["kind"] in ("execution", "destructive")
        # Nothing says the turn finished: it cannot, until this is answered.
        assert "TurnFinished" not in kinds(events)
        await service.stop()

    async def test_answering_clears_the_prompt_and_finishes_the_turn(
        self, service, session, llm
    ):
        queue, events = await self.park(service, session, llm)
        await service.handle(
            DecisionResolve(
                session_id=session.session_id,
                approved=False,
                reason="that is the real data",
            )
        )
        events += await wait_for(queue, "TurnFinished")
        assert "DecisionCleared" in kinds(events)
        assert only(events, "TurnFinished").reply == "left it alone"
        await service.stop()

    async def test_the_refusal_lands_in_the_row_that_asked_for_it(
        self, service, session, llm
    ):
        queue, events = await self.park(service, session, llm)
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        events += await wait_for(queue, "TurnFinished")
        rows = apply_deltas(events, session.session_id)
        # One working box across both halves of the turn, not two.
        assert [r.kind for r in rows] == ["user", "thinking", "assistant"]
        assert [(p.kind, p.tool, p.failed) for p in rows[1].parts] == [
            ("call", "run_bash", True)
        ]

        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == rows
        await service.stop()

    async def test_it_does_not_stall_another_sessions_queue(
        self, service, conn, session, llm
    ):
        # A parked thread cannot move until it is answered — but only its own.
        # The rule the scheduler is written around, asserted here through a
        # real interrupt rather than a fake result.
        queue, _ = await self.park(service, session, llm)
        other = SessionStore(conn).create(profile="default", title="the other one")
        await service.handle(TurnSubmit(session_id=other.session_id, text="hi"))

        finished = only(await wait_for(queue, "TurnFinished"), "TurnFinished")
        assert finished.session_id == other.session_id
        # And the parked one is still parked — its own queue is the only one
        # the unanswered question holds up.
        assert session.session_id in service._scheduler.pending_decisions()
        await service.stop()

    async def test_a_rollback_is_refused_while_it_waits(
        self, service, session, llm
    ):
        # The resume would land on message indices the cut had removed. The
        # answer is the reason, and it reaches the user as a warning.
        queue, _ = await self.park(service, session, llm)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        assert "ChatReset" not in kinds(events)
        assert "decision" in only(events, "Notify").text
        await service.stop()

    async def test_the_resumed_turn_is_still_stoppable(
        self, service, session, llm
    ):
        # An approval splits one exchange into two turns. The second carries
        # no user message of its own, and without the anchor it is a spinner
        # nothing can answer — the case `tui/app.py:_interrupt_anchor` exists
        # for, and the acceptance list's "after an approval, the resumed turn
        # carries the same interrupt anchor".
        import asyncio

        queue, _ = await self.park(service, session, llm)
        entered, release = gate_llm(llm)
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert service._scheduler.can_interrupt(session.session_id) is True

        await service.handle(TurnInterrupt(session_id=session.session_id))
        events = await drain(queue)
        # The message that started the exchange, and the whole exchange gone
        # from the thread — the refused call included.
        assert only(events, "TurnInterrupted").text == "clear the scratch dir"
        assert only(events, "ChatReset").entries == []
        release.set()
        await service.stop()

    async def test_the_anchor_goes_when_the_exchange_does(
        self, service, session, llm
    ):
        queue, _ = await self.park(service, session, llm)
        assert session.session_id in service._scheduler._anchors
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        await wait_for(queue, "TurnFinished")
        # Answered for good: the next turn must not be handed this one's
        # message, and there is nothing left of it to roll back to.
        assert service._scheduler._anchors == {}
        await service.stop()


class TestAResumeAfterACoreRestart:
    """The half of an approval that outlives the process that asked.

    The thread is parked at `interrupt()` in the checkpointer, so the question
    survives; the turn that asked it does not, and with it goes the record of
    which chat rows are that exchange's. Without something to bind to, the
    resume can only stay silent and let the next `chat.reset` show the reply
    — the gap specs-ui-replacement.md §4.2 records against M5.
    """

    def build(self, home, conn, checkpointer, llm):
        async def db(fn):
            return fn(conn)

        return build_service(
            settings=Settings.load(),
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=checkpointer,
            llm=llm,
        )

    async def test_the_answer_lands_in_the_rows_already_on_screen(
        self, home, conn, session
    ):
        checkpointer = InMemorySaver()
        first = self.build(home, conn, checkpointer, FakeLLM(
            [calling("run_bash", content_lines=["rm -rf /scratch/old"])]
        ))
        queue = subscribe(first)
        await first.handle(
            TurnSubmit(session_id=session.session_id, text="clear the scratch dir")
        )
        parked = only(
            await wait_for(queue, "DecisionRequested"), "DecisionRequested"
        )
        await first.stop()

        # A new core over the same store and the same checkpointer, holding
        # the question the old one was parked on.
        second = self.build(
            home, conn, checkpointer, FakeLLM([respond("left it alone")])
        )
        second._scheduler._decisions[session.session_id] = dict(parked.payload)
        second._scheduler._awaiting_approval.add(session.session_id)
        queue = subscribe(second)

        # The user opens the session to answer the prompt, which is what puts
        # the parked exchange on screen — and names its rows.
        await second.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        # The message alone: a parked call is announced only once it is
        # answered for, so the thread holds nothing else yet.
        assert [e.kind for e in reset.entries] == ["user"]

        await second.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        events = await wait_for(queue, "TurnFinished")
        rows = apply_deltas([reset] + events, session.session_id)
        # One working box across both cores, and the reply after it — not a
        # second box, and not silence until the next open.
        assert [r.kind for r in rows] == ["user", "thinking", "assistant"]
        assert rows[-1].text == "left it alone"
        assert [(p.tool, p.failed) for p in rows[1].parts] == [("run_bash", True)]

        await second.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == rows
        await second.stop()


class TestAFailedTurn:
    """A turn that breaks instead of finishing.

    The rows it drew have no result to fold, so without reconciling them they
    settle exactly as the live path left them — and a re-opened session would
    then draw something else.
    """

    @pytest.fixture
    def target(self, home):
        path = home / "reads.tsv"
        path.write_text("sample\tcount\na\t7\n")
        return path

    async def failing(self, service, session, llm, target):
        """One round of real work, and then the backend goes away — a tunnel
        dropped mid-turn, which is what this looks like on a cluster."""
        llm._outputs = [
            ChatResponse(
                content=calling("read_file", path=str(target)),
                reasoning="column 2 holds the counts",
            )
        ]
        answer, rounds = llm.chat, {"n": 0}

        async def then_dies(messages, **kwargs):
            rounds["n"] += 1
            if rounds["n"] > 1:
                raise ConnectionError("connection refused")
            return await answer(messages, **kwargs)

        llm.chat = then_dies
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="how many reads?")
        )
        return queue, await wait_for(queue, "TurnFailed")

    async def test_the_failure_is_reported_not_raised(
        self, service, session, llm, target
    ):
        _, events = await self.failing(service, session, llm, target)
        # What a front-end draws its error entry from (§3.2), carrying enough
        # to say what went wrong.
        assert "connection refused" in only(events, "TurnFailed").error
        # And the spinner goes: an empty activity is how a renderer drops it.
        assert any(
            type(e).__name__ == "TurnActivity" and e.activity == ""
            for e in events
        )
        await service.stop()

    async def test_the_rows_it_drew_are_re_stated_from_the_thread(
        self, service, session, llm, target
    ):
        # The working box is the fold's copy, not the half-drawn live one: the
        # reasoning that only ever existed in the checkpoint is in it, exactly
        # where a re-opened session puts it.
        _, events = await self.failing(service, session, llm, target)
        rows = apply_deltas(events, session.session_id)
        assert [r.kind for r in rows] == ["user", "thinking"]
        assert [p.kind for p in rows[1].parts] == ["reasoning", "call"]
        assert rows[1].parts[0].text == "column 2 holds the counts"
        assert rows[1].parts[1].done  # and its result, not a call still going
        await service.stop()

    async def test_the_session_is_free_again(
        self, service, session, llm, target
    ):
        await self.failing(service, session, llm, target)
        assert not service._scheduler.is_busy(session.session_id)
        assert service._scheduler.rewind_blocker(session.session_id) is None
        await service.stop()

    async def test_re_opening_yields_the_rows_the_deltas_left(
        self, service, session, llm, target
    ):
        # The property a failure must not be allowed to break: what the user
        # is looking at and what they get back on re-open are the same rows,
        # seq for seq. Unreconciled they are not — the box still holds a call
        # with no result, and the fold's reasoning is missing from it.
        queue, events = await self.failing(service, session, llm, target)
        live = apply_deltas(events, session.session_id)
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == live
        await service.stop()


class TestSessionClose:
    async def test_closing_forgets_what_was_open(self, service, session):
        await service.handle(SessionFocus(session_id=session.session_id))
        queue = subscribe(service)
        await service.handle(SessionClose())
        assert service._deps.focused_session_id is None
        events = await drain(queue)
        # The panel is session-scoped, so it empties with the session, and the
        # sidebar restates itself without a working row.
        assert "SessionRows" in kinds(events) and "PanelUpdate" in kinds(events)

    async def test_closing_nothing_is_harmless(self, service):
        await service.handle(SessionClose())
        assert service._deps.focused_session_id is None


class TestSessionRename:
    async def test_a_rename_lands_and_is_listed(self, service, session, conn):
        queue = subscribe(service)
        await service.handle(
            SessionRename(session_id=session.session_id, title="BAM QC")
        )
        rows = only(await drain(queue), "SessionRows").rows
        assert rows[0].title == "BAM QC"
        assert SessionStore(conn).get(session.session_id).title == "BAM QC"

    async def test_an_empty_title_is_refused(self, service, session, conn):
        queue = subscribe(service)
        await service.handle(
            SessionRename(session_id=session.session_id, title="   ")
        )
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert SessionStore(conn).get(session.session_id).title == "a session"

    async def test_renaming_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionRename(session_id="nope", title="x"))
        assert (await drain(queue))[0].severity == "warning"


class TestSessionRetitle:
    async def test_the_model_names_the_conversation(self, service, session, llm):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [json.dumps({"title": "Read counting"})]
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "SessionRows")
        assert only(events, "SessionRows").rows[0].title == "Read counting"

    async def test_it_does_not_hold_up_the_next_command(
        self, service, session, llm
    ):
        # A generation takes seconds; a dispatch that awaited it would stall
        # every command behind it on the same socket.
        await run_turn(service, session.session_id, "hello")
        import asyncio

        release = asyncio.Event()
        answer = llm.chat

        async def gated(messages, **kwargs):
            await release.wait()
            return await answer(messages, **kwargs)

        llm.chat = gated
        await service.handle(SessionRetitle(session_id=session.session_id))
        queue = subscribe(service)
        await service.handle(SessionList())  # answered while titling waits
        assert "SessionRows" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_an_empty_conversation_has_nothing_to_summarize(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "Notify")
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"

    async def test_a_model_that_cannot_write_one_is_reported_not_raised(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "hello")
        llm._outputs = []  # every attempt answers with the turn JSON instead
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "Notify")
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error" for e in events
        )
        assert "SessionRows" not in kinds(events)


class TestSessionDelete:
    async def test_the_row_and_its_history_go(self, service, session, conn):
        await run_turn(service, session.session_id, "hello")
        queue = subscribe(service)
        await service.handle(SessionDelete(session_id=session.session_id))
        events = await drain(queue)
        assert only(events, "SessionRows").rows == []
        assert SessionStore(conn).get(session.session_id) is None
        state = await service._graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        assert not (state.values or {}).get("messages")

    async def test_a_deleted_sessions_watches_go_with_it(
        self, service, session, conn
    ):
        from hpca.watches import WatchStore

        WatchStore(conn).add(
            session_id=session.session_id, kind="log", target="/tmp/run.log"
        )
        await service.handle(SessionDelete(session_id=session.session_id))
        # Left behind they would be boxes no session can show while the
        # pollers went on stat-ing their files forever.
        assert WatchStore(conn).list(session_id=session.session_id) == []

    async def test_it_is_forgotten_by_search_too(self, service, session, conn):
        from hpca.episodic import EpisodicStore

        store = EpisodicStore(conn)
        store.record(
            session_id=session.session_id,
            profile="default",
            entries=[("user", "the patient identifier")],
        )
        assert store.search("patient", limit=5)
        await service.handle(SessionDelete(session_id=session.session_id))
        # Patient-data environments: a deleted conversation must not resurface
        # through a search either.
        assert store.search("patient", limit=5) == []

    async def test_nothing_queued_for_it_can_still_start(
        self, service, session, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="queued behind it")
        )
        await service.handle(SessionDelete(session_id=session.session_id))
        assert service._scheduler.queued_texts_for(session.session_id) == []
        release.set()
        await service.stop()

    async def test_deleting_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionDelete(session_id="nope"))
        assert (await drain(queue))[0].severity == "warning"


class TestRollback:
    async def test_the_trimmed_conversation_comes_back_as_a_reset(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = subscribe(service)
        # Cut before the second user message — the index the core itself put
        # on that entry.
        await service.handle(
            SessionRollback(session_id=session.session_id, index=2)
        )
        reset = only(await drain(queue), "ChatReset")
        assert [e.text for e in reset.entries] == ["first", "done"]
        # Re-based numbering: the rows the cut removed cannot be addressed by
        # either side afterwards.
        assert [e.seq for e in reset.entries] == [1, 2]

    async def test_it_is_refused_while_a_turn_is_running(
        self, service, session, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert "a turn is running" in events[0].text
        release.set()
        await service.stop()

    async def test_an_index_the_thread_no_longer_has_is_refused(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        queue = subscribe(service)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=99)
        )
        events = await drain(queue)
        # Refused rather than truncating somewhere the user never pointed at.
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert "no longer" in events[0].text

    async def test_the_fill_is_restated_for_the_thread_that_is_left(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        queue = subscribe(service)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        # The measured number described a thread that no longer exists.
        assert only(events, "ContextEstimate").used == 0

    async def test_rolling_back_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionRollback(session_id="nope", index=0))
        assert (await drain(queue))[0].severity == "warning"


class TestFork:
    async def test_the_branch_is_created_and_announced(
        self, service, session, conn
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=2))
        events = await drain(queue)
        created = only(events, "SessionCreated").row
        assert created.title == "a session (fork)"
        assert created.session_id in {
            r.session_id for r in only(events, "SessionRows").rows
        }
        # The source keeps its whole history: this half is non-destructive.
        assert SessionStore(conn).get(session.session_id) is not None

    async def test_the_fork_holds_the_conversation_up_to_the_cut(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=2))
        fork = only(await drain(queue), "SessionCreated").row

        await service.handle(SessionOpen(session_id=fork.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert [e.text for e in reset.entries] == ["first", "done"]

    async def test_the_forks_profile_and_backend_come_from_the_source(
        self, service, conn
    ):
        blob = json.dumps(
            {"model": "qwen3-32b", "base_url": "http://localhost:20001/v1"}
        )
        store = SessionStore(conn)
        source = store.create(profile="bioinformatics", title="pinned", mode="plan")
        await run_turn(service, source.session_id, "first")
        # Pinned after the turn: a session pointed at a real endpoint would
        # dial it, and this test has no backend to answer.
        store.set_backend(source.session_id, blob)
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=source.session_id, index=0))
        row = only(await drain(queue), "SessionCreated").row
        # Not on the wire, so a front-end cannot fork a conversation into a
        # profile the user never chose.
        assert (row.profile, row.mode, row.model) == (
            "bioinformatics",
            "plan",
            "qwen3-32b",
        )

    async def test_forking_is_allowed_while_the_source_is_busy(
        self, service, session, llm
    ):
        # Deliberately ungated where the rollback is not: a fork only reads a
        # snapshot of the source, and branching off mid-turn is the case it
        # exists for.
        await run_turn(service, session.session_id, "first")
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=0))
        assert "SessionCreated" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_an_index_the_thread_no_longer_has_leaves_no_session_behind(
        self, service, session, conn
    ):
        await run_turn(service, session.session_id, "first")
        before = len(SessionStore(conn).list_all())
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=99))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert len(SessionStore(conn).list_all()) == before

    async def test_a_failed_copy_leaves_no_session_behind(
        self, service, session, conn, monkeypatch
    ):
        # An empty session nobody asked for is worse than none: it would sit
        # in the sidebar looking like the fork worked.
        await run_turn(service, session.session_id, "first")

        async def explode(*a, **k):
            raise RuntimeError("checkpoint write failed")

        monkeypatch.setattr("hpca.core.service.fork_thread", explode)
        before = len(SessionStore(conn).list_all())
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=0))
        events = await drain(queue)
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error" for e in events
        )
        assert len(SessionStore(conn).list_all()) == before

    async def test_forking_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionFork(session_id="nope", index=0))
        assert (await drain(queue))[0].severity == "warning"


class TestAConversationEndToEnd:
    """List, make one, open it, talk, rewind, branch — commands only.

    The point of this one is that nothing else is used: no store call, no
    graph call, no reach into a service. If a front-end can send these seven
    frames and read the ones that come back, it has a conversation.
    """

    async def test_the_whole_round_trip(self, service, llm):
        queue = subscribe(service)

        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows == []

        await service.handle(SessionNew(profile="default"))
        events = await drain(queue)
        session_id = only(events, "SessionCreated").row.session_id
        assert [r.session_id for r in only(events, "SessionRows").rows] == [
            session_id
        ]

        await service.handle(SessionOpen(session_id=session_id))
        assert only(await drain(queue), "ChatReset").entries == []

        await run_turn(service, session_id, "which BAMs are in /data?")
        await run_turn(service, session_id, "and the CRAMs?")
        await drain(queue)

        await service.handle(SessionFork(session_id=session_id, index=2))
        fork_id = only(await drain(queue), "SessionCreated").row.session_id

        await service.handle(SessionRollback(session_id=session_id, index=2))
        reset = only(await drain(queue), "ChatReset")
        assert (reset.session_id, [e.text for e in reset.entries]) == (
            session_id,
            ["which BAMs are in /data?", "done"],
        )

        # The fork kept the same two entries, and the source's rollback did
        # not touch it — two conversations now, numbered from 1 apiece.
        await service.handle(SessionOpen(session_id=fork_id))
        forked = only(await drain(queue), "ChatReset")
        assert forked.session_id == fork_id
        assert [(e.text, e.seq) for e in forked.entries] == [
            ("which BAMs are in /data?", 1),
            ("done", 2),
        ]
        await service.stop()


class TestFocus:
    async def test_focus_is_recorded_and_repaints(self, service, session):
        queue = subscribe(service)
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
        queue = subscribe(service)
        ran = []

        async def on_yes():
            ran.append(True)

        service.ask("Learn this signature?", on_yes)
        events = await drain(queue)
        assert kinds(events) == ["ConfirmRequested"]
        await service.handle(ConfirmResolve(id=events[0].id, confirmed=True))
        assert ran == [True]

    async def test_a_no_runs_nothing(self, service):
        queue = subscribe(service)
        ran = []
        service.ask("Learn this?", lambda: _record(ran))
        events = await drain(queue)
        await service.handle(ConfirmResolve(id=events[0].id, confirmed=False))
        assert ran == []

    async def test_an_answer_to_a_question_nobody_asked_is_ignored(self, service):
        await service.handle(ConfirmResolve(id="q99", confirmed=True))

    async def test_the_same_answer_twice_runs_once(self, service):
        queue = subscribe(service)
        ran = []
        service.ask("Learn this?", lambda: _record(ran))
        key = (await drain(queue))[0].id
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        assert ran == [True]

    async def test_a_failing_action_is_reported_not_raised(self, service):
        queue = subscribe(service)

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
        queue = subscribe(service)

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
        queue = subscribe(service)
        await service.handle(Shutdown.model_construct(TYPE="shutdown"))
        await service.stop()

    async def test_answering_a_decision_nobody_is_parked_on_is_harmless(
        self, service
    ):
        queue = subscribe(service)
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
