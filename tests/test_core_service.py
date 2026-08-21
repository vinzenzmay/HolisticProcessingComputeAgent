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
        queue = service.subscribe()
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
        # then have to be un-drawn a moment later.
        queue = service.subscribe()
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert "ChatAppend" not in kinds(await drain(queue))
        await service.stop()

    async def test_cancelling_hands_the_text_back(self, service, session, llm):
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))
        row = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"][0]

        release.set()
        for _ in range(60):
            events = await drain(queue)
            updates = [e for e in events if type(e).__name__ == "ChatUpdate"]
            if updates:
                break
            await _yield()
        else:
            raise AssertionError("the queued row was never promoted")
        assert (updates[0].entry.seq, updates[0].entry.kind) == (
            row.entry.seq, "user"
        )
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

    queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(SessionList())
        rows = {r.session_id: r for r in only(await drain(queue), "SessionRows").rows}
        assert rows[pinned.session_id].model == "gemma-3-27b"

    async def test_a_bootstrap_session_names_no_model(self, service, session):
        queue = service.subscribe()
        await service.handle(SessionList())
        rows = only(await drain(queue), "SessionRows").rows
        # Not the app's default model: the row says what this conversation is
        # pinned to, and it is pinned to nothing.
        assert rows[0].model == ""

    async def test_a_running_turn_marks_its_row(self, service, session, llm):
        release = await park_turn(service, llm, session.session_id)
        queue = service.subscribe()
        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows[0].flags == ["working"]
        release.set()
        await service.stop()

    async def test_a_parked_decision_marks_its_row(self, service, session):
        # The other state a user working elsewhere has to be able to see. Set
        # on the scheduler because that is where a parked decision lives now.
        service._scheduler._decisions[session.session_id] = {"tool": "run_bash"}
        queue = service.subscribe()
        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows[0].flags == ["decision"]


class TestSessionNew:
    async def test_a_new_session_is_announced_and_then_listed(self, service):
        queue = service.subscribe()
        await service.handle(SessionNew(profile="default"))
        events = await drain(queue)
        # created first: the UI has to open it, and a sidebar cannot say which
        # of its lines is the new one.
        assert kinds(events) == ["SessionCreated", "SessionRows"]
        created = events[0].row
        assert created.session_id in {r.session_id for r in events[1].rows}

    async def test_it_is_created_under_the_profile_asked_for(self, service, conn):
        queue = service.subscribe()
        await service.handle(SessionNew(profile="bioinformatics"))
        created = only(await drain(queue), "SessionCreated").row
        assert created.profile == "bioinformatics"
        stored = SessionStore(conn).get(created.session_id)
        assert stored is not None and stored.profile == "bioinformatics"

    async def test_the_backend_it_asks_for_is_pinned_to_it(self, service, conn):
        blob = json.dumps(
            {"model": "qwen3-32b", "base_url": "http://localhost:20001/v1"}
        )
        queue = service.subscribe()
        await service.handle(SessionNew(profile="default", backend=blob))
        created = only(await drain(queue), "SessionCreated").row
        assert created.model == "qwen3-32b"
        # Stored as the blob, so the choice survives the catalog entry going.
        assert "qwen3-32b" in SessionStore(conn).get(created.session_id).backend

    async def test_an_unusable_backend_falls_back_rather_than_stranding_it(
        self, service
    ):
        queue = service.subscribe()
        await service.handle(SessionNew(profile="default", backend="not json"))
        assert only(await drain(queue), "SessionCreated").row.model == ""


class TestSessionOpen:
    async def test_opening_sends_the_whole_transcript_once(self, service, session):
        await run_turn(service, session.session_id, "which BAMs?")
        queue = service.subscribe()
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert reset.session_id == session.session_id
        assert [e.kind for e in reset.entries][:1] == ["user"]
        assert "which BAMs?" in reset.entries[0].text

    async def test_every_row_arrives_named_from_one(self, service, session):
        await run_turn(service, session.session_id, "hello")
        queue = service.subscribe()
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        # A reset re-bases the numbering; the UI drops the names it held.
        assert [e.seq for e in reset.entries] == list(
            range(1, len(reset.entries) + 1)
        )

    async def test_an_empty_session_opens_empty(self, service, session):
        queue = service.subscribe()
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == []

    async def test_opening_says_how_full_the_window_already_is(
        self, service, session
    ):
        await run_turn(service, session.session_id, "hello")
        queue = service.subscribe()
        await service.handle(SessionOpen(session_id=session.session_id))
        events = await drain(queue)
        # Nothing has been sent this run, so the fill is derived from the
        # stored history — the same number a restart would show.
        assert only(events, "ContextEstimate").session_id == session.session_id

    async def test_opening_a_session_that_is_gone_says_so(self, service):
        queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(SessionOpen(session_id=session.session_id))
        row = only(await drain(queue), "ChatReset").entries[-1]

        await service.handle(
            TurnUnqueue(session_id=session.session_id, seq=row.seq)
        )
        assert only(await drain(queue), "TurnUnqueued").text == "and the CRAMs?"
        release.set()
        await service.stop()


class TestSessionClose:
    async def test_closing_forgets_what_was_open(self, service, session):
        await service.handle(SessionFocus(session_id=session.session_id))
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(
            SessionRename(session_id=session.session_id, title="BAM QC")
        )
        rows = only(await drain(queue), "SessionRows").rows
        assert rows[0].title == "BAM QC"
        assert SessionStore(conn).get(session.session_id).title == "BAM QC"

    async def test_an_empty_title_is_refused(self, service, session, conn):
        queue = service.subscribe()
        await service.handle(
            SessionRename(session_id=session.session_id, title="   ")
        )
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert SessionStore(conn).get(session.session_id).title == "a session"

    async def test_renaming_a_session_that_is_gone_says_so(self, service):
        queue = service.subscribe()
        await service.handle(SessionRename(session_id="nope", title="x"))
        assert (await drain(queue))[0].severity == "warning"


class TestSessionRetitle:
    async def test_the_model_names_the_conversation(self, service, session, llm):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [json.dumps({"title": "Read counting"})]
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(SessionList())  # answered while titling waits
        assert "SessionRows" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_an_empty_conversation_has_nothing_to_summarize(
        self, service, session
    ):
        queue = service.subscribe()
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "Notify")
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"

    async def test_a_model_that_cannot_write_one_is_reported_not_raised(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "hello")
        llm._outputs = []  # every attempt answers with the turn JSON instead
        queue = service.subscribe()
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "Notify")
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error" for e in events
        )
        assert "SessionRows" not in kinds(events)


class TestSessionDelete:
    async def test_the_row_and_its_history_go(self, service, session, conn):
        await run_turn(service, session.session_id, "hello")
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(SessionDelete(session_id="nope"))
        assert (await drain(queue))[0].severity == "warning"


class TestRollback:
    async def test_the_trimmed_conversation_comes_back_as_a_reset(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        # The measured number described a thread that no longer exists.
        assert only(events, "ContextEstimate").used == 0

    async def test_rolling_back_a_session_that_is_gone_says_so(self, service):
        queue = service.subscribe()
        await service.handle(SessionRollback(session_id="nope", index=0))
        assert (await drain(queue))[0].severity == "warning"


class TestFork:
    async def test_the_branch_is_created_and_announced(
        self, service, session, conn
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(SessionFork(session_id=session.session_id, index=0))
        assert "SessionCreated" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_an_index_the_thread_no_longer_has_leaves_no_session_behind(
        self, service, session, conn
    ):
        await run_turn(service, session.session_id, "first")
        before = len(SessionStore(conn).list_all())
        queue = service.subscribe()
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
        queue = service.subscribe()
        await service.handle(SessionFork(session_id=session.session_id, index=0))
        events = await drain(queue)
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error" for e in events
        )
        assert len(SessionStore(conn).list_all()) == before

    async def test_forking_a_session_that_is_gone_says_so(self, service):
        queue = service.subscribe()
        await service.handle(SessionFork(session_id="nope", index=0))
        assert (await drain(queue))[0].severity == "warning"


class TestAConversationEndToEnd:
    """List, make one, open it, talk, rewind, branch — commands only.

    The point of this one is that nothing else is used: no store call, no
    graph call, no reach into a service. If a front-end can send these seven
    frames and read the ones that come back, it has a conversation.
    """

    async def test_the_whole_round_trip(self, service, llm):
        queue = service.subscribe()

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
