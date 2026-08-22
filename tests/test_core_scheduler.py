"""The turn scheduler's invariants, without a graph or a widget in sight.

The point of extracting this from `HpcaApp` was that its rules — one turn per
session, many sessions at once, a parked approval blocks only its own queue —
were only ever reachable through a running Textual app. Here they are asserted
against a two-line fake, which is the argument for the extraction in one file.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from hpca.agent.graph import TurnResult
from hpca.core import scheduler as scheduler_module
from hpca.core.deps import CoreDeps
from hpca.core.scheduler import TurnPlan, TurnScheduler
from hpca.protocol import Entry


class FakeSession:
    def __init__(self, session_id: str, title: str = "a session") -> None:
        self.session_id = session_id
        self.title = title
        self.profile = "default"


@pytest.fixture
def sessions():
    return {"s1": FakeSession("s1"), "s2": FakeSession("s2", "the other one")}


@pytest.fixture
def events():
    return []


@pytest.fixture
def deps(events, tmp_path):
    async def _db(fn):
        raise AssertionError("the scheduler must not touch the database")

    return CoreDeps(
        settings=None,
        app_dir=tmp_path,
        db=_db,
        emit=events.append,
    )


class FakeGraph:
    """The one thing the scheduler asks of a graph object directly: what a
    thread holds. Everything else goes through the module-level entry points
    `graph_calls` replaces, which is why this is a single method."""

    def __init__(self, state: dict) -> None:
        self._state = state

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return SimpleNamespace(values=self._state.get(thread_id))


@pytest.fixture
def graph_calls(monkeypatch):
    """Replace the four graph entry points the scheduler uses.

    `run_turn` is driven per session id: a test pushes the result it wants,
    or an asyncio.Event to park on so it can assert what happens mid-turn.
    """
    calls = {
        "run_turn": [],
        "delivered": [],
        "stopped": [],
        "results": {},
        "gates": {},
        "counts": {},
        # What each thread holds, for the one read that is not a call:
        # `FakeGraph.aget_state`.
        "state": {},
    }

    async def fake_run_turn(graph, *, session_id, user_text=None, resume=None,
                            api_content=None):
        calls["run_turn"].append(
            {"session_id": session_id, "user_text": user_text, "resume": resume,
             "api_content": api_content}
        )
        if user_text is not None:
            # The message reaches the thread the moment the turn starts, and
            # the scheduler asks the thread how long it is to find out whether
            # a stopped turn left anything behind. A count that never moved
            # would make every stop here look like one that landed before its
            # message did (`TurnScheduler._keep_stopped_work`).
            calls["counts"][session_id] = calls["counts"].get(session_id, 3) + 1
        gate = calls["gates"].get(session_id)
        if gate is not None:
            await gate.wait()
        result = calls["results"].get(session_id)
        if isinstance(result, Exception):
            raise result
        return result or TurnResult(reply="done", interrupt=None)

    async def fake_thread_message_count(graph, *, session_id):
        return calls["counts"].get(session_id, 3)

    async def fake_stop_thread(graph, *, session_id):
        calls["stopped"].append(session_id)

    async def fake_deliver_event(graph, *, session_id, text):
        calls["delivered"].append((session_id, text))

    monkeypatch.setattr(scheduler_module, "run_turn", fake_run_turn)
    monkeypatch.setattr(
        scheduler_module, "thread_message_count", fake_thread_message_count
    )
    monkeypatch.setattr(scheduler_module, "stop_thread", fake_stop_thread)
    monkeypatch.setattr(scheduler_module, "deliver_event", fake_deliver_event)
    return calls


@pytest.fixture
def sched(deps, sessions, graph_calls):
    return TurnScheduler(
        deps,
        graph=FakeGraph(graph_calls["state"]),
        prepare=lambda session, *, user_text=None, forced_skill=None: TurnPlan(
            api_content=f"api:{user_text}" if user_text else None
        ),
        session_for=sessions.get,
    )


def kinds(events):
    return [type(e).__name__ for e in events]


def reset_rows(count):
    """A `chat.reset`'s entries, as far as `rebase_rows` reads them.

    Numbered from 1 and carrying the message index each row is, which is how
    the scheduler recognises a running turn's own message among them.
    """
    return [
        Entry(kind="user", text=f"m{i}", index=i, seq=i + 1) for i in range(count)
    ]


def queued_seqs(events, session_id="s1"):
    """The row names of the queued entries emitted for one session, in order."""
    return [
        e.entry.seq
        for e in events
        if type(e).__name__ == "ChatAppend"
        and e.session_id == session_id
        and e.entry.kind == "queued"
    ]


async def settle():
    """Let the drain that a finished turn schedules actually run."""
    for _ in range(6):
        await asyncio.sleep(0)


class TestSerialisation:
    async def test_a_second_message_waits_for_the_first(self, sched, graph_calls):
        # Two turns on one thread_id would interleave checkpoint writes.
        graph_calls["gates"]["s1"] = asyncio.Event()
        assert sched.submit_user("s1", "first") is False
        await sched.drain()
        await settle()  # let the turn task actually enter run_turn
        assert sched.submit_user("s1", "second") is True
        await sched.drain()
        await settle()
        assert len(graph_calls["run_turn"]) == 1
        assert sched.queued_texts_for("s1") == ["second"]

        graph_calls["gates"]["s1"].set()
        await settle()
        assert [c["user_text"] for c in graph_calls["run_turn"]] == ["first", "second"]

    async def test_another_session_starts_at_once(self, sched, graph_calls):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "slow")
        sched.submit_user("s2", "quick")
        await sched.drain()
        await settle()
        # One drain pass starts every free session, not just the first.
        assert {c["session_id"] for c in graph_calls["run_turn"]} == {"s1", "s2"}
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_busy_session_is_reported_busy(self, sched, graph_calls):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "x")
        await sched.drain()
        assert sched.is_busy("s1") and not sched.is_busy("s2")
        graph_calls["gates"]["s1"].set()
        await settle()


class TestQueuedRows:
    """The queued-message channel: the row, its name, and taking it back.

    In the Textual front-end the queue was UI state; here the scheduler owns
    it, so everything the user can see or do about it has to be emitted. The
    row's `Entry.seq` is what ties the three frames together — the append that
    draws it, the update that promotes it, the unqueue that removes it.
    """

    async def start_and_queue(self, sched, graph_calls, *texts):
        """Park a turn on s1 and queue ``texts`` behind it.

        Their row names come out of the emitted events (`queued_seqs`), which
        is also the only way the front-end can learn them.
        """
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "running")
        await sched.drain()
        await settle()
        for text in texts:
            sched.submit_user("s1", text)

    async def test_a_message_typed_during_a_turn_gets_a_row_of_its_own(
        self, sched, events, graph_calls
    ):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "running")
        await sched.drain()
        await settle()
        events.clear()
        assert sched.submit_user("s1", "second") is True

        appended = [e for e in events if type(e).__name__ == "ChatAppend"]
        assert len(appended) == 1
        entry = appended[0].entry
        assert (entry.kind, entry.text) == ("queued", "second")
        assert entry.seq >= 1  # named, so an update can find it later
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_message_that_starts_at_once_gets_no_queued_row(
        self, sched, events
    ):
        # Nothing is waiting, so the turn starts — and a "queued" row would
        # have to be un-drawn a moment later. It is still drawn: a turn's own
        # message is a chat row whether or not it had to wait for one.
        assert sched.submit_user("s1", "hello") is False
        await sched.drain()
        await settle()
        appended = [e for e in events if type(e).__name__ == "ChatAppend"]
        assert [(e.entry.kind, e.entry.text) for e in appended] == [
            ("user", "hello")
        ]

    async def test_every_row_gets_its_own_name_per_session(
        self, sched, events, graph_calls
    ):
        graph_calls["gates"]["s1"] = asyncio.Event()
        graph_calls["gates"]["s2"] = asyncio.Event()
        for sid in ("s1", "s2"):
            sched.submit_user(sid, "running")
        await sched.drain()
        await settle()
        events.clear()
        for sid in ("s1", "s2"):
            sched.submit_user(sid, "second")
            sched.submit_user(sid, "third")

        rows = [e for e in events if type(e).__name__ == "ChatAppend"]
        per_session: dict[str, list[int]] = {}
        for row in rows:
            per_session.setdefault(row.session_id, []).append(row.entry.seq)
        # Monotonic and per session: two conversations number their own rows
        # and never have to agree with each other.
        assert all(seqs == sorted(set(seqs)) for seqs in per_session.values())
        assert set(per_session) == {"s1", "s2"}
        for gate in ("s1", "s2"):
            graph_calls["gates"][gate].set()
        await settle()

    async def test_the_row_becomes_a_user_row_when_its_turn_starts(
        self, sched, events, graph_calls
    ):
        # The promotion is an update to the same row, not a second row and a
        # guess: `chat.update` carries the seq the append gave it.
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "running")
        await sched.drain()
        await settle()
        events.clear()
        sched.submit_user("s1", "second")
        seq = [e for e in events if type(e).__name__ == "ChatAppend"][0].entry.seq

        events.clear()
        graph_calls["gates"]["s1"].set()
        await settle()
        updates = [e for e in events if type(e).__name__ == "ChatUpdate"]
        assert len(updates) == 1
        assert (updates[0].entry.seq, updates[0].entry.kind) == (seq, "user")
        assert updates[0].entry.text == "second"
        # ...and it lands before the turn it belongs to is announced, so the
        # row is never drawn as waiting behind a turn that is already itself.
        assert kinds(events).index("ChatUpdate") < kinds(events).index("TurnStarted")

    async def test_a_queued_message_comes_back_out_with_its_text(
        self, sched, events, graph_calls
    ):
        await self.start_and_queue(sched, graph_calls, "second", "third")
        seqs = queued_seqs(events)
        assert sched.unqueue("s1", seqs[1]) == "third"
        assert sched.queued_texts_for("s1") == ["second"]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_two_identical_messages_are_two_rows(
        self, sched, events, graph_calls
    ):
        # The reason the wire carries a row name and not the text: cancelling
        # by value would take both copies, or the wrong one.
        await self.start_and_queue(sched, graph_calls, "same", "same")
        first, second = queued_seqs(events)
        assert first != second
        assert sched.unqueue("s1", first) == "same"
        assert sched.queued_texts_for("s1") == ["same"]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_row_that_is_no_longer_queued_is_refused(
        self, sched, events, graph_calls
    ):
        # The turn ahead can finish while the dialog is open. A row name that
        # no longer names a waiting message is None — "too late, it is already
        # running" — and never a neighbour cancelled by accident, which is
        # exactly what a queue *position* would have become.
        await self.start_and_queue(sched, graph_calls, "second")
        seq = queued_seqs(events)[0]
        assert sched.unqueue("s1", seq) == "second"
        assert sched.unqueue("s1", seq) is None  # gone, and stays gone
        assert sched.unqueue("s1", 999) is None
        assert sched.unqueue("s1", 0) is None  # 0 is "unnamed", never a row
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_row_belongs_to_the_session_that_queued_it(
        self, sched, events, graph_calls
    ):
        graph_calls["gates"]["s2"] = asyncio.Event()
        sched.submit_user("s2", "running")
        await sched.drain()
        await settle()
        await self.start_and_queue(sched, graph_calls, "second")
        seq = queued_seqs(events, session_id="s1")[0]
        # Two sessions number independently, so the same seq exists in both.
        sched.submit_user("s2", "another session's")
        assert sched.unqueue("s2", seq) != "second"
        assert sched.queued_texts_for("s1") == ["second"]
        for gate in ("s1", "s2"):
            graph_calls["gates"][gate].set()
        await settle()

    async def test_a_background_completion_is_not_a_queued_row(
        self, sched, events, graph_calls
    ):
        # Nobody typed it and no row was ever drawn for it, so it must not be
        # retractable — and must not disturb the rows that are.
        await self.start_and_queue(sched, graph_calls, "second")
        events.clear()
        sched.submit_event("s1", "a job finished")
        assert "ChatAppend" not in kinds(events)
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_unqueueing_from_a_session_with_no_queue_is_harmless(self, sched):
        assert sched.unqueue("s1", 1) is None


class TestLiveSteps:
    """A tool call as a row, and the result landing in that same row.

    The scheduler half of it: what `on_step` does to the chat, without a graph
    to produce the payloads. The end-to-end version (a real graph, a real tool)
    is in `test_core_service.py`.
    """

    async def working(self, sched, graph_calls):
        """A turn parked on the model, with its rows already drawn."""
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "what is in the BAM?")
        await sched.drain()
        await settle()

    def rows(self, events, kind=None):
        return [
            e.entry
            for e in events
            if type(e).__name__ in ("ChatAppend", "ChatUpdate")
            and (kind is None or type(e).__name__ == kind)
        ]

    async def test_a_call_is_drawn_the_moment_it_is_made(
        self, sched, events, graph_calls
    ):
        await self.working(sched, graph_calls)
        events.clear()
        sched.report_step("s1", {"kind": "call", "tool": "read_file"})

        assert kinds(events) == ["ChatAppend"]
        entry = events[0].entry
        assert entry.kind == "thinking" and entry.seq > 0
        # Nothing in the result half: the tool has not answered yet, and a row
        # that showed an empty result would read as a tool that answered with
        # nothing.
        assert [(p.kind, p.tool, p.done) for p in entry.parts] == [
            ("call", "read_file", False)
        ]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_the_result_fills_the_row_the_call_drew(
        self, sched, events, graph_calls
    ):
        await self.working(sched, graph_calls)
        events.clear()
        sched.report_step("s1", {"kind": "call", "tool": "read_file"})
        drawn = events[0].entry.seq
        sched.report_step(
            "s1", {"kind": "step", "text": "[tool result] read_file: 40 lines"}
        )

        # An update, not a second row: one exchange is one row (`Part.done`).
        assert kinds(events) == ["ChatAppend", "ChatUpdate"]
        entry = events[-1].entry
        assert entry.seq == drawn
        assert [(p.done, p.result) for p in entry.parts] == [(True, "40 lines")]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_two_calls_are_answered_in_the_order_they_were_asked(
        self, sched, events, graph_calls
    ):
        await self.working(sched, graph_calls)
        for tool in ("read_file", "list_scripts"):
            sched.report_step("s1", {"kind": "call", "tool": tool})
        events.clear()
        sched.report_step("s1", {"kind": "step", "text": "[tool result] x: first"})

        # The earliest call still waiting takes it — the same rule the fold
        # uses, so a re-open cannot pair them up differently.
        parts = events[-1].entry.parts
        assert [(p.tool, p.result) for p in parts] == [
            ("read_file", "first"),
            ("list_scripts", ""),
        ]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_step_for_a_session_with_no_turn_is_ignored(
        self, sched, events
    ):
        sched.report_step("s1", {"kind": "call", "tool": "read_file"})
        assert events == []


class TestRebasedRows:
    """What a `chat.reset` has to carry that the thread does not hold.

    Re-opening a session mid-turn is where this bites: the transcript in the
    checkpoint knows nothing about a message still waiting in the queue, and
    the front-end that used to re-add them from its own state has none.
    """

    async def test_the_counter_restarts_where_the_reset_ended(self, sched):
        # A reset renumbers from 1, so the next row the session mints must
        # follow the entries the reset carried, not the ones it replaced.
        sched.rebase_rows("s1", entries=reset_rows(7))
        assert sched._next_entry_seq("s1") == 8

    async def test_queued_messages_come_back_as_rows_after_the_transcript(
        self, sched, graph_calls
    ):
        graph_calls["gates"]["s1"] = asyncio.Event()
        graph_calls["counts"]["s1"] = 4
        sched.submit_user("s1", "running")
        await sched.drain()
        await settle()
        sched.submit_user("s1", "second")
        sched.submit_user("s1", "third")

        rows = sched.rebase_rows("s1", entries=reset_rows(5))
        # The running turn's message is in the copy — the reset carries an
        # entry for message 4, which is where the turn started — so only the
        # two waiting behind it are re-drawn, numbered after the transcript,
        # in the order they will run.
        assert [(r.kind, r.text, r.seq) for r in rows] == [
            ("queued", "second", 6),
            ("queued", "third", 7),
        ]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_re_drawn_queued_row_can_still_be_taken_back(
        self, sched, graph_calls
    ):
        # The whole point of renumbering the queue rather than only the
        # transcript: `turn.unqueue` names a row, and the name the UI now
        # holds is the one this reset gave it.
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "running")
        await sched.drain()
        await settle()
        sched.submit_user("s1", "second")
        old = sched._pending[-1].entry_seq

        row = sched.rebase_rows("s1", entries=reset_rows(2))[-1]
        assert row.seq != old, "the reset must hand out a fresh name"
        assert sched.unqueue("s1", old) is None
        assert sched.unqueue("s1", row.seq) == "second"
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_running_message_the_thread_has_not_stored_is_re_drawn(
        self, sched, graph_calls
    ):
        # The copy just read predates the turn's own message, which is what
        # `interrupt_keep` measures. Without this the user re-opens a working
        # session and their own sentence is missing from it.
        graph_calls["gates"]["s1"] = asyncio.Event()
        graph_calls["counts"]["s1"] = 4
        sched.submit_user("s1", "what is in the BAM?")
        await sched.drain()
        await settle()

        rows = sched.rebase_rows("s1", entries=reset_rows(4))
        assert [(r.kind, r.text) for r in rows] == [
            ("user", "what is in the BAM?")
        ]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_running_turns_rows_are_re_bound_to_the_resets_names(
        self, sched, graph_calls
    ):
        # The names the turn drew under are gone — the reset renumbered from 1.
        # If the turn kept them, the update carrying its next tool result would
        # address a row the front-end no longer has, and be dropped.
        graph_calls["gates"]["s1"] = asyncio.Event()
        graph_calls["counts"]["s1"] = 2
        sched.submit_user("s1", "what is in the BAM?")
        await sched.drain()
        await settle()
        sched.report_step("s1", {"kind": "call", "tool": "read_file"})

        entries = reset_rows(3) + [
            Entry(kind="thinking", text="", index=-1, seq=4)
        ]
        assert sched.rebase_rows("s1", entries=entries) == []
        live = sched._live["s1"]
        # The turn began at message 2, so the reset's third row is its message
        # and everything after it belongs to the turn as well.
        assert live.rows == [3, 4]
        assert live.thinking_seq == 4
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_turns_working_box_comes_back_with_its_message(
        self, sched, graph_calls
    ):
        # The copy predates the turn, so neither its message nor the call it
        # has already announced is in the reset. A call is announced before it
        # returns and its exchange is checkpointed only after — so a re-open in
        # that window would otherwise show a session doing nothing.
        graph_calls["gates"]["s1"] = asyncio.Event()
        graph_calls["counts"]["s1"] = 4
        sched.submit_user("s1", "what is in the BAM?")
        await sched.drain()
        await settle()
        sched.report_step("s1", {"kind": "call", "tool": "read_file"})

        rows = sched.rebase_rows("s1", entries=reset_rows(4))
        assert [(r.kind, r.seq) for r in rows] == [("user", 5), ("thinking", 6)]
        assert [(p.kind, p.tool, p.done) for p in rows[1].parts] == [
            ("call", "read_file", False)
        ]
        assert sched._live["s1"].rows == [5, 6]
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_an_idle_session_gets_nothing_extra(self, sched):
        assert sched.rebase_rows("s1", entries=reset_rows(3)) == []

    async def test_one_sessions_reset_leaves_another_alone(
        self, sched, graph_calls
    ):
        graph_calls["gates"]["s2"] = asyncio.Event()
        sched.submit_user("s2", "running")
        await sched.drain()
        await settle()
        sched.submit_user("s2", "second")
        before = sched._pending[-1].entry_seq

        sched.rebase_rows("s1", entries=reset_rows(9))
        assert sched._pending[-1].entry_seq == before
        graph_calls["gates"]["s2"].set()
        await settle()


class TestEvents:
    async def test_a_turn_announces_its_start_and_its_end(self, sched, events):
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        names = kinds(events)
        # The message is on screen before the spinner that belongs to it.
        assert names[:2] == ["ChatAppend", "TurnStarted"]
        assert names[-1] == "TurnFinished"
        assert events[-1].reply == "done"

    async def test_the_start_says_when_the_turn_began(
        self, sched, events, graph_calls
    ):
        # The elapsed clock a user reads is "how long since I sent it", so it
        # starts with the turn rather than with the first thing the turn gets
        # round to reporting — a wait on the backend's first answer is the
        # longest silence there is, and it is not free.
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "hello")
        await sched.drain()
        started = [e for e in events if type(e).__name__ == "TurnStarted"][0]
        activity = [e for e in events if type(e).__name__ == "TurnActivity"][0]
        datetime.fromisoformat(started.started_at)  # ISO 8601, as everything is
        # The same stamp the activity carries: two clocks for one turn would
        # disagree by however long the first round took.
        assert started.started_at == activity.started_at
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_the_clock_stops_when_the_turn_does(self, sched, events):
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        # An empty activity is how the renderer learns to drop the spinner.
        assert any(
            type(e).__name__ == "TurnActivity" and e.activity == "" for e in events
        )

    async def test_activity_is_reported_against_its_own_session(
        self, sched, events, graph_calls
    ):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "x")
        await sched.drain()
        events.clear()
        sched.report_activity("s1", "running read_file")
        assert events[0].session_id == "s1"
        assert events[0].activity == "running read_file"
        assert sched.activity_of("s1") == "running read_file"
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_activity_for_a_session_with_no_turn_is_ignored(self, sched, events):
        sched.report_activity("s1", "running read_file")
        assert events == []

    async def test_a_failed_turn_says_so_and_does_not_stall_the_queue(
        self, sched, events, graph_calls
    ):
        graph_calls["results"]["s1"] = RuntimeError("backend refused")
        sched.submit_user("s1", "x")
        await sched.drain()
        await settle()
        assert "TurnFailed" in kinds(events)
        assert events[[type(e).__name__ for e in events].index("TurnFailed")].error == (
            "backend refused"
        )
        assert not sched.is_busy("s1")


class TestAFailedTurnsRows:
    """What is left on screen when a turn breaks instead of finishing.

    There is no result to fold, so without this the rows settle exactly as the
    live path drew them — a working box whose last call is still spinning for
    a result that is never coming, and which a re-opened session would draw
    differently. The thread is the answer: what the graph checkpointed before
    it broke is what a `chat.reset` will show.
    """

    async def failing_turn(self, sched, graph_calls, *, state):
        """A turn that draws a call, then breaks. Returns once it has."""
        graph_calls["counts"]["s1"] = 0  # the thread starts empty
        graph_calls["gates"]["s1"] = asyncio.Event()
        graph_calls["results"]["s1"] = RuntimeError("connection refused")
        graph_calls["state"]["s1"] = state
        sched.submit_user("s1", "how many reads?")
        await sched.drain()
        await settle()
        sched.report_step("s1", {"kind": "call", "tool": "read_file"})
        graph_calls["gates"]["s1"].set()
        await settle()

    def state_with_a_finished_call(self):
        """A thread that got its tool result and its reasoning checkpointed —
        and then lost the backend on the round that would have answered."""
        return {
            "messages": [
                {"role": "user", "content": "how many reads?"},
                {"role": "user", "content": "[tool result] read_file: 40 lines"},
            ],
            "thinking": [{"after": 1, "reasoning": "column 2 holds the counts"}],
            "calls": [{"after": 1, "tool": "read_file", "arguments": {}}],
        }

    async def test_the_rows_are_re_stated_from_what_the_thread_kept(
        self, sched, events, graph_calls
    ):
        await self.failing_turn(
            sched, graph_calls, state=self.state_with_a_finished_call()
        )
        box = [
            e.entry
            for e in events
            if type(e).__name__ == "ChatUpdate" and e.entry.kind == "thinking"
        ][-1]
        # The fold's account, not the live one: the call has its result, and
        # the reasoning that only ever existed in the checkpoint is there.
        assert [(p.kind, p.done) for p in box.parts] == [
            ("reasoning", False),
            ("call", True),
        ]
        assert box.parts[0].text == "column 2 holds the counts"

    async def test_the_failure_itself_is_not_a_row(
        self, sched, events, graph_calls
    ):
        # It never entered the thread, so it travels as `turn.failed` — which
        # is also what stops the spinner — and a front-end draws it as an
        # ephemeral error entry of its own (§3.2). A row from here as well
        # would put two of them on screen.
        await self.failing_turn(
            sched, graph_calls, state=self.state_with_a_finished_call()
        )
        drawn = [
            e.entry.kind
            for e in events
            if type(e).__name__ in ("ChatAppend", "ChatUpdate")
        ]
        assert "error" not in drawn
        failure = events[[type(e).__name__ for e in events].index("TurnFailed")]
        assert "connection refused" in failure.error
        assert any(
            type(e).__name__ == "TurnActivity" and e.activity == "" for e in events
        )

    async def test_a_thread_it_cannot_read_still_reports_the_failure(
        self, sched, events, graph_calls
    ):
        # A backend that died may have taken more with it. The rows then stand
        # as they were drawn — there is nothing better to say — and the
        # failure still reaches the user.
        await self.failing_turn(sched, graph_calls, state=None)
        assert "TurnFailed" in kinds(events)

    async def test_the_next_turn_does_not_revise_the_failed_ones_rows(
        self, sched, events, graph_calls
    ):
        # The record is closed with the turn: those rows are settled.
        await self.failing_turn(
            sched, graph_calls, state=self.state_with_a_finished_call()
        )
        assert "s1" not in sched._live


class TestRewindGate:
    """When a thread may not be truncated (`session.rollback`).

    The core is the only side that can answer this: every one of these states
    lives here, and a front-end asking "is it safe to roll back?" from what it
    last drew would be answering from a screenshot.

    `session.fork` asks nothing of this gate — see `rewind_blocker`: the
    destructive half is gated, the recoverable one is not.
    """

    async def test_a_settled_session_can_be_rolled_back(self, sched):
        assert sched.rewind_blocker("s1") is None

    async def test_not_while_a_turn_is_running(self, sched, graph_calls):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "x")
        await sched.drain()
        await settle()
        assert "a turn is running" in (sched.rewind_blocker("s1") or "")
        # Only its own session: another conversation is free to be cut while
        # this one works.
        assert sched.rewind_blocker("s2") is None
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_not_while_a_decision_is_unanswered(self, sched, graph_calls):
        # The resume would land on message indices the cut had removed.
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "run_bash", "kind": "destructive"}
        )
        sched.submit_user("s1", "rm the scratch dir")
        await sched.drain()
        await settle()
        assert "decision" in (sched.rewind_blocker("s1") or "")

    async def test_not_while_a_message_is_still_queued(self, sched):
        # It becomes a turn the moment the session is free, and would then
        # append past the cut.
        sched.submit_user("s1", "typed ahead")
        assert "queued" in (sched.rewind_blocker("s1") or "")

    async def test_the_gate_lifts_when_the_turn_ends(self, sched, graph_calls):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "x")
        await sched.drain()
        await settle()
        graph_calls["gates"]["s1"].set()
        await settle()
        assert sched.rewind_blocker("s1") is None


class TestApprovals:
    async def test_a_parked_turn_asks_and_blocks_only_its_own_queue(
        self, sched, events, graph_calls
    ):
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "run_bash", "kind": "destructive"}
        )
        sched.submit_user("s1", "rm the scratch dir")
        await sched.drain()
        await settle()
        assert "DecisionRequested" in kinds(events)

        # Its own queue waits...
        sched.submit_user("s1", "actually, wait")
        await sched.drain()
        assert sched.queued_texts_for("s1") == ["actually, wait"]
        # ...and another session's does not.
        sched.submit_user("s2", "unrelated")
        await sched.drain()
        await settle()
        assert any(c["session_id"] == "s2" for c in graph_calls["run_turn"])

    async def test_a_decision_survives_for_a_client_that_arrives_later(
        self, sched, graph_calls
    ):
        # The bug this fixes: the pending decision used to be UI-process
        # memory, so a restart left the session parked with nothing to answer.
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "delete_path"}
        )
        sched.submit_user("s1", "delete it")
        await sched.drain()
        await settle()
        assert sched.pending_decisions()["s1"]["tool"] == "delete_path"

    async def test_answering_resumes_the_turn_and_clears_the_prompt(
        self, sched, events, graph_calls
    ):
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "delete_path"}
        )
        sched.submit_user("s1", "delete it")
        await sched.drain()
        await settle()
        graph_calls["results"]["s1"] = TurnResult(reply="deleted", interrupt=None)
        events.clear()

        assert sched.resolve_decision("s1", approved=True) is True
        await settle()
        assert "DecisionCleared" in kinds(events)
        assert sched.pending_decisions() == {}
        assert graph_calls["run_turn"][-1]["resume"] is not None

    async def test_a_refusal_carries_its_reason_back(self, sched, graph_calls):
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "delete_path"}
        )
        sched.submit_user("s1", "delete it")
        await sched.drain()
        await settle()
        graph_calls["results"]["s1"] = TurnResult(reply="ok", interrupt=None)

        sched.resolve_decision("s1", approved=False, reason="that is the real data")
        await settle()
        resume = graph_calls["run_turn"][-1]["resume"]
        assert resume.resume["approved"] is False
        assert resume.resume["reason"] == "that is the real data"

    async def test_answering_a_decision_nobody_is_waiting_on_does_nothing(self, sched):
        assert sched.resolve_decision("s1", approved=True) is False

    async def test_the_queue_runs_once_the_decision_is_answered(
        self, sched, graph_calls
    ):
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "delete_path"}
        )
        sched.submit_user("s1", "delete it")
        await sched.drain()
        await settle()
        sched.submit_user("s1", "and then this")
        await sched.drain()
        graph_calls["results"]["s1"] = TurnResult(reply="ok", interrupt=None)

        sched.resolve_decision("s1", approved=True)
        await settle()
        assert [c["user_text"] for c in graph_calls["run_turn"]][-1] == "and then this"


class TestAResumeFindsItsRows:
    """Which rows the answer to an approval revises.

    A parked exchange and its resume are one working box, so the resume has to
    know the names of the rows that box was drawn as. Two ways those names can
    change under it: the session is re-opened while it waits (a `chat.reset`
    renumbers every row), or the turn that parked belonged to a core that has
    since restarted and left no record at all.
    """

    # The reset a re-opened session gets while the exchange is parked: two
    # settled rows, then the message this exchange is about and its box.
    def reopened(self):
        return [
            Entry(kind="user", text="earlier", index=0, seq=1),
            Entry(kind="assistant", text="answered", index=1, seq=2),
            Entry(kind="user", text="clear the scratch dir", index=2, seq=3),
            Entry(kind="thinking", text="", index=-1, seq=4),
        ]

    def resumed_result(self):
        """What the graph returns once the answer goes back in: the whole
        thread, with the exchange's call answered and the reply after it."""
        return TurnResult(
            reply="left it alone",
            interrupt=None,
            messages=[
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "answered"},
                {"role": "user", "content": "clear the scratch dir"},
                {"role": "user", "content": "[tool result] run_bash: SKIPPED"},
                {"role": "assistant", "content": "left it alone"},
            ],
            calls=[{"after": 3, "tool": "run_bash", "arguments": {}}],
            first_new=4,
        )

    async def park(self, sched, graph_calls):
        graph_calls["counts"]["s1"] = 2  # two messages settled before this one
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "run_bash", "kind": "execution"}
        )
        sched.submit_user("s1", "clear the scratch dir")
        await sched.drain()
        await settle()

    def revised(self, events):
        return [
            e.entry.seq for e in events if type(e).__name__ == "ChatUpdate"
        ]

    async def test_a_reset_while_parked_re_binds_the_box_the_resume_fills(
        self, sched, events, graph_calls
    ):
        # The turn ended when it parked, so there is no `TurnState` to find it
        # by — but its rows are still on screen and still its own.
        await self.park(sched, graph_calls)
        sched.rebase_rows("s1", entries=self.reopened())
        graph_calls["results"]["s1"] = self.resumed_result()
        events.clear()

        sched.resolve_decision("s1", approved=False)
        await settle()
        # The reset's names, not the ones the live path handed out before it.
        assert self.revised(events) == [3, 4]

    async def test_a_resume_after_a_restart_draws_into_the_rows_on_screen(
        self, sched, events, graph_calls
    ):
        # The parked half belonged to a process that is gone: nothing here
        # drew these rows. What is left to go on is the `chat.reset` that put
        # the prompt on screen in the first place — without it the reply
        # appears only at the next reset, which is the gap §4.2 records.
        sched._decisions["s1"] = {"tool": "run_bash", "kind": "execution"}
        sched._awaiting_approval.add("s1")
        sched.rebase_rows("s1", entries=self.reopened())
        graph_calls["results"]["s1"] = self.resumed_result()
        events.clear()

        sched.resolve_decision("s1", approved=False)
        await settle()
        assert self.revised(events) == [3, 4]
        # And the reply is a row of its own, after the box — not a second copy
        # of the exchange.
        appended = [
            e.entry for e in events if type(e).__name__ == "ChatAppend"
        ]
        assert [(e.kind, e.seq) for e in appended] == [("assistant", 5)]

    async def test_with_nothing_on_screen_it_still_says_nothing(
        self, sched, events, graph_calls
    ):
        # No reset, so no row names: a UI that has never been shown this
        # session gets the reply from its next `chat.reset`. Guessing rows
        # from the resume alone would draw a second working box under one the
        # client may or may not have.
        sched._decisions["s1"] = {"tool": "run_bash", "kind": "execution"}
        sched._awaiting_approval.add("s1")
        graph_calls["results"]["s1"] = self.resumed_result()
        events.clear()

        sched.resolve_decision("s1", approved=False)
        await settle()
        assert self.revised(events) == []
        assert [
            e.entry.seq for e in events if type(e).__name__ == "ChatAppend"
        ] == []


class TestBackgroundEvents:
    async def test_the_open_session_reacts_at_once(self, sched, deps, graph_calls):
        deps.focused_session_id = "s1"
        sched.submit_event("s1", "[process finished] qc exited 0.")
        await sched.drain()
        await settle()
        assert graph_calls["run_turn"][-1]["user_text"].startswith("[process finished]")
        assert graph_calls["delivered"] == []

    async def test_a_session_the_user_left_only_gets_the_message(
        self, sched, deps, graph_calls, events
    ):
        deps.focused_session_id = "s2"
        sched.submit_event("s1", "[process failed] qc exited 1.")
        await sched.drain()
        await settle()
        # Written into the thread for the next turn to find — free, and it
        # never yanks the user back into a session they left.
        assert graph_calls["delivered"] == [("s1", "[process failed] qc exited 1.")]
        assert graph_calls["run_turn"] == []
        assert "Notify" in kinds(events)

    async def test_an_event_for_a_session_that_is_gone_still_reaches_the_thread(
        self, sched, graph_calls
    ):
        sched.submit_event("ghost", "[process finished]")
        await sched.drain()
        await settle()
        assert graph_calls["delivered"] == [("ghost", "[process finished]")]

    async def test_a_submitted_event_drains_itself(self, sched, deps, graph_calls):
        # A poll's cadence must not decide a turn's timing, so the poller does
        # not drain and the scheduler does. Note: no explicit drain() here.
        deps.focused_session_id = "s1"
        sched.submit_event("s1", "[process finished] qc exited 0.")
        await settle()
        assert len(graph_calls["run_turn"]) == 1

    async def test_a_burst_of_completions_costs_one_drain(self, sched, deps):
        deps.focused_session_id = None
        for i in range(12):
            sched.submit_event("s2", f"[process finished] job{i}")
        # One coalesced task, not twelve each walking the same queue.
        assert sched._drain_task is not None
        first = sched._drain_task
        sched.submit_event("s2", "[process finished] job12")
        assert sched._drain_task is first
        await settle()

    async def test_shutdown_stops_a_late_completion_starting_a_turn(
        self, sched, deps, graph_calls
    ):
        deps.focused_session_id = "s1"
        await sched.shutdown()
        sched.submit_event("s1", "[process finished] straggler")
        await settle()
        assert graph_calls["run_turn"] == []


class TestInterrupt:
    async def _park_on_the_model(self, sched, graph_calls, session_id="s1"):
        graph_calls["gates"][session_id] = asyncio.Event()
        sched.submit_user(session_id, "a long question")
        await sched.drain()
        await settle()
        # The activity the graph reports while a turn waits on the backend.
        sched.report_activity(session_id, "LLM processing")

    async def test_in_any_phase_of_the_sessions_own_turn(self, sched, graph_calls):
        # Not only the wait on the model: a script that runs for minutes, or a
        # chain of tool rounds gone astray, is exactly what a user wants to
        # stop, and refusing there leaves them watching a spinner they cannot
        # answer (`tui/app.py:_can_interrupt`, and the acceptance list's "a
        # turn currently running a tool is interruptible").
        await self._park_on_the_model(sched, graph_calls)
        assert sched.can_interrupt("s1") is True
        sched.report_activity("s1", "running read_file")
        assert sched.can_interrupt("s1") is True

    async def test_never_for_a_session_with_no_turn(self, sched):
        assert sched.can_interrupt("s1") is False

    async def test_not_before_the_start_of_the_exchange_is_known(
        self, sched, graph_calls
    ):
        # The two things a stop needs are a message of the user's own and the
        # point in the thread the exchange began at — the number that says
        # whether any of it ever got as far as being written down. A turn a
        # fraction of a second old has only the first.
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "x")
        await sched.drain()
        sched._turns["s1"].exchange_start = None
        assert sched.can_interrupt("s1") is False
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_a_background_task_the_turn_started_outlives_it(
        self, sched, graph_calls, monkeypatch
    ):
        # Cancelling ends the turn, not what the turn started: a script's
        # monitor is a task of the core's (`runner.ProcessRunner.start`), not a
        # child of the turn's, so the cancellation never reaches the work
        # already in flight.
        started: dict = {}

        async def long_lived():
            await asyncio.Event().wait()

        async def spawns_then_parks(graph, *, session_id, **kwargs):
            started["task"] = asyncio.ensure_future(long_lived())
            await asyncio.Event().wait()  # and then waits, as a tool call does

        monkeypatch.setattr(scheduler_module, "run_turn", spawns_then_parks)
        sched.submit_user("s1", "start the long thing")
        await sched.drain()
        await settle()
        try:
            assert await sched.interrupt("s1") is not None
            await settle()
            assert not started["task"].done()
        finally:
            started["task"].cancel()

    async def test_a_stopped_turn_keeps_what_it_already_did(
        self, sched, graph_calls
    ):
        # The change the user asked for: "not the whole turn should be thrown
        # away, the agent should just be stopped". Nothing is rolled out of
        # the thread; what is written is the note that closes it off, so the
        # next turn does not read an unfinished history as an instruction to
        # finish it.
        graph_calls["counts"]["s1"] = 7
        await self._park_on_the_model(sched, graph_calls)
        stopped = await sched.interrupt("s1")
        assert stopped is not None
        assert graph_calls["stopped"] == ["s1"]
        assert not sched.is_busy("s1")

    async def test_and_does_not_hand_the_message_back(self, sched, graph_calls):
        # It is in the conversation now. Handing it to the entry box as well
        # would have the user send the same sentence twice without meaning to.
        await self._park_on_the_model(sched, graph_calls)
        assert (await sched.interrupt("s1")).text is None

    async def test_but_a_turn_that_never_reached_the_thread_gives_it_back(
        self, sched, graph_calls, monkeypatch
    ):
        # The window between announcing a turn and its message landing in the
        # thread. There is no exchange to keep, so the alternative to handing
        # the sentence back is losing it — and nothing is written under a turn
        # nobody can see.
        async def never_gets_there(graph, *, session_id, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(scheduler_module, "run_turn", never_gets_there)
        sched.submit_user("s1", "a question that never landed")
        await sched.drain()
        await settle()
        stopped = await sched.interrupt("s1")
        assert stopped.text == "a question that never landed"
        assert graph_calls["stopped"] == []

    async def test_stopping_a_turn_ends_it_for_the_screen(
        self, sched, graph_calls, events
    ):
        # `turn.finished` is the only thing a front-end reads as a turn
        # ending. Without one the working row spins forever on a turn that no
        # longer exists — and goes on offering to stop it.
        await self._park_on_the_model(sched, graph_calls)
        events.clear()
        await sched.interrupt("s1")
        assert "TurnFinished" in kinds(events)
        # The spinner is cleared before the turn is called over, the order the
        # finished path uses: a client reading the empty activity on its own
        # would otherwise end a turn that is about to be ended again.
        assert kinds(events).index("TurnActivity") < kinds(events).index(
            "TurnFinished"
        )

    async def test_interrupting_nothing_returns_nothing(self, sched):
        assert await sched.interrupt("s1") is None


class TestStoppableAcrossAnApproval:
    """An approval splits one exchange into two turns, and both must be
    stoppable by the same gesture.

    The resume carries no user message of its own — it is a `Command(resume=)`
    on a thread that already holds the message — so without the anchor the
    second half of every approved turn is a spinner nothing can answer, the
    start of the exchange having been thrown away with the first turn's state
    (`tui/app.py:_interrupt_anchor`).
    """

    async def _park_on_a_decision(self, sched, graph_calls):
        graph_calls["counts"]["s1"] = 4
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "run_bash", "kind": "execution"}
        )
        sched.submit_user("s1", "clear the scratch dir")
        await sched.drain()
        await settle()

    async def test_the_resumed_turn_carries_the_same_anchor(
        self, sched, graph_calls
    ):
        await self._park_on_a_decision(sched, graph_calls)
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.resolve_decision("s1", approved=True)
        await settle()
        ts = sched._turns["s1"]
        # The message that started the exchange, and the point in the thread
        # it started from — borrowed, not invented.
        assert (ts.user_text, ts.exchange_start) == ("clear the scratch dir", 4)
        assert sched.can_interrupt("s1") is True
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_stopping_the_resume_keeps_the_whole_exchange(
        self, sched, graph_calls
    ):
        await self._park_on_a_decision(sched, graph_calls)
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.resolve_decision("s1", approved=True)
        await settle()

        stopped = await sched.interrupt("s1")
        # The exchange began before the resume did, and all of it stays: the
        # call the user approved and answered for is exactly the work they
        # would be most annoyed to lose. Nothing comes back to the entry box —
        # the message that opened it is in the conversation.
        assert stopped.text is None
        assert graph_calls["stopped"] == ["s1"]
        assert not sched.is_busy("s1")

    async def test_the_anchor_is_dropped_once_the_exchange_ends(
        self, sched, graph_calls
    ):
        await self._park_on_a_decision(sched, graph_calls)
        assert sched._anchors.get("s1") is not None  # the exchange is not over
        graph_calls["results"]["s1"] = TurnResult(reply="done", interrupt=None)
        sched.resolve_decision("s1", approved=True)
        await settle()
        # Answered for good: there is no exchange left to stop, and the next
        # turn must not be handed the last one's message.
        assert sched._anchors == {}
        assert sched.can_interrupt("s1") is False

    async def test_a_failed_turn_drops_its_anchor_too(self, sched, graph_calls):
        graph_calls["results"]["s1"] = RuntimeError("backend refused")
        sched.submit_user("s1", "x")
        await sched.drain()
        await settle()
        assert sched._anchors == {}


class TestLifecycle:
    async def test_a_deleted_session_takes_its_queue_with_it(self, sched, graph_calls):
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "first")
        await sched.drain()
        sched.submit_user("s1", "queued behind it")
        sched.forget_session("s1")
        assert sched.queued_texts_for("s1") == []
        graph_calls["gates"]["s1"].set()
        await settle()

    async def test_forgetting_drops_a_parked_decision(self, sched, graph_calls):
        graph_calls["results"]["s1"] = TurnResult(reply=None, interrupt={"tool": "x"})
        sched.submit_user("s1", "x")
        await sched.drain()
        await settle()
        sched.forget_session("s1")
        assert sched.pending_decisions() == {}

    async def test_shutdown_drops_the_queue_and_unwinds_the_turn(
        self, sched, graph_calls
    ):
        # Quitting must not kick off what was still queued: the databases and
        # the checkpointer are about to close under it.
        graph_calls["gates"]["s1"] = asyncio.Event()
        sched.submit_user("s1", "running")
        await sched.drain()
        sched.submit_user("s2", "never starts")

        await sched.shutdown()
        assert sched.busy_sessions() == set()
        await sched.drain()
        assert not any(c["user_text"] == "never starts" for c in graph_calls["run_turn"])


class TestPostTurn:
    async def test_the_result_hook_sees_the_turn_and_its_plan(
        self, deps, sessions, graph_calls
    ):
        seen = []

        async def on_result(session, result, plan):
            seen.append((session.session_id, result.reply, plan.api_content))

        sched = TurnScheduler(
            deps,
            graph=object(),
            prepare=lambda s, *, user_text=None, forced_skill=None: TurnPlan(
                api_content=f"api:{user_text}"
            ),
            session_for=sessions.get,
            on_turn_result=on_result,
        )
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        assert seen == [("s1", "done", "api:hello")]

    async def test_a_broken_result_hook_never_fails_the_turn(
        self, deps, sessions, graph_calls, events
    ):
        async def on_result(session, result, plan):
            raise RuntimeError("the episodic index is on fire")

        sched = TurnScheduler(
            deps,
            graph=object(),
            prepare=lambda s, *, user_text=None, forced_skill=None: TurnPlan(),
            session_for=sessions.get,
            on_turn_result=on_result,
        )
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        assert "TurnFinished" in kinds(events)

    async def test_the_error_hook_sees_a_turn_that_broke_and_where_it_began(
        self, deps, sessions, graph_calls, events
    ):
        # A failure produces no result, so the second hook is the only way the
        # transcript hears about it at all — and it carries the index the
        # turn's own messages start at, since reading its tail back out of the
        # checkpoint is the only place they still exist.
        seen = []

        async def on_error(session, plan, error, first_new):
            seen.append((session.session_id, str(error), first_new, plan.api_content))

        graph_calls["counts"]["s1"] = 4
        graph_calls["results"]["s1"] = RuntimeError("backend refused")
        sched = TurnScheduler(
            deps,
            graph=object(),
            prepare=lambda s, *, user_text=None, forced_skill=None: TurnPlan(
                api_content=f"api:{user_text}"
            ),
            session_for=sessions.get,
            on_turn_error=on_error,
        )
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        assert seen == [("s1", "backend refused", 4, "api:hello")]

    async def test_the_failure_is_announced_after_it_is_recorded(
        self, deps, sessions, graph_calls, events
    ):
        # The same order the finished path keeps: what a client is told about
        # has already been written down, so a front-end reacting to the
        # failure never races the record of it.
        order = []

        async def on_error(session, plan, error, first_new):
            order.append("recorded")

        graph_calls["results"]["s1"] = RuntimeError("backend refused")
        sched = TurnScheduler(
            deps,
            graph=object(),
            prepare=lambda s, *, user_text=None, forced_skill=None: TurnPlan(),
            session_for=sessions.get,
            on_turn_error=on_error,
        )
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        order += [type(e).__name__ for e in events if type(e).__name__ == "TurnFailed"]
        assert order == ["recorded", "TurnFailed"]

    async def test_a_broken_error_hook_never_swallows_the_failure(
        self, deps, sessions, graph_calls, events
    ):
        async def on_error(session, plan, error, first_new):
            raise RuntimeError("the log is on fire")

        graph_calls["results"]["s1"] = RuntimeError("backend refused")
        sched = TurnScheduler(
            deps,
            graph=object(),
            prepare=lambda s, *, user_text=None, forced_skill=None: TurnPlan(),
            session_for=sessions.get,
            on_turn_error=on_error,
        )
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        assert "TurnFailed" in kinds(events)
        assert not sched.is_busy("s1")

    async def test_a_resume_is_told_where_its_own_half_starts(
        self, deps, sessions, graph_calls, events
    ):
        # Not the anchor: that points at the message the whole exchange began
        # with, whose half was already recorded when the turn parked. What the
        # resume adds starts where the parked thread stopped.
        seen = []

        async def on_error(session, plan, error, first_new):
            seen.append(first_new)

        sched = TurnScheduler(
            deps,
            graph=object(),
            prepare=lambda s, *, user_text=None, forced_skill=None: TurnPlan(),
            session_for=sessions.get,
            on_turn_error=on_error,
        )
        graph_calls["counts"]["s1"] = 2  # the thread before the user message
        graph_calls["results"]["s1"] = TurnResult(
            reply=None, interrupt={"tool": "run_bash"}
        )
        sched.submit_user("s1", "clear the scratch dir")
        await sched.drain()
        await settle()
        graph_calls["counts"]["s1"] = 6  # the parked exchange, as checkpointed
        graph_calls["results"]["s1"] = RuntimeError("backend refused")
        sched.resolve_decision("s1", approved=True)
        await settle()
        assert seen == [6]
        # And the rollback point is still the anchor, untouched by the read.
        assert sched._anchors.get("s1") is None  # spent by the failure
