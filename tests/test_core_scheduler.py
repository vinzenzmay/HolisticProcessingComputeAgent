"""The turn scheduler's invariants, without a graph or a widget in sight.

The point of extracting this from `HpcaApp` was that its rules — one turn per
session, many sessions at once, a parked approval blocks only its own queue —
were only ever reachable through a running Textual app. Here they are asserted
against a two-line fake, which is the argument for the extraction in one file.
"""

from __future__ import annotations

import asyncio

import pytest

from hpca.agent.graph import TurnResult
from hpca.core import scheduler as scheduler_module
from hpca.core.deps import CoreDeps
from hpca.core.scheduler import LLM_WAIT_ACTIVITY, TurnPlan, TurnScheduler


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


@pytest.fixture
def graph_calls(monkeypatch):
    """Replace the four graph entry points the scheduler uses.

    `run_turn` is driven per session id: a test pushes the result it wants,
    or an asyncio.Event to park on so it can assert what happens mid-turn.
    """
    calls = {
        "run_turn": [],
        "delivered": [],
        "rolled_back": [],
        "results": {},
        "gates": {},
        "counts": {},
    }

    async def fake_run_turn(graph, *, session_id, user_text=None, resume=None,
                            api_content=None):
        calls["run_turn"].append(
            {"session_id": session_id, "user_text": user_text, "resume": resume,
             "api_content": api_content}
        )
        gate = calls["gates"].get(session_id)
        if gate is not None:
            await gate.wait()
        result = calls["results"].get(session_id)
        if isinstance(result, Exception):
            raise result
        return result or TurnResult(reply="done", interrupt=None)

    async def fake_thread_message_count(graph, *, session_id):
        return calls["counts"].get(session_id, 3)

    async def fake_rollback_thread(graph, *, session_id, keep):
        calls["rolled_back"].append((session_id, keep))
        return []

    async def fake_deliver_event(graph, *, session_id, text):
        calls["delivered"].append((session_id, text))

    monkeypatch.setattr(scheduler_module, "run_turn", fake_run_turn)
    monkeypatch.setattr(
        scheduler_module, "thread_message_count", fake_thread_message_count
    )
    monkeypatch.setattr(scheduler_module, "rollback_thread", fake_rollback_thread)
    monkeypatch.setattr(scheduler_module, "deliver_event", fake_deliver_event)
    return calls


@pytest.fixture
def sched(deps, sessions, graph_calls):
    return TurnScheduler(
        deps,
        graph=object(),
        prepare=lambda session, *, user_text=None, forced_skill=None: TurnPlan(
            api_content=f"api:{user_text}" if user_text else None
        ),
        session_for=sessions.get,
    )


def kinds(events):
    return [type(e).__name__ for e in events]


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
        # have to be un-drawn a moment later.
        assert sched.submit_user("s1", "hello") is False
        await sched.drain()
        await settle()
        assert "ChatAppend" not in kinds(events)

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


class TestEvents:
    async def test_a_turn_announces_its_start_and_its_end(self, sched, events):
        sched.submit_user("s1", "hello")
        await sched.drain()
        await settle()
        assert kinds(events)[0] == "TurnStarted"
        assert kinds(events)[-1] == "TurnFinished"
        assert events[-1].reply == "done"

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
        sched.report_activity(session_id, LLM_WAIT_ACTIVITY)

    async def test_only_while_parked_on_the_model(self, sched, graph_calls):
        await self._park_on_the_model(sched, graph_calls)
        assert sched.can_interrupt("s1") is True
        sched.report_activity("s1", "running read_file")
        assert sched.can_interrupt("s1") is False

    async def test_never_for_a_session_with_no_turn(self, sched):
        assert sched.can_interrupt("s1") is False

    async def test_it_hands_the_message_back_and_rolls_the_thread_back(
        self, sched, graph_calls
    ):
        graph_calls["counts"]["s1"] = 7
        await self._park_on_the_model(sched, graph_calls)
        text = await sched.interrupt("s1")
        assert text == "a long question"
        # Everything the aborted turn appended leaves the thread, or the model
        # meets the abandoned attempt again on the retry.
        assert graph_calls["rolled_back"] == [("s1", 7)]
        assert not sched.is_busy("s1")

    async def test_interrupting_nothing_returns_nothing(self, sched):
        assert await sched.interrupt("s1") is None


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
