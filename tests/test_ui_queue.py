"""M5b — typing ahead of a running turn, and taking it back.

The claims are `specs-ui-acceptance.md`'s "Queueing while a turn runs" and the
half of "Stopping a turn" that M5b owns: the confirm on the working row, the
message that comes back, and where it comes back *to*.

Several of that section's claims are the core's rather than the UI's now — a
queue that drains in order, one turn per session — and they are asserted
against the scheduler in `tests/test_core_scheduler.py`. What is left here is
what the front-end can get wrong: the row being drawn at all, the row not
vanishing when the turn it waited for finishes, and a cancel that names the
message by `seq` so that a turn finishing mid-dialog cannot take the wrong one.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from hpca import protocol
from hpca.ui.app import CHAT, INPUT, SESSIONS
from hpca.ui.overlays import QueuedOverlay
from tests.ui_harness import connected, on_entry, plain

STARTED = "2026-08-21T10:00:00+00:00"
EPOCH = datetime.fromisoformat(STARTED).timestamp()

ROWS = [
    protocol.SessionRow(session_id="s1", title="the first thing", mode="auto"),
    protocol.SessionRow(session_id="s2", title="the second thing", mode="auto"),
]


def entry(seq: int, kind: str = "user", text: str = "", **kw) -> protocol.Entry:
    return protocol.Entry(kind=kind, text=text or f"row {seq}", seq=seq, **kw)


@pytest.fixture
async def wire():
    async with connected() as w:
        await w.tell(protocol.Hello(profile="hpc"))
        await w.tell(protocol.SessionRows(rows=list(ROWS)))
        await w.tell(
            protocol.ChatReset(
                session_id="s1", entries=[entry(1, text="the first ask", index=0)]
            )
        )
        w.peer.clear()
        yield w


async def working(wire, session_id: str = "s1"):
    """A turn in flight, as the core announces one."""
    await wire.tell(protocol.TurnStarted(session_id=session_id))
    await wire.tell(
        protocol.TurnActivity(
            session_id=session_id, activity="LLM processing", started_at=STARTED
        )
    )
    return wire


async def queued(wire, seq: int = 7, text: str = "and also check the logs"):
    """The row the core sends back when a message is accepted mid-turn."""
    await wire.tell(
        protocol.ChatAppend(
            session_id="s1", entry=entry(seq, kind="queued", text=text)
        )
    )
    return wire


def rows_of(wire, kind: str) -> list[str]:
    return [x.text for x in wire.ui.chat.items if x.kind == kind]


# ----------------------------------------------------------- typing ahead


class TestAMessageSentWhileTheTurnRuns:
    async def test_it_is_accepted_not_refused(self, wire):
        await working(wire)
        wire.ui.focus = INPUT
        await wire.press(*"and also check the logs", "enter")
        assert wire.peer.last(protocol.TurnSubmit).text == "and also check the logs"

    async def test_and_the_box_is_cleared_at_once(self, wire):
        await working(wire)
        wire.ui.focus = INPUT
        await wire.press(*"and also check the logs", "enter")
        assert wire.ui.input.text() == ""

    async def test_the_queued_row_is_visible_straight_away(self, wire):
        await working(wire)
        await queued(wire)
        assert "and also check the logs" in wire.screen()
        assert rows_of(wire, "queued") == ["and also check the logs"]

    async def test_it_survives_the_reply_of_the_turn_it_waited_for(self, wire):
        # The queued-message bugs the Textual app carried were all one bug: a
        # UI that rebuilt its transcript from a snapshot older than the row.
        # Nothing here rebuilds — `chat.append` is the only way a chat grows
        # (§3.2) — so the row is still there afterwards.
        await working(wire)
        await queued(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=entry(8, "assistant", "the logs are clean")
            )
        )
        await wire.tell(protocol.TurnFinished(session_id="s1", reply="done"))
        assert rows_of(wire, "queued") == ["and also check the logs"]

    async def test_and_it_survives_arriving_before_that_reply_too(self, wire):
        await working(wire)
        await queued(wire)
        await wire.tell(protocol.TurnFinished(session_id="s1", reply="done"))
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=entry(8, "assistant", "the logs are clean")
            )
        )
        assert rows_of(wire, "queued") == ["and also check the logs"]

    async def test_a_slash_command_is_refused_rather_than_queued(self, wire):
        # It acts on the UI and runs its own exclusive worker, so there is
        # nothing sensible to run it behind.
        await working(wire)
        wire.ui.focus = INPUT
        await wire.press(*"/compact", "enter")
        assert wire.peer.took(protocol.TurnSubmit) == []
        assert "cannot be queued" in wire.ui.note

    async def test_and_still_goes_when_nothing_is_running(self, wire):
        # As a command, which is what M8 made of it: `command.run`, aimed at
        # the open session, and never a turn.
        wire.ui.focus = INPUT
        await wire.press(*"/compact", "enter")
        assert wire.peer.took(protocol.TurnSubmit) == []
        ran = wire.peer.last(protocol.CommandRun)
        assert (ran.name, ran.session_id) == ("compact", "s1")

    async def test_a_reply_landing_after_the_wire_closed_stays_quiet(self, wire):
        # The Textual claim was "a reply landing after the screen is gone must
        # not raise: keep the entries, stay quiet". There is no screen to be
        # gone here — a `RowUI` is a function from state to strings — so what
        # is left of it is the client: a frame applied after the connection
        # closed lands in state and nothing raises.
        await working(wire)
        await queued(wire)
        await wire.peer.conn.close()
        wire.client.apply(
            protocol.ChatAppend(
                session_id="s1", entry=entry(9, "assistant", "a late reply")
            )
        )
        assert rows_of(wire, "queued") == ["and also check the logs"]
        assert "a late reply" in wire.screen()

    async def test_a_session_parked_on_an_approval_does_not_stall_another(
        self, wire
    ):
        await wire.tell(
            protocol.DecisionRequested(session_id="s1", payload={"tool": "rm"})
        )
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 2
        await wire.press("enter")
        wire.ui.focus = INPUT
        await wire.press(*"carry on", "enter")
        sent = wire.peer.last(protocol.TurnSubmit)
        assert (sent.session_id, sent.text) == ("s2", "carry on")


# -------------------------------------------------------- taking it back


class TestTheCancelDialog:
    async def at_the_dialog(self, wire, seq: int = 7):
        await working(wire)
        await queued(wire, seq)
        wire.ui.focus = CHAT
        await wire.press("end")
        # `end` lands on the working row, which is the tail; the queued row is
        # the one above it.
        await wire.press("up", "enter")
        assert isinstance(wire.ui.overlay, QueuedOverlay), wire.ui.overlay
        return wire

    async def test_activating_a_queued_row_offers_it(self, wire):
        await self.at_the_dialog(wire)
        assert "still waiting to run" in wire.screen()
        assert "cancel it" in wire.screen()

    async def test_and_not_the_rewind_a_sent_message_gets(self, wire):
        await self.at_the_dialog(wire)
        assert "fork the session" not in wire.screen()

    async def test_x_cancels_it_by_seq(self, wire):
        await self.at_the_dialog(wire, seq=7)
        await wire.press("x")
        taken = wire.peer.last(protocol.TurnUnqueue)
        assert (taken.session_id, taken.seq) == ("s1", 7)

    async def test_the_right_copy_of_a_message_queued_twice(self, wire):
        # Two rows with the same words: the one the cursor was on is the one
        # that goes, and it is named by the core's `seq` rather than by a
        # position the turn ahead can shift.
        await working(wire)
        await queued(wire, 7, "run it again")
        await queued(wire, 8, "run it again")
        wire.ui.focus = CHAT
        # The first of the two, named as an entry rather than counted back in
        # lines from the working row — a queued message draws as many lines as
        # its text needs.
        on_entry(wire.ui.chat, len(wire.ui.chat.items) - 2)
        await wire.press("enter", "x")
        assert wire.peer.last(protocol.TurnUnqueue).seq == 7

    async def test_escape_leaves_it_queued(self, wire):
        await self.at_the_dialog(wire)
        await wire.press("esc")
        assert wire.peer.took(protocol.TurnUnqueue) == []
        assert rows_of(wire, "queued") == ["and also check the logs"]

    async def test_enter_copies_it_and_leaves_it_queued(self, wire):
        await self.at_the_dialog(wire)
        await wire.press("enter")
        assert wire.peer.took(protocol.TurnUnqueue) == []
        assert wire.ui.input.text() == "and also check the logs"

    async def test_the_cancelled_row_goes_and_the_text_comes_back(self, wire):
        await self.at_the_dialog(wire)
        await wire.press("x")
        await wire.tell(
            protocol.TurnUnqueued(session_id="s1", seq=7, text="and also check")
        )
        assert rows_of(wire, "queued") == []
        assert wire.ui.input.text() == "and also check"

    async def test_and_the_working_indicator_is_still_there(self, wire):
        await self.at_the_dialog(wire)
        await wire.press("x")
        await wire.tell(
            protocol.TurnUnqueued(session_id="s1", seq=7, text="and also check")
        )
        assert "LLM processing" in wire.screen()

    async def test_a_message_that_already_started_is_not_cancelled(self, wire):
        # The core refuses it with a warning; stopping the turn it became is
        # `turn.interrupt`, which is the working row's job.
        await self.at_the_dialog(wire)
        await wire.press("x")
        await wire.tell(
            protocol.Notify(
                severity="warning", text="Too late — that message is already running."
            )
        )
        assert "already running" in plain(wire.frame()[-1])
        assert rows_of(wire, "queued") == ["and also check the logs"]

    async def test_switching_session_while_it_is_open_cancels_nothing(self, wire):
        await self.at_the_dialog(wire)
        # The dialog holds the keys, so the sessions row cannot be reached
        # through it — and the answer that eventually comes still names the
        # session it was opened in rather than whatever is on screen.
        await wire.press("down", "enter")
        assert wire.peer.took(protocol.TurnUnqueue) == []
        assert wire.ui.active_id == "s1"

    async def test_the_cancel_names_the_session_it_was_opened_in(self, wire):
        await working(wire)
        await queued(wire, 7)
        wire.ui.focus = CHAT
        await wire.press("end", "up", "enter")
        wire.ui.open_session("s2")  # switched away while it sat open
        await wire.press("x")
        assert wire.peer.last(protocol.TurnUnqueue).session_id == "s1"


class TestATakenBackMessageGoesToItsOwnSession:
    async def test_not_to_whichever_one_is_on_screen(self, wire):
        await wire.tell(
            protocol.TurnUnqueued(session_id="s2", seq=3, text="the second ask")
        )
        assert wire.ui.active_id == "s1"
        assert wire.ui.input.text() == ""
        assert wire.ui.session_for("s2").draft.text() == "the second ask"

    async def test_and_says_where_it_is_waiting(self, wire):
        await wire.tell(
            protocol.TurnUnqueued(session_id="s2", seq=3, text="the second ask")
        )
        assert "the second thing" in wire.ui.note

    async def test_it_never_overwrites_a_draft(self, wire):
        wire.ui.focus = INPUT
        await wire.press(*"half a thought")
        await wire.tell(
            protocol.TurnUnqueued(session_id="s1", seq=3, text="the taken back one")
        )
        assert wire.ui.input.text() == "half a thought\nthe taken back one"


# ------------------------------------------------- stopping, and what returns


class TestTheInterruptHandsTheMessageBack:
    """What the UI does with `turn.interrupted` when one arrives.

    Rarely, now: a stopped turn keeps its work, so its message stays in the
    conversation and the core sends this only when nothing of the turn ever
    reached the thread (`protocol.TurnInterrupted`). The half asserted here is
    unchanged either way — where a message that does come back lands.
    """

    async def test_into_the_session_it_was_typed_in(self, wire):
        # Not into whichever entry is on screen when the answer arrives: the
        # message belongs to the conversation it was typed in, and this is the
        # kind of thing that looks right in a single-session test.
        await working(wire, "s2")
        await wire.tell(
            protocol.TurnInterrupted(
                session_id="s2", text="why did the merge stall"
            )
        )
        assert wire.ui.active_id == "s1"
        assert wire.ui.session_for("s2").draft.text() == "why did the merge stall"

    async def test_and_says_where_it_went(self, wire):
        await working(wire, "s2")
        await wire.tell(
            protocol.TurnInterrupted(session_id="s2", text="why did it stall")
        )
        assert "the second thing" in wire.ui.note

    async def test_and_into_the_box_when_that_is_the_open_one(self, wire):
        await working(wire)
        await wire.tell(
            protocol.TurnInterrupted(
                session_id="s1", text="why did the merge stall"
            )
        )
        assert wire.ui.input.text() == "why did the merge stall"
        assert wire.ui.focus == INPUT

    async def test_it_names_no_row_because_the_reset_settled_them(self, wire):
        # `turn.unqueued` names the row it removes; this one cannot, and the
        # `chat.reset` before it is what settles the screen.
        await working(wire)
        await wire.tell(
            protocol.ChatReset(
                session_id="s1", entries=[entry(1, text="the first ask", index=0)]
            ),
            protocol.TurnInterrupted(session_id="s1", text="the second ask"),
        )
        assert [x.text for x in wire.ui.chat.items] == ["the first ask"]
        assert wire.ui.input.text() == "the second ask"


class TestWhatEscapeMeansWhereItLands:
    async def test_a_reply_landing_while_the_dialog_is_open_is_a_no_op(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end", "enter")  # the confirm
        await wire.tell(protocol.TurnFinished(session_id="s1", reply="done"))
        await wire.press("y")
        assert wire.peer.took(protocol.TurnInterrupt) == []
        assert "finished while you were deciding" in wire.ui.note

    async def test_escape_on_an_open_modal_leaves_the_turn_alone(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("?")  # the key list
        await wire.press("esc", "esc")
        assert wire.peer.took(protocol.TurnInterrupt) == []

    async def test_escape_on_a_pending_decision_refuses_it_not_the_turn(self, wire):
        await wire.tell(
            protocol.DecisionRequested(session_id="s1", payload={"tool": "rm"})
        )
        await wire.press("esc")
        assert wire.peer.last(protocol.DecisionResolve).approved is False
        assert wire.peer.took(protocol.TurnInterrupt) == []
