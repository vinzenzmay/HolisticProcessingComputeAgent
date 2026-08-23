"""The client: events become state, keys become commands.

Driven through a real `InProcessConnection.pair()` — the transport is not
mocked. A scripted peer sends the events a core would and records the commands
that come back, and every assertion is either about a frame (a string) or about
a typed command on the wire.

The three things this file exists to hold on to, from specs-ui-replacement.md
§3.2 and `protocol.ChatUpdate`:

* the chat is append-only between resets, and a row is named by its `seq`;
* an update for a row the UI does not have is dropped, never invented;
* an event for a session that is not on screen changes the sidebar marker and
  nothing else.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from hpca import protocol
from hpca.transport import InProcessConnection
from hpca.ui.app import CHAT, INPUT, SESSIONS, WATCHERS, RowUI
from hpca.ui import state, theme
from hpca.ui.client import UIClient
from hpca.ui.overlays import HelpOverlay, InspectOverlay
from tests.ui_harness import Peer, Wire, clocked, on_entry, plain, settle, widths

ROWS = [
    protocol.SessionRow(
        session_id="s1", title="the first thing", profile="hpc", mode="agent"
    ),
    protocol.SessionRow(
        session_id="s2", title="the second thing", profile="hpc", mode="plan"
    ),
]


# A fixed instant, so a frame can be searched for the stamp rather than for
# whatever the clock says while the test runs. Read back through `state.when`,
# which is what turns the core's UTC into the local wall-clock the label
# carries — writing the expected text out here would make the assertion
# depend on the timezone the suite runs in.
AT = "2026-08-22T11:04:47+00:00"


def entry(seq: int, kind: str = "user", text: str = "", **kw) -> protocol.Entry:
    return protocol.Entry(kind=kind, text=text or f"row {seq}", seq=seq, **kw)


@pytest.fixture
async def wire():
    """A UI, a client and a scripted core, over a real pair of connections."""
    ui = RowUI()
    ours, theirs = InProcessConnection.pair()
    client = UIClient(ui, ours)
    peer = Peer(theirs)
    tasks = [asyncio.create_task(client.run()), asyncio.create_task(peer.listen())]
    try:
        yield Wire(ui, client, peer)
    finally:
        await ours.close()
        await theirs.close()
        for task in tasks:
            task.cancel()


async def started(wire: Wire, entries: list[protocol.Entry] | None = None) -> Wire:
    """The opening exchange: hello, the sidebar, and the first transcript."""
    await wire.tell(protocol.Hello(profile="hpc"))
    await wire.tell(protocol.SessionRows(rows=list(ROWS)))
    await wire.tell(
        protocol.ChatReset(
            session_id="s1",
            entries=entries
            if entries is not None
            else [entry(1, text="hello there", index=0)],
        )
    )
    wire.peer.clear()
    return wire


# ------------------------------------------------------------- the boundary


def test_the_app_does_not_import_the_protocol():
    """The rule §3.1 puts the whole layering on, checked the way
    `test_core_headless.py` checks the core's: in a clean interpreter, because
    an in-process check would pass on an import some other test already made.

    If a `RowUI` method took a `protocol.Entry`, the render tests would need
    pydantic models to say anything and the property that makes this UI
    testable — frames are strings, and strings compare — would be gone.
    """
    code = (
        "import sys, hpca.ui.app; "
        "assert 'hpca.protocol' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"hpca.ui.app drags in the protocol:\n{result.stderr}"
    )


def test_and_neither_does_the_state_layer():
    code = (
        "import sys, hpca.ui.state; "
        "assert 'hpca.protocol' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_client_is_the_one_that_does():
    # The other half of the same claim: the import has to live somewhere, and
    # a check that only ever says "not here" would pass on a UI wired to
    # nothing at all.
    code = "import sys, hpca.ui.client; assert 'hpca.protocol' in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------ the handshake


class TestHello:
    async def test_it_asks_for_the_session_list(self, wire):
        await wire.tell(protocol.Hello(profile="hpc"))
        assert wire.peer.took(protocol.SessionList)

    async def test_the_profile_reaches_the_header(self, wire):
        await wire.tell(protocol.Hello(profile="genomics"))
        assert "genomics" in wire.frame()[0]

    async def test_it_asks_how_often_each_command_has_been_run(self, wire):
        # The "/" menu is sorted the first time it is drawn, so the counts are
        # asked for on connect rather than when a slash is typed.
        await wire.tell(protocol.Hello(profile="hpc"))
        assert wire.peer.took(protocol.CommandList)

    async def test_a_core_of_another_version_says_so(self, wire):
        await wire.tell(protocol.Hello(version=99))
        assert [t.severity for t in wire.ui.toasts] == ["error"]

    async def test_and_the_frame_survives_saying_it(self, wire):
        await wire.tell(protocol.Hello(version=99))
        assert widths(wire.ui.render(40, 10)) == {40}


class TestTheSettingsTheUiDrawsWith:
    """`hello.display` and `display.settings` — the one part of the settings
    file a front-end is handed, and the path it is handed it down.

    The rule is unchanged: the UI reads no settings file (§4.2 rule 2). What
    is new is that a couple of keys are about nothing except what a frame
    looks like, and a `settings_digest` cannot answer "draw this how". So
    those keys, and strictly those, arrive as state — on the frame that lands
    before anything is drawn, and again whenever a save changes them.

    Deliberately not `settings.body`: that is the file as *text*, fetched
    because the config editor is opening over it. Hanging a chat label off a
    read path would leave the setting inert until somebody pressed `c`.
    """

    async def test_they_arrive_on_the_first_frame(self, wire):
        await wire.tell(
            protocol.Hello(
                profile="hpc",
                display=protocol.DisplaySettings(
                    chat_stamps=False, decision_pulse_seconds=4.0
                ),
            )
        )
        assert wire.ui.display.chat_stamps is False
        assert wire.ui.display.decision_pulse_seconds == 4.0
        # And the palette came with them, on the same frame: the colours are
        # display settings like any other, so a first frame that carried the
        # stamps but not the theme would draw one row in the built-in palette.
        assert wire.ui.display.palette["chrome"] == "73"

    async def test_and_before_the_first_chat_that_uses_them(self, wire):
        # The ordering that matters: `hello` is the first frame, and the
        # transcript can only arrive as the answer to a command `hello` itself
        # sends — so no row is ever built against the wrong setting.
        await wire.tell(
            protocol.Hello(display=protocol.DisplaySettings(chat_stamps=False)),
            protocol.SessionRows(rows=list(ROWS)),
            protocol.ChatReset(
                session_id="s1", entries=[entry(1, text="run it", at=AT)]
            ),
        )
        assert any(x.rstrip().endswith(" you") for x in wire.frame())
        assert state.when(AT) not in wire.screen()

    async def test_a_ui_with_no_core_behind_it_draws_the_defaults(self):
        # What a fresh install would say, so nothing below needs a None check.
        assert RowUI().display == state.Display()

    async def test_a_change_lands_without_a_restart(self, wire):
        await started(wire, [entry(1, text="run it", at=AT)])
        assert state.when(AT) in wire.screen()
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(chat_stamps=False)
            )
        )
        assert state.when(AT) not in wire.screen()
        assert any(x.rstrip().endswith(" you") for x in wire.frame())

    async def test_and_the_conversation_survives_it(self, wire):
        # `restyle`, not `reset`: nothing about what was *said* changed, so
        # the rows keep their text, their numbering and what is open in them.
        await started(wire, [entry(1, text="run it", at=AT)])
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(chat_stamps=False)
            )
        )
        assert "run it" in wire.screen()
        assert [x.key for x in wire.ui.chat.items] == ["1"]

    async def test_a_session_nobody_has_looked_at_yet_gets_them_too(self, wire):
        # Restyling on the way back into view would do the work at the one
        # moment the user is watching.
        await started(wire)
        await wire.tell(
            protocol.ChatReset(
                session_id="s2", entries=[entry(2, text="later", at=AT)]
            ),
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(chat_stamps=False)
            ),
        )
        assert wire.ui.session_for("s2").display.chat_stamps is False
        assert state.when(AT) not in " ".join(
            x.head for x in wire.ui.session_for("s2").chat.items
        )

    async def test_a_session_opened_after_the_change_is_drawn_with_it(self, wire):
        await started(wire)
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(chat_stamps=False)
            ),
            protocol.ChatReset(
                session_id="s3", entries=[entry(9, text="new one", at=AT)]
            ),
        )
        assert wire.ui.session_for("s3").chat.items[0].head == "you"


# -------------------------------------------------------------- the sidebar


class TestTheSessionList:
    async def test_the_rows_are_the_sidebar(self, wire):
        await wire.tell(
            protocol.Hello(), protocol.SessionRows(rows=list(ROWS))
        )
        assert "the second thing" in wire.screen()

    async def test_the_first_one_is_opened(self, wire):
        await wire.tell(
            protocol.Hello(), protocol.SessionRows(rows=list(ROWS))
        )
        assert wire.peer.last(protocol.SessionOpen).session_id == "s1"

    async def test_and_the_core_is_told_what_is_on_screen(self, wire):
        await wire.tell(
            protocol.Hello(), protocol.SessionRows(rows=list(ROWS))
        )
        assert wire.peer.last(protocol.SessionFocus).session_id == "s1"

    async def test_a_second_list_does_not_re_open_it(self, wire):
        await started(wire)
        await wire.tell(protocol.SessionRows(rows=list(ROWS)))
        assert wire.peer.took(protocol.SessionOpen) == []

    async def test_nor_does_it_disturb_the_chat(self, wire):
        await started(wire)
        before = wire.screen()
        await wire.tell(protocol.SessionRows(rows=list(ROWS)))
        assert wire.screen() == before

    async def test_a_repaint_keeps_the_open_session_open(self, wire):
        await started(wire)
        await wire.tell(
            protocol.SessionRows(
                rows=[
                    protocol.SessionRow(session_id="new", title="a fork elsewhere"),
                    *ROWS,
                ]
            )
        )
        assert wire.ui.active_id == "s1"


class TestTheCursorIsKeptByIdentity:
    """A row inserted above the cursor must not move the selection (§3.2)."""

    async def positioned(self, wire) -> Wire:
        await started(wire)
        wire.ui.focus = SESSIONS
        # Row 0 is `(new session)`, so the second conversation is line 2.
        wire.ui.session_pane.cursor = 2
        assert wire.ui.session_pane.items[
            wire.ui.session_pane.current(wire.inner)
        ].key == "s2"
        return wire

    async def test_a_row_inserted_above_does_not_move_it(self, wire):
        await self.positioned(wire)
        await wire.tell(
            protocol.SessionRows(
                rows=[
                    protocol.SessionRow(session_id="new", title="made in the background"),
                    *ROWS,
                ]
            )
        )
        pane = wire.ui.session_pane
        assert pane.items[pane.current(wire.inner)].key == "s2"

    async def test_which_is_a_different_line_than_before(self, wire):
        # The point of keying by id: the line moved, the selection did not.
        await self.positioned(wire)
        await wire.tell(
            protocol.SessionRows(
                rows=[protocol.SessionRow(session_id="new", title="x"), *ROWS]
            )
        )
        assert wire.ui.session_pane.cursor == 3

    async def test_an_open_row_stays_open_under_the_row_that_arrived(self, wire):
        await self.positioned(wire)
        wire.ui.session_pane.expand(wire.inner)
        await wire.tell(
            protocol.SessionRows(
                rows=[protocol.SessionRow(session_id="new", title="x"), *ROWS]
            )
        )
        assert wire.ui.session_pane.expanded == {"s2"}

    async def test_a_row_that_left_takes_its_state_with_it(self, wire):
        await self.positioned(wire)
        wire.ui.session_pane.expand(wire.inner)
        await wire.tell(protocol.SessionRows(rows=[ROWS[0]]))
        assert wire.ui.session_pane.expanded == set()


# ----------------------------------------------------------------- the chat


class TestTheChat:
    async def test_a_reset_fills_it(self, wire):
        await started(wire, [entry(1, text="the first line"), entry(2, "assistant")])
        assert "the first line" in wire.screen()

    async def test_an_append_grows_it(self, wire):
        await started(wire, [entry(1)])
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=entry(2, "assistant", "and then this")
            )
        )
        assert len(wire.ui.chat.items) == 2
        assert "and then this" in wire.screen()

    async def test_an_update_revises_a_row_in_place(self, wire):
        await started(wire, [entry(1), entry(2, "assistant", "still working")])
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1", entry=entry(2, "assistant", "finished, actually")
            )
        )
        assert len(wire.ui.chat.items) == 2
        assert "finished, actually" in wire.screen()
        assert "still working" not in wire.screen()

    async def test_an_update_for_a_row_we_do_not_have_is_dropped(self, wire):
        await started(wire, [entry(1)])
        await wire.tell(
            protocol.ChatUpdate(session_id="s1", entry=entry(97, text="from nowhere"))
        )
        assert len(wire.ui.chat.items) == 1
        assert "from nowhere" not in wire.screen()

    async def test_and_says_that_it_dropped_it(self, wire):
        await started(wire, [entry(1)])
        await wire.tell(protocol.ChatUpdate(session_id="s1", entry=entry(97)))
        assert wire.client.dropped["chat.update"] == 1

    async def test_an_unnumbered_row_can_never_be_updated(self, wire):
        # seq 0 means "not numbered" (`protocol.Entry.seq`), so an update
        # carrying 0 must not land on the first unnumbered row it finds.
        await started(wire, [entry(0, text="unnumbered")])
        await wire.tell(
            protocol.ChatUpdate(session_id="s1", entry=entry(0, text="revised"))
        )
        assert "revised" not in wire.screen()

    async def test_a_reset_re_bases_the_numbering(self, wire):
        await started(wire, [entry(1, text="one"), entry(2, text="two")])
        await wire.tell(
            protocol.ChatReset(session_id="s1", entries=[entry(1, text="only this")])
        )
        assert [x.key for x in wire.ui.chat.items] == ["1"]
        assert "two" not in wire.screen()

    async def test_and_the_rows_it_dropped_are_no_longer_addressable(self, wire):
        await started(wire, [entry(1, text="one"), entry(2, text="two")])
        await wire.tell(
            protocol.ChatReset(session_id="s1", entries=[entry(1, text="only this")])
        )
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1", entry=entry(2, text="back from the dead")
            )
        )
        assert "back from the dead" not in wire.screen()
        assert wire.client.dropped["chat.update"] == 1

    async def test_the_same_seq_after_a_reset_is_the_new_row(self, wire):
        await started(wire, [entry(1, text="one"), entry(2, text="two")])
        await wire.tell(
            protocol.ChatReset(session_id="s1", entries=[entry(1, text="a new one")])
        )
        await wire.tell(
            protocol.ChatUpdate(session_id="s1", entry=entry(1, text="revised"))
        )
        assert [x.text for x in wire.ui.session.entries] == ["revised"]


class TestSessionsThatAreNotOnScreen:
    """Property 1 of §3.2: most events are not for the visible session."""

    async def test_an_append_elsewhere_changes_nothing_on_screen(self, wire):
        await started(wire)
        before = wire.screen()
        await wire.tell(
            protocol.ChatAppend(session_id="s2", entry=entry(1, text="not for you"))
        )
        assert wire.screen() == before

    async def test_but_it_did_land(self, wire):
        await started(wire)
        await wire.tell(
            protocol.ChatAppend(session_id="s2", entry=entry(1, text="kept for later"))
        )
        assert wire.ui.session_for("s2").entries[0].text == "kept for later"

    async def test_a_reset_for_another_session_does_not_touch_this_one(self, wire):
        await started(wire, [entry(1, text="mine")])
        await wire.tell(protocol.ChatReset(session_id="s2", entries=[entry(1)]))
        assert "mine" in wire.screen()

    async def test_a_decision_elsewhere_shows_as_a_sidebar_mark(self, wire):
        await started(wire)
        await wire.tell(
            protocol.DecisionRequested(session_id="s2", payload={"tool": "run_bash"})
        )
        assert "!" in wire.screen()

    async def test_and_only_the_sidebar_changed(self, wire):
        await started(wire)
        before = wire.frame()
        await wire.tell(protocol.DecisionRequested(session_id="s2", payload={}))
        changed = [i for i, line in enumerate(wire.frame()) if line != before[i]]
        rows = wire.ui._heights(wire.height, wire.width)
        assert changed and all(i <= rows[0] for i in changed)

    async def test_the_mark_goes_when_the_decision_is_cleared(self, wire):
        await started(wire)
        await wire.tell(protocol.DecisionRequested(session_id="s2", payload={}))
        await wire.tell(protocol.DecisionCleared(session_id="s2"))
        assert "!" not in wire.screen()

    async def test_a_turn_elsewhere_shows_as_the_working_mark(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        assert "⟳" in wire.screen()

    async def test_and_goes_when_it_finishes(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        assert "⟳" not in wire.screen()


# ----------------------------------------------------------------- the turn


class TestTheTurn:
    async def test_it_starts_and_stops(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s1"))
        assert wire.ui.session.turn.working
        await wire.tell(protocol.TurnFinished(session_id="s1"))
        assert not wire.ui.session.turn.working

    async def test_an_activity_carries_the_cores_own_stamp(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="reading", started_at="2026-08-21T10:00:00"
            )
        )
        assert wire.ui.session.turn.started_at == "2026-08-21T10:00:00"

    async def test_the_same_activity_again_does_not_restart_the_clock(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="reading", started_at="2026-08-21T10:00:00"
            )
        )
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="reading", started_at="2026-08-21T10:00:09"
            )
        )
        assert wire.ui.session.turn.started_at == "2026-08-21T10:00:00"

    async def test_but_a_different_one_does(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="reading", started_at="2026-08-21T10:00:00"
            )
        )
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="writing", started_at="2026-08-21T10:00:09"
            )
        )
        assert wire.ui.session.turn.started_at == "2026-08-21T10:00:09"

    async def test_a_failure_becomes_an_entry_in_the_chat(self, wire):
        await started(wire, [entry(1)])
        await wire.tell(
            protocol.TurnFailed(session_id="s1", error="the backend hung up")
        )
        assert "the backend hung up" in wire.screen()

    async def test_the_failure_row_is_not_addressable(self, wire):
        # The core did not number it, so nothing may revise it later.
        await started(wire, [entry(1)])
        await wire.tell(protocol.TurnFailed(session_id="s1", error="boom"))
        assert wire.ui.session.entries[-1].seq == 0

    async def test_and_the_turn_is_over(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s1"))
        await wire.tell(protocol.TurnFailed(session_id="s1", error="boom"))
        assert not wire.ui.session.turn.working


class TestTheContextMeter:
    async def test_an_estimate_is_marked_as_one(self, wire):
        await started(wire)
        await wire.tell(
            protocol.ContextEstimate(session_id="s1", used=1000, window=10000)
        )
        assert "~1,000 / 10,000 (10%)" in wire.screen()

    async def test_a_measurement_is_not(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=2000, max_model_len=10000
            )
        )
        assert "2,000 / 10,000 (20%)" in wire.screen()
        assert "~2,000" not in wire.screen()

    async def test_a_measurement_beats_an_estimate(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=2000, max_model_len=10000
            ),
            protocol.ContextEstimate(session_id="s1", used=9000, window=10000),
        )
        assert "2,000 / 10,000 (20%)" in wire.screen()

    async def test_until_the_thread_it_measured_is_gone(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=2000, max_model_len=10000
            )
        )
        await wire.tell(protocol.ChatReset(session_id="s1", entries=[entry(1)]))
        await wire.tell(
            protocol.ContextEstimate(session_id="s1", used=500, window=10000)
        )
        assert "~500 / 10,000 (5%)" in wire.screen()

    # ---------------------------------------------- the speed (§6, §3.14 #9)

    async def test_the_speed_is_the_pair_the_core_sent_divided(self, wire):
        # `client._usage` dropped both of these, so `· tok/s` could never
        # appear — while `state.py` said "nothing on the wire carries either".
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1",
                prompt_tokens=2000,
                max_model_len=10000,
                completion_tokens=210,
                request_seconds=50.0,
            )
        )
        assert "· 4.2 tok/s" in wire.screen()

    async def test_a_fast_turn_drops_the_decimal(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1",
                prompt_tokens=2000,
                max_model_len=10000,
                completion_tokens=840,
                request_seconds=10.0,
            )
        )
        assert "· 84 tok/s" in wire.screen()

    async def test_a_restatement_with_no_generation_keeps_the_last_rate(
        self, wire
    ):
        # A session restated after a backend switch has a prompt size and no
        # fresh generation behind it (`protocol.TurnUsage`), and 0/None must
        # not become "0.0 tok/s".
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1",
                prompt_tokens=2000,
                max_model_len=10000,
                completion_tokens=210,
                request_seconds=50.0,
            ),
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=2500, max_model_len=10000
            ),
        )
        assert "· 4.2 tok/s" in wire.screen()

    async def test_and_a_session_that_generated_nothing_shows_no_rate(self, wire):
        await started(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=2000, max_model_len=10000
            )
        )
        assert "tok/s" not in wire.screen()


# -------------------------------------------------------------- the panels


PANEL = [
    protocol.PanelRow(
        key="w7",
        title="job 4821000",
        text="RUNNING\n/scratch/run/step0.log\n[12:41] merging shard 3",
        classes="watch watch-live",
        kind=protocol.PANEL_WATCH,
        ref="7",
    ),
    protocol.PanelRow(
        key="w9",
        title="job 4821001",
        text="FAILED\n/scratch/run/step1.log",
        classes="watch watch-dead",
        kind=protocol.PANEL_WATCH,
        ref="9",
    ),
]


class TestTheWatchers:
    async def test_the_rows_are_the_column(self, wire):
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        assert "job 4821000" in wire.screen()

    async def test_they_are_keyed_by_the_key_the_core_gave_them(self, wire):
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        assert [x.key for x in wire.ui.watchers.items] == ["w7", "w9"]

    async def test_a_repaint_keeps_the_cursor_on_its_box(self, wire):
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        wire.ui.focus = WATCHERS
        wire.ui.watchers.cursor = 1
        await wire.tell(
            protocol.PanelUpdate(
                session_id="s1",
                rows=[
                    protocol.PanelRow(key="w1", title="job 4820000", text="RUNNING"),
                    *PANEL,
                ],
            )
        )
        pane = wire.ui.watchers
        assert pane.items[pane.current(wire.inner)].key == "w9"

    async def test_and_keeps_an_open_box_open(self, wire):
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        wire.ui.focus = WATCHERS
        wire.ui.watchers.cursor = 0
        wire.ui.watchers.expand(wire.inner)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        assert wire.ui.watchers.expanded == {"w7"}

    async def test_a_panel_for_another_session_stays_off_screen(self, wire):
        await started(wire)
        before = wire.screen()
        await wire.tell(protocol.PanelUpdate(session_id="s2", rows=list(PANEL)))
        assert wire.screen() == before
        assert len(wire.ui.session_for("s2").watchers.items) == 2


class TestReorderingSurvivesTheNextFrame:
    """alt+↑/↓ asks the core, and the core's answer is the order.

    The bug this is the test for: both keys reordered the front-end's own
    copy and sent nothing, so the arrangement lasted exactly until the next
    frame — the column is repainted by a poll twice a second, and the sidebar
    on almost every command — and then snapped back. Nothing was wrong with
    the swap; there was no one to tell.
    """

    async def _column(self, wire) -> Wire:
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        wire.ui.focus = WATCHERS
        wire.ui.watchers.cursor = 0
        wire.peer.clear()
        return wire

    async def test_alt_down_on_a_box_asks_the_core(self, wire):
        await self._column(wire)
        await wire.press("alt-down")
        asked = wire.peer.last(protocol.WatchMove)
        assert (asked.watch_id, asked.delta) == (7, +1)

    async def test_alt_up_is_the_other_direction(self, wire):
        await self._column(wire)
        wire.ui.watchers.cursor = 1
        await wire.press("alt-up")
        asked = wire.peer.last(protocol.WatchMove)
        assert (asked.watch_id, asked.delta) == (9, -1)

    async def test_the_ref_is_an_int_by_the_time_it_is_a_command(self, wire):
        # `PanelRow.ref` is a string for every kind of row and a watch id is
        # an int; the conversion is the sender's errand (`protocol.PanelRow`),
        # which is `client.py`'s side of the line and not `app.py`'s.
        await self._column(wire)
        await wire.press("alt-down")
        assert isinstance(wire.peer.last(protocol.WatchMove).watch_id, int)

    async def test_the_column_is_not_reordered_before_the_answer(self, wire):
        # Nothing optimistic: the order is the store's, the answer is the
        # column whole, and a swap made here would only be overwritten by it.
        await self._column(wire)
        await wire.press("alt-down")
        assert [x.key for x in wire.ui.watchers.items] == ["w7", "w9"]

    async def test_and_the_frame_that_answers_is_the_new_order(self, wire):
        # The whole point, and the exact thing that used to fail: the order
        # the core sends after the move is the order that stays.
        await self._column(wire)
        await wire.press("alt-down")
        await wire.tell(
            protocol.PanelUpdate(session_id="s1", rows=[PANEL[1], PANEL[0]])
        )
        assert [x.key for x in wire.ui.watchers.items] == ["w9", "w7"]

    async def test_the_cursor_rides_the_box_it_moved(self, wire):
        # Which is what makes holding the key down walk one box past several:
        # the second press has to be aimed at the same watch as the first.
        await self._column(wire)
        await wire.press("alt-down")
        await wire.tell(
            protocol.PanelUpdate(session_id="s1", rows=[PANEL[1], PANEL[0]])
        )
        pane = wire.ui.watchers
        assert pane.items[pane.current(wire.inner)].key == "w7"
        await wire.press("alt-down")
        asked = wire.peer.last(protocol.WatchMove)
        assert (asked.watch_id, asked.delta) == (7, +1), "the same box again"

    async def test_two_presses_ahead_of_the_answer_still_name_one_box(self, wire):
        # An offset and not a slot: the core swaps with whichever row is the
        # neighbour when the command arrives, so a second press that went out
        # before the first was answered still walks the same box one further.
        await self._column(wire)
        await wire.press("alt-down", "alt-down")
        asked = wire.peer.took(protocol.WatchMove)
        assert [(x.watch_id, x.delta) for x in asked] == [(7, +1), (7, +1)]

    async def test_the_bottom_box_asks_for_nothing(self, wire):
        await self._column(wire)
        wire.ui.watchers.cursor = 1
        await wire.press("alt-down")
        assert wire.peer.took(protocol.WatchMove) == []
        assert wire.ui.note == ""

    async def test_a_sidebar_row_asks_the_core_too(self, wire):
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1  # row 0 is "+ new session"
        await wire.press("alt-down")
        asked = wire.peer.last(protocol.SessionMove)
        assert (asked.session_id, asked.delta) == ("s1", +1)

    async def test_and_the_sidebar_the_core_sends_back_is_the_order(self, wire):
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
        await wire.press("alt-down")
        assert [x.session_id for x in wire.ui.sessions] == ["s1", "s2"], "not yet"
        await wire.tell(protocol.SessionRows(rows=[ROWS[1], ROWS[0]]))
        assert [x.session_id for x in wire.ui.sessions] == ["s2", "s1"]

    async def test_the_open_session_is_still_the_open_one(self, wire):
        # `active` is a position in `self.sessions`, and the frame that
        # reorders them moves it out from under it.
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
        await wire.press("alt-down")
        await wire.tell(protocol.SessionRows(rows=[ROWS[1], ROWS[0]]))
        assert wire.ui.active_id == "s1"
        assert wire.ui.active == 1

    async def test_the_new_session_row_never_moves(self, wire):
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 0
        await wire.press("alt-down")
        assert wire.peer.took(protocol.SessionMove) == []
        assert wire.ui.note == ""


# ------------------------------------------------------ keys become commands


class TestKeysBecomeCommands:
    async def test_enter_in_the_box_submits_the_turn(self, wire):
        await started(wire)
        wire.ui.focus = INPUT
        wire.ui.input.set_text("which BAMs?")
        await wire.press("enter")
        sent = wire.peer.last(protocol.TurnSubmit)
        assert (sent.session_id, sent.text) == ("s1", "which BAMs?")

    async def test_and_the_chat_does_not_grow_until_the_core_says_so(self, wire):
        await started(wire, [entry(1)])
        wire.ui.focus = INPUT
        wire.ui.input.set_text("which BAMs?")
        await wire.press("enter")
        assert len(wire.ui.chat.items) == 1

    async def test_an_empty_box_sends_nothing(self, wire):
        await started(wire)
        wire.ui.focus = INPUT
        await wire.press("enter")
        assert wire.peer.took(protocol.TurnSubmit) == []

    async def test_esc_esc_interrupts_the_open_session(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s1"))
        clocked(wire.ui)
        await wire.press("esc")
        wire.ui._now += 0.2
        await wire.press("esc")
        assert wire.peer.last(protocol.TurnInterrupt).session_id == "s1"

    async def test_and_with_no_turn_running_it_sends_nothing(self, wire):
        # §4: the gesture used to report a stop it had not made, on an idle
        # session and even with nothing open at all.
        await started(wire)
        clocked(wire.ui)
        await wire.press("esc")
        wire.ui._now += 0.2
        await wire.press("esc")
        assert wire.peer.took(protocol.TurnInterrupt) == []
        assert "stopped" not in wire.ui.note

    async def test_one_esc_interrupts_nothing(self, wire):
        await started(wire)
        clocked(wire.ui)
        await wire.press("esc")
        assert wire.peer.took(protocol.TurnInterrupt) == []

    async def test_enter_on_a_session_opens_it(self, wire):
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 2
        await wire.press("enter")
        assert wire.peer.last(protocol.SessionOpen).session_id == "s2"

    async def test_and_says_which_one_is_on_screen(self, wire):
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 2
        await wire.press("enter")
        assert wire.peer.last(protocol.SessionFocus).session_id == "s2"

    async def test_going_back_does_not_ask_for_the_transcript_twice(self, wire):
        await started(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 2
        await wire.press("enter")
        await wire.tell(protocol.ChatReset(session_id="s2", entries=[entry(1)]))
        wire.peer.clear()
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
        await wire.press("enter")
        assert wire.peer.took(protocol.SessionOpen) == []
        assert wire.peer.last(protocol.SessionFocus).session_id == "s1"

    async def test_a_watcher_peek_names_the_watch(self, wire):
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        wire.ui.focus = WATCHERS
        wire.ui.watchers.cursor = 1
        await wire.press("enter")
        assert wire.peer.last(protocol.WatchPeek).watch_id == 9

    async def test_and_d_drops_it(self, wire):
        await started(wire)
        await wire.tell(protocol.PanelUpdate(session_id="s1", rows=list(PANEL)))
        wire.ui.focus = WATCHERS
        wire.ui.watchers.cursor = 0
        await wire.press("d")
        assert wire.peer.last(protocol.WatchDrop).watch_id == 7


class TestTheRewind:
    """The one place three different numbers meet, so the one to get wrong."""

    ENTRIES = [
        entry(1, text="the first ask", index=0),
        entry(2, "thinking"),
        entry(3, text="the second ask", index=1),
    ]

    async def at_the_second_ask(self, wire) -> Wire:
        await started(wire, list(self.ENTRIES))
        wire.ui.focus = CHAT
        on_entry(wire.ui.chat, 2)
        await wire.press("enter")
        assert wire.ui.overlay is not None, "enter on your own message opens the rewind"
        return wire

    async def test_a_fork_carries_the_message_index(self, wire):
        # Not the seq (3) and not the row's position in the pane (2): the wire
        # wants the thread message the cut is made at.
        await self.at_the_second_ask(wire)
        await wire.press("f")
        assert wire.peer.last(protocol.SessionFork).index == 1

    async def test_a_rollback_carries_it_too(self, wire):
        await self.at_the_second_ask(wire)
        await wire.press("r")
        assert wire.peer.last(protocol.SessionRollback).index == 1

    async def test_it_names_the_session_it_was_opened_in(self, wire):
        await self.at_the_second_ask(wire)
        await wire.press("f")
        assert wire.peer.last(protocol.SessionFork).session_id == "s1"

    async def test_enter_is_the_fork_and_carries_the_same_index(self, wire):
        # The dialog opens on Enter and answers Enter with the choice that
        # keeps the conversation whole.
        await self.at_the_second_ask(wire)
        await wire.press("enter")
        assert wire.peer.last(protocol.SessionFork).index == 1

    async def test_and_c_is_not_one_of_its_answers(self, wire):
        # It is the chat row's copy key, and the dialog in front of the row
        # neither takes it nor closes on it.
        await self.at_the_second_ask(wire)
        await wire.press("c")
        assert wire.peer.commands == []
        assert wire.ui.overlay is not None
        assert wire.ui.input.text() == ""

    async def test_a_row_that_is_not_a_message_cannot_be_rewound(self, wire):
        # A `thinking` row folds several messages and is `index` -1, so there
        # is nothing to cut at: the answer is to say so, not to send a cut
        # aimed at whatever -1 resolves to.
        await started(wire, list(self.ENTRIES))
        wire.ui.focus = CHAT
        on_entry(wire.ui.chat, 1)
        await wire.press("enter")
        assert wire.ui.overlay is None
        assert wire.ui.focus == INPUT

    async def test_and_an_own_message_with_no_index_is_refused(self, wire):
        # `index` -1 on a row that *is* the user's own words: the rewind opens
        # (it is a message), and the cut is refused when it turns out to name
        # no thread message.
        await started(wire, [entry(1, text="not in the thread yet")])
        wire.ui.focus = CHAT
        wire.ui.chat.cursor = 0
        await wire.press("enter", "f")
        assert wire.peer.took(protocol.SessionFork) == []
        assert "nothing to rewind to" in wire.ui.note

    async def test_a_fork_is_opened_when_the_core_makes_it(self, wire):
        await self.at_the_second_ask(wire)
        await wire.press("f")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="f1", title="a fork of it")
            )
        )
        assert wire.ui.active_id == "f1"

    async def test_and_it_is_in_the_sidebar_before_its_row_arrives(self, wire):
        await self.at_the_second_ask(wire)
        await wire.press("f")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="f1", title="a fork of it")
            )
        )
        assert "a fork of it" in wire.screen()

    async def test_and_its_transcript_is_asked_for(self, wire):
        await self.at_the_second_ask(wire)
        await wire.press("f")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="f1", title="a fork of it")
            )
        )
        assert wire.peer.last(protocol.SessionOpen).session_id == "f1"


# ------------------------------------------------------- what is not drawn


class TestWhatTheClientDrops:
    async def test_an_event_naming_a_row_we_do_not_have_is_counted(self, wire):
        # §3.2's rule is that a client ignores what it cannot draw, and the
        # count is how that stays a decision rather than an oversight. A
        # `turn.unqueued` for a row this chat never had is one of those: there
        # is nothing to remove.
        await started(wire)
        await wire.tell(
            protocol.TurnUnqueued(session_id="s1", seq=44, text="taken back")
        )
        assert wire.client.dropped["turn.unqueued"] == 1

    async def test_but_the_text_still_comes_back(self, wire):
        # The row is the UI's business and may already have been redrawn; the
        # *text* is the core's and is why the event carries it at all
        # (`protocol.TurnUnqueued`).
        await started(wire)
        await wire.tell(
            protocol.TurnUnqueued(session_id="s1", seq=44, text="taken back")
        )
        assert wire.ui.input.text() == "taken back"

    async def test_a_frame_that_is_not_a_message_at_all_is_dropped(self, wire):
        await started(wire)
        await wire.peer.conn.send(protocol.SessionList())  # a command, from a core
        await settle()
        assert wire.client.dropped["session.list"] == 1

    async def test_a_junk_envelope_costs_that_frame_and_no_more(self, wire):
        await started(wire)
        wire.client.apply(protocol.Envelope(type="nonsense.event", payload={}))
        assert wire.client.dropped["unparseable"] == 1
        await wire.tell(
            protocol.ChatAppend(session_id="s1", entry=entry(2, text="still here"))
        )
        assert "still here" in wire.screen()


class TestNotify:
    async def test_it_reaches_the_footer(self, wire):
        await started(wire)
        await wire.tell(protocol.Notify(text="the scratch quota is nearly full"))
        assert "the scratch quota is nearly full" in plain(wire.frame()[-1])

    async def test_a_toast_longer_than_the_terminal_does_not_break_the_frame(
        self, wire
    ):
        await started(wire)
        await wire.tell(protocol.Notify(text="verbose " * 60))
        assert widths(wire.ui.render(80, 24)) == {80}

    async def test_a_peek_answers_in_a_window_and_not_on_a_timer(self, wire):
        # It used to answer where a toast answers, which expired before a log
        # could be read and could not be selected out of on the way past.
        await started(wire)
        await wire.tell(
            protocol.WatchPeeked(watch_id=7, title="job 4821000", text="shard 4 of 8")
        )
        assert isinstance(wire.ui.overlay, InspectOverlay)
        assert "shard 4 of 8" in wire.screen()

    async def test_a_peeked_log_keeps_its_lines(self, wire):
        # The regression: the tail was flattened with `" ".join(split())`, and
        # a traceback run into one paragraph is not a traceback.
        await started(wire)
        await wire.tell(
            protocol.WatchPeeked(
                watch_id=7,
                title="job 4821000",
                text="[12:41:07] merging shard 3\n[12:41:44] merging shard 4",
            )
        )
        body = [x.strip() for x in wire.frame()]
        assert "[12:41:07] merging shard 3" in body
        assert "[12:41:44] merging shard 4" in body

    async def test_the_window_is_titled_after_the_watch(self, wire):
        await started(wire)
        await wire.tell(
            protocol.WatchPeeked(watch_id=7, title="job 4821000", text="running")
        )
        assert wire.ui.overlay.title == "job 4821000"

    async def test_an_empty_peek_stays_a_toast(self, wire):
        # A job that has not written yet. A blank full screen costs an escape
        # to say what the footer says for free.
        await started(wire)
        await wire.tell(
            protocol.WatchPeeked(watch_id=7, title="job 4821000", text="   \n")
        )
        assert wire.ui.overlay is None
        assert "nothing there yet" in plain(wire.frame()[-1])

    async def test_a_peek_does_not_take_away_a_screen_the_user_opened(self, wire):
        # The reply arrives after a round trip, and a peek is pressed from the
        # rows: `RowUI.window` parks rather than clearing the stack.
        await started(wire)
        wire.ui.focus = SESSIONS
        await wire.press("?")
        await wire.tell(
            protocol.WatchPeeked(watch_id=7, title="job 4821000", text="running")
        )
        assert isinstance(wire.ui.overlay, HelpOverlay)
        await wire.press("esc")
        assert isinstance(wire.ui.overlay, InspectOverlay)


class TestWithNothingOpenYet:
    """The UI draws before the first `session.rows`, so it has to survive it."""

    async def test_it_still_renders(self, wire):
        assert widths(wire.ui.render(80, 24)) == {80}

    async def test_and_a_keypress_sends_nothing(self, wire):
        wire.ui.focus = INPUT
        wire.ui.input.set_text("into the void")
        await wire.press("enter")
        assert wire.peer.commands == []
        assert wire.ui.note == "no session open"

    async def test_an_event_for_an_unknown_session_is_still_kept(self, wire):
        # It arrives before the sidebar that lists it; the state is made on the
        # spot rather than the frame being lost.
        await wire.tell(
            protocol.ChatAppend(session_id="stranger", entry=entry(1, text="early"))
        )
        assert wire.ui.session_for("stranger").entries[0].text == "early"


# --------------------------------------------------- the catalog and profiles

# M7. Both lists are asked for on connect, because `m`, `a` and the new-session
# picker must not each begin with a round trip.


CATALOG = [
    protocol.LLMEntry(
        label="qwen3-27b-fp8",
        model="qwen3-27b-fp8",
        base_url="http://10.12.4.31:20001/v1",
        max_model_len=112000,
        active=True,
    ),
    protocol.LLMEntry(
        label="llama-3.3-70b",
        model="llama-3.3-70b",
        base_url="http://10.12.4.55:20001/v1",
        needs_key=True,
    ),
]


class TestTheCatalog:
    async def test_hello_asks_for_it(self, wire):
        await wire.tell(protocol.Hello(profile="hpc"))
        assert wire.peer.took(protocol.LLMList)

    async def test_and_asks_for_the_probes_too(self, wire):
        # This client draws ● / ○, and the first frame is identical either way
        # (`protocol.LLMList`), so asking costs nothing before it can draw.
        await wire.tell(protocol.Hello(profile="hpc"))
        assert wire.peer.last(protocol.LLMList).probe is True

    async def test_it_fills_the_ui_s_catalog(self, wire):
        await wire.tell(protocol.LLMCatalog(entries=CATALOG))
        assert [x.label for x in wire.ui.catalog] == [
            "qwen3-27b-fp8",
            "llama-3.3-70b",
        ]

    async def test_an_unprobed_entry_keeps_its_third_state(self, wire):
        await wire.tell(protocol.LLMCatalog(entries=CATALOG))
        assert wire.ui.catalog[0].reachable is None

    async def test_a_second_catalog_replaces_the_first(self, wire):
        await wire.tell(protocol.LLMCatalog(entries=CATALOG))
        probed = [x.model_copy(update={"reachable": True}) for x in CATALOG]
        await wire.tell(protocol.LLMCatalog(entries=probed, probed=True))
        assert [x.reachable for x in wire.ui.catalog] == [True, True]

    async def test_it_reaches_the_new_session_picker(self, wire):
        await wire.tell(protocol.LLMCatalog(entries=CATALOG))
        assert [x.text for x in wire.ui._backend_rows()] == [
            "qwen3-27b-fp8",
            "llama-3.3-70b",
        ]

    async def test_a_catalog_that_lands_while_manage_llms_is_open_restates_it(
        self, wire
    ):
        wire.ui.focus = SESSIONS
        await wire.press("m")
        await wire.tell(protocol.LLMCatalog(entries=CATALOG))
        assert "llama-3.3-70b" in wire.screen()

    async def test_the_key_never_crosses(self, wire):
        # `protocol.LLMEntry` has no api_key field at all; what the UI holds
        # is whether there is one.
        await wire.tell(protocol.LLMCatalog(entries=CATALOG))
        assert wire.ui.catalog[1].needs_key is True
        assert not hasattr(wire.ui.catalog[1], "api_key")


class TestTheProfiles:
    async def test_hello_asks_for_them(self, wire):
        await wire.tell(protocol.Hello(profile="hpc"))
        assert wire.peer.took(protocol.ProfileList)

    async def test_the_rows_reach_the_screen(self, wire, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        await wire.tell(
            protocol.ProfileRows(
                rows=[
                    protocol.ProfileRow(name="default", memories=0, is_default=True),
                    protocol.ProfileRow(
                        name="hpc", memories=12, copied_from="default", working=True
                    ),
                ]
            )
        )
        assert [x.name for x in wire.ui.profiles] == ["default", "hpc"]
        assert wire.ui.profiles[1].copied_from == "default"
        assert wire.ui.profiles[0].default is True
        assert wire.ui.profiles[1].working is True

    async def test_a_new_profile_lands_on_the_open_screen(self, wire):
        """The bug this pair is here for: create a profile, and the screen you
        created it from went on showing the list it opened with."""
        wire.ui.focus = SESSIONS
        await wire.press("a")
        await wire.tell(_rows("default", "hpc"))
        assert "hpc" in wire.screen()

    async def test_and_is_still_there_when_the_screen_closes(self, wire):
        # The other half of it: the screen hands its list back on the way out
        # (`RowUI._closed`), so a stale copy there was the answer being undone
        # after it had already arrived.
        wire.ui.focus = SESSIONS
        await wire.press("a")
        await wire.tell(_rows("default", "hpc"))
        await wire.press("esc")
        assert [x.name for x in wire.ui.profiles] == ["default", "hpc"]


def _rows(*names: str) -> protocol.ProfileRows:
    return protocol.ProfileRows(
        rows=[
            protocol.ProfileRow(name=name, memories=0, is_default=name == "default")
            for name in names
        ]
    )


# ------------------------------------------------------- the M7 intents

# Each screen's decision, as the command that carries it. The screens
# themselves are tested in `test_ui_overlays.py`; this is the translation.


class TestTheScreenCommands:
    async def test_thinking_set(self, wire):
        wire.client.intent(state.SetThinking("s1", "medium"))
        await wire.client.flush()
        await settle()
        sent = wire.peer.last(protocol.ThinkingSet)
        assert (sent.session_id, sent.effort) == ("s1", "medium")

    async def test_backend_set_for_one_session(self, wire):
        wire.client.intent(state.SetBackend({"model": "m"}, "s1"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.BackendSet).session_id == "s1"

    async def test_and_none_for_the_default(self, wire):
        # None rather than "": the two halves of `backend.set` are told apart
        # by whether a session is named, and "" would name one that is not there.
        wire.client.intent(state.SetBackend({"model": "m"}))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.BackendSet).session_id is None

    async def test_profile_save(self, wire):
        wire.client.intent(state.SaveProfile("hpc", "memories", "text\n"))
        await wire.client.flush()
        await settle()
        sent = wire.peer.last(protocol.ProfileSave)
        assert (sent.name, sent.kind, sent.text) == ("hpc", "memories", "text\n")

    async def test_profile_create(self, wire):
        wire.client.intent(state.CreateProfile("bench"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.ProfileCreate).name == "bench"

    async def test_a_created_profile_is_asked_about_again(self, wire):
        # Nothing announces one: the core says so in a `notify` and restates
        # the sessions, and `profile.rows` only ever comes when it is asked for.
        wire.client.intent(state.CreateProfile("bench"))
        await wire.client.flush()
        await settle()
        assert wire.peer.took(protocol.ProfileList)

    async def test_and_so_is_a_deleted_one(self, wire):
        wire.client.intent(state.DeleteProfile("bench"))
        await wire.client.flush()
        await settle()
        assert wire.peer.took(protocol.ProfileList)

    async def test_profile_duplicate(self, wire):
        wire.client.intent(state.CopyProfile("hpc", "hpc-gpu"))
        await wire.client.flush()
        await settle()
        sent = wire.peer.last(protocol.ProfileDuplicate)
        assert (sent.source, sent.name) == ("hpc", "hpc-gpu")

    async def test_profile_delete(self, wire):
        wire.client.intent(state.DeleteProfile("writing"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.ProfileDelete).name == "writing"

    async def test_skill_save_and_delete(self, wire):
        wire.client.intent(state.SaveSkill("hpc", "merge", "body"))
        wire.client.intent(state.DeleteSkill("hpc", "merge"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.SkillSave).text == "body"
        assert wire.peer.last(protocol.SkillSave).level == "profile"
        assert wire.peer.last(protocol.SkillDelete).name == "merge"

    async def test_a_save_carries_the_level_the_creator_chose(self, wire):
        wire.client.intent(state.SaveSkill("hpc", "etiquette", "body", "global"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.SkillSave).level == "global"

    async def test_the_menu_asks_for_every_skill_the_profile_can_call(
        self, wire
    ):
        # The wide scope, which is what makes `/plan` work on a fresh install:
        # the profile's own list has nothing in it and HPCA's shipped skills
        # are still callable.
        wire.client.ask_skills("hpc")
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.SkillList).scope == "visible"

    async def test_and_an_editor_asks_only_for_what_it_may_write(self, wire):
        wire.client.intent(state.FetchSkills("hpc"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.SkillList).scope == "own"

    async def test_the_two_answers_fill_two_different_lists(self, wire):
        # Letting either land in the other is how a screen ends up offering to
        # delete a skill that ships with HPCA.
        wire.ui.profiles = [state.ProfileInfo(name="hpc")]
        await wire.tell(
            protocol.SkillRows(
                profile="hpc",
                scope="visible",
                skills=[
                    protocol.SkillRow(name="plan", level="builtin"),
                    protocol.SkillRow(name="merge", level="profile"),
                ],
            )
        )
        assert [x.name for x in wire.ui._skills["hpc"]] == ["plan", "merge"]
        assert wire.ui.profiles[0].skills == [], "not the editable list"
        await wire.tell(
            protocol.SkillRows(
                profile="hpc",
                scope="own",
                skills=[protocol.SkillRow(name="merge", level="profile")],
            )
        )
        assert [x.name for x in wire.ui.profiles[0].skills] == ["merge"]

    async def test_a_draft_is_asked_for_and_opens_the_form(self, wire):
        wire.client.intent(state.DraftSkill("hpc", "watch a run", "s1"))
        await wire.client.flush()
        await settle()
        asked = wire.peer.last(protocol.SkillDraft)
        assert (asked.request, asked.session_id) == ("watch a run", "s1")
        await wire.tell(
            protocol.SkillDrafted(
                profile="hpc",
                request="watch a run",
                name="watch-run",
                description="when a run needs watching",
                body="1. squeue",
            )
        )
        assert wire.ui.overlay is not None
        assert "watch-run" in wire.screen()

    async def test_and_a_failed_one_opens_it_empty(self, wire):
        await wire.tell(
            protocol.SkillDrafted(
                profile="hpc", request="watch a run", error="no backend"
            )
        )
        assert wire.ui.overlay is not None
        assert wire.ui.overlay.value("name") == ""
        assert "no backend" in wire.screen()

    async def test_a_request_with_nothing_open_names_no_session(self, wire):
        # Null, not "": the wire spells "no conversation" as a missing id.
        wire.client.intent(state.DraftSkill("hpc", "watch a run"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.SkillDraft).session_id is None

    async def test_the_counts_sort_the_menu(self, wire):
        await wire.tell(protocol.CommandCounts(counts={"thinking": 7}))
        assert wire.ui.command_counts == {"thinking": 7}
        wire.ui.focus = INPUT
        await wire.press("/")
        assert [x.name for x in wire.ui.menu()][0] == "thinking"

    async def test_memory_resolve_is_positional(self, wire):
        wire.client.intent(state.ResolveMemory("s1", (True, False)))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.MemoryResolve).approved == [True, False]

    async def test_a_profile_scoped_command_names_no_session(self, wire):
        wire.client.intent(state.RunCommand("skills-list"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.CommandRun).session_id is None

    async def test_a_session_scoped_one_does(self, wire):
        wire.client.intent(state.RunCommand("compact", session_id="s1"))
        await wire.client.flush()
        await settle()
        assert wire.peer.last(protocol.CommandRun).session_id == "s1"


class TestTheSettingsFile:
    """`settings.get` / `settings.save`, and the split the two of them make.

    The editor checks that the text is JSON, because that is answerable here
    with the standard library and is what it needs synchronously to refuse to
    close. Whether it is valid *settings* — and applying the change, which
    only the core can do — is the core's half, and comes back as
    `settings.body` with an ``error`` on it.
    """

    async def test_the_first_c_asks_for_the_file(self, wire):
        # Asked for as the screen opens, not carried from startup: the core
        # writes this file too, so a copy from last time can be wrong.
        wire.ui.focus = SESSIONS
        await wire.press("c")
        assert wire.peer.took(protocol.SettingsGet)

    async def test_and_the_answer_fills_the_editor(self, wire):
        wire.ui.focus = SESSIONS
        await wire.press("c")
        await wire.tell(protocol.SettingsBody(text='{"llm": {"model": "lazy"}}'))
        assert "lazy" in wire.ui.overlay.editor.text()

    async def test_until_it_lands_there_is_nothing_to_type_into(self, wire):
        # An empty box would be a file this screen could save over.
        wire.ui.focus = SESSIONS
        await wire.press("c")
        assert "fetching" in wire.screen()
        await wire.press("X")
        assert wire.ui.overlay.editor.text() == ""

    async def test_saving_sends_the_text(self, wire):
        wire.client.intent(state.SaveSettings('{"llm": {"model": "new"}}'))
        await wire.client.flush()
        await settle()
        sent = wire.peer.last(protocol.SettingsSave)
        assert sent.text == '{"llm": {"model": "new"}}'

    async def test_and_what_landed_comes_back(self, wire):
        # Not the bytes that were sent: the core writes the validated model
        # back out, so the file says more than the editor did.
        await wire.tell(protocol.SettingsBody(text='{"llm": {"model": "new"}}'))
        assert "new" in wire.ui.settings_json

    async def test_a_refused_save_says_why_and_keeps_the_file(self, wire):
        await wire.tell(
            protocol.SettingsBody(
                text='{"kept": true}',
                error="database.sync_interval_s: not a whole number",
            )
        )
        assert "sync_interval_s" in wire.ui.note
        assert wire.ui.settings_json == '{"kept": true}'

    async def test_the_json_check_reaches_the_editor(self, wire):
        wire.ui.focus = SESSIONS
        await wire.press("c")
        await wire.tell(protocol.SettingsBody(text="{}"))
        wire.ui.overlay.editor.set_text("{not json")
        await wire.press("esc")
        assert wire.ui.overlay is not None, "broken json keeps it open"
        assert "invalid JSON" in wire.screen()


class TestThinkingOnTheSidebar:
    async def test_the_level_reaches_the_meter(self, wire):
        await wire.tell(
            protocol.SessionRows(
                rows=[
                    protocol.SessionRow(
                        session_id="s1", title="t", thinking="medium"
                    )
                ]
            )
        )
        assert wire.ui.session_for("s1").thinking == "medium"
        assert wire.ui.session_for("s1").context.effort == "medium"

    async def test_two_sessions_keep_their_own(self, wire):
        await wire.tell(
            protocol.SessionRows(
                rows=[
                    protocol.SessionRow(session_id="s1", title="a", thinking="low"),
                    protocol.SessionRow(session_id="s2", title="b", thinking="xhigh"),
                ]
            )
        )
        assert wire.ui.session_for("s1").thinking == "low"
        assert wire.ui.session_for("s2").thinking == "xhigh"

    async def test_a_session_that_chose_nothing_shows_no_level(self, wire):
        await wire.tell(
            protocol.SessionRows(
                rows=[protocol.SessionRow(session_id="s1", title="a")]
            )
        )
        assert "think" not in wire.screen()


class TestMemoryProposals:
    PROPOSALS = [
        protocol.Proposal(scope="profile", kind="fact", text="scratch is /scratch"),
    ]

    async def test_they_open_the_review_for_the_session_on_screen(self, wire):
        await started(wire)
        await wire.tell(
            protocol.MemoryProposals(session_id="s1", proposals=self.PROPOSALS)
        )
        assert "scratch is /scratch" in wire.screen()

    async def test_another_session_s_offer_only_toasts(self, wire):
        # A review that covered somebody else's chat would be the modal mistake
        # the inline approval prompt is careful about, one screen over.
        await started(wire)
        await wire.tell(
            protocol.MemoryProposals(session_id="s2", proposals=self.PROPOSALS)
        )
        assert wire.ui.overlay is None
        assert "to review" in plain(wire.frame()[-1])

    async def test_and_is_still_held_for_when_it_is_opened(self, wire):
        await started(wire)
        await wire.tell(
            protocol.MemoryProposals(session_id="s2", proposals=self.PROPOSALS)
        )
        assert len(wire.ui.session_for("s2").proposals) == 1

    async def test_the_verdicts_go_back_positionally(self, wire):
        await started(wire)
        await wire.tell(
            protocol.MemoryProposals(session_id="s1", proposals=self.PROPOSALS)
        )
        await wire.press("y")
        sent = wire.peer.last(protocol.MemoryResolve)
        assert (sent.session_id, sent.approved) == ("s1", [True])


class TestThePaletteArrivesTheSameWay:
    """A theme is a display setting, so it travels the path they all do.

    Which is the whole of "hot reload": the core already restates the display
    section whenever a save changes it (`core.service._apply_settings`), and
    the colours ride that. There is no file watcher and nothing polls — the
    editor saves, the core diffs, and the next frame is drawn in the new
    palette.
    """

    def teardown_method(self):
        # The palette is module state; a test that swapped it and walked away
        # would be the next test's surprise.
        theme.reset()

    async def test_a_saved_colour_reaches_the_next_frame(self, wire):
        await started(wire, [entry(1, text="run it", at=AT)])
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(
                    palette=protocol.Palette(user="#ff0000")
                )
            )
        )
        assert theme.user == "\x1b[38;2;255;0;0m"
        assert any("38;2;255;0;0" in row for row in wire.ui.render(120, 40))

    async def test_and_the_conversation_survives_it(self, wire):
        # `restyle`, not `reset`: recolouring is not a reason to lose what was
        # said, what is open in it, or where the cursor was.
        await started(wire, [entry(1, text="run it", at=AT)])
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(
                    palette=protocol.Palette(agent="#00ff00")
                )
            )
        )
        assert "run it" in wire.screen()
        assert [x.key for x in wire.ui.chat.items] == ["1"]

    async def test_a_colour_the_theme_cannot_draw_costs_only_that_colour(
        self, wire
    ):
        # The settings model refuses these where the user can read why. This
        # is the far side of the wire, in the process holding a terminal in
        # raw mode, and there a palette is not worth a lost screen.
        await started(wire, [entry(1, text="run it", at=AT)])
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(
                    palette=protocol.Palette(user="rubbish", chrome="#0000ff")
                )
            )
        )
        assert theme.user == theme.sgr("215"), "the bad one kept the built-in"
        assert theme.chrome == "\x1b[38;2;0;0;255m", "the good one applied"
        assert "run it" in wire.screen(), "and the frame still draws"

    async def test_the_flash_hold_travels_with_it(self, wire):
        await started(wire, [entry(1, text="run it", at=AT)])
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(focus_flash_seconds=0.4)
            )
        )
        assert theme.flash_hold == 0.4
