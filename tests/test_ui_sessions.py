"""M6: making, naming, deleting and rewinding a conversation.

The acceptance sections this file answers, claim by claim:

* "Titles, rename, delete" — the `r`/`t`/`d` half of it, and the delete's
  consequences for the chat and the draft. What is *not* here, because the
  core does not do it yet, is written down at the bottom of this docstring.
* "Rewind and fork" — the cuts that `RewindOverlay` has been offering since
  before the port and that nothing carried out until `session.fork` and
  `session.rollback` existed.
* "Drafts" — every claim, since a draft lives on `SessionState` and most of
  them should already be true; a claim that is true by accident is a claim
  that is one refactor from being false.
* "Navigating the entry", the two session bullets: Enter on `(new session)`
  asks for a profile and escaping it opens nothing.

Driven through `ui_harness.connected`: a real `InProcessConnection.pair()`
with a scripted core on the far end, so every assertion is about a frame or
about a typed command on the wire.

**Not asserted here, and why.** The automatic titling claims — the column
shows the model's summary, a session is titled once, a reopened session is
not retitled, a slash command alone does not trigger one, a hand-written name
is never overwritten, a failed title call leaves the name alone — are all
about work the core does after a turn. `TurnScheduler` takes an
`on_turn_result` hook for exactly that and nothing passes one
(`core.service.build_service`), so there is no automatic titling to test from
either side of the wire yet. `t` (`session.retitle`) is the whole of titling
today, and it is asserted below.
"""

from __future__ import annotations

import pytest

from hpca import protocol
from hpca.ui.app import (
    CHAT,
    INPUT,
    NEW_SESSION_KEY,
    NEW_SESSION_ROW,
    SESSIONS,
    RowUI,
)
from hpca.ui.overlays import (
    BACKEND,
    EMPTY_REFUSAL,
    PROFILE,
    NewSessionOverlay,
    RenameOverlay,
    RewindOverlay,
    choice,
)
from tests.ui_harness import Wire, connected, frame, plain, widths

# Two conversations under two profiles, one of them the default — which is the
# one the sidebar does *not* tag.
ROWS = [
    protocol.SessionRow(
        session_id="s1", title="the first thing", profile="hpc", mode="auto"
    ),
    protocol.SessionRow(
        session_id="s2", title="the second thing", profile="default", mode="auto"
    ),
]

# Where each conversation is in the sidebar. Row 0 is `(new session)`.
FIRST, SECOND = 1, 2

WIDTHS = [80, 100, 137]

BACKENDS = [
    choice("cluster-qwen", "qwen3-27b-fp8", value='{"model": "qwen3-27b-fp8"}'),
    choice("big-llama", "llama-3.3-70b", value='{"model": "llama-3.3-70b"}'),
]


def entry(seq: int, kind: str = "user", text: str = "", **kw) -> protocol.Entry:
    return protocol.Entry(kind=kind, text=text or f"row {seq}", seq=seq, **kw)


# q1 / a1 / q2 / a2, the two exchanges every rewind test cuts into.
EXCHANGES = [
    entry(1, text="q1", index=0),
    entry(2, "assistant", "a1", index=1),
    entry(3, text="q2", index=2),
    entry(4, "assistant", "a2", index=3),
]


@pytest.fixture
async def wire():
    async with connected() as w:
        yield w


@pytest.fixture
async def with_backends():
    """A UI that has been handed a catalog, which a real run has not yet."""
    async with connected(RowUI(backends=list(BACKENDS))) as w:
        yield w


async def started(w: Wire, entries: list[protocol.Entry] | None = None) -> Wire:
    """Hello, the sidebar, and the first transcript."""
    await w.tell(protocol.Hello(profile="hpc"))
    await w.tell(protocol.SessionRows(rows=list(ROWS)))
    await w.tell(
        protocol.ChatReset(
            session_id="s1",
            entries=list(EXCHANGES) if entries is None else entries,
        )
    )
    w.peer.clear()
    return w


def listed(w: Wire) -> list[str]:
    """The conversations the sidebar is showing, by id, in order.

    Read off the pane rather than the frame, because the footer note says
    what was just deleted — a screen-wide search would find the row's title
    in the sentence about its going away.
    """
    return [
        x.key for x in w.ui.session_pane.items if x.key != NEW_SESSION_KEY
    ]


async def on_row(w: Wire, row: int) -> Wire:
    """Put the sidebar cursor on that line and leave the focus there."""
    w.ui.focus = SESSIONS
    w.ui.session_pane.cursor = row
    return w


# --------------------------------------------------------- the (new session) row


class TestTheRowThatIsNotASession:
    async def test_the_sidebar_offers_one(self, wire):
        await started(wire)
        assert NEW_SESSION_ROW in wire.screen()

    async def test_it_is_the_first_row(self, wire):
        await started(wire)
        assert wire.ui.session_pane.items[0].key == NEW_SESSION_KEY

    async def test_a_fresh_install_has_nothing_else(self, wire):
        # The hole M6 exists to fill: no sessions at all, and still a way in.
        await wire.tell(protocol.Hello(profile="hpc"))
        await wire.tell(protocol.SessionRows(rows=[]))
        assert [x.key for x in wire.ui.session_pane.items] == [NEW_SESSION_KEY]

    async def test_the_highlight_follows_the_open_session_not_this_row(self, wire):
        # A cursor left on row 0 would answer the next Enter by making a
        # conversation instead of opening the one that was just opened.
        await started(wire)
        assert wire.ui.session_pane.here() == "s1"

    async def test_the_footer_offers_only_what_works_here(self, wire):
        await started(wire)
        await on_row(wire, 0)
        footer = plain(wire.ui.render(160, 40)[-1])
        assert "enter start a session" in footer
        assert "rename" not in footer and "delete" not in footer


class TestStartingOne:
    async def test_enter_asks_for_a_profile(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter")
        assert isinstance(wire.ui.overlay, NewSessionOverlay)
        assert wire.ui.overlay.stage == PROFILE

    async def test_and_asks_for_nothing_from_the_core_yet(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter")
        assert wire.peer.commands == []

    async def test_escaping_the_picker_creates_nothing(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "esc")
        assert wire.ui.overlay is None
        assert wire.peer.took(protocol.SessionNew) == []

    async def test_the_profiles_offered_are_the_ones_the_ui_knows(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter")
        offered = [x.text for x in wire.ui.overlay.pane.items]
        # The default leads; then the core's own profile, then the sidebar's.
        assert offered[0] == "default"
        assert "hpc" in offered

    async def test_the_make_one_row_of_the_profiles_screen_is_not_a_profile(self):
        # `(new profile)` is that screen's own "make one" line. Offering it
        # here would start a conversation under a profile of that name;
        # creating a profile from inside the picker is §4.3 item 27 (M7).
        from hpca.ui.pane import Item

        ui = RowUI(
            profiles=[
                Item(head=f"{'hpc':<18}12 memories"),
                Item(head="(new profile)"),
            ]
        )
        assert [x.text for x in ui._profile_rows()] == ["default", "hpc"]

    async def test_choosing_one_asks_the_core_for_a_session(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "down", "enter")  # "hpc"
        assert wire.peer.last(protocol.SessionNew).profile == "hpc"

    async def test_the_session_runs_under_the_profile_that_was_chosen(self, wire):
        # The other half — that the profile's memories reach the prompt — is
        # the core's (`core.service._new_session` hands the name to
        # `SessionStore.create`); what the UI owes is the name it sends.
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "enter")  # "default"
        assert wire.peer.last(protocol.SessionNew).profile == "default"

    async def test_with_no_backends_configured_no_llm_is_asked_for(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "enter")
        assert wire.ui.overlay is None, "one keypress, one session"
        assert wire.peer.last(protocol.SessionNew).backend is None

    async def test_and_the_session_is_still_created(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "enter")
        assert len(wire.peer.took(protocol.SessionNew)) == 1

    async def test_the_new_session_is_opened(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "enter")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="n1", title="(untitled)")
            )
        )
        assert wire.ui.active_id == "n1"
        assert wire.peer.last(protocol.SessionOpen).session_id == "n1"

    async def test_and_lands_in_the_message_box(self, wire):
        await started(wire)
        await on_row(wire, 0)
        await wire.press("enter", "enter")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="n1", title="(untitled)")
            )
        )
        assert wire.ui.focus == INPUT

    async def test_a_fresh_install_can_start_one_from_a_keypress(self, wire):
        await wire.tell(protocol.Hello(profile="hpc"))
        await wire.tell(protocol.SessionRows(rows=[]))
        wire.peer.clear()
        await on_row(wire, 0)
        await wire.press("enter", "enter")
        assert wire.peer.last(protocol.SessionNew).profile == "default"


class TestTheLlmPicker:
    async def test_a_configured_catalog_is_asked_about(self, with_backends):
        await started(with_backends)
        await on_row(with_backends, 0)
        await with_backends.press("enter", "enter")
        assert with_backends.ui.overlay is not None
        assert with_backends.ui.overlay.stage == BACKEND

    async def test_and_nothing_is_created_until_it_is_answered(self, with_backends):
        await started(with_backends)
        await on_row(with_backends, 0)
        await with_backends.press("enter", "enter")
        assert with_backends.peer.took(protocol.SessionNew) == []

    async def test_escaping_it_creates_nothing(self, with_backends):
        await started(with_backends)
        await on_row(with_backends, 0)
        await with_backends.press("enter", "enter", "esc")
        assert with_backends.ui.overlay is None
        assert with_backends.peer.took(protocol.SessionNew) == []

    async def test_choosing_one_pins_the_session_to_it(self, with_backends):
        await started(with_backends)
        await on_row(with_backends, 0)
        await with_backends.press("enter", "enter", "down", "enter")
        sent = with_backends.peer.last(protocol.SessionNew)
        assert sent.backend == BACKENDS[1].text
        assert sent.profile == "default"

    async def test_the_ui_never_looks_inside_the_backend_string(self, with_backends):
        # Whatever the catalog handed over comes back untouched: what
        # identifies a backend is `core.backends`' business.
        await started(with_backends)
        await on_row(with_backends, 0)
        await with_backends.press("enter", "enter", "enter")
        assert with_backends.peer.last(protocol.SessionNew).backend == (
            '{"model": "qwen3-27b-fp8"}'
        )

    @pytest.mark.parametrize("width", WIDTHS)
    def test_every_row_is_exactly_the_width(self, width: int):
        screen = NewSessionOverlay(
            [choice("default", "the fallback")], list(BACKENDS)
        )
        assert widths(screen.render(width, 20)) == {width}
        screen.handle("enter", width, 20)
        assert widths(screen.render(width, 20)) == {width}


# ------------------------------------------------------------------ renaming


class TestRename:
    async def test_r_opens_a_prompt_with_the_current_name(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r")
        assert isinstance(wire.ui.overlay, RenameOverlay)
        assert wire.ui.overlay.editor.text() == "the first thing"

    async def test_the_cursor_is_at_the_end_so_the_name_is_edited(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r", *" again")
        assert wire.ui.overlay.editor.text() == "the first thing again"

    async def test_enter_saves_it(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r", *" again", "enter")
        sent = wire.peer.last(protocol.SessionRename)
        assert (sent.session_id, sent.title) == ("s1", "the first thing again")

    async def test_and_the_sidebar_says_so_before_the_core_answers(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r", "ctrl-u", *"STAR OOM", "enter")
        assert "STAR OOM" in wire.screen()

    async def test_escape_keeps_the_old_name(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r", *" again", "esc")
        assert wire.peer.took(protocol.SessionRename) == []
        assert wire.ui.session_for("s1").title == "the first thing"

    async def test_an_empty_name_is_refused(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r", "ctrl-u", "enter")
        assert wire.peer.took(protocol.SessionRename) == []
        assert isinstance(wire.ui.overlay, RenameOverlay), "and says so, on screen"
        assert EMPTY_REFUSAL in wire.screen()

    async def test_whitespace_is_not_a_name_either(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("r", "ctrl-u", " ", " ", "enter")
        assert wire.peer.took(protocol.SessionRename) == []

    async def test_it_renames_the_row_it_was_opened_on(self, wire):
        # Not the open session: `r` acts on what the sidebar is pointing at.
        await started(wire)
        await on_row(wire, SECOND)
        await wire.press("r", "ctrl-u", *"elsewhere", "enter")
        assert wire.peer.last(protocol.SessionRename).session_id == "s2"

    @pytest.mark.parametrize("width", WIDTHS)
    def test_every_row_is_exactly_the_width(self, width: int):
        screen = RenameOverlay("a name that has to fit", "s1")
        assert widths(screen.render(width, 12)) == {width}


class TestRetitle:
    async def test_t_asks_the_model(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("t")
        assert wire.peer.last(protocol.SessionRetitle).session_id == "s1"

    async def test_it_asks_about_the_highlighted_row(self, wire):
        await started(wire)
        await on_row(wire, SECOND)
        await wire.press("t")
        assert wire.peer.last(protocol.SessionRetitle).session_id == "s2"

    async def test_nothing_is_renamed_until_the_core_says_so(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("t")
        assert wire.ui.session_for("s1").title == "the first thing"

    async def test_a_background_titling_lights_the_row_and_not_the_chat(self, wire):
        # The claim is that retitling a highlighted-but-not-open session lights
        # its sidebar row and never the chat spinner. The UI half is asserted
        # here; the core does not yet *say* it is titling — `session.retitle`
        # emits no `turn.activity` (`core.service._retitle`) — so the event is
        # scripted rather than provoked.
        await started(wire)
        await on_row(wire, SECOND)
        await wire.press("t")
        await wire.tell(
            protocol.TurnActivity(session_id="s2", activity="writing a title")
        )
        assert "⟳" in TestTheSidebarMarkers.row_for(wire, "the second thing")
        assert "writing a title" not in wire.screen(), "not the open chat's spinner"

    async def test_titling_the_open_session_parks_the_chat_spinner(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("t")
        await wire.tell(
            protocol.TurnActivity(session_id="s1", activity="writing a title")
        )
        assert "writing a title" in wire.screen()

    async def test_a_refusal_is_shown_and_leaves_the_name_alone(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("t")
        await wire.tell(
            protocol.Notify(severity="error", text="The model could not write a title.")
        )
        assert "could not write a title" in wire.ui.note
        assert wire.ui.session_for("s1").title == "the first thing"


class TestTheSessionKeysAreInertOnTheNewSessionRow:
    @pytest.mark.parametrize("key", ["r", "t", "d"])
    async def test_it_does_nothing(self, wire, key: str):
        await started(wire)
        await on_row(wire, 0)
        await wire.press(key)
        assert wire.ui.overlay is None
        assert wire.ui.confirm is None
        assert wire.peer.commands == []


# ------------------------------------------------------------------ deleting


class TestDelete:
    async def test_d_asks_first(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d")
        assert wire.ui.confirm is not None
        assert "the first thing" in wire.ui.confirm.question
        assert "log" in wire.ui.confirm.question, "the dialog says the log stays"
        assert wire.peer.took(protocol.SessionDelete) == []

    async def test_and_then_deletes(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "y")
        assert wire.peer.last(protocol.SessionDelete).session_id == "s1"

    async def test_declining_keeps_the_session(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "n")
        assert wire.peer.took(protocol.SessionDelete) == []
        assert "the first thing" in wire.screen()

    async def test_escape_is_also_no(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "esc")
        assert wire.peer.took(protocol.SessionDelete) == []

    async def test_the_row_goes_at_once(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "y")
        assert listed(wire) == ["s2"]

    async def test_deleting_the_open_session_empties_the_chat(self, wire):
        await started(wire)
        assert wire.ui.active_id == "s1"
        await on_row(wire, FIRST)
        await wire.press("d", "y")
        assert wire.ui.active_id == ""
        assert wire.ui.chat.items == []

    async def test_and_leaves_the_cursor_where_there_is_something_to_do(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "y")
        assert wire.ui.focus == SESSIONS

    async def test_and_there_is_nothing_to_send_to(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "y")
        wire.ui.focus = INPUT
        wire.ui.input.set_text("into the void")
        await wire.press("enter")
        assert wire.peer.took(protocol.TurnSubmit) == []
        assert "no session open" in wire.ui.note

    async def test_deleting_another_leaves_the_open_one_alone(self, wire):
        await started(wire)
        await on_row(wire, SECOND)
        await wire.press("d", "y")
        assert wire.ui.active_id == "s1"
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1", "q2", "a2"]

    async def test_the_chat_history_goes_with_it(self, wire):
        await started(wire)
        await on_row(wire, FIRST)
        await wire.press("d", "y")
        # Not merely off screen: the state is gone, so a later `session.rows`
        # that still listed it would rebuild an empty one rather than restore
        # the transcript.
        assert "s1" not in wire.ui._states

    async def test_a_later_session_list_does_not_bring_it_back(self, wire):
        await started(wire)
        await on_row(wire, SECOND)
        await wire.press("d", "y")
        await wire.tell(protocol.SessionRows(rows=[ROWS[0]]))
        assert listed(wire) == ["s1"]


# -------------------------------------------------------------------- markers


class TestTheSidebarMarkers:
    async def test_a_decision_lights_the_row(self, wire):
        await started(wire)
        await wire.tell(
            protocol.DecisionRequested(session_id="s2", payload={"tool": "rm"})
        )
        assert "!" in self.row_for(wire, "the second thing")

    async def test_the_core_flag_lights_it_too(self, wire):
        # Both are consulted: the flag is what the core knows, the local
        # decision is what has arrived since the last `session.rows`.
        await started(wire)
        await wire.tell(
            protocol.SessionRows(
                rows=[
                    ROWS[0],
                    ROWS[1].model_copy(update={"flags": ["decision"]}),
                ]
            )
        )
        assert "!" in self.row_for(wire, "the second thing")

    async def test_a_running_turn_lights_it(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        assert "⟳" in self.row_for(wire, "the second thing")

    async def test_a_silent_backend_call_lights_it_too(self, wire):
        # `turn.activity` with no `turn.started`: the titler, a compaction, a
        # silent `/conclude`. There is nothing to stop, and there is still
        # something happening in that conversation.
        await started(wire)
        await wire.tell(
            protocol.TurnActivity(session_id="s2", activity="writing a title")
        )
        assert "⟳" in self.row_for(wire, "the second thing")

    async def test_and_goes_out_when_it_ends(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        assert "⟳" not in self.row_for(wire, "the second thing")

    async def test_a_non_default_profile_is_tagged(self, wire):
        await started(wire)
        assert "hpc" in self.row_for(wire, "the first thing")

    async def test_the_default_one_is_not(self, wire):
        await started(wire)
        assert "default" not in self.row_for(wire, "the second thing")

    @staticmethod
    def row_for(w: Wire, title: str) -> str:
        rows = [x for x in w.frame() if title in x]
        assert rows, f"no sidebar row for {title!r}"
        return rows[0]


# --------------------------------------------------------------- the rewind


class TestTheRewindIsConnected:
    """`RewindOverlay` has existed since before the port; these are the two
    choices that did nothing until `session.fork`/`session.rollback`."""

    async def at_q2(self, w: Wire) -> Wire:
        await started(w)
        w.ui.focus = CHAT
        w.ui.chat.cursor = 2  # q2
        await w.press("enter")
        assert isinstance(w.ui.overlay, RewindOverlay)
        return w

    async def test_enter_on_your_own_message_opens_it(self, wire):
        await self.at_q2(wire)
        assert wire.ui.overlay.message == "q2"

    async def test_escape_changes_nothing(self, wire):
        await self.at_q2(wire)
        await wire.press("esc")
        assert wire.peer.commands == []
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1", "q2", "a2"]

    async def test_rollback_names_the_message_the_cut_is_made_at(self, wire):
        await self.at_q2(wire)
        await wire.press("r")
        sent = wire.peer.last(protocol.SessionRollback)
        assert (sent.session_id, sent.index) == ("s1", 2)

    async def test_rollback_trims_the_conversation(self, wire):
        await self.at_q2(wire)
        await wire.press("r")
        # The core answers a rollback with the transcript that is left
        # (`protocol.SessionRollback`); the UI is what draws the trim.
        await wire.tell(
            protocol.ChatReset(session_id="s1", entries=list(EXCHANGES[:2]))
        )
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1"]
        assert "q2" not in wire.screen()

    async def test_rollback_to_the_first_message_empties_the_thread(self, wire):
        await started(wire)
        wire.ui.focus = CHAT
        wire.ui.chat.cursor = 0
        await wire.press("enter", "r")
        assert wire.peer.last(protocol.SessionRollback).index == 0
        await wire.tell(protocol.ChatReset(session_id="s1", entries=[]))
        assert wire.ui.session.entries == []

    async def test_the_next_turn_runs_on_the_trimmed_thread(self, wire):
        await self.at_q2(wire)
        await wire.press("r")
        await wire.tell(
            protocol.ChatReset(session_id="s1", entries=list(EXCHANGES[:2]))
        )
        wire.ui.focus = INPUT
        await wire.press(*"q2 but better", "enter")
        sent = wire.peer.last(protocol.TurnSubmit)
        # One message, on the session that was trimmed: nothing the UI holds
        # resends the cut rows, and what the model sees is the core's to say
        # (`test_core_service.py` drives the graph for that half).
        assert (sent.session_id, sent.text) == ("s1", "q2 but better")
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1"]

    async def test_a_refused_rollback_is_shown_and_cuts_nothing(self, wire):
        # Refusing is the core's job — `TurnScheduler.rewind_blocker` is the
        # one place that knows whether a turn, a decision or a queued message
        # is in the way, and it phrases the answer to be shown. What the UI
        # owes is to show it and to leave the transcript alone.
        await self.at_q2(wire)
        await wire.press("r")
        await wire.tell(
            protocol.Notify(
                severity="warning",
                text="Cannot roll back: a turn is running in this session.",
            )
        )
        assert "Cannot roll back" in wire.ui.note
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1", "q2", "a2"]

    async def test_a_fork_is_not_gated_the_way_a_rollback_is(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s1"))
        wire.ui.focus = CHAT
        wire.ui.chat.cursor = 2
        await wire.press("enter", "f")
        assert wire.peer.last(protocol.SessionFork).index == 2

    async def test_a_fork_opens_the_copy(self, wire):
        await self.at_q2(wire)
        await wire.press("f")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(
                    session_id="f1", title="the first thing (fork)", profile="hpc"
                )
            )
        )
        await wire.tell(
            protocol.ChatReset(session_id="f1", entries=list(EXCHANGES[:2]))
        )
        assert wire.ui.active_id == "f1"
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1"]

    async def test_the_source_is_untouched(self, wire):
        await self.at_q2(wire)
        await wire.press("f")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="f1", title="a fork")
            )
        )
        assert [e.text for e in wire.ui.session_for("s1").entries] == [
            "q1",
            "a1",
            "q2",
            "a2",
        ]

    async def test_both_sessions_are_in_the_sidebar(self, wire):
        await self.at_q2(wire)
        await wire.press("f")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="f1", title="a fork of it")
            )
        )
        screen = wire.screen()
        assert "the first thing" in screen and "a fork of it" in screen

    async def test_a_copy_still_only_copies(self, wire):
        await self.at_q2(wire)
        await wire.press("c")
        assert wire.peer.commands == []
        assert wire.ui.input.text() == "q2"


# ---------------------------------------------------------------------- drafts


class TestDrafts:
    """Every claim in the acceptance list's "Drafts" section.

    They should mostly fall out of a draft living on `SessionState` — which is
    exactly why they are asserted rather than assumed.
    """

    async def two_sessions(self, w: Wire) -> Wire:
        await started(w)
        await w.tell(protocol.ChatReset(session_id="s2", entries=[entry(1)]))
        return w

    async def switch_to(self, w: Wire, row: int) -> Wire:
        await on_row(w, row)
        await w.press("enter")
        return w

    async def test_a_draft_does_not_follow_into_the_next_session(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"half typed")
        await self.switch_to(wire, SECOND)
        assert wire.ui.input.text() == ""

    async def test_returning_to_a_session_restores_its_draft(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"half typed")
        await self.switch_to(wire, SECOND)
        await self.switch_to(wire, FIRST)
        assert wire.ui.input.text() == "half typed"

    async def test_each_session_keeps_its_own(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"for the first")
        await self.switch_to(wire, SECOND)
        wire.ui.focus = INPUT
        await wire.press(*"for the second")
        await self.switch_to(wire, FIRST)
        assert wire.ui.input.text() == "for the first"
        await self.switch_to(wire, SECOND)
        assert wire.ui.input.text() == "for the second"

    async def test_a_multi_line_draft_comes_back_whole(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"one", "shift-enter", *"two")
        await self.switch_to(wire, SECOND)
        await self.switch_to(wire, FIRST)
        assert wire.ui.input.text() == "one\ntwo"

    async def test_a_sent_message_leaves_no_draft_behind(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"send me", "enter")
        assert wire.ui.input.text() == ""
        await self.switch_to(wire, SECOND)
        await self.switch_to(wire, FIRST)
        assert wire.ui.input.text() == ""

    async def test_a_new_session_starts_with_an_empty_entry(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"half typed")
        await on_row(wire, 0)
        await wire.press("enter", "enter")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="n1", title="(untitled)")
            )
        )
        assert wire.ui.input.text() == ""

    async def test_closing_and_reopening_a_session_keeps_the_draft(self, wire):
        # "Closed" is what leaving one is here: the core is told nothing is on
        # screen, and coming back does not re-read the transcript.
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"kept")
        await self.switch_to(wire, SECOND)
        wire.peer.clear()
        await self.switch_to(wire, FIRST)
        assert wire.peer.took(protocol.SessionOpen) == []
        assert wire.ui.input.text() == "kept"

    async def test_deleting_a_session_drops_its_draft(self, wire):
        await self.two_sessions(wire)
        await self.switch_to(wire, SECOND)
        wire.ui.focus = INPUT
        await wire.press(*"goes with it")
        await on_row(wire, SECOND)
        await wire.press("d", "y")
        # Back as a stranger: whatever the core says about `s2` next builds a
        # session with nothing typed in it.
        assert wire.ui.session_for("s2").draft.text() == ""

    async def test_a_reused_message_becomes_that_sessions_draft(self, wire):
        await self.two_sessions(wire)
        wire.ui.focus = CHAT
        wire.ui.chat.cursor = 2
        await wire.press("enter", "c")
        await self.switch_to(wire, SECOND)
        await self.switch_to(wire, FIRST)
        assert wire.ui.input.text() == "q2"

    async def test_a_parked_slash_draft_does_not_bring_its_menu_back(self, wire):
        # The claim is "a parked `/…` draft brings its autocomplete menu back
        # with it". The menu is §4.3 item 25 and belongs to M8; there is no
        # menu in this build for a draft to bring back. The *draft* survives,
        # which is the half this milestone owns — recorded here so the missing
        # half is visible rather than merely absent.
        await self.two_sessions(wire)
        wire.ui.focus = INPUT
        await wire.press(*"/comp")
        await self.switch_to(wire, SECOND)
        await self.switch_to(wire, FIRST)
        assert wire.ui.input.text() == "/comp"


# ------------------------------------------------------------- the whole frame


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("keys", [("enter",), ("enter", "enter"), ("r",)])
async def test_a_session_screen_fills_the_frame_exactly(width: int, keys):
    """The picker (both stages) and the rename box, drawn as whole frames."""
    async with connected(RowUI(backends=list(BACKENDS))) as w:
        await started(w)
        await on_row(w, 0 if keys[0] == "enter" else FIRST)
        await w.press(*keys)
        assert w.ui.overlay is not None
        assert widths(frame(w.ui, width, 40)) == {width}


@pytest.mark.parametrize("width", WIDTHS)
async def test_the_sidebar_row_is_still_exactly_the_width(width: int):
    async with connected() as w:
        await started(w)
        await w.tell(protocol.TurnStarted(session_id="s2"))
        assert widths(frame(w.ui, width, 40)) == {width}
