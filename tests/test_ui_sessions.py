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
from hpca.ui.state import BackendInfo, ProfileInfo
from tests.ui_harness import Wire, connected, frame, on_entry, plain, widths

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

# The catalog, as `llm.catalog` carries it. `SessionNew.backend` is an
# `LLMEntry.label` and nothing else: it was once documented as a label and
# implemented as serialised backend JSON, which pinned nothing at all, silently.
CATALOG = [
    BackendInfo(
        label="qwen3-27b-fp8",
        model="qwen3-27b-fp8",
        base_url="http://10.12.4.31:20001/v1",
        context=112000,
        active=True,
    ),
    BackendInfo(
        label="llama-3.3-70b",
        model="llama-3.3-70b",
        base_url="http://10.12.4.55:20001/v1",
    ),
]
BACKEND_ROWS = [choice(x.label, x.model, value=x.label) for x in CATALOG]


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
    async with connected(RowUI(catalog=list(CATALOG))) as w:
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

    async def test_the_profiles_screen_s_rows_reach_the_picker(self):
        # `(new profile)` is the profiles screen's own "make one" line and is
        # added by that screen, so it can no longer leak into the picker and
        # start a conversation under a profile of that name.
        ui = RowUI(profiles=[ProfileInfo(name="hpc", memories=12)])
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
        assert sent.backend == CATALOG[1].label
        assert sent.profile == "default"

    async def test_what_goes_back_is_the_label_the_catalog_gave(self, with_backends):
        # The label and nothing else (`protocol.SessionNew`): what identifies a
        # backend is the core's business, and it mints the name so that both
        # sides cannot disagree about what a backend is called.
        await started(with_backends)
        await on_row(with_backends, 0)
        await with_backends.press("enter", "enter", "enter")
        assert with_backends.peer.last(protocol.SessionNew).backend == (
            "qwen3-27b-fp8"
        )

    @pytest.mark.parametrize("width", WIDTHS)
    def test_every_row_is_exactly_the_width(self, width: int):
        screen = NewSessionOverlay(
            [choice("default", "the fallback")], list(BACKEND_ROWS)
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
        # its sidebar row and never the chat spinner. The event is scripted
        # rather than provoked because this file's core is a scripted peer, not
        # because the real one is silent: `session.retitle` does emit
        # `turn.activity` (`core.service._retitle`, asserted in
        # `test_core_service.py`). The comment here used to claim it did not.
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

    async def test_a_reply_to_a_session_you_left_flags_the_row(self, wire):
        # §3.4: `⟳` went out the instant the turn finished — the exact moment
        # there is something new to read — and nothing took its place, so two
        # background conversations were indistinguishable.
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        assert "*" in self.row_for(wire, "the second thing")

    async def test_a_turn_that_failed_there_flags_it_too(self, wire):
        # §3.4: the marker was set on `turn.finished` and not on `turn.failed`,
        # so a background turn that broke cleared its "⟳", left the row blank,
        # and put the error in a transcript nothing pointed at. The old TUI
        # marked both paths, and a failure is the one you most want telling.
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        await wire.tell(
            protocol.TurnFailed(session_id="s2", error="the backend hung up")
        )
        assert "*" in self.row_for(wire, "the second thing")

    async def test_and_opening_it_clears_that_one_as_well(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnFailed(session_id="s2", error="boom"))
        await on_row(wire, SECOND)
        await wire.press("enter")
        assert "*" not in self.row_for(wire, "the second thing")

    async def test_a_failure_in_the_open_session_flags_nothing(self, wire):
        # It is already on screen — the error row is the notification.
        await started(wire)
        await wire.tell(protocol.TurnFailed(session_id="s1", error="boom"))
        assert "*" not in self.row_for(wire, "the first thing")

    async def test_and_does_not_force_that_session_open(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        assert wire.ui.active_id == "s1"

    async def test_opening_it_clears_the_flag(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        await on_row(wire, SECOND)
        await wire.press("enter")
        assert "*" not in self.row_for(wire, "the second thing")

    async def test_a_reply_to_the_open_session_flags_nothing(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnStarted(session_id="s1"))
        await wire.tell(protocol.TurnFinished(session_id="s1"))
        assert "*" not in self.row_for(wire, "the first thing")

    async def test_a_still_working_row_says_that_instead(self, wire):
        # Two turns in one background session: the second is still running,
        # and "there is something new" is the less useful of the two things
        # to say about it.
        await started(wire)
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        assert "⟳" in self.row_for(wire, "the second thing")

    async def test_a_row_the_core_calls_working_says_that_too(self, wire):
        # §9.14: the "⟳" is drawn from the core's flag *or* the local turn,
        # and the "*" used to be suppressed only by the local half — so a
        # session working with no local `turn.started` (a reconnect, or a
        # second front-end) could be drawn as finished while it ran.
        await started(wire)
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        await wire.tell(
            protocol.SessionRows(
                rows=[ROWS[0], ROWS[1].model_copy(update={"flags": ["working"]})]
            )
        )
        row = self.row_for(wire, "the second thing")
        assert "⟳" in row and "*" not in row

    async def test_and_a_parked_decision_wins_over_both(self, wire):
        await started(wire)
        await wire.tell(protocol.TurnFinished(session_id="s2"))
        await wire.tell(
            protocol.DecisionRequested(session_id="s2", payload={"tool": "rm"})
        )
        assert "!" in self.row_for(wire, "the second thing")

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


class TestTheSidebarSaysWhenEachSessionWasLastWorkedIn:
    """The overview's answer to "which of these thirty is still alive".

    The core owns the fact (`SessionStore.last_active`) and the front-end owns
    the clock it is written in: the stamp crosses the wire as UTC, because a
    core need not be on the same machine as the UI, and `state.when` is the
    only thing that turns it into the time on a person's own wall.
    """

    AT = "2026-08-21T12:34:56+00:00"

    async def listed(self, w: Wire, at: str = AT) -> Wire:
        await started(w)
        await w.tell(
            protocol.SessionRows(
                rows=[ROWS[0].model_copy(update={"last_active": at}), ROWS[1]]
            )
        )
        return w

    async def test_the_row_carries_the_time(self, wire):
        await self.listed(wire)
        assert "21-08-2026 " in self.row_for(wire, "the first thing")

    async def test_a_session_the_core_said_nothing_about_shows_none(self, wire):
        # Not a placeholder date: a row with no stamp says nothing about when
        # rather than something false.
        await self.listed(wire)
        assert "-2026 " not in self.row_for(wire, "the second thing")

    async def test_the_mode_comes_first(self, wire):
        # They are cut in that order on a narrow terminal, and which one is
        # lost matters: "full-auto" is a safety fact and "worked in on
        # Tuesday" is not.
        await self.listed(wire)
        row = self.row_for(wire, "the first thing")
        assert row.index("auto") < row.index("21-08-2026")

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
        on_entry(w.ui.chat, 2)  # q2
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
        assert "q2" not in "\n".join(x.head for x in wire.ui.chat.items)

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
        # The cut message is already in the box, cursor at the end of it, so
        # saying it differently is typing the difference — which is what a
        # rewind is for.
        await wire.press(*" but better", "enter")
        sent = wire.peer.last(protocol.TurnSubmit)
        # One message, on the session that was trimmed: nothing the UI holds
        # resends the cut rows, and what the model sees is the core's to say
        # (`test_core_service.py` drives the graph for that half).
        assert (sent.session_id, sent.text) == ("s1", "q2 but better")
        assert [e.text for e in wire.ui.session.entries] == ["q1", "a1"]

    async def test_the_cut_message_comes_back_to_the_box(self, wire):
        # A rewind is made in order to say that message differently, and
        # retyping it by hand was the whole of the friction.
        await self.at_q2(wire)
        await wire.press("r")
        assert wire.ui.input.text() == "q2"
        assert wire.ui.focus == INPUT

    async def test_and_does_not_overwrite_what_was_being_written(self, wire):
        # `_into_draft`'s rule, which is why the message is added to the draft
        # rather than replacing it: a cut must not cost the user a sentence.
        await self.at_q2(wire)
        wire.ui.input.set_text("half a thought")
        await wire.press("r")
        assert wire.ui.input.text() == "half a thought\nq2"

    async def test_a_fork_hands_the_message_to_the_copy(self, wire):
        # Not to the source, which still has the message in its log — the
        # copy is the conversation the message is going to be said in again.
        await self.at_q2(wire)
        await wire.press("f")
        assert wire.ui.session_for("s1").draft.text() == ""
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="f1", title="a fork")
            )
        )
        assert wire.ui.active_id == "f1"
        assert wire.ui.input.text() == "q2"

    async def test_a_session_made_from_the_picker_inherits_nothing(self, wire):
        # The fork's text waits on the app for the `session.created` that
        # answers it, so a session created any other way must not pick it up.
        await self.at_q2(wire)
        await wire.press("f")
        wire.ui._forked_draft = ""  # as `_closed` clears it for `session.new`
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="n1", title="a new one")
            )
        )
        assert wire.ui.input.text() == ""

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
        on_entry(wire.ui.chat, 2)
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

    async def test_enter_forks_from_the_message_it_was_opened_on(self, wire):
        # Enter opened the dialog, so Enter answers it with the one choice
        # that cannot lose anything — the reflex key is not the cut.
        await self.at_q2(wire)
        await wire.press("enter")
        sent = wire.peer.last(protocol.SessionFork)
        assert (sent.session_id, sent.index) == ("s1", 2)

    async def test_and_c_is_left_to_the_chat_row(self, wire):
        await self.at_q2(wire)
        await wire.press("c")
        assert wire.peer.commands == []
        assert wire.ui.overlay is not None


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

    async def rewind_answered_elsewhere(self, w: Wire) -> Wire:
        """Open the rewind in s1, land on s2, and take the fork.

        The switch is `open_session` rather than a keypress because the dialog
        owns the keyboard while it is up — which is exactly the situation
        `_rewind`'s docstring is about: "the session it was opened in … is not
        necessarily the one on screen by the time it closes". The copy that
        used to make this point left the dialog for the chat row's `c`; the
        two cuts carry the same promise and are what asserts it now.
        """
        await self.two_sessions(w)
        w.ui.focus = CHAT
        on_entry(w.ui.chat, 2)
        await w.press("enter")  # the rewind, opened in s1
        w.ui.open_session("s2")
        await w.press("f")  # …and answered while s2 is on screen
        return w

    async def test_a_cut_belongs_to_the_session_it_was_opened_in(self, wire):
        # §4: the answer used to land on whatever was on screen when the
        # dialog closed, contradicting `_rewind`'s own docstring — and a test
        # that never switches sessions cannot catch it.
        await self.rewind_answered_elsewhere(wire)
        assert wire.peer.last(protocol.SessionFork).session_id == "s1"

    async def test_a_parked_slash_draft_does_not_bring_its_menu_back(self, wire):
        # Half of "a parked `/…` draft brings its autocomplete menu back with
        # it": the *draft* survives a switch, which is the half that lives on
        # `SessionState`. The menu half arrived with M8 and is asserted where
        # the menu is — `test_ui_commands.py::TestAParkedDraft`. This comment
        # used to say there was no menu in this build, which the two files
        # then contradicted each other about (specs-ui-coverage.md §6).
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
    async with connected(RowUI(catalog=list(CATALOG))) as w:
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
