"""The rows themselves: layout, focus and key dispatch.

``RowUI`` takes keys and returns frames and does no I/O of its own, which is
what makes the whole UI testable by calling ``render()`` and comparing strings.
"""

from __future__ import annotations

import time
from collections import namedtuple
from collections.abc import Callable

from hpca import __version__ as VERSION
from hpca.ui import commands, toasts
from hpca.ui.ansi import BOLD, CYAN, DIM, GREEN, RED, RESET, REVERSE, YELLOW
from hpca.ui.ansi import cell_width, cut, footer_line, pad, rule, safe
from hpca.ui.approval import decision_height, render_decision
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS, is_paste, paste_text
from hpca.ui.overlays import (
    COPY,
    FORK,
    NO_SESSION,
    ROLLBACK,
    UNQUEUE,
    ConfigOverlay,
    HelpOverlay,
    InspectOverlay,
    LlmOverlay,
    MemoryReviewOverlay,
    NewSessionOverlay,
    Overlay,
    ProfilesOverlay,
    QueuedOverlay,
    RenameOverlay,
    RewindOverlay,
    SkillCreatorOverlay,
    SkillRemoveOverlay,
    SwitchLlmOverlay,
    ThinkingOverlay,
    choice,
)
from hpca.ui.pane import Item, Pane
from hpca.ui.state import (
    MODE_COLOURS,
    OWN_MESSAGE_KINDS,
    Answer,
    BackendInfo,
    Confirm,
    CycleMode,
    Decide,
    DeleteSession,
    DraftSkill,
    Drop,
    Fork,
    Intent,
    Interrupt,
    NewSession,
    OpenSession,
    Peek,
    ProfileInfo,
    Rename,
    Retitle,
    Rollback,
    RunCommand,
    SaveSettings,
    SaveSkill,
    SessionState,
    SkillInfo,
    SidebarRow,
    Submit,
    Toast,
    Unqueue,
    mode_line,
)

# How long a first escape stays armed for a second one to complete the stop
# gesture. The same 1.0s the Textual app uses, and for the same reason a single
# escape must keep meaning nothing: ESC is the byte a terminal also sends as the
# prefix of every arrow key and of bracketed paste, so one arriving on its own
# is not evidence that the user wants the turn dead. Two in a second are.
ESC_STOP_WINDOW = 1.0

# The four rows, and the prompt that is not a row. DECISION is a focus target
# without a pane: the inline approval sits at the foot of the chat column
# (§4.3 item 21), it takes keys while it is unanswered, and it is deliberately
# not in the ↑/↓ ring — you arrive at it because a decision arrived, and you
# leave it by answering.
SESSIONS, CHAT, INPUT, WATCHERS, DECISION = range(5)

# The question the aimed half of the stop gesture asks first. Enter on the
# working row is a key that can be hit while steering through a log the agent
# is writing into, so it confirms; `esc esc` does not, because the doubling is
# already the confirmation (specs-ui-acceptance.md, "Stopping a turn").
INTERRUPT_QUESTION = "Interrupt this turn and re-edit your last message?"

# What the prompt may take of the screen. Half of what is left after the header
# and the footer, which is the row UI's version of DecisionBar's `max-height:
# 60%`: a decision has to be readable, and the conversation it is about has to
# stay on screen behind it.
DECISION_SHARE = 2

# How many commands the "/" menu offers at once. Eight is every built-in plus
# a skill, which is what an unfiltered menu shows on a fresh install; past that
# the answer to "I cannot see mine" is to type another letter, not to give the
# list half the screen.
MENU_ROWS = 8

# What quitting asks, and what ctrl+q is worth. `q` confirms because it is one
# letter away from every other key on the sessions column; ctrl+q is not bound
# at all because it belongs to zellij, which is what a cluster user runs this
# inside (specs-ui-acceptance.md, "Backends").
QUIT_QUESTION = "Really quit?"

# What a completed stop gesture says when there was no turn under it, in the
# two shapes that has. A backend call that is not a turn — a compaction, the
# titler, a silent `/conclude` — has no turn to roll back and says so; an idle
# session has nothing at all. Both are `esc esc` and Enter on the working row
# giving the same answer to the same question.
NOT_A_TURN = "this is a backend call, not a turn — nothing to stop"
NOTHING_TO_STOP = "nothing running to stop"

# The title over the tunnel recipe an empty scan comes back with — the words
# `tui/manage_llms.py` put on the same window, so a user who has seen it once
# recognises it. The recipe itself is the core's (`autoconnect.offcluster_help`).
NO_ENDPOINTS = "no llm endpoints found"

# What the config editor's fetch is addressed by. There is one settings file,
# so the key is a constant — it exists so that the answer can be routed the
# same way a profile's or a skill's is (`RowUI.body_arrived`).
SETTINGS_KEY = ("settings", ())


# The sidebar markers §4.3 item 15 asks for, and the flag strings the core
# uses for the same states. Both are consulted: the flag is what the core
# knows, and the local turn/decision state is what has arrived since the last
# `session.rows` — a marker that waited for the next sidebar repaint would lag
# a whole poll behind the event that caused it.
DECISION_MARK, WORKING_MARK = "!", "⟳"

# And the third state §4.3 item 15 asks for, which the core has no flag for
# and could not have one: a reply landed in this conversation while the user
# was reading another. Without it a background turn's row shows "⟳" while it
# runs and goes blank the instant it finishes — the exact moment there is
# something new in it — and two conversations become indistinguishable.
UPDATED_MARK = "*"

# The row that is not a session. First, where it is in the Textual sidebar and
# for the same reason: on a cluster this is the only way to start a
# conversation, and a fresh install's sidebar is otherwise empty. Its key is in
# the `#` namespace `Pane.key_at` keeps for rows that are not addressed by an
# id, so it can never collide with a session's.
NEW_SESSION_KEY = "#new"
NEW_SESSION_ROW = "(new session)"

# The profile every session has unless it was given another one
# (`hpca.profiles.DEFAULT_PROFILE`, copied rather than imported so the key
# dispatch keeps its one-way dependency on `ui/`). Sessions under it are drawn
# without a tag: naming the default on every row spends the sidebar's width
# saying the same word fourteen times.
DEFAULT_PROFILE = "default"

# What deleting one asks first, and it says what survives: the transcript on
# disk is deliberately kept (`core.service._delete_session`), and a user who
# does not know that will not delete anything.
DELETE_QUESTION = (
    "Really delete “{title}”? The chat is dropped; its log on disk is kept."
)

# How the status row spends the width it has (§4.3 items 18 and 19). Three
# things want a permanent row of their own — the meter, the mode and the model
# — and on an 80x24 terminal three rows out of twenty-two is a tenth of the
# screen spent on things that do not change. So they share one row and give
# ground in this order, richest first: the mode's key hint goes, then its
# sentence, then the meter's picture, and what is left is the two facts that
# cannot be worked out from anything else — which mode is on, and how full the
# window is. The model rides on the message row's own rule and costs nothing.
StatusTier = namedtuple("StatusTier", "hint switch cells")
STATUS_TIERS = (
    StatusTier(hint=True, switch=True, cells=28),
    StatusTier(hint=True, switch=False, cells=28),
    StatusTier(hint=True, switch=False, cells=14),
    StatusTier(hint=False, switch=False, cells=14),
    StatusTier(hint=False, switch=False, cells=0),
)

# What the meter's severity paints it (`ui.meter.severity`).
METER_STYLES = {"warn": YELLOW, "danger": BOLD + RED}


class RowUI:
    MIN_CHAT = 4
    MAX_INPUT = 6

    def __init__(
        self,
        sessions: list[SessionState] | None = None,
        *,
        settings_json: str = "",
        profiles: list[ProfileInfo] | None = None,
        catalog: list[BackendInfo] | None = None,
        send: Callable[[Intent], None] | None = None,
    ) -> None:
        # The sidebar's order, and the store behind it. Two things, because an
        # event can name a session the sidebar has not been told about yet (a
        # fork's `session.created` arrives before the `session.rows` that lists
        # it), and because a session must not lose its chat and its draft
        # merely by scrolling out of the list.
        self.sessions = list(sessions or [])
        self._states = {x.session_id: x for x in self.sessions}
        # What the rows draw against before the first `session.rows` arrives.
        # Real, so that nothing below here needs a None check.
        self._blank = SessionState("")
        self.active = 0
        # Where a keypress goes when it means something the core has to do.
        # Recorded rather than dropped when nothing is listening, so a test can
        # read the intent off the UI without a client at all.
        self.intents: list[Intent] = []
        self.send: Callable[[Intent], None] = send or self.intents.append
        self.session_pane = Pane("sessions", [])
        self.refresh_sidebar()
        self.focus = CHAT
        self.frame_ms = 0.0
        self._note = ""
        self.note_style = YELLOW
        # Set from `hello`, and what the header falls back to when no session
        # is open to have a profile of its own.
        self.core_profile = ""
        self.confirm: Confirm | None = None
        self.toasts: list[Toast] = []
        # Whether the answer to the last question was "yes, quit". Read by
        # `handle`, because ending the loop is something only a return value
        # can say and a confirmation is answered a keypress later than it is
        # asked (§4.3 item 38).
        self.quitting = False
        # The screens over the rows, innermost last. A stack rather than one
        # slot because the profiles screen is genuinely three deep — the list,
        # a profile's skills, one skill's file — and escaping the file has to
        # land back on the skills. `overlay` is still the one the keys go to.
        self.overlays: list[Overlay] = []
        # When the last escape landed, so the next one can tell whether it is
        # the second half of a stop. Injectable so the headless check can drive
        # the clock instead of sleeping through the window.
        self.clock = time.monotonic
        # And the wall clock, which is a different question and needs a
        # different answer: "how long has this turn been running" is measured
        # against a stamp the *core* took (`TurnActivity.started_at`), and the
        # two processes share a wall clock and not a monotonic one.
        self.wall = time.time
        self._esc_armed_at: float | None = None
        # Everything the overlays draw and nothing else reads. Handed in
        # rather than fetched, for the reason every screen in `overlays/` is:
        # the UI opens no files and no sockets (rule 2 of §4.2). Both now come
        # down the wire — `profile.rows` and `llm.catalog` — and `client.py`
        # fills these in from them; a UI built without one (a test, the demo's
        # first frame) draws what it was handed and is corrected by the first
        # event that says otherwise.
        # What the config editor shows before the file it asks for arrives —
        # the last one that did, or whatever a UI with no wire behind it was
        # handed. The editor fetches on open either way (`settings.get`).
        self.settings_json = settings_json
        # A validator for the config editor, injected because what a settings
        # *file* means belongs to `hpca.config` and this module has never heard
        # of it (§3.1). None means "JSON syntax is the whole check".
        self.validate_settings: Callable[[str], str] | None = None
        self.profiles = list(profiles or [])
        # Every LLM the core knows about (`protocol.LLMCatalog`), configured
        # and discovered in one list with a flag — which is how the wire
        # carries it, and splitting it into two here would be a second copy to
        # keep in step. Empty until the first catalog arrives, which is the
        # case the new-session flow answers by skipping the LLM picker.
        self.catalog = list(catalog or [])
        # The four things M8 needs that are I/O, and are therefore injected —
        # `RowUI` opens no files, runs no processes and writes to no terminal
        # (§3.1). Each is None until something that owns the corresponding
        # resource wires it up, and the key that needs one says so rather than
        # doing nothing when it is missing.
        #
        # Ask for a profile's skills (`skill.list`). Called on the first `/`
        # rather than at startup, so a UI nobody types a slash into makes no
        # round trip; the answer comes back through `skills_listed`.
        self.skills_loader: Callable[[str], None] | None = None
        self._skills: dict[str, list[SkillInfo]] = {}
        # How often each slash command has been run, as `command.counts` last
        # said. Empty until something answers, which sorts the menu in
        # definition order — the same order an install nobody has typed into
        # yet would produce anyway.
        self.command_counts: dict[str, int] = {}
        # `y` on a chat row: text in, a sentence about where it went out
        # (`hpca.clipboard.ClipboardManager.copy(...).message`). A callable and
        # not the manager itself, because the manager writes OSC 52 straight to
        # the terminal and this module has never seen one.
        self.clipboard: Callable[[str], str] | None = None
        # ctrl+e, in two halves belonging to two different layers. `suspend`
        # takes the terminal down and puts it back (`ui/run.py`, which owns
        # it); `edit_profile` fetches the profile, runs the editor over it and
        # saves it back (`ui/client.py`, which is the side with a wire). Both
        # ends of it are asynchronous, so it answers in a toast rather than by
        # returning anything.
        self.suspend: Callable[[Callable[[], None]], None] | None = None
        self.edit_profile: Callable[[str], None] | None = None
        # A window that arrived while a screen was open, waiting for the
        # screen to close (`window`). One slot: two of these queued at once
        # would be a stack of modals nobody asked for, and the newer answer is
        # the one still worth reading.
        self._waiting_window: tuple[str, str] | None = None

    # ------------------------------------------------------ the open session

    # ------------------------------------------------------------- the screens

    @property
    def overlay(self) -> Overlay | None:
        """The screen the keys go to: the innermost one, or None."""
        return self.overlays[-1] if self.overlays else None

    @overlay.setter
    def overlay(self, screen: Overlay | None) -> None:
        """Open a screen over the rows, replacing whatever was there.

        Assignment stays the way a screen is opened — `self.overlay = X` reads
        as what it does — and it starts a fresh stack: every top-level screen
        is opened from the rows, so there is never one to go back to.
        """
        self.overlays = []
        if screen is not None:
            self.push(screen)

    def push(self, screen: Overlay) -> None:
        """Put a screen on the stack and let it ask the core for things.

        The one thing a screen is given beyond its content: where its intents
        go. Without it a screen keeps them in `sent`, which is what makes one
        testable on its own with no app around it.
        """
        screen.send = self.send
        self.overlays.append(screen)
        # Only now, with `send` wired: a screen that asks for something the
        # moment it opens — the body its editor is about to show, the listing
        # its list is about to draw — would otherwise be talking to the
        # throwaway list a screen carries when it is constructed alone.
        screen.opened()

    def _pop(self) -> None:
        """The innermost screen has closed. Whoever opened it hears about it."""
        done = self.overlays.pop()
        if self.overlays:
            parent = self.overlays[-1]
            parent.child_closed(done)
            self._adopt(parent)
            # A window that arrived while this child was up: the screen under
            # it may be one that takes it (the recipe, over manage-LLMs).
            self._open_waiting_window()
            return
        self._closed(done)
        self._open_waiting_window()

    def _adopt(self, screen: Overlay) -> None:
        """A screen asked for a screen of its own. Give it one."""
        if screen.child is not None:
            child, screen.child = screen.child, None
            self.push(child)

    @property
    def session(self) -> SessionState:
        """The conversation on screen, or a blank one when there is none.

        A UI drawn before the first `session.rows` has nothing open and still
        has to render four rows, so "nothing open" is a session with an empty
        chat rather than a special case in every method below.
        """
        if 0 <= self.active < len(self.sessions):
            return self.sessions[self.active]
        return self._blank

    @property
    def active_id(self) -> str:
        return self.session.session_id

    @property
    def note(self) -> str:
        return self._note

    @note.setter
    def note(self, text: str) -> None:
        # Anything set as an ordinary note is an ordinary note; a toast that
        # wants another colour sets both, and the next note takes it back.
        self._note, self.note_style = text, YELLOW

    @property
    def chat(self) -> Pane:
        return self.session.chat

    @property
    def watchers(self) -> Pane:
        return self.session.watchers

    @property
    def input(self) -> Editor:
        return self.session.draft

    @property
    def profile(self) -> str:
        return self.session.profile or self.core_profile

    @property
    def mode(self) -> str:
        """The open session's mode, and no default of its own.

        There is no such thing as the app's mode: it is a per-session dial,
        and a header that invented one would be naming a mode no session is
        in — which was survivable while the fallback happened to be a mode
        that existed, and stopped being so when `plan` was retired.
        """
        return self.session.mode

    @property
    def model(self) -> str:
        """The backend the open session is pinned to, as `session.rows` said.

        Per session and not per app: two conversations can be on two different
        servers, which is the whole reason `ctrl+l` switches one of them. Empty
        with no session open, and empty for a session on the bootstrap client.
        """
        return self.session.model

    @property
    def panes(self) -> list[Pane]:
        """The three list rows, top to bottom, for the session on screen."""
        return [self.session_pane, self.chat, self.watchers]

    def session_for(self, session_id: str) -> SessionState:
        """The state for that conversation, made if this is the first word of it.

        Every event handler starts here (§3.2 property 1): an event names a
        session, and it is answered whether or not that session is the one on
        screen.
        """
        session = self._states.get(session_id)
        if session is None:
            session = self._states[session_id] = SessionState(session_id)
        return session

    def adopt(self, session: SessionState) -> None:
        """Put a session in the sidebar now, ahead of the core saying so.

        `session.created` names a conversation the user is about to be looking
        at; waiting for the next `session.rows` to list it would mean opening
        something the sidebar does not show.
        """
        self._states[session.session_id] = session
        if session in self.sessions:
            return
        was = self.active_id
        self.sessions.insert(0, session)
        self._select(was)
        self.refresh_sidebar()

    def sync_sessions(self, rows: list[SidebarRow]) -> None:
        """The sidebar, as the core last described it.

        The cursor is preserved **by session id** rather than by index: a
        session created in the background inserts a row, and a selection that
        moved because of it is a selection the user did not make. So is the
        open session — `active` is an index into a list that has just been
        rebuilt, and it is recomputed from the id rather than carried over.

        Nothing else about a session is touched. The chat, the draft, the
        cursor line and the open entries are the UI's, and a repaint of the
        sidebar is not an event about any of them.
        """
        was = self.active_id
        self.sessions = []
        for row in rows:
            session = self.session_for(row.session_id)
            session.title = row.title
            session.profile = row.profile
            session.mode = row.mode
            session.model = row.model
            session.thinking = row.thinking
            # The meter's `. think medium` (§4.3 item 18) reads this: a session
            # left on xhigh looks identical to one on off until the first wait.
            session.context.effort = row.thinking
            session.flags = list(row.flags)
            self.sessions.append(session)
        keep = {x.session_id for x in self.sessions} | {was}
        self._states = {k: v for k, v in self._states.items() if k in keep}
        self._select(was)
        self.refresh_sidebar()

    def _select(self, session_id: str) -> None:
        """Point `active` at that session, or at nothing if it is not here.

        -1 rather than 0, because "the session that was open has gone" and
        "the first session is open" are different facts and only one of them
        is true after a delete: snapping to row 0 would draw a ● on a
        conversation the core was never told is on screen, whose chat has
        never been asked for. Nothing open is a state the rows already draw
        (`session` falls back to `_blank`), and the first list to arrive is
        answered by `client._rows` opening its first row for real.
        """
        self.active = next(
            (i for i, x in enumerate(self.sessions) if x.session_id == session_id),
            -1,
        )

    def _marks(self, session: SessionState) -> str:
        """The sidebar markers: what this session wants, and what it is doing.

        The one thing a session that is *not* on screen may change about the
        frame (§3.2 property 1) — so it is read off that session's own state
        rather than off anything the open conversation knows.
        """
        flags = " ".join(session.flags)
        # `is not None`: an interrupt whose payload happens to be empty is
        # still a decision waiting for an answer.
        parked = session.decision is not None or "decision" in flags
        marks = DECISION_MARK if parked else " "
        # `busy`, not `working`: a silent backend call — a compaction, the
        # titler, a `/conclude` — is something in flight in that conversation
        # and the row has to say so, even though there is no turn to stop
        # (`state.Turn.busy` is the same distinction the spinner draws).
        marks += WORKING_MARK if session.turn.busy or "working" in flags else " "
        # Last, and only where the other two are not: a row cannot be both
        # still working and finished, and a decision is the more urgent of the
        # two things to say about a session that is not on screen.
        if session.updated and not (parked or session.turn.busy):
            marks = marks[0] + UPDATED_MARK
        return marks

    def replied(self, session_id: str) -> None:
        """A turn finished. If it was not the one on screen, flag its row.

        The half of §4.3 item 15 M6 shipped without: "⟳" says a conversation
        is busy and nothing said it had *answered*. A reply must not pull the
        user out of what they are reading (the acceptance list is explicit
        that it does not force the session open), so the whole of it is one
        marker in the sidebar, cleared by opening the session.
        """
        session = self.session_for(session_id)
        if session_id != self.active_id:
            session.updated = True
        self.refresh_sidebar()

    def refresh_sidebar(self) -> None:
        """Redraw the session list: which one is open, and what each is doing.

        Rebuilt rather than patched because it is fourteen rows, not fourteen
        hundred — the cost that matters is the chat's, and that one is never
        rebuilt at all. `Pane.replace` keeps the cursor on the row it was on,
        by id.
        """
        self.session_pane.replace(
            [
                # The one row that is not a conversation, and the only way to
                # start one. First, so it is on screen without scrolling a
                # sidebar that shows a quarter of the terminal.
                Item(head=NEW_SESSION_ROW, key=NEW_SESSION_KEY),
                *(
                    Item(
                        head=(
                            f"{'●' if i == self.active else '○'} "
                            f"{self._marks(session)} {session.title[:40]:<42}"
                            + " · ".join(
                                x for x in (self._tag(session), session.mode) if x
                            )
                        ),
                        body=[
                            f"session {session.session_id}",
                            f"{len(session.entries)} entries"
                            + (
                                f" · {session.context.label()}"
                                if session.context.label()
                                else ""
                            ),
                            f"{session.watch_count} watches"
                            + (" · open" if i == self.active else ""),
                        ],
                        accent=GREEN if i == self.active else "",
                        key=session.session_id,
                    )
                    for i, session in enumerate(self.sessions)
                ),
            ]
        )

    def _tag(self, session: SessionState) -> str:
        """The profile a row is tagged with, and nothing for the default one.

        A tag is there to say "this conversation is not under the profile you
        would assume"; the default on every row says only that there are rows.
        """
        return "" if session.profile == DEFAULT_PROFILE else session.profile

    def open_session(
        self,
        session_id: str,
        *,
        announce: bool = True,
        land_in_box: bool = False,
    ) -> None:
        """Look at this conversation, and tell the core that we are.

        ``announce`` is off for the one the client opens by itself when the
        first sidebar arrives: the footer note answers a keypress, and there
        was none.

        ``land_in_box`` is for a session that exists *because* the user asked
        for one — a new session, a fork — where the next thing to do is type
        in it. Off by default, because opening a conversation from the sidebar
        decides that for itself (`_enter_session`), and the client's own first
        open answers no keypress at all.
        """
        self._select(session_id)
        # Opening it is the one thing that can be read as having read it, so
        # the "*" goes here rather than on the next frame that draws the chat.
        self.session_for(session_id).updated = False
        self.refresh_sidebar()
        # The highlight follows what is open, the way the Textual sidebar set
        # its index after every reload: the sidebar's first row is
        # `(new session)`, and a cursor left sitting on it would answer the
        # next Enter by making a conversation rather than opening this one.
        self.session_pane.show(session_id)
        if self.session.decision is not None and self.focus != WATCHERS:
            # It was flagged with a "!" while it was in the background; being
            # opened is what reveals it, and the cursor lands where it can be
            # answered.
            self.focus = DECISION
        elif land_in_box:
            self.focus = INPUT
        if announce:
            self.note = f"opened “{self.session.title}”"
        self.send(OpenSession(session_id))

    def _switch(self, session_id: str) -> None:
        """Open the session with that id. By id, because the sidebar's rows
        are not one-for-one with `self.sessions` — the first of them is not a
        session at all — and because a repaint can move a row under the
        cursor between the keypress and this."""
        if not session_id or session_id not in self._states:
            return
        if session_id == self.active_id:
            self.note = "already open"
            return
        self.open_session(session_id)

    def toast(
        self,
        text: str,
        severity: str = "information",
        timeout: float | None = None,
        title: str = "",
    ) -> None:
        """Something the core said, in the two places it is said (§4.3 item 35).

        Both, and not one or the other. The block over the frame is where a
        heading, a second line and a severity colour can be read, and it goes
        on a timer because a notification that stayed would be a modal. The
        footer note is what is left afterwards — one line, cut to the width —
        so that a message which expired while the user was reading the chat is
        still answerable with "what did that say".
        """
        self.toasts.append(
            Toast(text, severity, timeout, title=title, at=self.clock())
        )
        # Only what fits on one line, and the heading first where there is one:
        # the footer is a rule, not a paragraph. Through `safe` for the same
        # reason the block is — an escape sequence in the footer would move
        # the cursor just as readily as one in the body, and this is the copy
        # that outlives the toast.
        head = f"{title}: " if title else ""
        self._note = head + " ".join(safe(text).split())
        self.note_style = RED if severity == "error" else YELLOW
        # And, when it is a heading over a block, the window as well. `title`
        # is the signal (`protocol.Notify.title`): the core sets it exactly
        # where an answer is a heading plus a body — the skills a profile can
        # see, the summary `/compact` just wrote — and a body of more lines
        # than a toast can carry is one that was being read as
        # "… more in the log", where it never was.
        if title and len(text.splitlines()) > toasts.MAX_BODY:
            self.window(title, text)

    def window(self, title: str, body: str) -> None:
        """Text too long to be a toast, in a window that waits for escape (31).

        `RowUI.inspect` with one rule on top of it: it does not land on a
        screen the user opened for something else. The tunnel recipe is the
        case that made the rule — a scan can finish while the connection form
        it was started from is being typed into, and "a modal must not land on
        top of someone mid-typing" is what `tui/manage_llms.py` parked it for.
        So it is remembered and placed the moment it can be, which is the
        acceptance list's "waits for a form the user already opened".

        A screen that is *expecting* an answer of its own takes it straight
        away (`Overlay.welcomes_window`): manage-LLMs asked for the scan, and
        parking the recipe until manage-LLMs closed would hide it behind the
        one screen it is about.
        """
        self._waiting_window = (title, body)
        self._open_waiting_window()

    def _open_waiting_window(self) -> None:
        """Place a waiting window, if there is one and it may be placed."""
        top = self.overlay
        if self._waiting_window is None or (
            top is not None and not top.welcomes_window
        ):
            return
        title, body = self._waiting_window
        self._waiting_window = None
        if top is None:
            self.inspect(title, body)
        else:
            self.push(InspectOverlay(body, title=title))

    # ------------------------------------------------- the slash commands

    def visible_skills(self) -> list[SkillInfo]:
        """Every skill the open profile can call, each with its level.

        Whatever the last `skill.rows` for this profile said, and *asked for*
        the first time the menu wants it — the answer arrives a frame later and
        the menu, being a function of what has arrived, fills itself when it
        does. Asked once per profile rather than per keystroke, because the
        question is asked on every character of a command being typed.

        The *visible* scope, not the profile's own: HPCA ships skills, and a
        shared or project one is callable too, so a menu built from the own
        list reports `/plan` unknown on a fresh install. What each row may
        have done to it is `SkillInfo.level`'s business, not this list's.
        """
        profile = self.profile
        if profile not in self._skills:
            self._skills[profile] = []
            if self.skills_loader is not None:
                self.skills_loader(profile)
        return self._skills[profile]

    def skills(self) -> list[tuple[str, str]]:
        """The same list as the menu takes it: name and description."""
        return [(x.name, x.description) for x in self.visible_skills()]

    def skills_listed(self, profile: str, skills: list[SkillInfo]) -> None:
        """A `visible`-scoped `skill.rows`: what this profile can call.

        The own-scoped answer goes elsewhere (`own_skills_listed`): the two
        scopes fill two lists, and letting either one land in the other is how
        a screen ends up offering to delete a skill that ships with HPCA.
        """
        self._skills[profile] = list(skills)

    def own_skills_listed(self, profile: str, skills: list[SkillInfo]) -> None:
        """An `own`-scoped `skill.rows`: what that profile may edit and delete.

        Kept on the `ProfileInfo` because that is what the skills screen is
        opened *about*, and it may well be a profile other than the open one.
        """
        info = next((x for x in self.profiles if x.name == profile), None)
        if info is not None:
            info.skills = list(skills)

    def forget_skills(self, name: str = "") -> None:
        """A skill was written or removed; the menu's copy is out of date.

        The row goes now and the list is asked for again, which is the same
        bargain the sidebar's rename makes: what is shown immediately is what
        the user just did, and the next answer from the core is what makes it
        true — or takes it back, when the core refused.
        """
        info = next((x for x in self.profiles if x.name == self.profile), None)
        if name:
            self._skills[self.profile] = [
                x for x in self._skills.get(self.profile, []) if x.name != name
            ]
            if info is not None:
                info.skills = [x for x in info.skills if x.name != name]
        if self.skills_loader is not None:
            self.skills_loader(self.profile)

    def _remember_skill(self, name: str, description: str, level: str) -> None:
        """A skill the user just wrote, in the lists that already offer skills.

        The menu takes it whatever level it was written at — every level is
        callable. The profile's own list takes it only if it is the profile's
        to remove: a global skill is visible and is not "own", and putting one
        there would offer it to a picker whose Enter cannot delete it.
        """
        fresh = SkillInfo(name=name, description=description, level=level)
        known = self._skills.setdefault(self.profile, [])
        if not any(x.name == name for x in known):
            known.append(fresh)
        info = next((x for x in self.profiles if x.name == self.profile), None)
        if (
            fresh.removable
            and info is not None
            and not any(x.name == name for x in info.skills)
        ):
            info.skills.append(fresh)

    def menu(self) -> list[commands.Command]:
        """The commands the draft is currently naming, best first, or none.

        A function of the draft and nothing else — which is what makes "a
        parked `/…` draft brings its autocomplete menu back with it" true
        without anything being parked: switching session swaps the draft, and
        the menu follows because it was never state of its own. Only the
        highlighted row is remembered, on the session (`SessionState.menu_at`).

        Most-used first, then definition order. The counts are the core's —
        rule 2 of §4.2 keeps this side out of the database — and they arrive
        as `command.counts`, asked for on connect and restated whenever one
        changes. Until one arrives the mapping is empty and the sort collapses
        to definition order, which is what a menu nobody has typed into yet
        would show anyway.
        """
        typed = commands.typed_name(self.input.text())
        if typed is None:
            return []
        return commands.matching(
            typed, commands.all_commands(self.skills()), self.command_counts
        )

    def _menu_at(self, matches: list[commands.Command]) -> int:
        """Which row is highlighted, clamped to a list that has since narrowed."""
        if not matches:
            return 0
        return max(0, min(self.session.menu_at, len(matches) - 1))

    def _menu_h(self, width: int, height: int) -> int:
        """Rows the menu wants: its rule plus its matches, capped.

        Taken out of the chat's share the way the decision prompt is, rather
        than drawn over the frame the way a toast is. It belongs to the box
        being typed into and has to sit next to it, and unlike a toast it is
        not on a timer: it is up for exactly as long as a command is being
        named, so pushing the conversation up costs one relayout on the way in
        and one on the way out.
        """
        matches = self.menu()
        if not matches:
            return 0
        room = max(2, (max(8, height - 2) // 3))
        return min(1 + len(matches), room, 1 + MENU_ROWS)

    def _render_menu(self, width: int, height: int) -> list[str]:
        matches = self.menu()
        rows = self._menu_h(width, height)
        if not matches or rows < 2:
            return []
        return [commands.menu_title(matches, width)] + commands.menu_rows(
            matches, self._menu_at(matches), width, rows - 1
        )

    # ------------------------------------------------ what arrives afterwards

    def body_arrived(self, key, text: str, error: str = "") -> None:
        """A `profile.body`, `skill.body` or `settings.body`, to whoever asked.

        Innermost screen first, and the first one that recognises the key takes
        it: the stack is three deep at its deepest and two editors could in
        principle be waiting, but only one of them can be waiting for *this*.
        A body nobody claims is a body whose screen has since closed, and
        dropping it is the whole point of matching on the key.
        """
        for screen in reversed(self.overlays):
            if screen.fill(key, text, error):
                return

    def list_arrived(self, key, rows) -> None:
        """A `skill.rows`, to whoever asked. Same rule as `body_arrived`."""
        for screen in reversed(self.overlays):
            if screen.fill_list(key, rows):
                return

    def invalidate(self) -> None:
        """Every pane the UI holds, open or not."""
        self.session_pane.invalidate()
        for session in (*self._states.values(), self._blank):
            session.invalidate()

    # ------------------------------------------------------------- geometry

    def _input_h(self, width: int) -> int:
        """Screen rows the message box wants, title included.

        Counted after wrapping, not from the number of typed lines: one long
        sentence is several rows, and sizing the box from the logical count is
        what let the text run off the edge of the row.
        """
        rows = self.input.height(self._input_body(width))
        return 1 + max(1, min(self.MAX_INPUT, rows))

    def _input_body(self, width: int) -> int:
        """The width the message text is folded at — the marker column costs
        two, and the editor is handed the rest."""
        return max(4, width - 2)

    def _heights(self, height: int, width: int) -> list[int]:
        """How the rows split the screen.

        A quarter each for sessions and watchers and the rest to the chat — but
        only as much of a quarter as the pane actually has to show, so a short
        session list or an empty watcher row costs nothing instead of holding a
        quarter of the screen open. The message box takes what it needs up to
        six lines. Everything left over goes to the chat, which is the row that
        can use it.
        """
        avail = max(8, height - 2)  # header and footer
        inp = (
            self._input_h(width)
            + self._status_h()
            + self._decision_h(width, height)
            + self._menu_h(width, height)
        )
        quarter = max(2, avail // 4)
        inner = max(8, width - 2)
        top = max(2, min(1 + len(self.panes[0].flat(inner)), quarter))
        bottom = max(2, min(1 + len(self.panes[2].flat(inner)), quarter))
        while avail - top - bottom - inp < self.MIN_CHAT and (top > 2 or bottom > 2):
            if top >= bottom and top > 2:
                top -= 1
            elif bottom > 2:
                bottom -= 1
            else:
                break
        middle = avail - top - bottom - inp
        if middle < 1:  # a terminal too short for the design at all
            top = bottom = 2
            inp = (
                2
                + self._status_h()
                + self._decision_h(width, height)
                + self._menu_h(width, height)
            )
            middle = max(1, avail - 4 - inp)
        return [top, middle, inp, bottom]

    def _decision_h(self, width: int, height: int) -> int:
        """Rows the inline approval wants, or none because there is none.

        Taken out of the chat's share rather than given a row of its own,
        because that is what "inline, at the foot of the chat column" means in
        a layout made of rows: the prompt pushes the conversation up and the
        conversation is still there behind it — the property a modal would
        lose, and the reason this is not one (`ui/approval.py`).
        """
        decision = self.session.decision
        if decision is None:
            return 0
        return decision_height(
            decision, width, max(3, max(8, height - 2) // DECISION_SHARE)
        )

    def _status_h(self) -> int:
        """Whether the mode/meter row is on screen at all.

        Nothing to say costs nothing: with no session open there is no mode
        and no window, and the row would be a blank line between the chat and
        the message box.
        """
        return 1 if self._status_left() or self._status_right(28) else 0

    def _status_left(self, *, hint: bool = True, switch: bool = True) -> str:
        """The mode bar. Per-session, so it is hidden without a session."""
        if not self.active_id:
            return ""
        return mode_line(self.session.mode, hint=hint, switch=switch)

    def _status_right(self, cells: int) -> str:
        """The context meter, or nothing before the core has said anything."""
        context = self.session.context
        if not self.active_id or not (context.known or context.window):
            return ""
        return context.bar(cells)

    # -------------------------------------------------------------- drawing

    def render(self, width: int, height: int) -> list[str]:
        out = [self._header(width)]
        if self.overlay is not None:
            out += self.overlay.render(width, height - 2)
            out.append(footer_line(self.overlay.footer(), width))
            while len(out) < height:
                out.insert(len(out) - 1, " " * width)
            # A confirmation can be asked *about* an overlay — remove this
            # skill, delete this profile — so it is drawn over that too, and a
            # toast lands on a screen as readily as on the rows.
            out = self._over_toasts(out[:height], width)
            return self._over_confirm(out, width)
        heights = self._heights(height, width)
        order = [
            (SESSIONS, self.panes[0], heights[0]),
            (CHAT, self.panes[1], heights[1]),
            (INPUT, None, heights[2]),
            (WATCHERS, self.panes[2], heights[3]),
        ]
        # The spinner is a function of the clock, so the frame asks the clock
        # for it here rather than anything pushing frames at the UI.
        self.session.tick(self.wall())
        for slot, pane, pane_h in order:
            if slot == INPUT:
                prompt = self._render_decision(width, height)
                status = self._render_status(width)
                menu = self._render_menu(width, height)
                out += prompt + menu + status
                out += self._render_input(
                    width, pane_h - len(status) - len(prompt) - len(menu)
                )
            else:
                out += pane.render(width, pane_h, focused=self.focus == slot)
        note, style = self.note, self.note_style
        if self._esc_armed():
            note, style = "esc again to stop", RED
        out.append(footer_line(self._keys(), width, note, style))
        while len(out) < height:
            out.insert(len(out) - 1, " " * width)
        out = self._over_toasts(out[:height], width)
        return self._over_confirm(out, width)

    def _over_toasts(self, out: list[str], width: int) -> list[str]:
        """What the core said, over the finished frame (§4.3 item 35).

        Directly under the header, and full-width rows replaced whole. Both
        halves of that are about the differential repaint. Full rows, because
        splicing a box into the middle of an already-styled line means cutting
        SGR sequences, and a cut escape is a colour that never ends. Under the
        header, because that is the one band of the frame whose height never
        depends on what is in it — put the block over the chat and a toast
        arriving mid-turn would cover the rows the user is reading; put it over
        the message box and it would cover what they are typing.
        """
        if not self.toasts:
            return out
        rows = toasts.render(self.toasts, self.clock(), width, max(0, len(out) - 2))
        if not rows:
            # Nothing live: drop what has expired so the list cannot grow for
            # the length of a session.
            self.toasts = []
            return out
        out[1 : 1 + len(rows)] = rows
        return out

    def _render_decision(self, width: int, height: int) -> list[str]:
        """The open session's approval, and only the open session's.

        A decision waiting in a background conversation is somebody else's
        question: it lights the sidebar's "!" (`_marks`) and puts nothing on
        this screen, which is the whole reason the prompt is inline instead of
        modal. Opening that session is what reveals it.
        """
        decision = self.session.decision
        if decision is None:
            return []
        return render_decision(
            decision,
            width,
            self._decision_h(width, height),
            focused=self.focus == DECISION,
        )

    def _over_confirm(self, out: list[str], width: int) -> list[str]:
        """The generic yes/no, drawn over the finished frame (§4.3 item 22).

        Over rather than in: a confirmation is asked *about* what is on screen
        — really quit, interrupt this turn, apply this signature — and it can
        arrive on top of an overlay, which is what the Textual app's
        ConfirmScreen did by being pushed on the screen stack. It is also the
        one thing here that is deliberately modal: it is a question with two
        answers and no third thing to be doing meanwhile.
        """
        question = self.confirm
        if question is None:
            return out
        rows = [
            YELLOW + rule("confirm", width) + RESET,
            BOLD + pad(f"  {question.question}", width) + RESET,
            DIM + pad("  (y) yes · (n) no · (esc) no", width) + RESET,
        ]
        rows = rows[: len(out)]  # a terminal too short for the question
        at = max(0, (len(out) - len(rows)) // 2)
        out[at : at + len(rows)] = rows
        return out

    def _render_status(self, width: int) -> list[str]:
        """The mode bar and the context meter, sharing one row.

        Which is the layout answer to five new things wanting rows: two of
        them take one row between them, the model takes none (it is on the
        message rule), and the spinner and the live steps take none either —
        they are rows *of the chat*, which is the pane that has the slack.
        """
        if not self._status_h():
            return []
        left, right = "", ""
        for tier in STATUS_TIERS:
            left = self._status_left(hint=tier.hint, switch=tier.switch)
            right = self._status_right(tier.cells)
            if cell_width(left) + 1 + cell_width(right) <= width:
                break
        else:  # narrower than the poorest tier: the mode alone, clipped
            left, right = cut(left, max(0, width - 1)), ""
        gap = width - cell_width(left) - cell_width(right)
        meter = METER_STYLES.get(self.session.context.severity, DIM)
        return [
            MODE_COLOURS.get(self.session.mode, DIM)
            + left
            + RESET
            + " " * gap
            + meter
            + right
            + RESET
        ]

    def _render_input(self, width: int, height: int) -> list[str]:
        focused = self.focus == INPUT
        # The model line (§4.3 item 20): which backend this session is pinned
        # to, on the rule of the row you type into, because that is the row
        # the answer will be written by. Empty — and so absent — when no
        # session is open to be pinned to anything.
        title = rule("message", width, self.model)
        out = [(BOLD + CYAN if focused else DIM) + title + RESET]
        rows = max(1, height - 1)
        body = self.input.render(self._input_body(width), rows, focused=focused)
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((CYAN if focused else DIM) + marker + RESET + line)
        return out[:height]

    def _header(self, width: int) -> str:
        left = "  ·  ".join(
            x for x in (f" HPCA {VERSION}", self.profile, self.mode) if x
        )
        right = f"{self.frame_ms:5.2f}ms  ·  ? keys  "
        gap = width - cell_width(left) - cell_width(right)
        text = left + " " * gap + right if gap > 0 else left
        return REVERSE + pad(text, width) + RESET

    def _keys(self) -> list[tuple[str, str]]:
        """Only what applies where the cursor is — the footer's whole job.

        The Textual app gets this from ``check_action`` per binding; here it is
        one function, which is easier to read and impossible to get out of step
        with what the keys actually do.
        """
        # esc esc works from every row, so it belongs in the part every row
        # shows — and ahead of "? keys" and "q quit", because a footer drops
        # whole pairs off its end and this is the one that must not be the pair
        # that goes. Leaving the message box is ^↑, not escape: escape has a
        # job now.
        common = [("^↑^↓", "row"), ("esc esc", "stop"), ("?", "keys")]
        if self.confirm is not None:
            return [("y", "yes"), ("n", "no"), ("esc", "no")]
        if self.menu():
            # While a command is being named the menu owns ↑/↓ and the two keys
            # that fill one in, and saying so is the only way anybody finds tab.
            return [
                ("↑↓", "pick"),
                ("⇥", "complete"),
                ("enter", "complete / run"),
                ("esc esc", "stop"),
                ("?", "keys"),
            ]
        if self.focus == DECISION:
            decision = self.session.decision
            if decision is not None and not decision.asking:
                return [
                    ("enter", "send the reason"),
                    ("esc", "no reason"),
                    ("⇧enter", "new line"),
                ]
            return [
                ("y", "approve"),
                ("n", "deny"),
                ("esc", "deny, no reason"),
                ("^↑", "row"),
            ]
        if self.focus == INPUT:
            # Spelled out rather than built from ``common`` so that send and
            # stop come first — the message box is where you sit while a turn
            # runs, and those are the two keys that matter there.
            return [
                ("enter", "send"),
                ("esc esc", "stop"),
                ("^↑^↓", "row"),
                ("⇧enter", "new line"),
                ("^←→", "word"),
                ("^l", "switch llm"),
                ("⇧←→", "select"),
                ("^⌫ ^del", "cut word"),
                ("^u", "clear"),
                ("^e", "$editor"),
                ("?", "keys"),
            ]
        rows = [("↑↓", "line"), ("→←", "open"), ("⇧→←", "open all")]
        if self.focus == SESSIONS and self.session_pane.here() == NEW_SESSION_KEY:
            # The three session keys do nothing on this row, so the footer
            # does not offer them: a key list that lies is worse than a short
            # one (specs-ui-acceptance.md, "rename keys are inert").
            rows += [("enter", "start a session")]
        elif self.focus == SESSIONS:
            rows += [("enter", "switch"), ("r", "rename"), ("t", "retitle"), ("d", "delete")]
        elif self.focus == CHAT:
            rows += [
                ("i", "write"),
                ("enter", "reuse"),
                ("y", "copy"),
                ("⇧tab", "mode"),
            ]
        else:
            rows += [("enter", "peek"), ("d", "unwatch"), ("alt-↑↓", "move")]
        # Only where they do something. `m` and `a` are the sessions row's, `c`
        # is everywhere but the chat, and ctrl+l is the chat's — a key list
        # that lies is worse than a short one.
        if self.focus == SESSIONS:
            rows += [
                ("m", "llms"),
                ("a", "profiles"),
                ("c", "config"),
                ("q", "quit"),
            ]
        elif self.focus == CHAT:
            rows += [("^l", "switch llm")]
        else:
            rows += [("c", "config")]
        return rows + common

    # --------------------------------------------------------------- input

    def handle(self, key: str, width: int, height: int) -> bool:
        if is_paste(key):
            self._paste(paste_text(key))
            return True
        if self.confirm is not None:
            # First, and above the overlay: a confirmation is asked over
            # whatever raised it, and it is answered before anything else can
            # be done (see `_over_confirm`).
            return self._handle_confirm(key)
        if self.overlays:
            overlay = self.overlays[-1]
            alive = overlay.handle(key, width, height - 2)
            # A screen that opened a screen has not closed, and one that closed
            # cannot also have opened one — so the two are exclusive and the
            # order only decides which is checked first.
            if overlay.child is not None:
                self._adopt(overlay)
            elif not alive:
                self._pop()
            return True
        if self.focus == DECISION:
            # After the overlay, because a decision can arrive while a screen
            # is open: the screen keeps the keys until it closes, and the
            # prompt — which is underneath it, not over it — has them the
            # moment it does.
            return self._handle_decision(key)
        if self.focus == INPUT:
            return self._handle_input(key)
        return self._handle_row(key, width, height)

    def _paste(self, text: str) -> None:
        """A block the terminal handed over whole, inserted literally.

        Where it goes is decided by where the cursor is, with one deliberate
        exception: from one of the list rows it goes to the message box and
        takes the focus with it. A paste is an unambiguous "I am entering
        text", the rows have nowhere to put one, and silently dropping a
        multi-kilobyte payload because the cursor happened to be in the chat is
        the worse of the two answers. A screen that has no editor open — the
        key list, the session picker — does drop it, because there the paste
        would have to close the screen to land anywhere.
        """
        if self.overlay is not None:
            self.overlay.paste(text)
            return
        self.focus = INPUT
        self.input.insert_text(text)

    def _closed(self, overlay: Overlay) -> None:
        """A screen that answered with something the rows have to act on."""
        if isinstance(overlay, QueuedOverlay) and overlay.choice:
            self._queued(overlay)
        elif isinstance(overlay, RewindOverlay) and overlay.choice:
            self._rewind(overlay)
        elif isinstance(overlay, NewSessionOverlay) and overlay.chosen:
            # Only now, at the end of both stages: the command that makes a
            # conversation is sent once, and escaping either picker sends none.
            self.send(NewSession(overlay.profile, overlay.backend))
            self.note = f"new session under “{overlay.profile}”"
        elif isinstance(overlay, ConfigOverlay) and overlay.saved:
            self.send(SaveSettings(overlay.text))
            self.settings_json = overlay.text
            self.note = "settings saved"
        elif isinstance(overlay, ProfilesOverlay):
            # The screen sent its own commands; what comes back is the list it
            # now believes in, so the next `a` opens what the last one left.
            self.profiles = list(overlay.profiles)
        elif isinstance(overlay, LlmOverlay):
            self.catalog = overlay.discovered + overlay.configured
        elif isinstance(overlay, SwitchLlmOverlay) and overlay.chosen is not None:
            # Shown before the core answers, like the mode bar: the next
            # `session.rows` is what makes it true, and what puts it back.
            self.session_for(overlay.session_id).model = overlay.chosen.model
            self.refresh_sidebar()
            self.note = f"this session now talks to {overlay.chosen.label}"
        elif isinstance(overlay, ThinkingOverlay) and overlay.effort:
            session = self.session_for(overlay.session_id)
            session.thinking = overlay.effort
            session.context.effort = overlay.effort
            self.note = f"thinking effort: {overlay.effort}"
        elif isinstance(overlay, SkillCreatorOverlay) and overlay.saved:
            self.send(
                SaveSkill(
                    self.profile, overlay.name, overlay.text, overlay.level
                )
            )
            # Shown before the core answers, and remembered locally, because
            # the menu is what a new skill has to appear in and there is no
            # event that says a skill file was written.
            self._remember_skill(
                overlay.name, overlay.description, overlay.level
            )
            self.note = f"saved skill “{overlay.name}” · {overlay.where}"
        elif isinstance(overlay, SkillRemoveOverlay) and overlay.removed:
            self.forget_skills(overlay.removed)
            self.note = f"removed skill “{overlay.removed}”"
        elif isinstance(overlay, RenameOverlay) and overlay.name:
            self.send(Rename(overlay.session_id, overlay.name))
            # Shown before the core answers, like the mode bar: the next
            # `session.rows` is what makes it true, and what puts it back if
            # the core refuses.
            self.session_for(overlay.session_id).title = overlay.name
            self.refresh_sidebar()
            self.note = f"renamed to “{overlay.name}”"

    # -------------------------------------------------- the generic yes/no

    def ask(
        self,
        question: str,
        on_answer: Callable[[bool], None] | None = None,
        *,
        confirm_id: str = "",
    ) -> None:
        """Put a yes/no on screen. The answer goes wherever it belongs.

        Two kinds of caller, one dialog. The UI's own questions pass
        ``on_answer`` and nothing is waiting on the other side of a socket for
        them. `confirm.requested` passes ``confirm_id``: the core is holding a
        continuation under that id and only the verdict crosses back
        (`protocol.ConfirmResolve`), which is why the question need not name a
        session — a triage offer comes from a poll, not a conversation.
        """
        self.confirm = Confirm(
            id=confirm_id, question=question, on_answer=on_answer
        )

    def _handle_confirm(self, key: str) -> bool:
        """y, n, escape. Anything else is ignored rather than passed on: a
        modal that leaked its keys would act on the screen behind it."""
        if key == "quit":
            return False
        if key == "y":
            self._answer(True)
        elif key in ("n", "esc"):
            # Escape is "no" and not "ask me later": the question is a gate,
            # and a gate with a way past it that answers neither is a turn
            # parked on nothing.
            self._answer(False)
        # The one answer that ends the loop rather than changing the frame.
        return not self.quitting

    def _quit_answer(self, yes: bool) -> None:
        """"Really quit?", answered. `handle` reads it on the way out."""
        self.quitting = yes

    def _answer(self, confirmed: bool) -> None:
        question, self.confirm = self.confirm, None
        if question is None:  # pragma: no cover - guarded by the caller
            return
        if question.id:
            self.send(Answer(question.id, confirmed))
        if callable(question.on_answer):
            question.on_answer(confirmed)

    def confirm_requested(self, confirm_id: str, question: str) -> None:
        """`confirm.requested`, which the Textual UI never drew at all."""
        self.ask(question, confirm_id=confirm_id)

    # ------------------------------------------------------ the decision

    def decision_arrived(self, session_id: str) -> None:
        """A session is parked on an approval — this one, or another one.

        Another one changes exactly one thing about the frame: the "!" in the
        sidebar (§3.2 property 1). This one puts the prompt up and lands on it
        so its keys work at once — but only from the chat column, because a
        decision must never pull the cursor out of the sessions or watchers
        row the user is working in.
        """
        self.refresh_sidebar()
        if session_id == self.active_id and self.focus in (CHAT, INPUT):
            self.focus = DECISION

    def decision_cleared(self, session_id: str) -> None:
        """The core says that decision is gone (answered, or its turn died)."""
        self.refresh_sidebar()
        if session_id == self.active_id and self.focus == DECISION:
            self.focus = INPUT

    def _handle_decision(self, key: str) -> bool:
        """The two stages, and the keys each of them owns.

        The reason box takes the letters the y/n stage was using — it is a
        text field, and "n" in the middle of "not this path" is not a verdict
        — which is why the stage gates the keys rather than both being live at
        once (`DecisionBar.check_action` did the same with `check_action`).
        """
        if key == "quit":
            return False
        decision = self.session.decision
        if decision is None:  # answered, or its session went away
            self.focus = INPUT
            return True
        if decision.asking:
            if key == "y":
                self._resolve(True)
            elif key == "n":
                # Not an answer yet: the refusal is sent once the box says
                # why, or says nothing.
                decision.decline()
            elif key == "esc":
                self._resolve(False)
            elif key == "ctrl-up":
                self.focus = CHAT
            elif key in ("ctrl-down", "tab", "i"):
                self.focus = INPUT
            return True
        if key == "enter":
            self._resolve(False, decision.reason_text())
        elif key == "esc":
            self._resolve(False)
        elif key in NEWLINE_KEYS:
            decision.reason.newline()
        elif key == "ctrl-up":
            self.focus = CHAT  # the half-written reason stays where it is
        elif key == "ctrl-down":
            self.focus = INPUT
        else:
            decision.reason.handle(key)
        return True

    def _resolve(self, approved: bool, reason: str = "") -> None:
        """Answer the open session's decision and let the turn go on.

        Cleared here rather than waiting for `decision.cleared` to come back:
        the prompt has been answered, and a question that stays on screen
        until the core agrees is a question the user can answer twice.
        """
        session = self.session
        if session.decision is None:
            return
        session.clear_decision()
        self.refresh_sidebar()
        self.focus = INPUT
        self.send(Decide(session.session_id, approved, reason))
        self.note = "approved" if approved else "declined"

    # ------------------------------------------------------- the queue

    def _queued(self, overlay: QueuedOverlay) -> None:
        """What the queued-message dialog decided, for the session it was
        opened in — which need not be the one on screen when it closes."""
        if overlay.choice == UNQUEUE:
            # By seq: a turn finishing while the dialog sat open shifts every
            # position in the queue, and the row's own name cannot drift. The
            # text comes back on `turn.unqueued`, from the core, because by
            # then the row may have been redrawn.
            self.send(Unqueue(overlay.session_id, overlay.seq))
            self.note = "cancelling that message"
            return
        self.hand_back(overlay.session_id, overlay.message, note="copied")

    def hand_back(self, session_id: str, text: str, *, note: str = "") -> None:
        """A message the core gave back, into *that* session's draft.

        Not the visible one. An interrupt's rollback and a cancelled queued
        message both answer asynchronously, and the user can be looking at
        another conversation by the time they do — dropping the text into
        whichever entry happens to be on screen would put one session's words
        into another's turn.
        """
        session = self.session_for(session_id)
        self._into_draft(session, text)
        if session_id == self.active_id:
            self.focus = INPUT
            self.note = note or "the message is back in the box"
        else:
            title = session.title or session_id
            self.note = f"“{title}” — the message is waiting there"

    def _into_draft(self, session: SessionState, text: str) -> None:
        """Added to whatever is already being written rather than replacing
        it, so nothing the user typed can be lost by a message coming back. It
        starts its own line, except after a draft left ending in whitespace —
        that space is how you say "continue here"."""
        draft = session.draft.text()
        if draft and not draft[-1].isspace():
            draft += "\n"
        session.draft.set_text(draft + text)

    # -------------------------------------------------------- the chat rewind

    def _activate_chat(self, width: int) -> None:
        """Enter in the chat log.

        On the working row it stops the turn; on one of your own messages it
        opens the rewind. On anything else it moves to the message box, which
        is what app.py answers an Enter it has nothing better to do with.
        """
        position = self.chat.current(width)
        if self.chat.is_tail(position):
            self._stop_from_the_row()
            return
        entry = self.session.entry_at(position)
        if entry is not None and entry.kind == "queued":
            # A message that has not reached the model yet: the offer is to
            # take it back, not to rewind to it (§4.3 item 34).
            self.overlay = QueuedOverlay(entry.text, entry.seq, self.active_id)
        elif entry is not None and entry.kind in OWN_MESSAGE_KINDS:
            # The row's own name, not its position: the cut is decided by the
            # user now and carried out by the core later, and a turn appending
            # in between moves every position after it.
            self.overlay = RewindOverlay(entry.text, entry.seq, self.active_id)
        else:
            self.focus = INPUT

    def _stop_from_the_row(self) -> None:
        """Enter on the working row: the aimed half of the stop gesture.

        Not every spinner can be stopped. A backend call that is not a turn —
        a silent `/conclude`, a compaction, the titler — has no turn to roll
        back, and the honest answer to Enter on it is to say so: a key that
        looked like it did something and did not is worse than one that
        explains itself.
        """
        if self.session.turn.interruptible:
            # Asked, unlike `esc esc`: this key is aimed at a row in a log the
            # agent is writing into, so it is the one that can be hit by
            # accident. The dialog can sit open long enough for the reply to
            # land, so what it decided is re-checked before it is sent.
            session_id = self.active_id
            self.ask(
                INTERRUPT_QUESTION,
                lambda yes: self._confirmed_stop(session_id, yes),
            )
        else:
            self.note = NOT_A_TURN
            self.note_style = DIM

    def _confirmed_stop(self, session_id: str, yes: bool) -> None:
        """The answer to the interrupt dialog. A reply that landed while it
        was open makes it a no-op — there is no longer a turn to stop."""
        if not yes:
            return
        if not self.session_for(session_id).turn.interruptible:
            self.note = "that turn finished while you were deciding"
            self.note_style = DIM
            return
        self.send(Interrupt(session_id))
        self.note = "stopped the turn"

    def _rewind(self, overlay: RewindOverlay) -> None:
        """What the rewind decided, as an intent aimed at the session it was
        opened in — which is not necessarily the one on screen by the time it
        closes."""
        if overlay.choice == COPY:
            self.reuse_message(overlay.message, overlay.session_id)
            return
        if overlay.choice == FORK:
            self.send(Fork(overlay.session_id, overlay.seq))
        elif overlay.choice == ROLLBACK:
            self.send(Rollback(overlay.session_id, overlay.seq))
        # Either cut leaves you at the point the conversation now ends, which
        # is a place to say the next thing from. `reuse_message` above puts the
        # cursor in the box for itself.
        self.focus = INPUT

    def reuse_message(self, text: str, session_id: str = "") -> None:
        """Put one of your own past messages back in the box, to send again or
        edit into the next one — usually a command that needs a word changed,
        which is otherwise retyped off the screen.

        Into the draft of the session it was *taken from*, which is what the
        rewind promises ("an intent aimed at the session it was opened in —
        which is not necessarily the one on screen by the time it closes").
        That is the whole of `hand_back`'s errand already, so this goes
        through it rather than keeping a second copy of the rule that once
        disagreed with it.

        Added to whatever is already being written rather than replacing it, so
        activating a message can never lose a draft. It starts its own line,
        except after a draft left ending in whitespace — that space is how you
        say "continue here" (``rerun this: `` + the old command).
        """
        self.hand_back(
            session_id or self.active_id, text, note="copied into the message box"
        )

    def _esc_armed(self) -> bool:
        """Whether a first escape is still waiting for its second.

        Read by the footer, which says so in red. app.py deliberately stays
        silent here, but its objection is to a *toast*: a notification for
        every stray escape sequence would be noise on top of the screen. A
        word in the footer costs nothing, cannot cover anything, and goes away
        on its own — ``next_wake`` books the repaint that clears it, so the
        hint expires with the window rather than sitting there until the next
        key.
        """
        return (
            self._esc_armed_at is not None
            and self.clock() - self._esc_armed_at <= ESC_STOP_WINDOW
        )

    def next_wake(self) -> float | None:
        """Seconds until this frame goes stale on its own, or None.

        The loop repaints because something happened — a key, an event, a
        resize — and not on a timer, so anything that changes by the clock
        alone has to say when it will. Two things do, and the answer is the
        sooner of them: the armed escape, which is only true for
        ``ESC_STOP_WINDOW`` and has no keypress coming to wipe it, and the
        spinner, which turns.

        The spinner deliberately goes through here rather than being given a
        timer of its own. A widget that repaints itself is a poll by another
        name, and it is a poll that runs whether or not the terminal is even
        being looked at; this way the one loop still wakes exactly once per
        thing that will have changed, and an idle UI with no turn in flight
        still costs nothing at all.

        Asked after every frame, so a value that has already expired is None
        rather than zero — zero would be a repaint that schedules another
        repaint.
        """
        waits = [
            self.session.next_wake(self.wall()),
            # The third thing that changes with no keypress behind it: a toast
            # goes on its own, and a frame drawn with one on it stops being
            # true the moment it expires.
            toasts.next_wake(self.toasts, self.clock()),
        ]
        if self._esc_armed_at is not None:
            left = ESC_STOP_WINDOW - (self.clock() - self._esc_armed_at)
            # A hair past the window rather than exactly on it: `_esc_armed`
            # is true *at* the boundary, so waking there would redraw the same
            # frame and then have nothing left to schedule — the hint would
            # stick.
            waits.append(left + 0.01 if left > 0 else None)
        due = [x for x in waits if x is not None]
        return min(due) if due else None

    def _escape(self) -> bool:
        """Two escapes in quick succession stop the turn. One does nothing.

        Deliberately nothing, which is what app.py's ``action_stop_turn``
        settled on: the doubling *is* the confirmation, and a lone escape is
        too easy to arrive by accident — it is also the first byte of every
        arrow key — to end a turn on.

        It works from wherever the cursor is, the message box included, and
        that is the whole point of the gesture: a turn calling tool after tool
        is writing rows into the very list you would otherwise have to aim at.
        Escape needs no target.
        """
        now = self.clock()
        first, self._esc_armed_at = self._esc_armed_at, now
        if first is None or now - first > ESC_STOP_WINDOW:
            return False
        self._esc_armed_at = None  # spent: a third press opens a fresh pair
        if not (self.active_id and self.session.turn.interruptible):
            # The gesture completed and there was nothing for it to do. Said
            # rather than claimed: the footer used to report a stop it had not
            # made, on an idle session and even with nothing open at all —
            # and the same sentence is what four tests took as their proof
            # that the interrupt reached the core. Which of the two answers it
            # is matters: a spinner that cannot be stopped is a different
            # thing from no spinner, and `_stop_from_the_row` already tells
            # them apart.
            self.note = NOT_A_TURN if self.session.turn.busy else NOTHING_TO_STOP
            self.note_style = DIM
            return True
        self.send(Interrupt(self.active_id))
        self.note = "stopped the turn"
        return True

    def _handle_input(self, key: str) -> bool:
        if key == "esc":
            self._escape()
        elif self._menu_key(key):
            pass  # the "/" menu had it: see `_menu_key`
        elif key == "enter":
            self._send()
        elif key in NEWLINE_KEYS:
            self.input.newline()
        elif key == "shift-tab":
            # Cycling the mode is the one thing shift+tab does, and it has to
            # work from here: deciding the agent may act unasked is a thought
            # you have *while writing the message*, not one you leave the box
            # to act on. The Textual app bound it `priority=True` for exactly
            # that reason. ctrl+↑ is how you leave the box.
            self._cycle_mode()
        elif key == "ctrl-up":
            self.focus = CHAT
        elif key == "ctrl-down":
            self.focus = WATCHERS
        elif key == "ctrl-l":
            # From the box too: which model answers is a thought you have while
            # writing the message, like the mode — and ctrl+l is not a
            # printable character, so it cannot be something being typed.
            self._switch_llm()
        elif key == "ctrl-e":
            self._edit_profile()
        elif key == "quit":
            return False
        else:
            self.input.handle(key)
        return True

    def _menu_key(self, key: str) -> bool:
        """The keys the "/" menu takes while a command is being named.

        ↑/↓ pick, tab fills, and enter fills a partial command and runs a
        complete one — the same four the Textual entry answered, and the
        reason the menu closes the moment the token contains whitespace: from
        there on ↑/↓ belong to the draft, which may be a multi-line message
        that merely opens with a slash.
        """
        matches = self.menu()
        if not matches:
            return False
        at = self._menu_at(matches)
        if key in ("up", "down"):
            self.session.menu_at = (at + (1 if key == "down" else -1)) % len(matches)
            return True
        if key not in ("tab", "enter"):
            return False
        chosen = matches[at]
        stripped = self.input.text().lstrip()
        if stripped[1:] == chosen.name:
            # Already fully typed: tab has nothing to add and enter runs it.
            return key == "tab"
        # A space after the name, which is also what closes the menu — so one
        # enter fills the command in and the next one sends it.
        self.input.set_text(f"{stripped[0]}{chosen.name} ")
        self.session.menu_at = 0
        return True

    def _send(self) -> None:
        """Ask for the turn. The row it becomes comes back as a `chat.append`.

        Nothing is written into the chat here, and that is the append-only
        invariant (§3.2) seen from the writing end: a UI that drew its own copy
        of the message would have two rows to reconcile the moment the core
        sent the real one — which is exactly how the queued-message bugs the
        Textual app carried were made.
        """
        text = self.input.text().strip()
        if not text:
            return
        named = commands.split(text)
        if named is not None:
            self._command(named[0], named[1], text)
            return
        if not self.active_id:
            self.note = "no session open"
            return
        self.send(Submit(self.active_id, text))
        self.input.clear()
        self.note = "sent"

    # The three built-ins this side answers by drawing something instead of
    # sending `command.run`. `/thinking` is a chooser the core expects a
    # front-end to put up; `/skill-creator` is a form the core says outright it
    # does not own; a bare `/skill-remove` is a picker, because "the chosen
    # skill" should be a row you point at rather than a name you retype (the
    # core still answers `/skill-remove <name>` and that path is left alone).
    SCREEN_COMMANDS = ("thinking", "skill-creator", "skill-remove")

    def _command(self, name: str, args: str, text: str) -> None:
        """A typed `/name …`: a skill, a built-in, or a typo (§4.3 item 24)."""
        if self._skill_named(name) is not None:
            # An ordinary turn carrying a skill, not a command: it queues and
            # runs concurrently like any message, and what the skill adds goes
            # into the model's copy rather than the transcript (the core's
            # job, `protocol.TurnSubmit.forced_skill`).
            if not self.active_id:
                self.note = "no session open"
                return
            self.send(Submit(self.active_id, text, forced_skill=name))
            self.input.clear()
            self.note = f"sent, with the “{name}” skill"
            return
        if name not in commands.BUILTIN_NAMES:
            # Almost always a typo, so the draft stays: the user fixes the
            # spelling — or reopens the menu to look the name up — instead of
            # retyping the sentence they attached to it.
            self.toast(commands.unknown(name), "warning")
            return
        if name in self.SCREEN_COMMANDS and self._screen_command(name, args):
            # Ahead of the busy check on purpose: these three draw a screen
            # and ask the core for nothing, so there is no worker for a
            # running turn to collide with. Picking a thinking level while the
            # model is thinking is the case that makes the point.
            self.input.clear()
            return
        if self.session.turn.busy:
            # A command that *does* reach the core acts on the UI and runs its
            # own exclusive worker, so there is nothing sensible to queue it
            # behind — and a `/compact` that ran an hour later against a thread
            # the turn had since changed would be worse than one that was
            # refused. An ordinary message queues; this one waits for the user.
            self.note = "wait for this turn — a command cannot be queued"
            self.note_style = DIM
            return
        command = next(x for x in commands.BUILTINS if x.name == name)
        if command.session and not self.active_id:
            self.note = f"no session open — /{name} works on one conversation"
            return
        # The session id goes with every command that has one, the
        # profile-scoped ones included: it is how the core decides *which*
        # profile is asking (`core.service._list_skills`), and it is empty —
        # null on the wire — exactly when there is nothing open.
        self.send(RunCommand(name, args, self.active_id))
        if command.folds:
            # `/compact` rewrites the thread without taking a message out of
            # it, so no `chat.reset` follows and nothing else would ever
            # unstick the measured fill: the bar would keep showing the
            # pre-fold 92%, unmarked, for as long as the session stayed quiet.
            # Said here, where the command is known, rather than waited for —
            # the core's fresh `context.estimate` is a round trip away and the
            # number on screen is wrong the moment the command goes out.
            self.session.context.superseded()
        self.input.clear()
        self.note = f"/{name}"

    def _screen_command(self, name: str, args: str) -> bool:
        """Draw the answer rather than send it. False falls through to the core."""
        if name == "thinking":
            self.thinking()
            return True
        if name == "skill-creator":
            if args:
                # A draft is a model call, so it happens on the core's side
                # (`protocol.SkillDraft`); the form opens when it answers, and
                # opens empty if it could not. Nothing is drawn in between —
                # the wait is a spinner in the conversation, which is where
                # the core reports it.
                self.send(DraftSkill(self.profile, args, self.active_id))
                self.input.clear()
                self.note = "drafting a skill…"
                return True
            self.overlay = self._skill_form()
            return True
        if name == "skill-remove" and not args:
            own = self._own_skills()
            if not own:
                self.toast(
                    f"profile “{self.profile}” has no skills of its own to remove",
                    "warning",
                )
                return True
            self.overlay = SkillRemoveOverlay(self.profile, tuple(own))
            return True
        return False

    def _skill_form(
        self, drafted: SkillInfo | None = None, request: str = ""
    ) -> Overlay:
        """The creator's form, empty or filled with the model's draft.

        One form either way, which is the claim: a draft is a head start
        inside the screen the user was going to fill in anyway, edited and
        confirmed exactly like a hand-typed skill.
        """
        return SkillCreatorOverlay(
            self.profile,
            taken=tuple((x.name, x.level) for x in self.visible_skills()),
            request=request,
            name=drafted.name if drafted else "",
            description=drafted.description if drafted else "",
            body=drafted.text if drafted else "",
        )

    def skill_drafted(
        self, drafted: SkillInfo | None, request: str = "", error: str = ""
    ) -> None:
        """`skill.drafted` arrived: open the form over what came back.

        A failed draft opens the *empty* form rather than losing the command —
        the user asked for a skill about something, and the worst answer is
        the one that takes the request away and says nothing.
        """
        if error:
            self.toast(f"could not draft that skill ({error})", "warning")
        self.overlay = self._skill_form(drafted, request)

    def _skill_named(self, name: str) -> tuple[str, str] | None:
        """The visible skill that `/name` names, or None. Built-ins win."""
        if not name or name in commands.BUILTIN_NAMES:
            return None
        return next((x for x in self.skills() if x[0] == name), None)

    def _own_skills(self) -> list[SkillInfo]:
        """The open profile's removable skills — its own and this project's.

        Read off the menu's list rather than fetched again, and filtered by
        level: `skill.list` answers one question per round trip, and asking
        the same profile twice on the same keystroke to get a subset of what
        already arrived would be a second answer to disagree with the first.
        A shipped or global skill is callable and is not one profile's to
        delete, which is exactly what `SkillInfo.removable` says.
        """
        return [x for x in self.visible_skills() if x.removable]

    def _handle_row(self, key: str, width: int, height: int) -> bool:
        if key == "quit":
            return False  # ctrl+c and ctrl+d: out, and no question asked
        if key == "q":
            # Confirmed, and from the sessions column only (§5). Everywhere
            # else it is inert: on the watchers column it is a letter with
            # nothing to do, and in the chat it is a letter somebody is about
            # to type into the box they just left.
            if self.focus == SESSIONS:
                self.ask(QUIT_QUESTION, self._quit_answer)
            return True
        inner = max(8, width - 2)
        slots = [SESSIONS, CHAT, INPUT, WATCHERS]
        view = max(1, self._heights(height, width)[slots.index(self.focus)] - 1)
        pane = {
            SESSIONS: self.session_pane,
            CHAT: self.chat,
            WATCHERS: self.watchers,
        }[self.focus]
        self.note = ""
        if key == "esc":
            self._escape()
        elif key == "?":
            self.overlay = HelpOverlay()
        elif key == "m" and self.focus == SESSIONS:
            # Sessions row only, like `a` (§5). Both are global screens opened
            # from the one row whose keys are about the app rather than about a
            # conversation — and in the chat `m` and `a` are letters somebody
            # may be about to type into the box they just left.
            self.overlay = LlmOverlay(self.catalog)
        elif key == "a" and self.focus == SESSIONS:
            self.overlay = ProfilesOverlay(self.profiles)
        elif key == "c" and self.focus != CHAT:
            # Anywhere but the chat column (§5), which is the one row where the
            # cursor is on a conversation and `c` reads as a letter.
            # Fetched as it opens, like every other editable body: the file is
            # also written by the core — `settings.save` normalises what lands
            # on disk — so a copy kept from the last time this screen was open
            # is a copy that can already be wrong.
            self.overlay = ConfigOverlay(
                self.settings_json,
                validate=self.validate_settings,
                awaiting=SETTINGS_KEY,
            )
        elif key == "ctrl-l" and self.focus == CHAT:
            self._switch_llm()
        elif key == "shift-tab" and self.focus == CHAT:
            # The mode is a per-session dial, so the key means something only
            # where a session's conversation is: here and in the message box
            # (see `_handle_input`). From the sessions and watchers rows
            # shift+tab keeps moving between rows, which is what the Textual
            # app's `check_action` decided for the same reason.
            self._cycle_mode()
        elif key in ("ctrl-down", "tab"):
            self.focus = slots[(slots.index(self.focus) + 1) % len(slots)]
        elif key in ("ctrl-up", "shift-tab"):
            self.focus = slots[(slots.index(self.focus) - 1) % len(slots)]
        elif key == "ctrl-e":
            self._edit_profile()
        elif key == "y" and self.focus == CHAT:
            # §5: new, and colliding with nothing — the decision prompt owns
            # `y` only while a decision is pending, and it takes keys first.
            self._copy_row(inner)
        elif key == "i" and self.focus == CHAT:
            self.focus = INPUT
        elif key == "up":
            pane.move(-1, view, inner)
        elif key == "down":
            pane.move(1, view, inner)
        elif key == "pgup":
            pane.move(-view, view, inner)
        elif key == "pgdn":
            pane.move(view, view, inner)
        elif key == "home":
            pane.move(-(10**9), view, inner)
        elif key == "end":
            pane.move(10**9, view, inner)
        elif key == "right":
            # Open it; on one already open, step into what it opened, the way
            # a file tree does. On an entry with no body, nothing.
            if not pane.expand(inner) and pane.is_open(pane.current(inner)):
                pane.move(1, view, inner)
        elif key == "left":
            pane.collapse(inner)
        elif key == "shift-right":
            pane.expand_all(inner)
        elif key == "shift-left":
            pane.collapse_all(inner)
        elif key in ("alt-up", "alt-down") and self.focus == WATCHERS:
            moved = pane.reorder(-1 if key == "alt-up" else 1, view, inner)
            self.note = "moved" if moved else ""
        elif key == "enter":
            if self.focus == SESSIONS:
                self._enter_session(self._sidebar_key(inner))
            elif self.focus == CHAT:
                self._activate_chat(inner)
            elif self.focus == WATCHERS:
                self._watch(Peek, "peeking at the log", inner)
        elif key == "r" and self.focus == SESSIONS:
            self._rename(self._sidebar_key(inner))
        elif key == "t" and self.focus == SESSIONS:
            self._retitle(self._sidebar_key(inner))
        elif key == "d":
            if self.focus == SESSIONS:
                self._confirm_delete(self._sidebar_key(inner))
            else:
                self._watch(Drop, "unwatched", inner)
        return True

    # ------------------------------------------------- the clipboard and $EDITOR

    def _copy_row(self, inner: int) -> None:
        """`y`: the chat row under the cursor, to the clipboard (§4.3 item 36).

        Through `hpca.clipboard.ClipboardManager`, which already knows about
        OSC 52, the multiplexer wrapping and the file fallback, and which
        already takes an injected ``emit`` — so nothing about copying is
        rewritten here and this method only decides *what* is copied. The
        entry's own text, not the row it is drawn as: `you   ` and the tool
        markers are decoration, and pasting them back into a shell is a paper
        cut every time.
        """
        if self.clipboard is None:
            self.toast("no clipboard is wired up here", "warning")
            return
        at = self.chat.current(inner)
        if not (0 <= at < len(self.chat.items)):
            self.note = "nothing to copy"
            return
        item = self.chat.items[at]
        text = item.text or "\n".join([item.head, *item.body])
        if not text.strip():
            self.note = "nothing to copy"
            return
        try:
            self.toast(self.clipboard(text))
        except Exception as e:  # a tier that raised rather than reporting
            self.toast(f"copy failed: {e}", "error")

    def _edit_profile(self) -> None:
        """ctrl+e: this profile's memories in `$EDITOR` (§4.3 item 37).

        Textual spelled this `self.suspend()`; here it means leaving the
        alternate screen, putting the line discipline back, running the editor
        on a terminal that behaves like a terminal, and coming back to a full
        repaint. Both halves are injected — `run.py` owns the terminal and
        `client.py` owns the wire the file comes down — so this method is the
        key binding, one guard, and nothing else.

        Nothing is returned and nothing is waited for: the body has to be
        fetched before there is anything to edit, so the answer arrives as a
        toast (`UIClient._profile_body`). The guard is what keeps the key
        honest in a UI with no wire behind it — a key that silently did
        nothing would look exactly like an editor that opened and closed.
        """
        if self.suspend is None or self.edit_profile is None:
            self.toast("no editor is wired up here", "warning")
            return
        self.edit_profile(self.profile)

    # ------------------------------------------------------- the session keys

    def _sidebar_key(self, inner: int) -> str:
        """What the sidebar cursor is pointing at: a session id, or the row
        that is not one. The three session keys and Enter all start here, so
        "inert on `(new session)`" is one comparison rather than four."""
        return self.session_pane.key_at(self.session_pane.current(inner))

    def _enter_session(self, key: str) -> None:
        """Enter in the sidebar: open that conversation, or start one."""
        if key == NEW_SESSION_KEY:
            self._new_session()
            return
        self._switch(key)
        # Straight to the box: opening a session is something you do in order
        # to say something in it — unless it is parked on a decision, which is
        # the thing to do in it first.
        self.focus = DECISION if self.session.decision is not None else INPUT

    def _new_session(self) -> None:
        """The two-stage picker (§4.3 item 14). Nothing is asked for until it
        closes with a choice, so escaping it creates nothing."""
        self.overlay = NewSessionOverlay(self._profile_rows(), self._backend_rows())

    def _backend_rows(self) -> list[Item]:
        """The catalog a new conversation may be pinned to.

        Empty means no second stage at all, which is the case a fresh install
        is in: the core then talks to the bootstrap client, and a picker
        holding one thing that cannot be declined asks for nothing.
        """
        return [
            # `value` is the label and nothing else: `protocol.SessionNew`
            # settled on `LLMEntry.label` as the one field a command may name
            # an entry by, and a front-end that sent anything else pinned
            # nothing at all — silently, until the core started warning.
            choice(
                x.label,
                " · ".join(
                    y
                    for y in (x.model, x.base_url, "the default" if x.active else "")
                    if y
                ),
                value=x.label,
            )
            for x in self.catalog
            if not x.discovered
        ]

    def _switch_llm(self) -> None:
        """`ctrl+l` (§4.3 item 29): this conversation's backend, not the app's."""
        if not self.active_id:
            self.note = "no session open"
            return
        configured = [x for x in self.catalog if not x.discovered]
        if not configured:
            self.note = "no backends configured — press m on the sessions row"
            return
        self.overlay = SwitchLlmOverlay(
            configured, session_id=self.active_id, current=self.session.model
        )

    def thinking(self) -> None:
        """`/thinking` (§4.3 item 30). A conversation's level, so it needs one."""
        if not self.active_id:
            self.note = NO_SESSION
            return
        self.overlay = ThinkingOverlay(
            self.session.thinking, session_id=self.active_id
        )

    def review_memories(self, session_id: str = "") -> None:
        """Put the proposals a session is holding up for review (§4.3 item 32).

        By session and not "the open one": `memory.proposals` names the
        conversation it came out of, and answering it against another one
        would resolve the wrong offer.
        """
        session = self.session_for(session_id or self.active_id)
        if not session.proposals:
            self.note = "nothing to review"
            return
        self.overlay = MemoryReviewOverlay(session.proposals, session.session_id)

    def inspect(self, title: str, body: str) -> None:
        """A read-only window over text too long to be a toast (item 31)."""
        self.overlay = InspectOverlay(body, title=title)

    # ------------------------------------------------ what a scan answers with

    def probed(
        self,
        base_url: str,
        models: list[BackendInfo],
        *,
        needs_key: bool = False,
    ) -> None:
        """`backend.probed`: what one endpoint answered, to whoever asked.

        Addressed by endpoint rather than delivered to whatever is on screen:
        a probe of a node that has gone away costs the full timeout, and by
        then the form may have been escaped or pointed somewhere else. A
        screen that recognises the URL takes it; nobody recognising it is a
        question whose asker has gone, and the answer is dropped rather than
        toasted at a user who has moved on.
        """
        for screen in list(self.overlays):
            answer = getattr(screen, "probed", None)
            if answer is not None:
                answer(base_url, models, needs_key)
        if self.overlays:
            # A probe that has to be *chosen* between opens a picker, and a
            # screen opened outside a keypress still has to be adopted — the
            # one in `handle` only ever sees what a key asked for.
            self._adopt(self.overlays[-1])

    def scanned(
        self,
        *,
        found: int = 0,
        cluster: int = 0,
        notice: str = "",
        help_text: str = "",
    ) -> None:
        """`backend.scanned`: the scan is over, and what its emptiness meant.

        The rows are already drawn — every hit restated the catalog — so this
        is the verdict, and the two texts are rendered differently on purpose
        (`protocol.BackendScanned`): ``notice`` is a passing remark and goes
        in a toast, ``help`` is the tunnel recipe and needs a window that
        holds a selection and waits to be dismissed. Both empty means the scan
        found things and there is nothing to explain.
        """
        for screen in self.overlays:
            done = getattr(screen, "scanned", None)
            if done is not None:
                done(found, cluster)
        if notice:
            self.toast(notice)
        if help_text:
            self.window(NO_ENDPOINTS, help_text)

    def _profile_rows(self) -> list[Item]:
        """The profiles a new conversation can be started under.

        Assembled from what the UI has already been told rather than asked
        for again: `profile.rows` fills `self.profiles`, and the picker adds
        what it knows besides — the core's own profile from `hello` and the
        profile of every session in the sidebar, so a conversation can be
        started under a profile the list has not arrived for. The default
        leads, then the rest in the order they were met — which puts the
        profile the user is working in at the top of the list they are about
        to pick from.
        """
        names: list[str] = []
        for name in (
            DEFAULT_PROFILE,
            self.core_profile,
            *(x.profile for x in self.sessions),
            *(x.name for x in self.profiles),
        ):
            if name and name not in names:
                names.append(name)
        counts: dict[str, int] = {}
        for session in self.sessions:
            counts[session.profile] = counts.get(session.profile, 0) + 1
        return [choice(name, self._profile_detail(name, counts)) for name in names]

    @staticmethod
    def _profile_detail(name: str, counts: dict[str, int]) -> str:
        """What a profile row says about itself: how much is already under it,
        and whether it is the one a session gets by not choosing."""
        said = []
        if counts.get(name):
            said.append(f"{counts[name]} session" + ("s" if counts[name] > 1 else ""))
        if name == DEFAULT_PROFILE:
            said.append("the fallback")
        return " · ".join(said)

    def _rename(self, key: str) -> None:
        """`r`: the name, prefilled and editable. Inert on `(new session)`."""
        session = self._states.get(key)
        if session is None:
            return
        self.overlay = RenameOverlay(session.title, key)

    def _retitle(self, key: str) -> None:
        """`t`: ask the model. Which model, and whether there is anything to
        summarise, are the core's to answer."""
        if key not in self._states:
            return
        self.send(Retitle(key))
        self.note = "asking the llm for a title"

    def _confirm_delete(self, key: str) -> None:
        """`d`: ask, then delete. Inert on `(new session)`."""
        session = self._states.get(key)
        if session is None:
            return
        title = session.title or key
        self.ask(
            DELETE_QUESTION.format(title=title),
            lambda yes: self._delete(key, title, yes),
        )

    def _delete(self, session_id: str, title: str, yes: bool) -> None:
        """The answer to the delete question.

        The row is taken off here rather than waited for: `session.rows` says
        what the core holds, and `sync_sessions` deliberately keeps the *open*
        session's state even when the core stops listing it — so a delete that
        only sent a command would leave the deleted conversation's chat and
        half-typed draft on screen until something else switched away from it.
        """
        if not yes:
            self.note = "kept"
            return
        self.send(DeleteSession(session_id))
        self.forget(session_id)
        self.note = f"deleted “{title}”"

    def forget(self, session_id: str) -> None:
        """Take a conversation off the UI: its row, its chat and its draft.

        Its draft in particular, which is the one piece of it that lives
        nowhere else (specs-ui-acceptance.md, "Drafts"): the chat can be asked
        for again and the row will come back in the next `session.rows`, but
        an unsent message belongs to the session and goes with it.

        Deleting the one on screen leaves nothing open, which is what the
        Textual app did by closing the session: the chat empties, the message
        box has nowhere to send to, and the cursor goes back to the sidebar —
        the only row with anything left to do.
        """
        if self._states.pop(session_id, None) is None:
            return
        was = self.active_id
        self.sessions = [x for x in self.sessions if x.session_id != session_id]
        self._select("" if was == session_id else was)
        self.refresh_sidebar()
        if was == session_id:
            self.focus = SESSIONS

    def _cycle_mode(self) -> None:
        """Ask for the next mode. Which one that is, this does not know.

        `client.py` resolves the cycle (`hpca.agent.next_mode`) and persists
        it, because the modes are the agent's list and a key dispatcher that
        held a copy of it would be a second place for it to be wrong.
        """
        if not self.active_id:
            self.note = "no session open"
            return
        self.send(CycleMode(self.active_id))

    def _watch(self, intent: type, note: str, inner: int) -> None:
        """Peek at or drop the watch box under the cursor.

        The row carries the core's own ``PanelRow.ref`` (see
        `client._panel_item`), so what leaves here names a watch rather than a
        position in a column that a poll repaints twice a second.
        """
        item = self.watchers.current(inner)
        if item < 0:
            return
        self.send(intent(self.watchers.items[item].text))
        self.note = note
