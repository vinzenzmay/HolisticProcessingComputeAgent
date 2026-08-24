"""The rows themselves: layout, focus and key dispatch.

``RowUI`` takes keys and returns frames and does no I/O of its own, which is
what makes the whole UI testable by calling ``render()`` and comparing strings.
"""

from __future__ import annotations

import time
from collections import namedtuple
from collections.abc import Callable

from hpca import __version__ as VERSION
from hpca.ui import commands, theme, toasts
from hpca.ui.ansi import (
    BOLD,
    PULSE_INTERVAL,
    RESET,
    REVERSE,
    cell_width,
    cut,
    fold,
    footer_lines,
    footer_wrap,
    pad,
    rule,
    safe,
)
from hpca.ui.approval import decision_height, render_decision
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS, is_paste, paste_text
from hpca.ui.overlays import (
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
from hpca.ui.rain import rain
from hpca.ui.state import (
    OWN_MESSAGE_KINDS,
    Answer,
    BackendInfo,
    Confirm,
    CycleMode,
    Decide,
    DeleteSession,
    Display,
    DraftSkill,
    Drop,
    Fork,
    Intent,
    Interrupt,
    MoveSession,
    MoveWatch,
    Offer,
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
    mode_colour,
    mode_line,
    when,
)

# How long a first escape stays armed for a second one to complete the stop
# gesture. The same 1.0s the Textual app uses, and for the same reason a single
# escape must keep meaning nothing: ESC is the byte a terminal also sends as the
# prefix of every arrow key and of bracketed paste, so one arriving on its own
# is not evidence that the user wants the turn dead. Two in a second are.
ESC_STOP_WINDOW = 1.0

# The four rows, and the prompt that stands in for one of them. DECISION is a
# focus target without a pane: the inline approval sits at the foot of the chat
# column (§4.3 item 21) and takes keys while it is unanswered. It is in the
# ctrl+↑/ctrl+↓ ring — in the message box's slot, because while a decision is
# pending the box is not drawn and the prompt is what is there (`_ring`). It
# was outside the ring once, and that was a lockout: its own keys moved the
# focus away and nothing could move it back, so the turn stayed parked on a
# question that could no longer be answered.
SESSIONS, CHAT, INPUT, WATCHERS, DECISION, OFFER = range(6)

# The other thing that can stand in the message box's slot: a question the
# core raised about this conversation — today, triage offering to remember a
# failed job's error signature (`state.Offer`). It is in the ring for the same
# reason DECISION is, and it is a slot rather than a modal for a reason of its
# own: it arrives from a poll, so the user is as likely as not reading another
# session when it lands, and a question that takes the screen from work it has
# nothing to do with is a question asked in the wrong place.

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

# What the footer may take of the screen once its hints stop fitting on one
# row. They wrap rather than falling off the right-hand end (`ansi.footer_wrap`),
# and something has to stop a 20-column terminal from turning the message box's
# eleven hints into eleven rows of footer with the conversation squeezed out
# above them.
#
# A quarter of the screen, and never more than five rows. A quarter because
# that is already what the sessions and the watcher columns are each allowed,
# so the footer is not claiming more of the layout than a band of it does. Five
# because that is what the longest key set actually costs at the narrowest
# width anybody drives this at: the message box offers eleven pairs, ~136 cells
# of them, which is one row at 160 columns, two at 100, three at 60 and five at
# 30. Under the ceiling the hints that still do not fit are dropped from the
# end as they always were — a UI with no room left in it is worse than a hint
# `?` will still list in full.
FOOTER_ROWS = 5
FOOTER_SHARE = 4

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

# And the third, which reads like a running turn and is not one: a turn parked
# on an approval has been handed back to the user, and the core has no
# `TurnState` left to interrupt. Answer it — either way — and it can be
# stopped again (specs-ui-coverage.md §4).
PARKED_ON_A_DECISION = "this turn is waiting for your answer — decide it first"

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
# And what a session with an unanswered question from the core wants. Its
# own glyph rather than the "!": one is a turn parked mid-flight and the
# other is an offer about work that has already finished, and a user who
# crosses the screen for the second expecting the first has been lied to.
OFFER_MARK = "?"

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

# What the meter's severity paints it (`ui.meter.severity`) — the palette role
# by name, resolved at paint time, and whether that tier is also bold. Names
# rather than sequences for the reason `state.MODE_ROLES` gives: a table built
# at import outlives the palette it was built from.
METER_ROLES = {"warn": ("warn", False), "danger": ("danger", True)}


def meter_style(severity: str) -> str:
    """The style the context meter's bar is drawn in at `severity`."""
    role, bold = METER_ROLES.get(severity, ("faint", False))
    return (BOLD if bold else "") + getattr(theme, role)


class RowUI:
    MIN_CHAT = 4
    MAX_INPUT = 6

    def __init__(
        self,
        sessions: list[SessionState] | None = None,
        *,
        settings_json: str = "",
        display: Display | None = None,
        profiles: list[ProfileInfo] | None = None,
        catalog: list[BackendInfo] | None = None,
        send: Callable[[Intent], None] | None = None,
    ) -> None:
        # The sidebar's order, and the store behind it. Two things, because an
        # event can name a session the sidebar has not been told about yet (a
        # fork's `session.created` arrives before the `session.rows` that lists
        # it), and because a session must not lose its chat and its draft
        # merely by scrolling out of the list.
        # The settings this UI *draws* with, and the only ones it ever sees:
        # the file is out of a front-end's reach (§4.2 rule 2), so these come
        # down the wire on `hello` and again after a save (`client._display`).
        # A default here rather than a None check everywhere below, and it is
        # the same default `config.DisplaySettings` carries — so a UI built
        # with no core behind it (a test, the demo's first frame) draws what a
        # fresh install would.
        self.display = display or Display()
        self.sessions = list(sessions or [])
        self._states = {x.session_id: x for x in self.sessions}
        # What the rows draw against before the first `session.rows` arrives.
        # Real, so that nothing below here needs a None check.
        self._blank = SessionState("", display=self.display)
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
        self.note_style = theme.warn
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
        # The message a fork was made at, waiting for the session that fork
        # creates. A rollback can hand its message straight back — the
        # conversation it belongs to is the one on screen — but a fork's copy
        # does not exist until the core answers, so the text waits here and
        # `adopt` puts it in the box of the session that arrives.
        self._forked_draft = ""
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
        # When focus last *moved*, for the flash that says where it went
        # (`_flash`). None until it has moved at all: the pane the app opens
        # on was not arrived at, so lighting it would be an answer to a
        # question nobody asked.
        self._focus_lit_at: float | None = None
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
        # `c` on a chat row: text in, a sentence about where it went out
        # (`hpca.clipboard.ClipboardManager.copy(...).message`). A callable and
        # not the manager itself, because the manager writes OSC 52 straight to
        # the terminal and this module has never seen one.
        self.clipboard: Callable[[str], str] | None = None
        # The external editor, in two halves belonging to two different
        # layers. `suspend` takes the terminal down and puts it back
        # (`ui/run.py`, which owns it); the three hooks under it fetch a body,
        # run the editor over it and send what comes back (`ui/client.py`,
        # which is the side with a wire). All of them are asynchronous — the
        # text has to arrive before there is anything to edit — so they answer
        # in a toast rather than by returning anything.
        #
        # Three hooks and not one because the three things `$EDITOR` is opened
        # on are three different round trips: a draft is already here and goes
        # nowhere near the core (`edit_text`), a profile is `profile.get` /
        # `profile.save`, and the settings file is `settings.get` /
        # `settings.save`. They are wired together by one constructor
        # (`UIClient.__init__`), which is what `external_editor` reads.
        self.suspend: Callable[[Callable[[], None]], None] | None = None
        self.edit_text: Callable[[str, Callable[[str], None]], None] | None = None
        self.edit_profile: Callable[[str, str], None] | None = None
        self.edit_settings: Callable[[], None] | None = None
        # A window that arrived while a screen was open, waiting for the
        # screen to close (`window`). One slot: two of these queued at once
        # would be a stack of modals nobody asked for, and the newer answer is
        # the one still worth reading.
        self._waiting_window: tuple[str, str] | None = None
        # And the same slot for a *screen* the core answered for — today only
        # the skill creator's form, which opens seconds after the request when
        # the model has drafted something. Same reason, same rule: it is
        # placed when it can be, rather than replacing whatever the user
        # opened in the meantime (`land`, and §9.8 of the coverage audit).
        self._waiting_screen: Overlay | None = None

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

    def land(self, screen: Overlay) -> None:
        """Open a screen nobody just pressed a key for, without taking one away.

        The asynchronous counterpart of the `overlay` setter, and the reason
        it cannot be the setter: an answer that took the model several seconds
        arrives long after the request, and in those seconds the user is free
        to have opened the config editor and typed half a settings file into
        it. `overlay =` would start a fresh stack and drop that on the floor
        with no way back, which is exactly the hazard `window` parks for.

        So it waits, and lands at the first moment it is not landing on top of
        somebody. `boot._open_llm_screen` makes the same judgement by giving
        up entirely; this one keeps the answer, because a drafted skill is
        something the user asked for by name.
        """
        if self.overlay is not None:
            self._waiting_screen = screen
            return
        self.overlay = screen

    def _open_waiting_screen(self) -> None:
        """Place a parked screen, if there is one and the rows are clear."""
        if self._waiting_screen is None or self.overlay is not None:
            return
        screen, self._waiting_screen = self._waiting_screen, None
        self.overlay = screen

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
        # After the window, and only onto empty rows: a window that landed
        # here keeps the screen parked one more close, which is the right
        # order — the window is an answer to read, and this is a form to fill
        # in.
        self._open_waiting_screen()

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
        self._note, self.note_style = text, theme.warn

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
            session = self._states[session_id] = SessionState(
                session_id, display=self.display
            )
        return session

    def _move_focus(self, slots: list[int], step: int) -> None:
        """Move focus one pane along the ring, and light the pane it lands on.

        The flash is here rather than in `render` because this is the only
        place that knows focus *changed* as opposed to merely being somewhere:
        a frame drawn for any other reason must not relight the pane, or the
        wash would come back every time a token arrived.
        """
        self.focus = slots[(slots.index(self.focus) + step) % len(slots)]
        self._focus_lit_at = self.clock()

    @staticmethod
    def _wash(rows: list[str], tint: str) -> list[str]:
        """`rows` with `tint` behind them, or `rows` if there is no tint.

        Two things make this more than a prefix. A row is written as runs and
        every run opens with a RESET that states the whole style (`ui.rain`,
        `pane.Item.paint`), so a background set once at the left edge would be
        cleared by the first one and the wash would stop partway across; the
        tint is therefore restated after every reset in the row.

        And the cursor's row is skipped, because it is drawn REVERSE — which
        swaps foreground and background, so a tint under it would come out as
        the *text* colour and the one row the user is pointing at would be the
        one row drawn wrong. It keeps its highlight, which is a stronger mark
        than the wash anyway.
        """
        if not tint:
            return rows
        return [
            row if REVERSE in row else tint + row.replace(RESET, RESET + tint) + RESET
            for row in rows
        ]

    def _flash_wake(self) -> float | None:
        """Seconds until the flash stops being true, or None if it already is.

        None rather than zero for a spent flash, which is the rule the whole
        `next_wake` list is built on: zero is a repaint that books another
        repaint, and this is the one contributor that would do it forever,
        since nothing ever clears the timestamp.
        """
        if theme.flash_hold <= 0 or self._focus_lit_at is None:
            return None
        left = theme.flash_hold - (self.clock() - self._focus_lit_at)
        return left if left > 0 else None

    def _flash(self, slot: int) -> str:
        """The background the pane in `slot` is washed in, or "" for none.

        Held and then gone — no ramp. A decay needs frames to decay over and
        this is over in one or two of them, so what the eye is caught by is the
        movement, which a step gives it more cheaply than a gradient would.

        Zero hold is the setting turned off, and it short-circuits before the
        clock is read: `focus_flash_seconds` of 0 should cost nothing at all,
        not a subtraction whose answer is always false.
        """
        if theme.flash_hold <= 0 or slot != self.focus:
            return ""
        if self._focus_lit_at is None:
            return ""
        return theme.flash if self.clock() - self._focus_lit_at < theme.flash_hold else ""

    def set_profiles(self, profiles: list[ProfileInfo]) -> None:
        """Adopt the profile list that has just arrived, wherever it is shown.

        Assignment alone was not enough, and the gap was visible: creating a
        profile sends `profile.create` and asks for the list again, but the
        screen that asked is holding its *own* copy — taken when it opened, so
        the row for the new profile could not appear on it. Worse, closing that
        screen wrote the stale copy back over this one (`_closed`), so the
        answer the core had already sent was lost and the profile stayed
        invisible until the next start.

        So the open screen is told too. Its list is the one the keys act on and
        the one `_closed` hands back, which is why it is replaced rather than
        merely redrawn — and `refresh` keeps the cursor on the row it was on,
        so a list that grows under the user does not move what they were
        pointing at.
        """
        self.profiles = list(profiles)
        for screen in self.overlays:
            if isinstance(screen, ProfilesOverlay):
                screen.profiles = list(profiles)
                screen.refresh()

    def set_display(self, display: Display) -> None:
        """Adopt display settings that have just arrived, and redraw for them.

        Every session, not only the one on screen: the others are not being
        looked at *yet*, and a chat that restyled itself on the way back into
        view would be doing the work at the one moment the user is watching.
        `restyle` is cheap — it rebuilds `Item`s from entries already held —
        and it keeps what is open open, which `reset` would not.

        The palette is swapped *first*, and that ordering is the whole of what
        makes a saved theme land in one frame: `restyle` rebuilds rows, and a
        row rebuilt before the swap would be built in the colours that are on
        their way out. Everything drawn after this line is drawn in the new
        palette, including the rows this call is about to rebuild.

        A colour this cannot draw keeps the built-in one for that role and no
        more (`theme._sound`), which is why there is nothing to catch here.
        """
        self.display = display
        theme.apply(**dict(display.palette), flash_hold=display.focus_flash_seconds)
        for session in [self._blank, *self._states.values()]:
            session.restyle(display)

    def adopt(self, session: SessionState) -> None:
        """Put a session in the sidebar now, ahead of the core saying so.

        `session.created` names a conversation the user is about to be looking
        at; waiting for the next `session.rows` to list it would mean opening
        something the sidebar does not show.

        It is also where a fork's message lands: `_rewind` could not put it
        anywhere when the choice was made, because the session it belongs to
        is the one this event brings. Nothing else creates a session without
        clearing that text first, so an arrival with it still set is the fork
        that asked for it.
        """
        if self._forked_draft:
            self._into_draft(session, self._forked_draft)
            self._forked_draft = ""
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
            session.last_active = row.last_active
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
        # The offer is second to the decision and not beside it: one column,
        # and of the two the parked turn is the one that stops work.
        marks = (
            DECISION_MARK
            if parked
            else OFFER_MARK
            if session.offer is not None
            else " "
        )
        # `busy`, not `working`: a silent backend call — a compaction, the
        # titler, a `/conclude` — is something in flight in that conversation
        # and the row has to say so, even though there is no turn to stop
        # (`state.Turn.busy` is the same distinction the spinner draws).
        working = session.turn.busy or "working" in flags
        marks += WORKING_MARK if working else " "
        # Last, and only where the other two are not: a row cannot be both
        # still working and finished, and a decision is the more urgent of the
        # two things to say about a session that is not on screen. The same
        # `working` the mark above was drawn from, or the "*" would overwrite
        # a "⟳" the core's flag put there and nothing local knew about — a
        # reconnect, or a second front-end (specs-ui-coverage.md §9.14).
        if session.updated and not (parked or working):
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
                            # Mode before the clock, because the two are cut
                            # in that order on a narrow terminal and which of
                            # them is lost matters: "full-auto" is a safety
                            # fact and "last worked in on Tuesday" is not.
                            + " · ".join(
                                x
                                for x in (
                                    self._tag(session),
                                    session.mode,
                                    when(session.last_active),
                                )
                                if x
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
                        accent=theme.ok if i == self.active else "",
                        key=session.session_id,
                    )
                    for i, session in enumerate(self.sessions)
                ),
            ]
        )

    def _move_session(self, delta: int, width: int) -> bool:
        """Ask the core to shift the session under the cursor one place.

        `alt+↑/↓`, the same gesture the watchers column answers — asked for so
        the two reorderable rows behave alike, and now sent rather than
        performed. The order is a fact about the database, which rule 2 of
        §4.2 puts out of a front-end's reach; the answer is a whole new
        `session.rows` in the order the store now holds, and a sidebar that
        swapped its own rows first would be overwritten by it a frame later.
        That was the bug: the list moved, and the next frame put it back.

        Nothing is optimistic here for a second reason, which is what makes
        holding the key down work. `MoveSession` names a row by id and carries
        an offset, so the core swaps it with whichever row is its neighbour
        when the command *arrives*: two presses walk one session past two
        others whether or not the first answer has landed. A local swap would
        be aiming the second press at a list the core has not agreed to yet.

        What is decided here is only whether there is a move to ask for. The
        row above every real session is "+ new session", which is not a member
        of `self.sessions` at all — index 0 in the pane is index -1 here — so
        it refuses to move, the same way the top of the list refuses to move
        further up. Out of range at the far end is a silent no-op in the core
        too; answering it here as well is what keeps the footer's "moved" from
        being said about a row that did not.
        """
        index = self.session_pane.current(width) - 1
        target = index + delta
        if index < 0 or not 0 <= target < len(self.sessions):
            return False
        self.send(MoveSession(self.sessions[index].session_id, delta))
        return True

    def _move_watch(self, delta: int, width: int) -> bool:
        """The same, for the watch box under the cursor.

        The row carries the core's own ``PanelRow.ref`` (see
        `client._panel_item`), so what leaves here names a watch rather than a
        position in a column that a poll repaints twice a second — which is
        the other reason not to reorder locally: this column is rewritten by
        `panel.update` on a timer, and a swap made here would survive only
        until the next poll.
        """
        index = self.watchers.current(width)
        target = index + delta
        if index < 0 or not 0 <= target < len(self.watchers.items):
            return False
        self.send(MoveWatch(self.watchers.items[index].text, delta))
        return True

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
        self.note_style = theme.danger if severity == "error" else theme.warn
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
        room = max(2, self._avail(width, height) // 3)
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

    def _footer_cap(self, height: int) -> int:
        """The most rows the footer may have on a screen this tall
        (`FOOTER_ROWS`)."""
        return max(1, min(FOOTER_ROWS, height // FOOTER_SHARE))

    def _footer_note(self) -> tuple[str, str]:
        """What the footer says beside the keys, and in which colour.

        Read here rather than in `render` because the note is part of what
        decides how tall the footer is — it shares the row while there is one
        and takes the bottom line once the hints wrap (`ansi.footer_lines`) —
        so the measurement and the drawing ask the same question of it.
        """
        if self._esc_armed():
            return "esc again to stop", theme.danger
        return self.note, self.note_style

    def _footer_h(self, width: int, height: int) -> int:
        """How many rows the footer needs for the keys this row offers."""
        note, _ = self._footer_note()
        return len(
            footer_wrap(self._keys(), width, note, self._footer_cap(height))
        )

    def _avail(self, width: int, height: int, footer_h: int | None = None) -> int:
        """Rows the four bands share: the screen, less the header and however
        many rows the footer wants at this width.

        The footer used to be one row by definition and every band's share was
        measured from ``height - 2``. It can be several now, so the number is
        asked for in one place and every share is measured from that — a frame
        where the layout and the footer disagreed about the footer's height
        would be a frame with a clipped bottom. ``footer_h`` is for the caller
        that has already built the rows and can hand over the count instead of
        having it worked out again.
        """
        if footer_h is None:
            footer_h = self._footer_h(width, height)
        return max(8, height - 1 - footer_h)

    def _heights(
        self, height: int, width: int, footer_h: int | None = None
    ) -> list[int]:
        """How the rows split the screen.

        A quarter each for sessions and watchers and the rest to the chat — but
        only as much of a quarter as the pane actually has to show, so a short
        session list or an empty watcher row costs nothing instead of holding a
        quarter of the screen open. The message box takes what it needs up to
        six lines. Everything left over goes to the chat, which is the row that
        can use it.
        """
        avail = self._avail(width, height, footer_h)
        inp = self._entry_h(width, height) + self._status_h()
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
            # The prompt keeps its full share here and the box is cut to two
            # rows, which is the difference between them on a screen with no
            # room: a message can be typed a line at a time, and a question
            # nobody can read is a question nobody can answer.
            standing = (
                self.session.decision is not None or self.session.offer is not None
            )
            inp = self._status_h() + (
                self._entry_h(width, height) if standing else 2
            )
            middle = max(1, avail - 4 - inp)
        return [top, middle, inp, bottom]

    def _entry_h(self, width: int, height: int) -> int:
        """Rows the third band wants — the message box, or the prompt instead.

        Instead, and not as well. Both were on screen once, the prompt pushed
        in above the box, and it cost twice: the two of them shared the slot
        the ring pointed one cursor at, and the box sat there offering to take
        a message in a session whose turn is stopped dead until the question
        above it is answered. So the prompt takes the slot whole while it is
        up, and the box comes back — with the half-typed draft still in it,
        which lives on the `SessionState` and never depended on being drawn —
        the moment it is answered.

        The "/" menu goes with the box for the same reason: it is the box's
        own autocomplete, and a menu for a field that is not on screen is a
        list of commands nothing can run.
        """
        if self.session.decision is not None:
            return self._decision_h(width, height)
        if self.session.offer is not None:
            return self._offer_h(width, height)
        return self._input_h(width) + self._menu_h(width, height)

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
            decision, width, max(3, self._avail(width, height) // DECISION_SHARE)
        )

    def _offer_rows(self, offer: Offer, width: int) -> list[tuple[str, str]]:
        """The core's question, as ``(style, text)`` rows.

        Deliberately plainer than the decision above it. That one is a turn
        parked mid-flight and says so in red or yellow; this one is an offer
        about work that has already finished, and painting the two alike would
        say they are equally urgent — which would be a lie in the direction
        that costs most, since the decision is the one holding a turn.
        """
        rows = [("", rule("offer", width))]
        for line in fold(offer.question, max(8, width - 4)):
            rows.append((BOLD, f"  {line}"))
        rows.append((theme.faint, "  (y) yes · (n) no"))
        return rows

    def _offer_h(self, width: int, height: int) -> int:
        """Rows the offer wants, under the same cap the decision has."""
        offer = self.session.offer
        if offer is None:
            return 0
        cap = max(3, self._avail(width, height) // DECISION_SHARE)
        return max(3, min(cap, len(self._offer_rows(offer, width))))

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
        # Before anything is measured: a decision that arrived between two
        # keypresses changes which row the middle slot is, and a frame drawn
        # with the cursor on the row it used to be would draw nothing focused.
        self._settle_focus()
        out = [self._header(width)]
        if self.overlay is not None:
            footer = self._screen_footer(self.overlay, width, height)
            body = self._screen_h(self.overlay, width, height)
            out += self.overlay.render(width, body)
            out = self._frame(out, footer, width, height)
            # A confirmation can be asked *about* an overlay — remove this
            # skill, delete this profile — so it is drawn over that too, and a
            # toast lands on a screen as readily as on the rows.
            out = self._over_toasts(out, width, len(footer))
            return self._over_confirm(out, width)
        # Built once and its height handed to the layout, because how many rows
        # the hints need is a function of the width *and* of which keys this row
        # offers: two answers worked out separately are two answers that can
        # differ, and the difference would come off the bottom of the frame.
        footer = self._footer(width, height)
        heights = self._heights(height, width, len(footer))
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
            at = len(out)
            if slot == INPUT:
                status = self._render_status(width)
                prompt = self._render_decision(width, height) or self._render_offer(
                    width, height
                )
                if prompt:  # standing where the box would be (`_entry_h`)
                    out += prompt + status
                    out[at:] = self._wash(out[at:], self._flash(slot))
                    continue
                menu = self._render_menu(width, height)
                out += menu + status
                out += self._render_input(width, pane_h - len(status) - len(menu))
            else:
                out += pane.render(width, pane_h, focused=self.focus == slot)
            # The pane's whole rectangle, now that it is built: the flash is a
            # background under rows that were drawn without one, which is the
            # only way to add one to a UI that has never had a background at
            # all (`ui.theme.sgr`, `background=True`).
            out[at:] = self._wash(out[at:], self._flash(slot))
        out = self._frame(out, footer, width, height)
        out = self._over_toasts(out, width, len(footer))
        return self._over_confirm(out, width)

    def _footer(self, width: int, height: int) -> list[str]:
        """The key hints for this row, as the rows they will be drawn on."""
        note, style = self._footer_note()
        return footer_lines(
            self._keys(), width, note, style, self._footer_cap(height)
        )

    def _screen_footer(self, screen: Overlay, width: int, height: int) -> list[str]:
        """The same for an open screen, which has its own keys and no note."""
        return footer_lines(
            screen.footer(), width, max_rows=self._footer_cap(height)
        )

    def _screen_h(self, screen: Overlay, width: int, height: int) -> int:
        """Rows a screen's body gets, the header and its footer taken off.

        Asked for by the drawing and by the keys alike (`handle`): what a page
        key scrolls by has to be what was drawn, or page-down moves by a
        different amount than the screen showed.
        """
        return max(1, height - 1 - len(self._screen_footer(screen, width, height)))

    def _frame(
        self, out: list[str], footer: list[str], width: int, height: int
    ) -> list[str]:
        """The bands and the footer as exactly ``height`` rows.

        The frame's one hard invariant is settled in this one place: whatever
        the bands came to, the rows above the footer are padded out or cut
        down to what is left, and the footer goes on last. That order is the
        point — a band that asked for more rows than the screen has costs a
        clipped pane, never the key hints the footer was widened to show.
        """
        footer = footer[:height]
        rows = max(0, height - len(footer))
        return (out + [" " * width] * rows)[:rows] + footer

    def _over_toasts(self, out: list[str], width: int, footer_h: int) -> list[str]:
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
        rows = toasts.render(
            self.toasts, self.clock(), width, max(0, len(out) - 1 - footer_h)
        )
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
            # The answer line breathes, and the clock is what it breathes on
            # — the same arrangement the spinner has, and the reason
            # `next_wake` books a frame while this is up.
            now=self.clock(),
            period=self.display.decision_pulse_seconds,
        )

    def _over_confirm(self, out: list[str], width: int) -> list[str]:
        """The generic yes/no, and nothing else (§4.3 item 22).

        Instead of over the finished frame: a confirmation is the one thing
        here that is deliberately modal — a question with two answers and no
        third thing to be doing meanwhile — and three rows spliced into the
        middle of a full screen read as one more band of it rather than as a
        gate. So the frame it was asked over is built, measured, and then
        cleared: every row it came to is replaced by blanks and the question
        is the only thing left to read.

        Built and then cleared, rather than skipped, is what makes "no" cost
        nothing: the panes keep the heights and the scroll they had, and the
        frame after the answer is the frame that would have been drawn had the
        question never been asked.
        """
        question = self.confirm
        if question is None:
            return out
        rows = [
            theme.warn + rule("confirm", width) + RESET,
            BOLD + pad(f"  {question.question}", width) + RESET,
            theme.faint + pad("  (y) yes · (n) no · (esc) no", width) + RESET,
        ]
        # A terminal too short for all three keeps them in the order they are
        # worth: the question, then the way to answer it, then the rule, which
        # is decoration on a screen that has nothing left on it to divide.
        rows = [rows[i] for i in sorted([1, 2, 0][: len(out)])]
        at = max(0, (len(out) - len(rows)) // 2)
        # What the cleared screen is made of. Black, unless this is the
        # question you do not come back from (`ui.rain` says why that one is
        # different). The question is spliced over it whole either way — its
        # rows are padded to the width, so nothing of the field shows through
        # the three lines that matter.
        under = (
            rain(width, len(out), self.clock())
            if self._raining()
            else [" " * width] * len(out)
        )
        under[at : at + len(rows)] = rows
        return under

    def _render_offer(self, width: int, height: int) -> list[str]:
        """The offer, in exactly ``_offer_h`` rows — or none, if none is up.

        The rule says focus the way every other region's does, and the last
        row is the one kept when there is no room: a question with no visible
        way to answer it is worse than a question with no visible middle.
        """
        offer = self.session.offer
        if offer is None:
            return []
        focused = self.focus == OFFER
        rows = self._offer_rows(offer, width)
        rows[0] = (BOLD + theme.chrome if focused else theme.faint, rows[0][1])
        out = [f"{style}{pad(text, width)}{RESET}" for style, text in rows]
        room = self._offer_h(width, height)
        if len(out) > room:
            out = out[: max(0, room - 1)] + out[-1:]
        while len(out) < room:
            out.append(" " * width)
        return out[:room]

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
        meter = meter_style(self.session.context.severity)
        return [
            mode_colour(self.session.mode)
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
        out = [(BOLD + theme.chrome if focused else theme.faint) + title + RESET]
        rows = max(1, height - 1)
        body = self.input.render(self._input_body(width), rows, focused=focused)
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((theme.chrome if focused else theme.faint) + marker + RESET + line)
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
        common = [("^↑^↓", "panel"), ("esc esc", "stop"), ("?", "keys")]
        # Nothing for `self.confirm`: the dialog blanks the footer along with
        # the rest of the frame and carries its own two answers on its own
        # row. Keys that stay the keys of the row underneath are what lets the
        # layout — and so the frame a "no" comes back to — stay put.
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
        if self.focus == OFFER:
            return [("y", "yes"), ("n", "no"), ("^↑^↓", "panel"), ("?", "keys")]
        if self.focus == DECISION:
            decision = self.session.decision
            if decision is not None and not decision.asking:
                return [
                    ("enter", "send the reason"),
                    ("esc", "no reason"),
                    ("⇧enter", "new line"),
                    # Offered here too, because they work here too: the box
                    # keeps what is in it while the cursor is away, and the
                    # ring always comes back to this row (`_ring`).
                    ("^↑^↓", "panel"),
                ]
            return [
                ("y", "approve"),
                ("n", "deny"),
                ("esc", "deny, no reason"),
                ("^↑^↓", "panel"),
            ]
        if self.focus == INPUT:
            # Spelled out rather than built from ``common`` so that send and
            # stop come first — the message box is where you sit while a turn
            # runs, and those are the two keys that matter there.
            return [
                ("enter", "send"),
                ("esc esc", "stop"),
                ("^↑^↓", "panel"),
                ("⇧enter", "new line"),
                ("⇧tab", "mode"),
                # No word-motion, selection or cut-word pair here: they are
                # the editing keys somebody either already has in their
                # fingers or looks up once, and a footer that lists them
                # spends its width — the row wraps, and the pairs that wrap
                # off the end are the ones at the back — on keys nobody scans
                # the footer for. "? keys" still names all three.
                ("^l", "switch llm"),
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
            rows += [("enter", "rollback/fork"), ("c", "copy")]
        else:
            rows += [("enter", "peek"), ("d", "unwatch"), ("alt-↑↓", "move")]
        # Only where they do something. `m` and `a` are the sessions row's and
        # ctrl+l is the chat's; `c` is live everywhere, but it is the config
        # editor outside the chat and the row copy inside it (`_copy_row`), so
        # it is offered once per row with the label that is true there — a key
        # list that lies is worse than a short one.
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

    def _ring(self) -> list[int]:
        """The rows ctrl+↑ and ctrl+↓ walk, in the order they are drawn.

        The third one is the message box, or the decision prompt standing in
        its place while one is pending: the prompt is *in* that slot rather
        than beside it (`_entry_h`), so the ring says what the layout says and
        the two cannot disagree about how many rows there are.

        Which is also the fix for a hard lockout. DECISION used to be outside
        the ring while its own keys still moved the focus out of it, so one
        ctrl+↑ off an unanswered prompt left a parked turn that no key
        sequence could reach again — re-opening the session was the only way
        back, and nothing on screen said so.
        """
        if self.session.decision is not None:
            middle = DECISION
        elif self.session.offer is not None:
            middle = OFFER
        else:
            middle = INPUT
        return [SESSIONS, CHAT, middle, WATCHERS]

    def _settle_focus(self) -> None:
        """Put the cursor on a row that exists, before anything reads it.

        Two states are not allowed, and they are the two the ring cannot name:
        the prompt with no decision left behind it, and the message box while
        a decision is standing in front of it. Both are one line away from
        every path that sets INPUT — a paste, a rollback handing a message
        back, ctrl+↓ out of the chat, a decision arriving while the box has the
        cursor — and the one that forgot would leave the cursor on a row that
        is not drawn, or walk a key into `_handle_row`'s pane lookup with a
        focus it has no entry for. Asked once, here, rather than remembered
        eleven times.
        """
        middle = self._ring()[2]
        if self.focus in (INPUT, DECISION, OFFER) and self.focus != middle:
            self.focus = middle

    def handle(self, key: str, width: int, height: int) -> bool:
        self._settle_focus()
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
            alive = overlay.handle(key, width, self._screen_h(overlay, width, height))
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
        if self.focus == OFFER:
            return self._handle_offer(key)
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
        decision = self.session.decision
        if decision is not None:
            # The box this would land in is not on screen — the prompt is in
            # its slot — so the focus cannot follow the text. At the reason
            # stage there *is* a visible editor and it takes it; at the
            # question there is not, and the text goes into the draft it was
            # aimed at and waits there with it. Dropping a payload because a
            # gate happened to be open is the worse of the two answers.
            if decision.asking:
                self.input.insert_text(text)
            else:
                decision.reason.insert_text(text)
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
            # A fork whose copy never arrived must not hand its message to
            # this one — a session made from the picker is not that fork.
            self._forked_draft = ""
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
        rain: bool = False,
    ) -> None:
        """Put a yes/no over everything. The answer runs ``on_answer``.

        The UI's own questions only — really quit, interrupt this turn, delete
        this session — which is why taking the whole screen is fair: each of
        them answers a key that was just pressed, so there is nothing else the
        user was in the middle of. What the *core* asks does not come through
        here; it waits in the session it is about (`confirm_requested`).
        """
        self.confirm = Confirm(question=question, on_answer=on_answer, rain=rain)

    def _raining(self) -> bool:
        """Whether the cleared screen behind the open question is falling.

        Asked in the two places that must agree — what is drawn, and whether a
        frame is booked to draw it again — because a field painted once and
        never repainted hangs mid-drop, and a repaint booked for a screen that
        is black is a wakeup with nothing to do.
        """
        return (
            self.confirm is not None
            and self.confirm.rain
            and self.display.quit_rain
        )

    def _rain_interval(self) -> float:
        """Seconds between two frames of the field, from the settings.

        Clamped here rather than trusted: the value crossed a wire, this is a
        repaint loop, and a zero would divide by zero somewhere with the
        terminal in raw mode. The bounds are the ones `config` documents.
        """
        return 1.0 / min(120, max(1, self.display.quit_rain_fps))

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
        if callable(question.on_answer):
            question.on_answer(confirmed)

    def confirm_requested(self, session_id: str) -> None:
        """`confirm.requested`, which the Textual UI never drew at all.

        The question itself is already on its session (`client._confirm`);
        what is left is the two things the *frame* has to say about it. A
        session that is not on screen says it with the "?" in the sidebar and
        nothing else (§3.2 property 1) — this arrives from a poll, and a poll
        may not reach across and take the row somebody is working in.

        For the open session the cursor moves onto it, from the chat or the
        box only, which is the same bargain `decision_arrived` makes and for
        the same reason: the offer stands in the box's slot (`_entry_h`), so
        a cursor left in the box would be a cursor on a row that is no longer
        drawn. The draft is untouched and comes back with the box.
        """
        self.refresh_sidebar()
        if session_id == self.active_id and self.focus in (CHAT, INPUT):
            self.focus = OFFER

    # ------------------------------------------------------ the decision

    def decision_arrived(self, session_id: str) -> None:
        """A session is parked on an approval — this one, or another one.

        Another one changes exactly one thing about the frame: the "!" in the
        sidebar (§3.2 property 1). This one puts the prompt up and lands on it
        so its keys work at once — but only from the chat column, because a
        decision must never pull the cursor out of the sessions or watchers
        row the user is working in.

        From the message box it is not a courtesy but the only answer: the
        prompt takes the box's slot (`_entry_h`), so leaving the cursor there
        would leave it on a row that is no longer drawn. What was being typed
        is not lost — it is the session's draft, and the box comes back with
        it once this is answered.
        """
        self.refresh_sidebar()
        if session_id == self.active_id and self.focus in (CHAT, INPUT):
            self.focus = DECISION

    def decision_cleared(self, session_id: str) -> None:
        """The core says that decision is gone (answered, or its turn died).

        The cursor goes back to the box the prompt was standing in front of,
        which is where it came from and where the draft has been waiting.
        Only from the prompt: a user who had walked off to the sessions column
        meanwhile is left where they are.
        """
        self.refresh_sidebar()
        if session_id == self.active_id and self.focus == DECISION:
            self.focus = INPUT

    def _handle_decision(self, key: str) -> bool:
        """The two stages, and the keys each of them owns.

        The reason box takes the letters the y/n stage was using — it is a
        text field, and "n" in the middle of "not this path" is not a verdict
        — which is why the stage gates the keys rather than both being live at
        once (`DecisionBar.check_action` did the same with `check_action`).

        ctrl+↑ and ctrl+↓ leave the way they leave any row, because this is a
        row of the ring now (`_ring`): up to the chat the question is about,
        down to the watchers. What they do *not* do any more is drop into the
        message box — it is not on screen while this is, and tab used to aim
        at it. Nothing here can strand the prompt: every step of the ring
        comes back to it.
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
            elif key in ("ctrl-up", "shift-tab"):
                self.focus = CHAT
            elif key in ("ctrl-down", "tab"):
                self.focus = WATCHERS
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
            self.focus = WATCHERS
        else:
            decision.reason.handle(key)
        return True

    def _handle_offer(self, key: str) -> bool:
        """Two answers and the ring, which is all this row has.

        No third key, and in particular no "later": the core is holding a
        continuation under this id and a question that can be walked past
        without answering is a continuation nothing ever frees. Walking off
        with ctrl+↑/↓ is not walking past it — the offer is still here, the
        sidebar still says so, and the ring comes back.
        """
        if key == "quit":
            return False
        offer = self.session.offer
        if offer is None:  # answered elsewhere, or its session went away
            self.focus = INPUT
            return True
        if key == "y":
            self._answer_offer(offer, True)
        elif key in ("n", "esc"):
            self._answer_offer(offer, False)
        elif key in ("ctrl-up", "shift-tab"):
            self.focus = CHAT
        elif key in ("ctrl-down", "tab"):
            self.focus = WATCHERS
        return True

    def _answer_offer(self, offer: Offer, accepted: bool) -> None:
        """Answer it and give the slot back — to the box, or to the next one.

        Taken off here rather than when the core agrees, the same as a
        decision: a question that stays up until an answer comes back is a
        question that can be answered twice, and this one's continuation is
        freed by the first answer.
        """
        session = self.session
        session.drop_offer(offer.id)
        self.refresh_sidebar()
        self.focus = self._ring()[2]
        self.send(Answer(offer.id, accepted))
        self.note = "accepted" if accepted else "declined"

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
            self.note = self._why_not_stoppable()
            self.note_style = theme.faint

    def _why_not_stoppable(self) -> str:
        """Why the stop gesture did nothing, in the three shapes that has.

        One answer for both halves of the gesture — `esc esc` and Enter on the
        working row ask the same question — and all three are said rather than
        papered over: "waiting for you" and "nothing running" are different
        situations with different next steps, and the one thing none of them
        may say is that a turn was stopped.
        """
        turn = self.session.turn
        if turn.parked:
            return PARKED_ON_A_DECISION
        return NOT_A_TURN if turn.busy else NOTHING_TO_STOP

    def _confirmed_stop(self, session_id: str, yes: bool) -> None:
        """The answer to the interrupt dialog. A reply that landed while it
        was open makes it a no-op — there is no longer a turn to stop."""
        if not yes:
            return
        if not self.session_for(session_id).turn.interruptible:
            self.note = "that turn finished while you were deciding"
            self.note_style = theme.faint
            return
        self.send(Interrupt(session_id))
        self.note = "stopped the turn"

    def _rewind(self, overlay: RewindOverlay) -> None:
        """What the rewind decided, as an intent aimed at the session it was
        opened in — which is not necessarily the one on screen by the time it
        closes.

        Either cut ends the conversation just before the message it was made
        at, and the reason to make one is almost always to say that message
        differently — so the message comes back to the box, the way a
        cancelled queued one does (`hand_back`). Added to the draft rather
        than replacing it, so nothing typed in between is lost.
        """
        # Both halves put the message where it goes *before* the command
        # leaves, and both need to: the fork's copy can be created and adopted
        # inside `send` (the demo's peer answers synchronously), and whatever
        # the core says about the cut is the more specific news, so it must be
        # the note left standing when it arrives.
        if overlay.choice == FORK:
            # Not into this session's box: the fork's own is the one to type
            # in, and it arrives with `session.created` (see `adopt`).
            self._forked_draft = overlay.message
            self.send(Fork(overlay.session_id, overlay.seq))
        elif overlay.choice == ROLLBACK:
            # Handed back as the cut is sent rather than when the trimmed
            # transcript lands: a rollback has no reply of its own
            # (`protocol.SessionRollback`), and a core that refuses it answers
            # with a warning the user reads with the message still in the box
            # — which is the recoverable half of being wrong here, since the
            # draft is added to and never overwritten.
            self.hand_back(overlay.session_id, overlay.message)
            self.send(Rollback(overlay.session_id, overlay.seq))
        # Either cut leaves you at the point the conversation now ends, which
        # is a place to say the next thing from.
        self.focus = INPUT

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

    def activity_label(self) -> str:
        """What to blame a loop stall on, for the lag probe's spike list.

        The same answer the spinner gives — "running read_file", "LLM
        processing" — because that is already the app's own word for what it
        is doing, so a spike in `looplag.log` names the step that caused it
        rather than a time nobody can place (specs-core-process.md §8).

        Every busy session, not just the one on screen: a turn in a
        conversation the user has left blocks this loop exactly as hard as the
        open one. Sorted and de-duplicated so that two sessions running the
        same tool read as one step rather than two problems.

        Called from inside a stall, on state that may be half-built, and its
        answer is only ever a string in a log — so it is deliberately total:
        no session, no turn, no blame.
        """
        live = sorted({x.turn.label for x in self.sessions if x.turn.busy})
        return ", ".join(live) if live else "idle"

    def next_wake(self) -> float | None:
        """Seconds until this frame goes stale on its own, or None.

        The loop repaints because something happened — a key, an event, a
        resize — and not on a timer, so anything that changes by the clock
        alone has to say when it will. Four things do, and the answer is the
        sooner of them: the armed escape, which is only true for
        ``ESC_STOP_WINDOW`` and has no keypress coming to wipe it; the
        spinner, which turns; the toast, which expires; and the answer line of
        an open decision, which breathes.

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
            # And the fourth: the prompt's answer line is a function of the
            # clock (`ansi.pulse`), so without a frame booked here it would be
            # painted once in whatever colour the keypress that drew it landed
            # on and sit there. A fixed interval rather than "when the ramp
            # next steps", because the ramp has seven colours and working out
            # which second of the sweep is the slow one costs more than the
            # frame it would save. Only while a decision is on *this* screen:
            # a background session's prompt is not drawn, so nothing about it
            # changes with the clock.
            PULSE_INTERVAL if self.session.decision is not None else None,
            # And the fifth: the field behind an open confirmation falls by
            # the clock and by nothing else, so without a frame booked here it
            # would be painted once and hang there mid-drop.
            self._rain_interval() if self._raining() else None,
            # And the sixth: the focus flash is held for a fixed time and then
            # is not, so the frame that clears it has to be booked or the wash
            # stays until the next keypress happens to redraw. One wake and
            # then nothing — unlike the four above, this one is spent: once
            # the hold is over it returns None for good, which is the whole of
            # why a flash costs two repaints and an animation costs ten a
            # second.
            self._flash_wake(),
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
            # that the interrupt reached the core. Which of the three answers
            # it is matters: a spinner that cannot be stopped is a different
            # thing from no spinner, and `_stop_from_the_row` gives the same
            # three answers to the same question.
            self.note = self._why_not_stoppable()
            self.note_style = theme.faint
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
            # The message box is the one place the mode can be changed from,
            # and the right one: deciding the agent may act unasked is a
            # thought you have *while writing the message*, not one you leave
            # the box to act on. The Textual app bound it `priority=True` for
            # exactly that reason. Everywhere else shift+tab is the way back
            # up the ring, as tab is the way down; ctrl+↑ leaves the box.
            self._cycle_mode()
        elif key == "tab":
            # The ring does not stop here. Tab walked the focus down every
            # other row and then went quiet at the box, where it fell through
            # to the draft as whitespace — so the one row it is hardest to
            # leave by accident was also the only one it could not be walked
            # out of, and the way on was a key (ctrl+↓) nothing on screen
            # names. Shift+tab still belongs to the mode here, which is why
            # this is not symmetric: the way back up is ctrl+↑.
            self.focus = WATCHERS
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
            self._edit_draft()
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
        screen = name in self.SCREEN_COMMANDS
        if screen and not self._asks_the_core(name, args):
            # Ahead of the busy check on purpose, and only while it is true
            # that these draw a screen and ask the core for nothing: with no
            # worker to collide with there is nothing for a running turn to
            # refuse. Picking a thinking level while the model is thinking is
            # the case that makes the point.
            if self._screen_command(name, args):
                self.input.clear()
                return
        if self.session.turn.busy:
            # A command that *does* reach the core acts on the UI and runs its
            # own exclusive worker, so there is nothing sensible to queue it
            # behind — and a `/compact` that ran an hour later against a thread
            # the turn had since changed would be worse than one that was
            # refused. An ordinary message queues; this one waits for the user.
            self.note = "wait for this turn — a command cannot be queued"
            self.note_style = theme.faint
            return
        if screen and self._screen_command(name, args):
            # The screen commands that *do* reach the core — a
            # `/skill-creator <request>`, which is a model call — answered
            # here, behind the guard, where the rest of the core-bound
            # commands are.
            self.input.clear()
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

    def _asks_the_core(self, name: str, args: str) -> bool:
        """Whether this screen command is a round trip after all (§9.7).

        One of the three is: `/skill-creator <request>` drafts the skill with
        the model, so it sends `skill.draft` and the core answers it by
        putting *this session* to work. Run mid-turn that rewrites the running
        turn's step label and restarts its elapsed clock — twice, once for the
        draft and once for the blank that ends it — while the acceptance list
        is explicit that the elapsed count answers "how long since I sent it".
        So it waits behind the same guard every other core-bound command does.
        """
        return name == "skill-creator" and bool(args)

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
                    f"no skills to remove — “{self.profile}” has none of its "
                    "own, and nor has this project",
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

        `land`, not `overlay =`: this is the one screen in the app that opens
        because the *core* answered, seconds after the key that asked for it,
        and by then there may be a screen of somebody's own in front of it
        (§9.8).
        """
        if error:
            self.toast(f"could not draft that skill ({error})", "warning")
        self.land(self._skill_form(drafted, request))

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
                self.ask(QUIT_QUESTION, self._quit_answer, rain=True)
            return True
        inner = max(8, width - 2)
        slots = self._ring()
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
            # Told whether `$EDITOR` can be reached, because that is what
            # decides where the profile under the cursor opens: the user's own
            # editor, or the in-app one that stands in when there is no
            # terminal to hand over (`external_editor`).
            self.overlay = ProfilesOverlay(
                self.profiles, external=self.external_editor
            )
        elif key == "c" and self.focus == CHAT:
            # The chat column's `c` copies the row under the cursor (§4.3 item
            # 36). It is the one row where `c` is about a conversation rather
            # than about the app, which is why the config editor gives it up
            # here and keeps every other row.
            self._copy_row(inner)
        elif key == "c":
            # Anywhere but the chat column (§5), where `c` is the row copy.
            self._edit_config()
        elif key == "ctrl-l" and self.focus == CHAT:
            self._switch_llm()
        elif key in ("ctrl-down", "tab"):
            self._move_focus(slots, +1)
        elif key in ("ctrl-up", "shift-tab"):
            self._move_focus(slots, -1)
        elif key == "ctrl-e":
            # The draft, from here too — and the focus goes with it, the way a
            # paste does (`_edit_draft`, `_paste`).
            self._edit_draft()
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
            moved = self._move_watch(-1 if key == "alt-up" else 1, inner)
            self.note = "moved" if moved else ""
        elif key in ("alt-up", "alt-down") and self.focus == SESSIONS:
            # Said before the core has answered, like the mode bar: the
            # keypress needs feedback, the frame that arrives is what makes it
            # true, and the one case the core would refuse — no neighbour to
            # trade with — has already been ruled out above.
            moved = self._move_session(-1 if key == "alt-up" else 1, inner)
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
        """`c`: the chat row under the cursor, to the clipboard (§4.3 item 36).

        Through `hpca.clipboard.ClipboardManager`, which already knows about
        OSC 52, the multiplexer wrapping and the file fallback, and which
        already takes an injected ``emit`` — so nothing about copying is
        rewritten here and this method only decides *what* is copied. The
        entry's own text, not the lines it is drawn as: the label line and the
        tool markers are decoration, and pasting them back into a shell is a
        paper cut every time.

        Still worth its key now that an opened message's lines are drawn flush
        and a mouse selection of them is already clean — because this one
        copies the *row*, whole, from a closed one, without opening it and
        without dragging across a reply that runs off the bottom of the pane.
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

    @property
    def external_editor(self) -> bool:
        """Whether `$EDITOR` can be reached from here at all.

        Read by the keys that now have two ways to do the same job — the
        config editor, and a profile's files off the profiles screen — so
        that a UI with no terminal to hand over (a test, the demo, anything
        driving `RowUI` headless) keeps the in-app form rather than being
        told there is no editor and left with no way in.

        One question for three hooks and a terminal, because they are wired
        by one constructor (`UIClient.__init__`) against the one terminal
        `run.py` owns: a UI holding some of them and not the others would be
        a UI somebody had taken apart by hand.
        """
        return self.suspend is not None and self.edit_text is not None

    def _edit_draft(self) -> None:
        """ctrl+e: the message being written, in `$EDITOR` (§4.3 item 37).

        Textual spelled this `self.suspend()`; here it means leaving the
        alternate screen, putting the line discipline back, running the editor
        on a terminal that behaves like a terminal, and coming back to a full
        repaint. Both halves are injected — `run.py` owns the terminal and
        `client.py` owns the process — so this method is the key binding, one
        guard, and where the text lands when it comes back.

        The draft and not the profile, which is what this key used to open:
        the footer offers it in the message box and nowhere else, and a key
        advertised next to "send" and "new line" that opened a *memories* file
        was the one hint in the footer that named the wrong thing entirely.
        A profile is edited where a profile is chosen (`a`, then the row), and
        the settings file where the settings are (`c`).

        It works from the rows too, and takes the focus back to the box with
        it, for the reason `_paste` does the same: editing the message is an
        unambiguous "I am writing", the rows have no draft of their own, and
        coming back from the editor to a cursor parked on a chat row would
        hide the very text that was just edited.

        The editor is captured rather than looked up again when the text comes
        back, so what was edited is what is written to — the answer arrives
        from `client.py` and is not obliged to arrive before the next key.
        """
        if self.suspend is None or self.edit_text is None:
            self.toast("no editor is wired up here", "warning")
            return
        draft = self.input
        self.edit_text(draft.text(), lambda text: self._draft_edited(draft, text))

    def _draft_edited(self, draft: Editor, text: str) -> None:
        """What `$EDITOR` left in the file, as the draft it was opened on.

        Said out loud only when it comes back empty: everything else about
        this is visible — the box is the feedback, and it now holds what the
        editor holds. An empty buffer is applied rather than refused, because
        deleting the message and saving is a thing a person does on purpose
        and the alternative is a key that silently ignores it; but the draft
        it replaced is gone, so the one case that can lose work says so.
        """
        draft.set_text(text)
        self.focus = INPUT
        if not text.strip():
            self.toast("the editor left the message empty", "warning")

    def _edit_config(self) -> None:
        """`c`: the settings file, in `$EDITOR` — or in the screen, failing that.

        The file is the interface either way (`overlays/config.py` says why at
        length: the settings model grows a field whenever anything does, and a
        hand-built form is the copy of it that falls behind). All that changes
        here is which editor holds the text, and `$EDITOR` is the better one
        for a file: it has the user's keys, their search, their JSON mode.

        The in-app screen is kept as the way in when there is no terminal to
        hand over, and as the place a refusal lands: text the settings model
        would not accept is opened in it with the reason on its rule
        (`fix_settings`), rather than dropped after a minute of typing. Nothing malformed reaches disk in either case: the core
        validates `settings.save` again and refuses it (`_save_settings`),
        which is what keeps a hand-edited file from being one the next start
        cannot read.
        """
        if self.external_editor and self.edit_settings is not None:
            self.edit_settings()
            return
        # Fetched as it opens, like every other editable body: the file is
        # also written by the core — `settings.save` normalises what lands on
        # disk — so a copy kept from the last time this screen was open is a
        # copy that can already be wrong.
        self.overlay = ConfigOverlay(
            self.settings_json,
            validate=self.validate_settings,
            awaiting=SETTINGS_KEY,
        )

    def fix_settings(self, text: str, reason: str) -> None:
        """Put rejected settings back on screen, in the editor that refuses to
        lose them.

        Called by `client.py` when a file that came back out of `$EDITOR`
        cannot be saved — either it is not JSON, or the core would not take it
        — and it is the whole of the answer to "what happens when they save
        something malformed": the text is not on disk, not on the wire, and
        not thrown away either.

        ``was`` is set to the file that is still there rather than to the text
        the screen opens with, and that is the load-bearing line. It is what
        `EditorOverlay` measures "changed" against, so the rejected text reads
        as an unsaved edit: escape asks to keep it, and `ConfigOverlay.refuse`
        will not let the screen close at all while it is still not JSON.
        Opened over itself, one escape would drop the minute of typing this
        exists to preserve.
        """
        screen = ConfigOverlay(text, validate=self.validate_settings)
        screen.was = self.settings_json
        screen.note = reason
        self.overlay = screen
        self.toast(reason, "error")

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
        started under a profile the list has not arrived for.

        The default leads, then the rest in the order they were met. Note what
        that does *not* say: the working profile is second when it is not
        `default`, and the cursor still starts on `default` — so a user
        working under `hpc` picks `hpc` with one press of ↓ and gets the
        fallback by pressing Enter twice. Putting the working profile first
        instead is `specs-ui-coverage.md` §7's suggestion and is not what this
        does today.
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
