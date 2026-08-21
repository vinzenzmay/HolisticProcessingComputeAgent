"""The rows themselves: layout, focus and key dispatch.

``RowUI`` takes keys and returns frames and does no I/O of its own, which is
what makes the whole UI testable by calling ``render()`` and comparing strings.
"""

from __future__ import annotations

import time
from collections import namedtuple
from collections.abc import Callable

from hpca import __version__ as VERSION
from hpca.ui.ansi import BOLD, CYAN, DIM, GREEN, RED, RESET, REVERSE, YELLOW
from hpca.ui.ansi import cell_width, cut, footer_line, pad, rule
from hpca.ui.approval import decision_height, render_decision
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS, is_paste, paste_text
from hpca.ui.overlays import (
    COPY,
    FORK,
    ROLLBACK,
    UNQUEUE,
    ConfigOverlay,
    HelpOverlay,
    LlmOverlay,
    Overlay,
    ProfilesOverlay,
    QueuedOverlay,
    RewindOverlay,
)
from hpca.ui.pane import Item, Pane
from hpca.ui.state import (
    MODE_COLOURS,
    OWN_MESSAGE_KINDS,
    Answer,
    Confirm,
    CycleMode,
    Decide,
    Drop,
    Fork,
    Intent,
    Interrupt,
    OpenSession,
    Peek,
    Rollback,
    SessionState,
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


# The sidebar markers §4.3 item 15 asks for, and the flag strings the core
# uses for the same states. Both are consulted: the flag is what the core
# knows, and the local turn/decision state is what has arrived since the last
# `session.rows` — a marker that waited for the next sidebar repaint would lag
# a whole poll behind the event that caused it.
DECISION_MARK, WORKING_MARK = "!", "⟳"

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
        learnings: dict[str, str] | None = None,
        settings_json: str = "",
        llms: tuple[list[Item], list[Item]] | None = None,
        profiles: list[Item] | None = None,
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
        self.overlay: Overlay | None = None
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
        self._learnings = learnings or {}
        self._settings_json = settings_json
        self._llms = llms or ([], [])
        self._profiles = profiles or []

    # ------------------------------------------------------ the open session

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
            session.flags = list(row.flags)
            self.sessions.append(session)
        keep = {x.session_id for x in self.sessions} | {was}
        self._states = {k: v for k, v in self._states.items() if k in keep}
        self._select(was)
        self.refresh_sidebar()

    def _select(self, session_id: str) -> None:
        """Point `active` at that session, or at the first one if it is gone."""
        self.active = next(
            (i for i, x in enumerate(self.sessions) if x.session_id == session_id),
            0,
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
        marks += WORKING_MARK if session.turn.working or "working" in flags else " "
        return marks

    def refresh_sidebar(self) -> None:
        """Redraw the session list: which one is open, and what each is doing.

        Rebuilt rather than patched because it is fourteen rows, not fourteen
        hundred — the cost that matters is the chat's, and that one is never
        rebuilt at all. `Pane.replace` keeps the cursor on the row it was on,
        by id.
        """
        self.session_pane.replace(
            [
                Item(
                    head=(
                        f"{'●' if i == self.active else '○'} "
                        f"{self._marks(session)} {session.title[:40]:<42}"
                        + " · ".join(
                            x for x in (session.profile, session.mode) if x
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
            ]
        )

    def open_session(self, session_id: str, *, announce: bool = True) -> None:
        """Look at this conversation, and tell the core that we are.

        ``announce`` is off for the one the client opens by itself when the
        first sidebar arrives: the footer note answers a keypress, and there
        was none.
        """
        self._select(session_id)
        self.refresh_sidebar()
        if self.session.decision is not None and self.focus != WATCHERS:
            # It was flagged with a "!" while it was in the background; being
            # opened is what reveals it, and the cursor lands where it can be
            # answered.
            self.focus = DECISION
        if announce:
            self.note = f"opened “{self.session.title}”"
        self.send(OpenSession(session_id))

    def _switch(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        if index == self.active:
            self.note = "already open"
            return
        self.open_session(self.sessions[index].session_id)

    def toast(
        self, text: str, severity: str = "information", timeout: float | None = None
    ) -> None:
        """Something the core said. The footer says it; M8 makes it a toast."""
        self.toasts.append(Toast(text, severity, timeout))
        self._note = text
        self.note_style = RED if severity == "error" else YELLOW

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
            inp = 2 + self._status_h() + self._decision_h(width, height)
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
            # skill, delete this profile — so it is drawn over that too.
            return self._over_confirm(out[:height], width)
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
                out += prompt + status
                out += self._render_input(
                    width, pane_h - len(status) - len(prompt)
                )
            else:
                out += pane.render(width, pane_h, focused=self.focus == slot)
        note, style = self.note, self.note_style
        if self._esc_armed():
            note, style = "esc again to stop", RED
        out.append(footer_line(self._keys(), width, note, style))
        while len(out) < height:
            out.insert(len(out) - 1, " " * width)
        out = out[:height]
        return self._over_confirm(out, width)

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
        common = [("^↑^↓", "row"), ("esc esc", "stop"), ("?", "keys"), ("q", "quit")]
        if self.confirm is not None:
            return [("y", "yes"), ("n", "no"), ("esc", "no")]
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
                ("⇧←→", "select"),
                ("^⌫ ^del", "cut word"),
                ("^u", "clear"),
                ("?", "keys"),
                ("q", "quit"),
            ]
        rows = [("↑↓", "line"), ("→←", "open"), ("⇧→←", "open all")]
        if self.focus == SESSIONS:
            rows += [("enter", "switch"), ("r", "rename"), ("t", "retitle"), ("d", "delete")]
        elif self.focus == CHAT:
            rows += [("i", "write"), ("enter", "reuse"), ("⇧tab", "mode")]
        else:
            rows += [("enter", "peek"), ("d", "unwatch"), ("alt-↑↓", "move")]
        return rows + [("m", "llms"), ("a", "profiles"), ("c", "config")] + common

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
        if self.overlay is not None:
            overlay = self.overlay
            if not overlay.handle(key, width, height - 2):
                self.overlay = None
                self._closed(overlay)
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
        return True

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
            self.note = "this is a backend call, not a turn — nothing to stop"
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
            self.reuse_message(overlay.message)
            return
        if overlay.choice == FORK:
            self.send(Fork(overlay.session_id, overlay.seq))
        elif overlay.choice == ROLLBACK:
            self.send(Rollback(overlay.session_id, overlay.seq))
        # Either cut leaves you at the point the conversation now ends, which
        # is a place to say the next thing from. `reuse_message` above puts the
        # cursor in the box for itself.
        self.focus = INPUT

    def reuse_message(self, text: str) -> None:
        """Put one of your own past messages back in the box, to send again or
        edit into the next one — usually a command that needs a word changed,
        which is otherwise retyped off the screen.

        Added to whatever is already being written rather than replacing it, so
        activating a message can never lose a draft. It starts its own line,
        except after a draft left ending in whitespace — that space is how you
        say "continue here" (``rerun this: `` + the old command).
        """
        self._into_draft(self.session, text)
        self.focus = INPUT  # cursor behind the reused text, ready to send
        self.note = "copied into the message box"

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
        waits = [self.session.next_wake(self.wall())]
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
        if self.active_id:
            self.send(Interrupt(self.active_id))
        self.note = "stopped the turn"
        return True

    def _handle_input(self, key: str) -> bool:
        if key == "esc":
            self._escape()
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
        elif key == "quit":
            return False
        else:
            self.input.handle(key)
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
        if not self.active_id:
            self.note = "no session open"
            return
        if text.startswith("/") and self.session.turn.busy:
            # A slash command acts on the UI and runs its own exclusive
            # worker, so there is nothing sensible to queue it behind — and a
            # `/compact` that ran an hour later against a thread the turn had
            # since changed would be worse than one that was refused. An
            # ordinary message queues; this one waits for the user.
            self.note = "wait for this turn — a command cannot be queued"
            self.note_style = DIM
            return
        self.send(Submit(self.active_id, text))
        self.input.clear()
        self.note = "sent"

    def _handle_row(self, key: str, width: int, height: int) -> bool:
        if key in ("q", "quit"):
            return False
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
        elif key == "m":
            self.overlay = LlmOverlay(*self._llms)
        elif key == "a":
            self.overlay = ProfilesOverlay(list(self._profiles), self._learnings)
        elif key == "c":
            self.overlay = ConfigOverlay(self._settings_json)
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
                self._switch(self.session_pane.current(inner))
                # Straight to the box: opening a session is something you do
                # in order to say something in it — unless it is parked on a
                # decision, which is the thing to do in it first.
                self.focus = (
                    DECISION if self.session.decision is not None else INPUT
                )
            elif self.focus == CHAT:
                self._activate_chat(inner)
            elif self.focus == WATCHERS:
                self._watch(Peek, "peeking at the log", inner)
        elif key == "r" and self.focus == SESSIONS:
            self.note = "rename: a modal in the real app"
        elif key == "t" and self.focus == SESSIONS:
            self.note = "asking the llm for a title"
        elif key == "d":
            if self.focus == SESSIONS:
                self.note = "delete session (confirm)"
            else:
                self._watch(Drop, "unwatched", inner)
        return True

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
