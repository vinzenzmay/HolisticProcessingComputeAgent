"""The rows themselves: layout, focus and key dispatch.

``RowUI`` takes keys and returns frames and does no I/O of its own, which is
what makes the whole UI testable by calling ``render()`` and comparing strings.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from hpca import __version__ as VERSION
from hpca.ui.ansi import BOLD, CYAN, DIM, GREEN, RED, RESET, REVERSE, YELLOW
from hpca.ui.ansi import cell_width, footer_line, pad, rule
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS, is_paste, paste_text
from hpca.ui.overlays import (
    COPY,
    FORK,
    ROLLBACK,
    ConfigOverlay,
    HelpOverlay,
    LlmOverlay,
    Overlay,
    ProfilesOverlay,
    RewindOverlay,
)
from hpca.ui.pane import Item, Pane
from hpca.ui.state import (
    OWN_MESSAGE_KINDS,
    Confirm,
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
)

# How long a first escape stays armed for a second one to complete the stop
# gesture. The same 1.0s the Textual app uses, and for the same reason a single
# escape must keep meaning nothing: ESC is the byte a terminal also sends as the
# prefix of every arrow key and of bracketed paste, so one arriving on its own
# is not evidence that the user wants the turn dead. Two in a second are.
ESC_STOP_WINDOW = 1.0

SESSIONS, CHAT, INPUT, WATCHERS = range(4)


# The sidebar markers §4.3 item 15 asks for, and the flag strings the core
# uses for the same states. Both are consulted: the flag is what the core
# knows, and the local turn/decision state is what has arrived since the last
# `session.rows` — a marker that waited for the next sidebar repaint would lag
# a whole poll behind the event that caused it.
DECISION_MARK, WORKING_MARK = "!", "⟳"


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
        return self.session.mode or "agent"

    @property
    def model(self) -> str:
        """Empty until something puts a model on the wire.

        `protocol.SessionRow` carries the title, the profile, the mode and the
        render flags, and nothing about the backend a session is pinned to — so
        the model line (§4.3 item 20) has nothing to read yet, and draws as the
        absence rather than as a guess.
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
                        f"{session.profile} · {session.mode or 'agent'}"
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
        inp = self._input_h(width)
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
            inp = 2
            middle = max(1, avail - 6)
        return [top, middle, inp, bottom]

    # -------------------------------------------------------------- drawing

    def render(self, width: int, height: int) -> list[str]:
        out = [self._header(width)]
        if self.overlay is not None:
            out += self.overlay.render(width, height - 2)
            out.append(footer_line(self.overlay.footer(), width))
            while len(out) < height:
                out.insert(len(out) - 1, " " * width)
            return out[:height]
        heights = self._heights(height, width)
        order = [
            (SESSIONS, self.panes[0], heights[0]),
            (CHAT, self.panes[1], heights[1]),
            (INPUT, None, heights[2]),
            (WATCHERS, self.panes[2], heights[3]),
        ]
        for slot, pane, pane_h in order:
            if slot == INPUT:
                out += self._render_input(width, pane_h)
            else:
                out += pane.render(width, pane_h, focused=self.focus == slot)
        note, style = self.note, self.note_style
        if self._esc_armed():
            note, style = "esc again to stop", RED
        out.append(footer_line(self._keys(), width, note, style))
        while len(out) < height:
            out.insert(len(out) - 1, " " * width)
        return out[:height]

    def _render_input(self, width: int, height: int) -> list[str]:
        focused = self.focus == INPUT
        # Built from what is actually known: the model comes from nothing on
        # the wire yet, and the context meter is empty until the core has
        # either measured or estimated one.
        right = " · ".join(
            x for x in (self.mode, self.model, self.session.context.label()) if x
        )
        title = rule("message", width, right)
        out = [(BOLD + CYAN if focused else DIM) + title + RESET]
        rows = max(1, height - 1)
        body = self.input.render(self._input_body(width), rows, focused=focused)
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((CYAN if focused else DIM) + marker + RESET + line)
        return out[:height]

    def _header(self, width: int) -> str:
        left = f" HPCA {VERSION}  ·  {self.profile}  ·  {self.mode}"
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
            rows += [("i", "write"), ("enter", "reuse")]
        else:
            rows += [("enter", "peek"), ("d", "unwatch"), ("alt-↑↓", "move")]
        return rows + [("m", "llms"), ("a", "profiles"), ("c", "config")] + common

    # --------------------------------------------------------------- input

    def handle(self, key: str, width: int, height: int) -> bool:
        if is_paste(key):
            self._paste(paste_text(key))
            return True
        if self.overlay is not None:
            overlay = self.overlay
            if not overlay.handle(key, width, height - 2):
                self.overlay = None
                self._closed(overlay)
            return True
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
        if isinstance(overlay, RewindOverlay) and overlay.choice:
            self._rewind(overlay)

    # -------------------------------------------------------- the chat rewind

    def _activate_chat(self, width: int) -> None:
        """Enter in the chat log.

        On one of your own messages it opens the rewind. On anything else it
        moves to the message box, which is what app.py answers an Enter it has
        nothing better to do with.
        """
        entry = self.session.entry_at(self.chat.current(width))
        if entry is not None and entry.kind in OWN_MESSAGE_KINDS:
            # The row's own name, not its position: the cut is decided by the
            # user now and carried out by the core later, and a turn appending
            # in between moves every position after it.
            self.overlay = RewindOverlay(entry.text, entry.seq, self.active_id)
        else:
            self.focus = INPUT

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
        draft = self.input.text()
        if draft and not draft[-1].isspace():
            draft += "\n"
        self.input.set_text(draft + text)
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
        alone has to say when it will. Today that is the armed escape and
        nothing else: "esc again to stop" is only true for
        ``ESC_STOP_WINDOW``, and no keypress is coming to wipe it. Asked after
        every frame, so a value that has already expired is None rather than
        zero — zero would be a repaint that schedules another repaint.
        """
        if self._esc_armed_at is None:
            return None
        left = ESC_STOP_WINDOW - (self.clock() - self._esc_armed_at)
        # A hair past the window rather than exactly on it: `_esc_armed` is
        # true *at* the boundary, so waking there would redraw the same frame
        # and then have nothing left to schedule — the hint would stick.
        return left + 0.01 if left > 0 else None

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
        elif key in ("ctrl-up", "shift-tab"):
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
                # in order to say something in it.
                self.focus = INPUT
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
