"""The rows themselves: layout, focus and key dispatch.

``RowUI`` takes keys and returns frames and does no I/O of its own, which is
what makes the whole UI testable by calling ``render()`` and comparing strings.
"""

from __future__ import annotations

import time

from hpca import __version__ as VERSION
from hpca.ui.ansi import BLUE, BOLD, CYAN, DIM, GREEN, RED, RESET, REVERSE, YELLOW
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

# How long a first escape stays armed for a second one to complete the stop
# gesture. The same 1.0s the Textual app uses, and for the same reason a single
# escape must keep meaning nothing: ESC is the byte a terminal also sends as the
# prefix of every arrow key and of bracketed paste, so one arriving on its own
# is not evidence that the user wants the turn dead. Two in a second are.
ESC_STOP_WINDOW = 1.0

SESSIONS, CHAT, INPUT, WATCHERS = range(4)


class SessionState:
    """Everything that belongs to one conversation rather than to the app.

    Switching sessions swaps this and nothing else, which is why the chat's
    cursor line, its open entries and a half-typed message all survive going
    away and coming back — the same property each row has, one level up. The
    Textual app parks drafts per session for the same reason; here it falls out
    of where the Editor lives instead of needing a store.

    Watchers belong to the session that registered them (WatchStore.list takes
    a session_id), so they swap too. A session with none simply shows an empty
    row, which the layout already charges nothing for.

    The chat and the watchers are built on first visit. The real app reads a
    thread from the checkpointer on switch, and this mirrors that: fourteen
    sessions of four hundred steps should not all exist because one is open.
    """

    def __init__(
        self,
        *,
        title: str,
        profile: str,
        model: str,
        started: str,
        size: int,
        watch_count: int,
        index: int,
    ) -> None:
        self.title = title
        self.profile = profile
        self.model = model
        self.started = started
        self.size = size
        self.watch_count = watch_count
        self.index = index
        self.draft = Editor(wrap=True)
        self._chat: Pane | None = None
        self._watchers: Pane | None = None

    @property
    def loaded(self) -> bool:
        return self._chat is not None

    @property
    def chat(self) -> Pane:
        # Imported here rather than at module scope: ``demo`` builds a RowUI,
        # so the app must not depend on the demo content at import time. Until
        # a real conversation arrives (specs-ui-replacement.md M3) this is
        # where it comes from.
        from hpca.ui.demo import sample_chat

        if self._chat is None:
            self._chat = Pane("chat", sample_chat(self.size, self.index, self.title))
            self._chat.cursor = 10**9  # open at the newest, as the app does
        return self._chat

    @property
    def watchers(self) -> Pane:
        from hpca.ui.demo import sample_watchers

        if self._watchers is None:
            self._watchers = Pane(
                "watchers", sample_watchers(self.watch_count, self.index)
            )
        return self._watchers

    def invalidate(self) -> None:
        for pane in (self._chat, self._watchers):
            if pane is not None:
                pane.invalidate()


class RowUI:
    MIN_CHAT = 4
    MAX_INPUT = 6

    def __init__(
        self,
        sessions: list[SessionState],
        *,
        learnings: dict[str, str],
        settings_json: str,
        llms: tuple[list[Item], list[Item]],
    ) -> None:
        self.sessions = sessions
        self.active = 0
        self.session_pane = Pane("sessions", [])
        self._refresh_sessions()
        self.focus = CHAT
        self.mode = "agent"
        self.frame_ms = 0.0
        self.note = ""
        self.overlay: Overlay | None = None
        # When the last escape landed, so the next one can tell whether it is
        # the second half of a stop. Injectable so the headless check can drive
        # the clock instead of sleeping through the window.
        self.clock = time.monotonic
        self._esc_armed_at: float | None = None
        self._learnings = learnings
        self._settings_json = settings_json
        self._llms = llms

    # ------------------------------------------------------ the open session

    @property
    def session(self) -> SessionState:
        return self.sessions[self.active]

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
        return self.session.profile

    @property
    def model(self) -> str:
        return self.session.model

    @property
    def panes(self) -> list[Pane]:
        """The three list rows, top to bottom, for the session on screen."""
        return [self.session_pane, self.chat, self.watchers]

    def _refresh_sessions(self) -> None:
        """Redraw the session list so the open one is marked.

        Rebuilt rather than patched because it is fourteen rows, not fourteen
        hundred — the cost that matters is the chat's, and that one is never
        rebuilt at all. The cursor is kept: which session you are *looking at*
        is not the same as which one is open, and moving the highlight must not
        follow the switch.
        """
        cursor = self.session_pane.cursor
        self.session_pane.items = [
            Item(
                head=(
                    f"{'●' if i == self.active else '○'} "
                    f"{state.title[:40]:<42}{state.started:>9}   {state.model}"
                ),
                body=[
                    f"session 9f3c{i:04x} · profile {state.profile} · mode agent",
                    f"{state.size} entries · {(i * 13) % 90}% of context used",
                    f"{state.watch_count} watches"
                    + (" · open" if i == self.active else ""),
                ],
                accent=GREEN if i == self.active else "",
            )
            for i, state in enumerate(self.sessions)
        ]
        self.session_pane.invalidate()
        self.session_pane.cursor = cursor

    def _switch(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        if index == self.active:
            self.note = "already open"
            return
        self.active = index
        self._refresh_sessions()
        self.note = f"opened “{self.session.title}”"

    def invalidate(self) -> None:
        """Every pane that has been built — a session never visited has none."""
        self.session_pane.invalidate()
        for state in self.sessions:
            state.invalidate()

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
        note, style = self.note, YELLOW
        if self._esc_armed():
            note, style = "esc again to stop", RED
        out.append(footer_line(self._keys(), width, note, style))
        while len(out) < height:
            out.insert(len(out) - 1, " " * width)
        return out[:height]

    def _render_input(self, width: int, height: int) -> list[str]:
        focused = self.focus == INPUT
        right = f"{self.mode} · {self.model} · 31% ctx"
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
            self._rewind(overlay.choice, overlay.index, overlay.message)

    # -------------------------------------------------------- the chat rewind

    def _activate_chat(self, width: int) -> None:
        """Enter in the chat log.

        On one of your own messages it opens the rewind. On anything else it
        moves to the message box, which is what app.py answers an Enter it has
        nothing better to do with.
        """
        index = self.chat.current(width)
        item = self.chat.items[index] if index >= 0 else None
        if item is not None and item.kind == "user":
            self.overlay = RewindOverlay(item.text, index)
        else:
            self.focus = INPUT

    def _rewind(self, choice: str, index: int, message: str) -> None:
        if choice == COPY:
            self.reuse_message(message)
        elif choice == FORK:
            self._fork_at(index)
        elif choice == ROLLBACK:
            self._rollback_to(index)

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

    def _fork_at(self, index: int) -> None:
        """A copy of this conversation that stops just before that message.

        The original stays whole — that is the difference from a rollback, and
        the reason both are offered instead of one being the safe version of
        the other.
        """
        source = self.session
        fork = SessionState(
            title=f"{source.title} (fork)",
            profile=source.profile,
            model=source.model,
            started="just now",
            size=0,
            watch_count=0,
            index=source.index,
        )
        fork._chat = Pane("chat", list(self.chat.items[:index]))
        fork._chat.cursor = 10**9
        self.sessions.insert(0, fork)
        self.active = 0
        self._refresh_sessions()
        self.focus = INPUT
        self.note = f"forked “{source.title}” — this copy stops before that message"

    def _rollback_to(self, index: int) -> None:
        """Drop this conversation back to just before that message."""
        chat = self.chat
        dropped = len(chat.items) - index
        del chat.items[index:]
        chat.expanded = {i for i in chat.expanded if i < index}
        chat.invalidate()
        chat.cursor = 10**9
        self.focus = INPUT
        self.note = f"rolled back to just before that message ({dropped} entries gone)"

    def _esc_armed(self) -> bool:
        """Whether a first escape is still waiting for its second.

        Read by the footer, which says so in red. app.py deliberately stays
        silent here, but its objection is to a *toast*: a notification for
        every stray escape sequence would be noise on top of the screen. A
        word in the footer costs nothing, cannot cover anything, and goes away
        on its own — the poll timeout repaints twice a second, so the hint
        expires with the window rather than sitting there until the next key.
        """
        return (
            self._esc_armed_at is not None
            and self.clock() - self._esc_armed_at <= ESC_STOP_WINDOW
        )

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
        text = self.input.text().strip()
        if not text:
            return
        chat = self.chat
        chat.items.append(
            Item(head=f"you   {text}", accent=BLUE, kind="user", text=text)
        )
        chat.items.append(
            Item(
                head="hpca  looking at that now…",
                body=["(there is no backend yet; this is where a turn would start)"],
                accent=YELLOW,
            )
        )
        chat.invalidate()
        chat.cursor = 10**9
        self.input.clear()
        self.note = "sent"

    def _handle_row(self, key: str, width: int, height: int) -> bool:
        # ``demo`` builds a RowUI, so it is imported here rather than at module
        # scope; see SessionState.chat.
        from hpca.ui.demo import sample_profiles

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
            self.overlay = ProfilesOverlay(sample_profiles(), self._learnings)
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
            if not pane.expand(inner) and pane.current(inner) in pane.expanded:
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
                self.note = "peeking at the log"
        elif key == "r" and self.focus == SESSIONS:
            self.note = "rename: a modal in the real app"
        elif key == "t" and self.focus == SESSIONS:
            self.note = "asking the llm for a title"
        elif key == "d":
            self.note = (
                "delete session (confirm)"
                if self.focus == SESSIONS
                else "unwatched"
            )
        return True
