"""Main application: three-column layout, focus model, agent wiring (§3, §4)."""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os import environ as os_environ
from pathlib import Path
from time import monotonic, time
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Offset
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Label, ListItem, ListView, Static, TextArea

from hpca.agent import compact
from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.explainer import explain_process_failure
from hpca.agent.graph import (
    build_graph,
    compact_now,
    deliver_event,
    rollback_thread,
    run_turn,
    thread_message_count,
)
from hpca.agent.job_tools import add_job_tools
from hpca.agent.watch_tools import add_watch_tools
from hpca.agent.tools import ToolRegistry
from hpca.autoconnect import plan_auto_connect
from hpca.clipboard import ClipboardManager, CopyResult
from hpca.cluster_endpoints import discover_cluster_endpoints
from hpca import curator
from hpca.config import LLMBackend, Settings, app_dir, llm_settings_for
from hpca.discover import DiscoveredBackend
from hpca.db import (
    DbIO,
    command_use_counts,
    connect,
    init_db,
    record_command_use,
)
from hpca.dbcache import DbCache, local_dir_for
from hpca.jobs import JobRow, JobStore, apply_statuses
from hpca.agent.conclude import MemoryProposal, propose_memories
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.memory_context import (
    build_memory_context,
    compose_api_content,
    note_line,
    retrieved_line,
)
from hpca.agent.memory_tools import add_memory_tools
from hpca.agent.modes import (
    add_plan_tool,
    next_mode,
)
from hpca.agent.prompts import (
    build_skill_directive,
    environment_facts,
    orchestrator_system_prompt,
)
from hpca.agent.reflect import Reflection, propose_reflections
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.titler import propose_title
from hpca.agent.struggle import (
    STRUGGLE_KIND,
    matching_struggles,
)
from hpca.editor import resolve_editor
from hpca.embeddings import EmbeddingClient
from hpca.episodic import EpisodicStore
from hpca.llm import LLMClient
from hpca.logs import LoggedLLM, SessionLog, open_log
from hpca.looplag import LoopLagProbe
from hpca.rag import RagStore
from hpca.transcript import ASSISTANT as ASSISTANT_ENTRY
from hpca.transcript import THINKING, Entry, Step, build_entries
from hpca.transcript import USER as USER_ENTRY
from hpca.profiles import DEFAULT_PROFILE, MemoryScope, Profile
from hpca.registry import PathRegistry
from hpca.runner import (
    ProcessRunner,
    analyse_process_failure,
    format_process_event,
    poll_processes,
    reconcile_orphans,
    running_session_ids,
)
from hpca.triage import Signature, append_user_signature
from hpca.sessions import Session, SessionStore
from hpca.skills import (
    Skill,
    copy_profile_skills,
    delete_own_skill,
    delete_profile_skills,
    load_own_skills,
    load_project_skills,
    load_skills,
    patched_body,
    skill_path,
    summarize_skills,
    write_skill,
)
from hpca.slurm import TERMINAL_STATES as SLURM_TERMINAL_STATES
from hpca.slurm import SlurmClient
from hpca.symbols import SymbolIndex
from hpca.trash import TrashManager
from hpca.tui.approval_screen import (
    approval_details,
    approval_hint,
    approval_kind,
    approval_reason_hint,
    approval_reason_title,
    approval_script,
    approval_title,
)
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.context_bar import ContextBar, ModelLine
from hpca.tui.inspect_screen import InspectScreen
from hpca.tui.manage_llms import ManageLLMsScreen
from hpca.tui.mode_bar import ModeBar
from hpca.memory_index import MemoryIndex
from hpca.memory_ops import (
    MemoryOp,
    MemoryOpError,
    apply_batch,
    drift_detected,
)
from hpca.tui.memory_screens import (
    MemoryBatchScreen,
    MemoryProposalScreen,
    ReflectionScreen,
)
from hpca.tui.profiles_screen import ProfilePickerScreen, ProfilesScreen
from hpca.tui.rename_screen import RenameScreen
from hpca.tui.settings_screen import SettingsScreen
from hpca.tui.switch_llm import SwitchLLMScreen
from hpca.watches import (
    KIND_JOB,
    KIND_LOG,
    LOG_GONE,
    Watch,
    WatchStore,
    apply_job_details,
    is_settled,
    peek,
    poll_log_watches,
    watch_class,
    watch_lines,
)

COLUMN_IDS = ("sessions", "chat", "watchers")
# How often a watched log is stat'ed. Cheap (one stat per box) and worth being
# responsive about: the whole point of a log box is that "still writing" and
# "stopped a minute ago" are visibly different.
LOG_WATCH_SECONDS = 5.0
# ...and how often watched jobs are refreshed. Slower: each sweep is an squeue
# call to the controller, and a job's state does not change on the second.
JOB_WATCH_SECONDS = 15.0
# The Enter peek flashes rather than parks: long enough to read 300 characters,
# short enough that it is a glance, not a screen to dismiss.
PEEK_TIMEOUT = 12.0


def _file_logger(name: str, filename: str) -> logging.Logger:
    """A logger writing to <app_dir>/<filename>, and only there.

    The handler is attached on first use (idempotent) and does not propagate:
    a TUI owns the terminal, so nothing may reach the root logger — whose
    last-resort handler writes to stderr and would shred the display.

    The app dir is created here rather than assumed: on a first run nothing has
    written settings yet, and these loggers are the first thing startup touches
    — a missing dir used to take the app down with a FileNotFoundError before
    the UI existed.
    """
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        app_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(app_dir() / filename)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _autoconnect_logger() -> logging.Logger:
    """The ``hpca.autoconnect`` logger. Discovery is best-effort and must never
    interrupt startup, so its failures are invisible in the TUI and this file
    is the debugging trail."""
    return _file_logger("hpca.autoconnect", "autoconnect.log")


def _dbcache_logger() -> logging.Logger:
    """The ``hpca.dbcache`` logger. A failed sync is retried on the next tick
    rather than surfaced, and recovery happens before the UI exists, so this
    file is where both leave their trail."""
    return _file_logger("hpca.dbcache", "dbcache.log")



@dataclass
class PanelRow:
    """One box of the watchers column.

    The column repaints on a timer while the user is arrowing through it.
    Naming every row with a stable ``key`` is what lets a repaint update the
    text in place — a watch's "last write 4s ago" ticks on its own — and
    rebuild only when the set of rows actually changes, so the cursor stops
    jumping back to the top twice a second.
    """

    key: str
    text: str
    classes: str
    title: str  # the box's border title
    watch: "Watch"


SESSION_TITLE_MAX = 40
UNTITLED_SESSION = "untitled"  # placeholder until the first message names it

# Sidebar in-flight marker (stage 3 / decision 6): prefixed on the row of any
# session with a live TurnState, so an OFF-SCREEN session that is still working
# is visible. A stable glyph (not an animated spinner) — appears/clears reliably
# and needs no timer. Distinct from the "! " pending-decision mark so the two do
# not collide; when a row is somehow both, the "! " decision mark wins (a
# decision needs the user, working is transient — see _session_row_text).
WORKING_MARK = "⟳ "

# Interrupt-while-waiting-for-the-LLM (§ interrupt): the activity string the
# graph reports while a turn is parked on the model — the one phase the
# interrupt is armed in. The user aborts it by selecting the working indicator
# in the chat log and confirming (see _maybe_interrupt_llm), so the message can
# be re-edited and sent again.
LLM_WAIT_ACTIVITY = "LLM processing"
# The chat kinds that are the user's own words — what Enter on a message in the
# log offers back for reuse. "queued" is the same text, typed ahead of a
# running turn; everything else in the log (the agent's replies, background
# events, recalled memory) is not something the user would send again.
OWN_MESSAGE_KINDS = ("user", "queued")
CHAT_TITLES = {
    "user": "you",
    "assistant": "agent",
    "error": "error",
    "event": "background",
    "queued": "queued",
    "recall": "recalled from memory",
    "notice": "context",
}


@dataclass
class PendingWork:
    """One turn's worth of input waiting for the orchestrator to be free."""

    session_id: str
    text: str
    kind: str  # "user" — typed and waiting | "event" — background completion
    # Set when the text is a "/<skill>" invocation: the named skill's procedure
    # is dropped into that turn's API copy so the model follows it directly.
    forced_skill: "Skill | None" = None


@dataclass
class TurnState:
    """Everything one in-flight turn owns, kept per session so turns on
    different sessions never read each other's client, context, memory,
    interrupt bookkeeping, or activity.

    A session "has a turn in flight" ⇔ its session_id is a key in
    ``HpcaApp._turns``. Cleared wholesale in ``_agent_turn``'s finally block.
    """

    session: "Session"
    worker: Any = None
    ctx: "ToolContext | None" = None
    memory: "Profile | None" = None       # that turn's frozen profile snapshot
    skills: "list[Skill] | None" = None
    interrupt_keep: int | None = None
    # The user message this turn is running (None for a resume or an event).
    # Two readers: the interrupt hands it back to the entry for editing, and a
    # transcript rebuild re-adds it while the graph copy still predates it.
    user_text: str | None = None
    activity: str = "working"             # what the spinner says (decision 7)
    # When this turn began. Lives here, not on the spinner widget, because the
    # widget is rebuilt every time the session is re-opened — reading the clock
    # off the widget restarted it at 0 on every visit, so a turn the user had
    # been waiting on for two minutes claimed to be three seconds old.
    started: float = field(default_factory=monotonic)


# Built-in chat commands ("/" or "\"): typing the prefix lists these above the
# entry, alongside the profile's skills (see ``_all_commands``). Built-ins run a
# UI worker; a "/<skill>" runs a normal turn with that skill's procedure forced.
COMMANDS = (
    ("memorize", "/memorize <note> — form memories from the note and this conversation"),
    ("conclude", "/conclude — propose memories from this conversation"),
    (
        "compact",
        "/compact [what to keep / what you do next] — fold this conversation "
        "into a summary and free the context",
    ),
    ("skill-creator", "/skill-creator — add a skill (name, description, body) to this profile"),
    ("skills-list", "/skills-list — list this profile's skills"),
    ("skill-remove", "/skill-remove — remove one of this profile's skills"),
)
# How the "/" menu marks its own commands apart from the profile's skills:
# the built-in's name is bold, everything else is left alone. A plain ANSI
# attribute, not a theme variable — a span style is parsed at paint time, and
# `$text` there raises UnresolvedVariableError (variables resolve in CSS and
# markup, not in assembled spans). The legend in the border title is what
# makes the mark readable — bold alone says an entry is special, not why.
BUILTIN_COMMAND_STYLE = "bold"
LOG_KINDS = {
    "user": "user",
    "assistant": "agent",
    "error": "error",
    "event": "background",
    "recall": "recalled from memory",
}


class TopBar(Static):
    """Top bar: app name | profile | settings hint.

    The per-session model moved to a dedicated line at the top of the chat
    column (``ModelLine``); it belongs to the session on screen, not to the
    app as a whole. ``profile`` stays here (it is also echoed on each sidebar
    row).
    """

    def __init__(self) -> None:
        super().__init__(id="top-bar")
        self._profile = ""

    def update_info(self, profile: str) -> None:
        self._profile = profile
        self.update(self.render_text())

    def render_text(self) -> str:
        from hpca import __version__

        return (
            f" HPCA v{__version__} │ profile: {self._profile} │ (c) config"
        )


class ColumnPanel(Vertical):
    """One of the three main columns; its ListView receives focus."""

    def __init__(self, title: str, *, id: str) -> None:
        super().__init__(id=id)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield ListView(id=f"{self.id}-list")


class SessionsList(ListView):
    """Left column; renaming applies to a session, not to "(new session)"."""

    BINDINGS = [
        # (q) quit first so the footer shows it leftmost (§ esc/quit ordering);
        # the app-level (q) is what handles it, this just orders the display.
        Binding("q", "confirm_quit", "quit"),
        Binding("r", "rename_session", "rename"),
        Binding("t", "retitle_session", "ask llm for a title"),
        Binding("d", "delete_session", "delete session"),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        if action in ("rename_session", "retitle_session", "delete_session"):
            return getattr(self.highlighted_child, "data_session", None) is not None
        return True

    def action_rename_session(self) -> None:
        self.app.rename_selected_session()

    def action_retitle_session(self) -> None:
        self.app.retitle_selected_session()

    def action_delete_session(self) -> None:
        self.app.confirm_delete_session()


class SessionsPanel(ColumnPanel):
    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield SessionsList(id="sessions-list")


class ChatInput(TextArea):
    """Chat entry: a wrapping, multi-line field that grows with the draft, so
    a long message is readable while it is being written.

    Enter sends; shift+enter, alt+enter or ctrl+j start a new line (terminals
    that cannot report shift+enter still have the other two). Arrow keys move
    the text cursor and only hand focus on at the edges of the draft: ← at the
    very start leaves for the sessions column, → at the very end for the
    watchers column, ↑ on the first *visual* row leaves to browse the message
    log. The row check is wrap-aware, so ↑/↓ still step through a long draft
    that soft-wraps onto several rows even though it is one logical line.
    The draft is kept, so the user can step away mid-sentence and come back.
    It belongs to the session, not to this widget: one entry serves them all,
    so switching sessions parks the text under the session being left and puts
    that session's own back (see ``HpcaApp._park_draft``/``_restore_draft``).
    """

    NEWLINE_KEYS = ("shift+enter", "alt+enter", "ctrl+j")

    class Submitted(Message):
        def __init__(self, chat_input: "ChatInput", text: str) -> None:
            super().__init__()
            self.chat_input = chat_input
            self.text = text

        @property
        def control(self) -> "ChatInput":
            return self.chat_input

    def __init__(self, **kwargs) -> None:
        super().__init__(soft_wrap=True, **kwargs)

    async def _on_key(self, event) -> None:
        # While the slash-command menu is open, ↑/↓ move the selection and
        # tab/enter fast-select it (enter only fills a partial command; a fully
        # typed one falls through to submit and runs).
        if self.app.command_menu_active():
            if event.key in ("up", "down"):
                event.stop()
                event.prevent_default()
                self.app.command_menu_move(-1 if event.key == "up" else 1)
                return
            if event.key == "tab":
                event.stop()
                event.prevent_default()
                self.app.command_menu_accept()
                return
            if event.key == "enter" and self.app.command_menu_accept():
                event.stop()
                event.prevent_default()
                return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
        elif event.key in self.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
        else:
            await super()._on_key(event)

    def action_cursor_left(self, select: bool = False) -> None:
        if not select and self.selection.is_empty and self.cursor_at_start_of_text:
            self.app.action_focus_column(-1)
        else:
            super().action_cursor_left(select)

    def action_cursor_right(self, select: bool = False) -> None:
        if not select and self.selection.is_empty and self.cursor_at_end_of_text:
            self.app.action_focus_column(1)
        else:
            super().action_cursor_right(select)

    def action_cursor_up(self, select: bool = False) -> None:
        # Wrap-aware: leave only from the first *visual* row. cursor_at_first_line
        # is true for the whole logical line, so with soft wrap it would fire
        # anywhere in a wrapped first paragraph and never let ↑ climb rows.
        if not select and self.navigator.is_first_wrapped_line(self.selection.end):
            self.app.browse_chat_messages()
        else:
            super().action_cursor_up(select)


class ThinkingBox(Static):
    """One turn's working — the model's reasoning and the tool steps it led
    to — folded into a single box. Collapsed by default; enter toggles it.

    Expanding no longer spells the whole box out inline: the app reveals each
    part as its own :class:`StepBox` row below this one, so the header only ever
    carries the summary and the expand/collapse hint."""

    def __init__(self, entry: Entry, *, expanded: bool = False) -> None:
        super().__init__(classes="chat-thinking")
        self.border_title = "thinking"
        self._entry = entry
        self._collapsed = not expanded
        self._render_entry()

    @property
    def entry(self) -> Entry:
        return self._entry

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_expanded(self._collapsed)

    def set_expanded(self, expanded: bool) -> None:
        self._collapsed = not expanded
        self._render_entry()

    def _render_entry(self) -> None:
        marker = "▶" if self._collapsed else "▼"
        hint = "enter to expand" if self._collapsed else "enter to collapse"
        self.update(Content(f"{marker} {self._entry.summary()}  ({hint})"))


class StepBox(Static):
    """One part of an expanded thinking box — a reasoning block or a tool step —
    collapsible on its own. Collapsed shows just its label; expanded appends the
    part's full text. Each is a ListView row, so up/down/enter navigate them
    with no special-casing."""

    def __init__(self, step: Step, *, expanded: bool = False) -> None:
        # A call is marked apart from the reasoning and results around it: it
        # is the row that holds the script, and the one the user comes back to
        # the log to read.
        classes = "chat-step chat-step-call" if step.kind == "call" else "chat-step"
        super().__init__(classes=classes)
        self._step = step
        self._collapsed = not expanded
        self._render_step()

    @property
    def expanded(self) -> bool:
        return not self._collapsed

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._render_step()

    def _render_step(self) -> None:
        marker = "▶" if self._collapsed else "▼"
        header = f"{marker} {self._step.label()}"
        body = "" if self._collapsed else "\n\n" + self._step.text.strip()
        self.update(Content(header + body))


@dataclass
class _ThinkingExpansion:
    """Whether a thinking box is open, and which of its parts are open. Held per
    Entry so an expanded view is not lost to an unrelated chat re-render."""

    open: bool = False
    parts: dict[int, bool] = field(default_factory=dict)  # part index → open


class ChatItem(ListItem):
    """A chat entry whose text can be marked with the mouse.

    Textual turns a press and release on one widget into a Click however far
    the mouse travelled between them, and ListItem reads any Click as
    "activate me" — which is why dragging across a message used to jump focus
    to the entry, or collapse the box under the cursor and take the selection
    with it. A click that moved marked text; it did not ask for anything to
    happen, so it suppresses ListItem's default action and nothing else.
    Marking and copying (ctrl+c) are Textual's own; they only need us out of
    the way.
    """

    def __init__(self, *children) -> None:
        super().__init__(*children)
        self._pressed_at: Offset | None = None

    def _on_mouse_down(self, event: events.MouseDown) -> None:
        self._pressed_at = event.screen_offset

    def _on_click(self, event: events.Click) -> None:
        pressed, self._pressed_at = self._pressed_at, None
        if pressed is not None and pressed != event.screen_offset:
            # Textual dispatches every _on_click up the MRO, so ListItem's
            # "activate me" is stopped by preventing it, not by not calling it.
            event.prevent_default()


class WorkingIndicator(Static):
    """Sits at the end of the log while a reply is on its way.

    A turn is silent for seconds, or minutes with thinking on, and a still
    screen looks like a hung one. It names the step the graph reports —
    thinking, or the tool in flight — and counts up from the moment the turn
    started, which is how you tell a slow answer from a lost one.

    The count belongs to the turn (``TurnState.started``), not to this widget:
    leaving a session and coming back rebuilds the widget, and a clock owned
    here would start again from zero each visit.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.08

    def __init__(self, activity: str = "working", started: float | None = None) -> None:
        super().__init__(classes="chat-working")
        self._frame = 0
        self._activity = activity
        # Supplied by the caller for a turn already running (see TurnState);
        # minted here only for a one-off backend call that has no turn.
        self._started = monotonic() if started is None else started
        self._width = -1  # last rendered width; see _render_frame

    def on_mount(self) -> None:
        self.set_interval(self.INTERVAL, self._advance)
        self._render_frame()

    @property
    def activity(self) -> str:
        return self._activity

    @property
    def elapsed(self) -> int:
        return int(monotonic() - self._started)

    def set_activity(self, activity: str) -> None:
        """Name the step now in flight, leaving the clock alone.

        The number answers "how long since I asked?", which is the question
        the user actually has while waiting. Timing each step separately read
        better in theory but hid the total: a turn that spent a minute across
        four steps never showed a number above twenty.
        """
        if activity == self._activity:
            return
        self._activity = activity
        self._render_frame()

    def _advance(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self._render_frame()

    def _frame_text(self) -> str:
        elapsed = f" {self.elapsed}s" if self.elapsed else ""
        # While parked on the model this line is selectable to abort the turn;
        # say so, mirroring the ThinkingBox's inline "(enter …)" hint.
        hint = "  (enter to interrupt)" if self._activity == LLM_WAIT_ACTIVITY else ""
        return f"{self.FRAMES[self._frame]} {self._activity}…{elapsed}{hint}"

    def _render_frame(self) -> None:
        # `Static.update` lays out by default, and a layout pass walks the
        # whole chat log — O(messages). At 12.5 frames a second that made the
        # TUI crawl for as long as a reply was in flight, and worse the longer
        # the conversation: measured event-loop lag went from 0.7ms to 44ms
        # (p95) at 300 messages, which is exactly the window where the user is
        # waiting and most likely to scroll or type. Between most frames only
        # the spinner glyph changes, and a line of the same width cannot move
        # anything below it, so only a change in width earns a layout.
        text = self._frame_text()
        self.update(Content(text), layout=len(text) != self._width)
        self._width = len(text)


class ReasonInput(TextArea):
    """The "why not" box, opened by refusing a gated call.

    Enter sends what is written (empty is allowed — that is the plain refusal);
    the same keys as the chat entry start a new line, for a reason worth more
    than one. It only exists while a refusal is being explained, so unlike the
    chat entry there is nothing to park: leaving the prompt answers it.
    """

    class Submitted(Message):
        def __init__(self, reason_input: "ReasonInput", text: str) -> None:
            super().__init__()
            self.reason_input = reason_input
            self.text = text

        @property
        def control(self) -> "ReasonInput":
            return self.reason_input

    def __init__(self, **kwargs) -> None:
        super().__init__(soft_wrap=True, **kwargs)

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
        elif event.key in ChatInput.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
        else:
            await super()._on_key(event)


class DecisionBar(Vertical):
    """Inline, non-modal decision prompt at the foot of the chat column.

    A parked turn's approval (destructive op, or a gated execution in manual
    mode) renders here — inside the chat column of the session it belongs
    to — instead of a full-screen modal that would cover every other column
    and block a session the user has switched to. Its keys fire only while it
    holds focus, i.e. only when the chat column is
    focused; a decision waiting in a background session shows nothing here and
    only lights the "!" in the sidebar until that session is opened.

    Saying no has two stages, tracked by ``self.stage``. "ask" is the y/n
    question; "n" moves to "reason", which keeps the call on screen — the
    reason is written *about* it — and adds a box for why. Nothing is sent
    until that box is answered, so the refusal and the reason reach the model
    together. Escape stays a one-key way out at either stage: it refuses
    without explaining, because the key that leaves a prompt must not open a
    second one to get out of.

    ``self.kind`` and ``self.stage`` gate the keys in ``check_action`` so the
    footer offers nothing while the bar is hidden, and does not still offer
    y/n once the answer is in. It answers by calling back into the app, which
    owns the pending-decision state and runs the resume path.
    """

    can_focus = True

    # esc listed first (esc/quit ordering), and priority so it denies the
    # call rather than being eaten by whatever holds focus — including the
    # reason box, which is a text field and would otherwise swallow it.
    BINDINGS = [
        Binding("y", "approve", "approve"),
        Binding("n", "deny", "deny"),
        Binding("escape", "cancel", "deny", priority=True),
    ]

    DEFAULT_CSS = """
    DecisionBar {
        display: none;
        height: auto;
        max-height: 60%;
        margin: 0 1;
        padding: 0 1;
        border: heavy $error;
    }
    DecisionBar.decision-execution {
        border: heavy $warning;
    }
    .decision-title {
        text-style: bold;
        color: $error;
    }
    DecisionBar.decision-execution .decision-title { color: $warning; }
    .decision-script {
        height: auto;
        max-height: 12;
        border: round $panel;
        padding: 0 1;
        margin: 1 0;
    }
    .decision-hint {
        color: $text-muted;
    }
    #decision-reason {
        height: auto;
        max-height: 6;
        margin: 1 0 0 0;
        border: round $panel;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kind = ""  # "" (hidden) | "approval"
        self.stage = "ask"  # "ask" (y/n) | "reason" (saying why not)
        self._payload = None

    def check_action(self, action: str, parameters) -> bool | None:
        if action in ("approve", "deny"):
            # Gone once the answer is "no": the question left is why, and the
            # box below takes those letters as text.
            return self.kind == "approval" and self.stage == "ask"
        if action == "cancel":  # esc refuses, at either stage
            return self.kind != ""
        return True

    async def _reset(self, kind: str, payload, stage: str = "ask") -> None:
        self.kind = kind
        self.stage = stage
        self._payload = payload
        self.remove_class("decision-execution")
        await self.remove_children()

    async def show_approval(
        self, payload: dict, *, stage: str = "ask", reason: str = ""
    ) -> None:
        """Render one decision at the given stage.

        Called afresh on every session switch, so it takes the stage and any
        half-written reason as arguments rather than keeping them: the app
        holds that per session, and this bar only ever shows one.

        Never takes focus — a decision surfacing in the open session must not
        pull the user out of whatever column they are in (see
        ``HpcaApp._set_pending_decision``); ``focus_prompt`` is what lands on
        it deliberately.
        """
        await self._reset("approval", payload, stage)
        if approval_kind(payload) == "execution":
            self.add_class("decision-execution")
        # Once the answer is in, the heading stops asking for it and asks what
        # to change instead — the question the box below is there to answer.
        title = (
            approval_reason_title(payload)
            if stage == "reason"
            else approval_title(payload)
        )
        widgets: list[Widget] = [
            Static(title, classes="decision-title"),
            Static(Content(approval_details(payload)), classes="decision-details"),
        ]
        script = approval_script(payload)
        if script:
            scroller = VerticalScroll(
                Static(Content(script)), classes="decision-script"
            )
            widgets.append(scroller)
        if stage == "reason":
            # The call stays above the box: the reason is written about it,
            # and a script the user can no longer see is one they cannot
            # explain what is wrong with.
            widgets.append(
                Static(approval_reason_hint(), classes="decision-hint")
            )
            widgets.append(ReasonInput(id="decision-reason"))
        else:
            widgets.append(Static(approval_hint(payload), classes="decision-hint"))
        await self.mount(*widgets)
        if stage == "reason":
            box = self.query_one("#decision-reason", ReasonInput)
            box.text = reason  # what was typed before switching away
            box.move_cursor(box.document.end)
        self.display = True

    async def clear_decision(self) -> None:
        await self._reset("", None)
        self.display = False

    def reason_box(self) -> ReasonInput | None:
        """The "why not" box, while one is open."""
        found = self.query("#decision-reason")
        return found.first(ReasonInput) if found else None

    def reason_text(self) -> str:
        """What has been typed into the box so far (empty when there is none)."""
        box = self.reason_box()
        return box.text if box is not None else ""

    def focus_prompt(self) -> None:
        """Focus whatever answers the prompt: the box once a reason is being
        written, otherwise the bar itself, where y/n live."""
        box = self.reason_box()
        if box is not None:
            box.focus()
        else:
            self.focus()

    def action_approve(self) -> None:
        self.app.resolve_decision("approval", True)

    def action_deny(self) -> None:
        # Not an answer yet: the refusal is sent once the box says why, or
        # says nothing.
        self.app.begin_decline("approval")

    def action_cancel(self) -> None:
        if self.kind == "approval":
            self.app.resolve_decision("approval", False)

    def _on_reason_input_submitted(self, event: ReasonInput.Submitted) -> None:
        event.stop()
        self.app.resolve_decision("approval", False, reason=event.text.strip())


class ChatList(ListView):
    """Message log; ↓ past the last message returns to the chat entry."""

    def action_cursor_down(self) -> None:
        if self.index is None or self.index >= len(self) - 1:
            self.app.focus_chat_input()
        else:
            super().action_cursor_down()


class ChatPanel(ColumnPanel):
    """Center column: message log plus the chat input.

    The input only exists inside a session, so an empty chat column cannot
    invite typing that has nowhere to go.
    """

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        # The model in use for the on-screen session, on its own row directly
        # above the context meter. Hidden until a session is open.
        model_line = ModelLine(id="model-line")
        model_line.display = False
        yield model_line
        yield ContextBar(id="context-bar")
        yield ChatList(id="chat-list")
        menu = Static(id="command-menu")
        menu.display = False
        yield menu
        # A parked turn's approval renders here, at the foot of
        # the chat log for the session it belongs to — never as a modal over
        # the whole TUI (deliverable 1). Hidden until there is one.
        yield DecisionBar(id="decision-bar")
        # The mode line sits directly above the entry, visible exactly when
        # the entry is: mode is a per-session property (§3.5).
        mode_bar = ModeBar(id="mode-bar")
        mode_bar.display = False
        yield mode_bar
        chat_input = ChatInput(placeholder="Message the agent…", id="chat-input")
        chat_input.display = False
        yield chat_input


class WatchersList(ListView):
    """Right-column list; its hotkeys appear in the footer when focused.

    Enter is a flash of the log's tail rather than a modal: the question a
    watch box provokes ("did it print an error, or is it just slow?") is
    answered by 300 characters, and a screen to open and dismiss is more
    ceremony than the answer is worth.

    alt+↑/alt+↓ carry the highlighted box past its neighbour. The column is
    registration-ordered by default, which has nothing to do with what the
    user is actually waiting on — three finished jobs can sit above the one
    log that matters. Moving beats sorting because only the reader knows which
    box that is, and it is not a property anything here could compute.

    The alt+arrows are on project.md's reserved list (zellij and tmux bind
    them for pane navigation), so shift+↑/shift+↓ do the same thing — the same
    belt-and-braces as ChatInput.NEWLINE_KEYS. Only the alt pair is shown in
    the footer, which is the binding that was asked for; the shift pair is
    there for the terminals that swallow it.
    """

    BINDINGS = [
        Binding("enter", "peek_watch", "peek"),
        Binding("d", "drop_watch", "unwatch"),
        Binding("alt+up", "move_watch(-1)", "move up"),
        Binding("alt+down", "move_watch(1)", "move down"),
        Binding("shift+up", "move_watch(-1)", "move up", show=False),
        Binding("shift+down", "move_watch(1)", "move down", show=False),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        # An empty column offers nothing: with no box highlighted every one of
        # these keys would be advertised in the footer and then do nothing.
        if action in ("peek_watch", "drop_watch", "move_watch"):
            return getattr(self.highlighted_child, "data_watch", None) is not None
        return True

    def action_peek_watch(self) -> None:
        self.app.peek_selected_watch()

    def action_drop_watch(self) -> None:
        self.app.drop_selected_watch()

    def action_move_watch(self, delta: int) -> None:
        self.app.move_selected_watch(delta)


class WatchersPanel(ColumnPanel):
    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield WatchersList(id="watchers-list")


class HpcaApp(App):
    TITLE = "HPCA"
    # Textual's command palette moves off ctrl+p (reserved — see the hotkey note
    # above BINDINGS) to a bare "p". check_action gates it to the sessions
    # column, the one place no typing happens, so "p" falls through as a letter
    # everywhere else.
    COMMAND_PALETTE_BINDING = "p"

    CSS = """
    #top-bar {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
    }
    #columns {
        height: 1fr;
        /* Narrower than this the columns keep these sizes and the terminal
           clips them. Without it the chat column — the one without a
           min-width of its own — is squeezed to nothing, and a bordered
           widget with zero content width crashes Rich's text wrapping
           (a resize to ~50 columns used to take the whole app down).
           A per-column min-width does not work: when it binds, every column
           balloons past the terminal instead. */
        min-width: 74;
    }
    ColumnPanel {
        border: solid $panel;
    }
    ColumnPanel:focus-within {
        border: heavy $accent;
    }
    .column-title {
        height: 1;
        text-style: bold;
        text-align: center;
        background: $boost;
    }
    #sessions {
        width: 1fr;
        min-width: 20;
    }
    #chat {
        width: 2fr;
    }
    #chat-list {
        height: 1fr;
    }
    /* Grows with the draft, then scrolls: a long message stays readable
       without ever crowding out the conversation above it. */
    #chat-input {
        height: auto;
        max-height: 10;
        border: round $panel;
    }
    #chat-input:focus {
        border: round $accent;
    }
    #watchers {
        width: 1fr;
        min-width: 24;
    }
    /* Each entry is a titled box; who is speaking is a colour, not a prefix. */
    #chat-list > ListItem {
        background: transparent;
    }
    .chat-user {
        border: round $accent;
        color: $text;
    }
    .chat-assistant {
        border: round $success;
        color: $text;
    }
    .chat-thinking {
        border: round $panel-lighten-2;
        color: $text-muted;
    }
    /* A single revealed part of an expanded thinking box: no frame and inset,
       so the rows read as belonging under the box above them. */
    .chat-step {
        color: $text-muted;
        margin-left: 2;
        padding: 0 1;
    }
    /* The call rows carry the script and the command; a rule down their left
       edge picks them out of a long turn without framing them like a message. */
    .chat-step-call {
        border-left: outer $warning;
    }
    .chat-error {
        border: round $error;
        color: $error;
    }
    .chat-event {
        border: round $warning;
        color: $text-muted;
    }
    .chat-queued {
        border: round $panel-lighten-2;
        color: $text-muted;
    }
    .chat-recall {
        border: round $panel-lighten-2;
        color: $text-muted;
    }
    /* Something the app did to the session itself (a /compact fold), not part
       of the conversation. */
    .chat-notice {
        border: round $panel-lighten-2;
        color: $text-muted;
    }
    /* A watch is a box, not a row: it is the whole content of this column now,
       and the frame is what keeps several of them legible as separate things
       in a narrow width. The border colour carries the state, so the column
       reads at a glance. */
    .watch {
        border: round $panel;
        border-title-color: $text;
        padding: 0 1;
        height: auto;
    }
    .watch-live { border: round $success; }
    /* Neutral, not amber. Every existing log box lands here now that a log is
       no longer labelled writing-or-idle, and a warning colour on all of them
       would make the same unsupported claim the words used to: that the age of
       a file's last write says whether the work behind it is still alive. */
    .watch-idle { border: round $panel; }
    .watch-done { border: round $accent; }
    .watch-dead {
        border: round $error;
        color: $text-muted;
    }
    .chat-working {
        color: $text-muted;
        padding: 0 1;
    }
    #command-menu {
        height: auto;
        border: round $accent;
        border-title-color: $accent;
        /* Not muted: the menu carries its own distinction in weight (a
           built-in's name is bold), so the text itself sits at full strength. */
        color: $text;
        padding: 0 1;
    }
    /* A reply landed in a session the user has left: frame it, never
       force it open. Cleared when the session is opened. */
    #sessions-list > ListItem.session-updated {
        border: round $success;
    }
    /* A decision is waiting in this session: frame it in
       warning and prefix its label with "!" (deliverable 2). Distinct from
       session-updated and, unlike it, held until the decision is answered —
       opening the session reveals the prompt but does not resolve it. */
    #sessions-list > ListItem.session-pending {
        border: round $warning;
    }
    /* A turn is in flight for this session (stage 3 / decision 6): frame it in
       accent and prefix its label with the working glyph, so an OFF-SCREEN
       session that is still working is visible. Listed LAST so, of equal CSS
       specificity, a live turn's accent border wins over a stale
       session-updated success frame while the turn runs; it clears (revealing
       any success frame) the moment the turn ends. */
    #sessions-list > ListItem.session-working {
        border: round $accent;
    }
    """

    # RESERVED HOTKEYS — do NOT bind these anywhere in the TUI (they are eaten
    # or made unreliable by terminals, zellij/tmux, or the flow-control layer):
    #   ctrl+[q p t n h s o g]  and  alt+[n f  ← ↑ → ↓  + -]
    # ctrl+s is terminal XOFF (freezes output), ctrl+q is XON/zellij, ctrl+p is
    # a common multiplexer prefix. Prefer bare letters (gated via check_action to
    # a non-typing column) or safe ctrl combos (l, e, r, …). Keep this list in
    # sync with the note in project.md.
    BINDINGS = [
        # (q) quit first, so the footer shows it leftmost (§ esc/quit ordering).
        # Not priority: (q) must reach the chat entry as a letter. Offered only
        # on the sessions column, where no typing happens (check_action).
        Binding("q", "confirm_quit", "quit"),
        Binding("left", "focus_column(-1)", "◀ column", show=False),
        Binding("right", "focus_column(1)", "column ▶", show=False),
        Binding("c", "open_settings", "config editor"),
        Binding("m", "manage_llms", "manage llms"),
        Binding("a", "manage_profiles", "profiles & learnings"),
        Binding("ctrl+l", "switch_llm", "switch llm"),
        Binding("ctrl+e", "edit_profile", "edit profile", show=False),
        # Priority so it also fires while the chat entry is focused. ctrl+m is
        # bound as asked, but most terminals send it as Enter (carriage
        # return), so shift+tab is the binding that works everywhere.
        Binding("shift+tab", "cycle_mode", "agent mode", priority=True),
        Binding("ctrl+m", "cycle_mode", "agent mode", show=False),
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: Any = None,
        tools: ToolRegistry | None = None,
        slurm: SlurmClient | None = None,
        profile: str = "default",
    ) -> None:
        super().__init__()
        self.settings = settings or Settings.load()
        self.profile = profile
        self._llm = llm
        self._owns_llm = llm is None
        # Per-session LLM clients, built lazily from each session's stored
        # backend and keyed by base_url||model so sessions on the same backend
        # share one. Closed en masse on unmount.
        self._session_clients: dict[str, LLMClient] = {}
        self.slurm = slurm or self._detect_slurm()
        self.skills = load_skills(profile)
        if tools is not None:
            self._tools = tools
        else:
            self._tools = add_ask_docs(
                add_doc_tools(add_file_tools(default_tool_registry()))
            )
            if self.slurm is not None:
                add_job_tools(self._tools)
            # Watching is useful with or without Slurm: watch_job needs a
            # cluster, but watch_log is just a file, and half the point is
            # seeing a log that a hand-submitted job is writing.
            add_watch_tools(self._tools)
            # Always registered: HPCA ships skills of its own, so read_skill
            # has something to fetch on any install and in any profile.
            add_skill_tools(self._tools)
            add_memory_tools(self._tools)
            add_plan_tool(self._tools)
        self.active_session: Session | None = None
        self._tool_ctx: ToolContext | None = None
        self._chat_entries: list[Entry] = []
        # Which thinking boxes (and which of their parts) the user has opened,
        # keyed by the Entry's identity so the expansion survives a chat
        # re-render but resets when a session is reloaded and fresh entries are
        # built. See _ThinkingExpansion / _entry_items.
        self._thinking_expanded: dict[int, _ThinkingExpansion] = {}
        # Slash-command autocomplete: the currently-shown (name, usage) matches
        # and which one the ↑/↓ selection is on.
        self._command_matches: list[tuple[str, str]] = []
        self._command_index: int = 0
        self._log: SessionLog | None = None
        # Unsent text per session (session_id → draft), for sessions that are
        # not the one on screen. One entry widget serves every session, so
        # without this a half-written message follows the user into the next
        # session — where it reads as that session's draft and is one Enter
        # away from being sent to the wrong thread. Held for the run only: a
        # draft is a thought in progress, not part of the transcript.
        self._drafts: dict[str, str] = {}
        self._untitled: set[str] = set()  # sessions awaiting their first title
        self._updated: set[str] = set()  # replies that landed while switched away
        # In-flight turns, keyed by session_id. Turns on different sessions run
        # concurrently (per-session backends, separate checkpoint keys); a
        # session "has a turn in flight" ⇔ its id is a key here. Replaces the
        # old app-wide singletons (_busy_turn/_turn_worker/_turn_ctx/…) so two
        # live turns never read each other's client, context or interrupt state.
        self._turns: dict[str, TurnState] = {}
        # Sessions busy with a silent backend call (e.g. an off-screen retitle)
        # that has no TurnState — a second source for the sidebar working glyph.
        self._busy_sessions: set[str] = set()
        self._interrupt_worker = None  # the rollback worker after a 3s hold
        # Work waiting for a session's orchestrator: messages the user typed
        # while that session's turn was running, and background completions
        # reporting in. Same-session turns still serialise — two on one
        # thread_id would interleave checkpoint writes — so anything arriving
        # for a busy session waits here; other sessions start straight away.
        self._pending_work: list[PendingWork] = []
        # Sessions whose thread is parked on a destructive-op approval. Their
        # queued messages wait for the resume; other sessions are unaffected.
        self._awaiting_approval: set[str] = set()
        # The inline decision each session is waiting on, keyed by session_id:
        # {"kind": "approval", "payload": ..., "stage": ..., "reason": ...}.
        # Drives both the inline DecisionBar (shown only for the active
        # session) and the sidebar "!" (shown for every session with one), so
        # switching sessions reveals or hides the right prompt without losing a
        # decision left behind. "stage"/"reason" carry the refusal in progress
        # — which half of the prompt is up, and the words typed into it so far
        # — for the same reason drafts are parked: one bar serves every
        # session, so what is on screen has to be restorable per session.
        self._pending_decision: dict[str, dict] = {}
        self._shutting_down = False
        # Frozen per-session memory views (redesign Phase 1): one snapshot per
        # profile, reused across turns so the system-prompt prefix stays
        # byte-stable for the backend's prefix cache. Refreshed on approved
        # writes and profile edits, never silently mid-session.
        self._memory_snapshots: dict[str, Profile] = {}
        # Memory the agent flagged mid-conversation with the `memory` tool,
        # per session. Nothing is written until the user runs /conclude, which
        # reviews these together with the self-review proposals and clears them.
        self._pending_memory: dict[str, list[MemoryOp]] = {}
        # Context meter state: the window the backend reports, and the last
        # measured prompt size (per session — a different thread is a
        # different context).
        self._discovered_window: int | None = None
        # The last measured prompt size per session (thread_id → prompt_tokens).
        # A background turn updates its own entry silently; the meter on screen
        # always reflects the active session's number (decision 9), so switching
        # to a session that ran off-screen shows its current fill, not a stale
        # zero or another session's count.
        self._context_used: dict[str, int] = {}
        # Last decision's generation speed per session (completion tokens over
        # request wall time) — kept alongside the context number and shown on
        # the same bar, since both describe "the last model call".
        self._turn_speed: dict[str, float] = {}
        self._refreshing_watchers = False
        # The right column's row keys as last painted; identical keys mean the
        # repaint can update text in place instead of rebuilding under the
        # user's cursor. See _paint_panel.
        self._panel_keys: list[str] | None = None
        self._conn = None
        # Sync sqlite off the event loop (NFS homes make each call slow
        # enough to eat keystrokes); the periodic pollers go through this.
        self._dbio: DbIO | None = None
        # ...and run the databases themselves on node-local disk, so each of
        # those calls is fast in the first place. DbIO decides who waits,
        # DbCache how long. Built in on_mount, before anything opens a
        # database, because it is what decides where the databases are.
        self._dbcache: DbCache | None = None
        self._syncing_db_cache = False
        self._db_sync_incomplete = False
        self._saver_ctx = None
        # Event-loop lag, measured (specs-core-process.md §8). The case for
        # moving the agent into its own process is that synchronous work on
        # this loop makes the UI stutter, and that has to be a number taken
        # before and after, not a feeling. Off unless $HPCA_LOOPLAG is set:
        # a disabled probe starts no task, so it costs nothing to leave here.
        self._looplag = LoopLagProbe(
            enabled=bool(os_environ.get("HPCA_LOOPLAG")),
            # What the app thinks it is doing when a stall is recorded. Read
            # off the active turn rather than passed in, so a spike in the log
            # names the tool that caused it.
            label=self._current_activity,
        )

    def _current_activity(self) -> str:
        """What to blame a loop stall on, for the lag probe's spike list.

        The turn's own activity string — "running read_file", "LLM processing"
        — is already what the spinner says, so a spike in looplag.log names the
        step that caused it. Reads any live turn, not just the visible one: a
        background turn blocks this loop exactly as hard as the open one.
        """
        turns = getattr(self, "_turns", {})
        if not turns:
            return "idle"
        return ", ".join(sorted({ts.activity for ts in turns.values()}))

    def notify(self, message: str, **kwargs) -> None:
        """Toasts carry dynamic text — LLM output, exception strings, memory
        excerpts — that may contain Textual markup (``[...]``, ``$(...)``).
        Rendering that as markup raises ``MarkupError`` deep inside the
        compositor and takes the whole app down, so default ``markup=False``.
        A caller that genuinely wants markup can still pass ``markup=True``."""
        kwargs.setdefault("markup", False)
        super().notify(message, **kwargs)

    def compose(self) -> ComposeResult:
        yield TopBar()
        with Horizontal(id="columns"):
            yield SessionsPanel("Sessions", id="sessions")
            yield ChatPanel("Chat", id="chat")
            yield WatchersPanel("Watchers", id="watchers")
        yield Footer()

    async def on_mount(self) -> None:
        # Before the DbCache, the graph and the registry are built: that
        # startup path is itself one of the suspects, and a probe started
        # after it would be measuring everything except.
        self._looplag.start()
        self.clipboard_manager = ClipboardManager(
            self.settings.clipboard, emit=self._emit_to_terminal
        )
        self._dbcache = self._open_db_cache()
        self._conn = connect(self._dbcache.path_for("hpca.db"))
        init_db(self._conn)
        self._dbio = DbIO(self._dbcache.path_for("hpca.db"))
        # Processes the previous run was watching when it exited would claim
        # to be running forever; harmless while the panel was empty on
        # restart, a standing lie now that it shows history.
        reconcile_orphans(self._conn)
        self.session_store = SessionStore(self._conn)
        self.episodic = EpisodicStore(self._conn)
        self.memory_index = MemoryIndex(self._conn)
        self._saver_ctx = AsyncSqliteSaver.from_conn_string(
            str(self._dbcache.path_for("checkpoints.db"))
        )
        checkpointer = await self._saver_ctx.__aenter__()
        self._checkpointer = checkpointer  # kept for graph rebuilds on switch
        if self._llm is None:
            self._llm = LLMClient(self.settings.llm)
        self.profile_memory = Profile.load(self.profile)
        self._memory_snapshots[self.profile] = self.profile_memory
        if self.profile_memory.problems:
            self.notify(
                "Profile file has problems: "
                + "; ".join(self.profile_memory.problems[:3]),
                severity="warning",
            )
        self._rebuild_graph()
        self.job_store = JobStore(self._conn)
        self.watch_store = WatchStore(self._conn)
        self.symbol_index = SymbolIndex(self._conn)
        self.rag_store = RagStore(self._dbcache.path_for("rag.db"))
        self.embedder = EmbeddingClient(
            base_url=self.settings.rag.embedding_base_url,
            model=self.settings.rag.embedding,
        )
        self.trash = TrashManager(
            app_dir() / "trash",
            backup_limit_bytes=int(self.settings.safety.backup_limit_gb * 1024**3),
        )
        removed = self.trash.cleanup(self.settings.safety.trash_ttl_days)
        if removed:
            self.notify(f"Trash: cleaned up {removed} expired entr"
                        f"{'y' if removed == 1 else 'ies'}")
        self._refresh_top_bar()
        self._refresh_model_line()
        self._refresh_context_bar()
        self.run_worker(self._discover_context_window(), group="llm-probe")
        # Pre-agent auto-connect: on the cluster, wire up whatever vLLM servers
        # are live (spec §4). Best-effort and off the critical path.
        self.run_worker(self._auto_connect_cluster(), group="auto-connect")
        await self._reload_sessions()
        self.run_curator_if_due()
        self.set_interval(2.0, self.refresh_watchers)
        self.set_interval(2.0, self.watch_processes)
        self.set_interval(LOG_WATCH_SECONDS, self.poll_watched_logs)
        sync_interval = self.settings.database.sync_interval_s
        if self._dbcache.active and sync_interval > 0:
            self.set_interval(sync_interval, self._sync_db_cache)
        if self.slurm is not None:
            self.set_interval(
                max(5, self.settings.cluster.job_poll_seconds), self.poll_jobs
            )
            self.set_interval(JOB_WATCH_SECONDS, self.poll_watched_jobs)
        self._focus_column("sessions")

    # ------------------------------------------------------------- db cache

    def _open_db_cache(self) -> DbCache:
        """Decide where the databases run, before any of them is opened.

        Local mode is the fast path; declining it (disabled in settings,
        another instance holding the lease, an unusable working dir) means
        running straight from home, which is slower but always correct — so
        the only thing to do about it is say so.
        """
        _dbcache_logger()  # before acquire(): recovery logs through it
        cache = DbCache(
            app_dir(),
            local_dir=(
                local_dir_for(app_dir(), configured=self.settings.database.local_dir)
            ),
            enabled=self.settings.database.local_cache,
        )
        if not cache.acquire() and self.settings.database.local_cache:
            self.notify(f"Databases: {cache.reason}", severity="warning")
        for warning in cache.drain_warnings():
            self.notify(f"Databases: {warning}", severity="warning")
        return cache

    async def _sync_db_cache(self) -> None:
        """Write the node-local databases back to home (the periodic tick).

        Skipped while a previous sync is still running: on NFS one can outlast
        its interval, and two backups of the same file at once is pointless
        work. Failures are reported and retried on the next tick — home being
        briefly unreachable must not take the app down.
        """
        cache = self._dbcache
        if cache is None or not cache.active or self._syncing_db_cache:
            return
        self._syncing_db_cache = True
        try:
            complete = await asyncio.to_thread(cache.sync)
            for warning in cache.drain_warnings():
                self.notify(f"Databases: {warning}", severity="warning")
            # Say it once when syncing starts failing, not on every tick: a
            # database that cannot reach home for long means the app dir's
            # copy is quietly falling behind, and dbcache.log alone proved
            # too easy to miss.
            if not complete and not self._db_sync_incomplete:
                self.notify(
                    "Database sync to home is failing; see dbcache.log",
                    severity="warning",
                )
            self._db_sync_incomplete = not complete
        except Exception as e:
            self.notify(f"Database sync to home failed: {e}", severity="warning")
        finally:
            self._syncing_db_cache = False

    def _detect_slurm(self) -> SlurmClient | None:
        """Job tools are available when sbatch exists or a submit host is set."""
        submit_host = self.settings.cluster.submit_host
        if submit_host or shutil.which("sbatch"):
            return SlurmClient(submit_host=submit_host)
        return None

    async def on_unmount(self) -> None:
        # Quitting must not kick off whatever was still queued: the database
        # and checkpointer are about to close under it.
        self._shutting_down = True
        self._pending_work.clear()
        if self._saver_ctx is not None:
            await self._saver_ctx.__aexit__(None, None, None)
        if self._dbio is not None:
            await self._dbio.close()
        if self._conn is not None:
            self._conn.close()
        if getattr(self, "rag_store", None) is not None:
            self.rag_store.close()
        # Last, and only once every connection above is closed: the final sync
        # should copy a quiesced database, and nothing may open a local file
        # after the working dir is removed.
        if self._dbcache is not None:
            await asyncio.to_thread(self._dbcache.release)
        if getattr(self, "embedder", None) is not None:
            await self.embedder.close()
        if self._owns_llm and self._llm is not None:
            await self._llm.close()
        for client in self._session_clients.values():
            await client.close()
        # Last: the report is written before the probe stops, because `elapsed`
        # freezes at stop and the block would otherwise claim a shorter run
        # than it measured. A disabled probe writes nothing and returns False,
        # so this needs no guard of its own.
        self._looplag.write_report(
            app_dir() / "looplag.log",
            note=os_environ.get("HPCA_LOOPLAG", ""),
        )
        await self._looplag.stop()

    # ------------------------------------------------------------- clipboard

    def _emit_to_terminal(self, sequence: str) -> None:
        """Write a raw escape sequence through the driver (bypasses compositor)."""
        driver = self._driver
        if driver is None:
            raise RuntimeError("no driver")
        driver.write(sequence)

    def copy_text(self, text: str) -> CopyResult:
        """Copy text via the tiered ClipboardManager and toast the outcome."""
        result = self.clipboard_manager.copy(text)
        self.notify(result.message, severity="information" if result.ok else "error")
        return result

    def copy_to_clipboard(self, text: str) -> None:
        """Textual's own copy path — ctrl+c on a marked selection, and the
        chat entry's copy — routed through the tiered manager. Textual emits
        OSC 52 only, which tmux and screen swallow without passthrough, and
        this agent is normally reached through both (§ clipboard).
        """
        if getattr(self, "clipboard_manager", None) is None:
            super().copy_to_clipboard(text)  # before on_mount; nothing to tier with
            return
        self.copy_text(text)

    def _memory_snapshot(self, profile: str) -> Profile:
        """The frozen memory view for a profile (loaded once, reused across
        turns). See ``_memory_snapshots`` for why it is not reloaded per turn.

        Loading is also when the tier-3 index is rebuilt: the markdown file is
        the source of truth, and it may have been edited by hand since.
        """
        snapshot = self._memory_snapshots.get(profile)
        if snapshot is None:
            snapshot = Profile.load(profile)
            self._memory_snapshots[profile] = snapshot
            index = getattr(self, "memory_index", None)
            if index is not None:
                try:
                    index.reindex(snapshot)
                except Exception:
                    pass  # retrieval is an optimization; never block a turn
        return snapshot

    def _refresh_memory_snapshot(self, profile: str) -> None:
        """Drop a profile's frozen view; the next turn reloads from disk and
        rebuilds the tier-3 index.

        Called after approved writes and profile edits — the deliberate
        refresh points where invalidating the backend's prefix cache is
        worth it."""
        self._memory_snapshots.pop(profile, None)

    def _render_system_prompt(self, session_id: str | None = None) -> str:
        """Per-call prompt assembly (§4.3): memories, skills, dynamic facts.

        Rendered for the running session (``session_id``), so a background
        turn reads its own session's profile memories and tags with its own
        model, not the profile/model on screen. Falls back to the on-screen
        defaults for the session with no live turn."""
        ts = self._turns.get(session_id) if session_id is not None else None
        if ts is not None:
            memory = ts.memory if ts.memory is not None else (
                self._memory_snapshot(ts.session.profile)
            )
            skills = ts.skills if ts.skills is not None else self.skills
            backend = self._model_of(ts.session)
        else:
            memory = self._memory_snapshot(self.profile)
            skills = self.skills
            backend = self._active_model()
        cap = self.settings.memory.system_prompt_token_cap
        return orchestrator_system_prompt(
            system_prompt_memories=memory.system_prompt_text(active_backend=backend),
            memory_meter=memory.usage_meter(cap),
            # The skill list is deliberately kept out of the prompt; the model
            # is only told skills exist and reaches one via read_skill (or gets
            # it inline when the user invokes "/<skill>").
            has_skills=bool(skills),
            session_search="session_search" in self._tools.names(),
            memory_tool="memory" in self._tools.names(),
            watch_tools="watch_log" in self._tools.names(),
        )

    def _recall_lines(
        self, user_text: str, memory: Profile, profile: str
    ) -> list[str]:
        """What this request recalls: matching struggle notes plus retrieved
        RAG memories.

        RAG is retrieved rather than injected wholesale, which is what lets it
        grow: situational memories cost context only on the turns they actually
        match.
        """
        lines = [note_line(m) for m in matching_struggles(memory.memories, user_text)]
        seen = set(lines)
        index = getattr(self, "memory_index", None)
        if index is not None:
            try:
                hits = index.search(
                    user_text,
                    profile=profile,
                    active_backend=self._active_model(),
                    limit=self.settings.memory.rag_prefetch_count,
                )
            except Exception:
                hits = []
            for hit in hits:
                line = retrieved_line(hit)
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
        return lines

    def warn_about_struggles(self, text: str) -> list:
        """§4.4: warn the *user* up front when a request matches a past
        struggle. The model gets its own copy as a fenced memory-context
        block when the turn starts (``_run_agent``)."""
        matches = matching_struggles(self.profile_memory.memories, text)
        for memory in matches:
            first_line = memory.text.splitlines()[0]
            self.notify(
                f"I have struggled with this before: {first_line}",
                severity="warning",
                timeout=10,
            )
        return matches

    # ------------------------------------------------------------ chat/agent

    @on(ChatInput.Submitted)
    async def _on_chat_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return
        forced_skill: "Skill | None" = None
        if text.startswith(("\\", "/")):
            # A "/<skill>" invocation is a real turn (queued/concurrent like any
            # message), so it falls through to the turn machinery below carrying
            # the skill. A built-in slash command instead acts on the UI and
            # runs its own exclusive worker: not a turn, not queued, and refused
            # only while THIS session's own turn is running (a background turn
            # in another session leaves the open session free to run one).
            forced_skill = self._slash_skill(text)
            if forced_skill is None:
                command = text[1:].partition(" ")[0]
                if command not in {name for name, _ in COMMANDS}:
                    # Neither a built-in nor a skill — almost always a typo, so
                    # leave the draft standing: the user fixes the spelling (or
                    # reopens the menu to look the command up) instead of having
                    # to type the whole thing again.
                    self.notify(f"Unknown command: /{command}", severity="warning")
                    return
                active_busy = (
                    self.active_session is not None
                    and self.active_session.session_id in self._turns
                )
                if active_busy:
                    self.notify(
                        f"Still working in “{self.active_session.title}” — "
                        "commands wait for that reply.",
                        severity="warning",
                    )
                    return
                event.chat_input.text = ""
                self._handle_slash_command(text)
                return
        event.chat_input.text = ""
        if self.active_session is None:
            await self.start_new_session()
        if self.active_session.title == UNTITLED_SESSION:
            self._name_session(text[:SESSION_TITLE_MAX])
            await self._reload_sessions()
        self.warn_about_struggles(text)
        # The message is accepted either way; only its turn may have to wait.
        # It goes in the transcript now so typing ahead looks like it worked.
        # Queued only when THIS session already has a turn in flight — a turn
        # busy in another session no longer blocks this one (concurrent turns).
        queued = self.active_session.session_id in self._turns
        await self._append_chat("queued" if queued else "user", text)
        self._pending_work.append(
            PendingWork(
                session_id=self.active_session.session_id,
                text=text,
                kind="user",
                forced_skill=forced_skill,
            )
        )
        await self.drain_work()

    @on(TextArea.Changed, "#chat-input")
    def _on_draft_changed(self, event: TextArea.Changed) -> None:
        self._update_command_menu(event.text_area.text)

    def _update_command_menu(self, draft: str) -> None:
        """List the chat commands above the entry while one is being *named*:
        substring match (so "/skill" finds every command with "skill" in it),
        most-used first, and ↑/↓ selectable (see ChatInput).

        Only while it is being named. Whitespace after the name means the
        command is settled and what follows is its arguments — or, after a
        shift+enter, the body of a multi-line message. The menu has nothing
        left to offer there, and an open menu owns ↑/↓: leaving it up made
        every draft that opens with "/" untraversable for as long as it was
        being written, which is exactly when those keys are wanted.
        """
        menu = self.query_one("#command-menu", Static)
        stripped = draft.lstrip()
        typed = stripped[1:]
        if not stripped.startswith(("/", "\\")) or any(c.isspace() for c in typed):
            menu.display = False
            self._command_matches = []
            return
        matches = self._matching_commands(typed)
        if not matches:  # typed something that matches no command: show nothing
            menu.display = False
            self._command_matches = []
            return
        # Keep the highlight on the same command across keystrokes when it is
        # still in the list; otherwise start at the top (the most-used match).
        previous = (
            self._command_matches[self._command_index][0]
            if 0 <= self._command_index < len(self._command_matches)
            else None
        )
        self._command_matches = matches
        self._command_index = next(
            (i for i, (name, _) in enumerate(matches) if name == previous), 0
        )
        menu.border_title = (
            "commands  (bold = built-in · ↑/↓ select · ⇥ complete)"
        )
        self._render_command_menu()
        menu.display = True

    def _all_commands(self) -> list[tuple[str, str]]:
        """Built-in chat commands plus the on-screen profile's skills, so both
        show in the "/" menu and tab-complete. Built-ins win a name clash. A
        skill whose name has whitespace is left out: the slash parser splits on
        the first space, so it could never be selected as "/<skill>" anyway."""
        builtin = {name for name, _ in COMMANDS}
        skill_cmds = [
            (
                s.name,
                f"/{s.name} — {s.description}" if s.description else f"/{s.name}",
            )
            for s in self.skills
            if s.name
            and s.name not in builtin
            and not any(ch.isspace() for ch in s.name)
        ]
        return list(COMMANDS) + skill_cmds

    def _slash_skill(self, text: str) -> "Skill | None":
        """The visible skill a leading "/<token>" names, or None. Built-in
        commands win a name clash, and a skill is reachable this way only when
        its name is a single token (the parser splits on the first space)."""
        token = text[1:].partition(" ")[0]
        if not token or token in {name for name, _ in COMMANDS}:
            return None
        return next((s for s in self.skills if s.name == token), None)

    def _matching_commands(self, typed: str) -> list[tuple[str, str]]:
        """(name, usage) pairs whose name contains ``typed``, most-used first
        then in definition order. Empty ``typed`` matches everything."""
        needle = typed.lower()
        commands = self._all_commands()
        order = {name: i for i, (name, _) in enumerate(commands)}
        counts = self._command_counts()
        matches = [(n, u) for n, u in commands if needle in n.lower()]
        matches.sort(key=lambda nu: (-counts.get(nu[0], 0), order[nu[0]]))
        return matches

    def _command_counts(self) -> dict[str, int]:
        if self._conn is None:
            return {}
        try:
            return command_use_counts(self._conn)
        except Exception:
            return {}

    def _render_command_menu(self) -> None:
        """Draw the match list, marking which entries are HPCA's own.

        The menu mixes two things the user cannot otherwise tell apart: HPCA's
        built-in commands and the profile's skills. Only a built-in's ``/name``
        is marked — bolding the description too would make half the menu shout
        and bury the names it is there to help pick between, and skills need no
        mark of their own once the built-ins carry one.

        Assembled as styled spans rather than markup: a skill's description is
        user-written, and a markup parse would eat its brackets.
        """
        menu = self.query_one("#command-menu", Static)
        builtin = {name for name, _ in COMMANDS}
        parts: list = []
        for i, (name, usage) in enumerate(self._command_matches):
            if i:
                parts.append("\n")
            parts.append("▶ " if i == self._command_index else "  ")
            if name not in builtin:
                parts.append(usage)
                continue
            head = f"/{name}"
            if usage.startswith(head):
                parts.append((head, BUILTIN_COMMAND_STYLE))
                parts.append(usage[len(head):])
            else:  # a usage string that does not open with its own name
                parts.append((usage, BUILTIN_COMMAND_STYLE))
        menu.update(Content.assemble(*parts))

    def command_menu_active(self) -> bool:
        """Whether the autocomplete menu is showing selectable matches."""
        return bool(self._command_matches)

    def command_menu_move(self, delta: int) -> None:
        if not self._command_matches:
            return
        self._command_index = (self._command_index + delta) % len(
            self._command_matches
        )
        self._render_command_menu()

    def command_menu_accept(self) -> bool:
        """Fill the entry with the highlighted command so arguments can be
        added. Returns False (let Enter submit) when it is already fully typed.
        """
        if not self._command_matches:
            return False
        name = self._command_matches[self._command_index][0]
        chat_input = self.query_one("#chat-input", ChatInput)
        stripped = chat_input.text.lstrip()
        typed = stripped[1:].split(maxsplit=1)[0] if stripped[1:] else ""
        if typed == name:
            return False  # already complete: Enter runs it
        chat_input.text = f"/{name} "
        chat_input.move_cursor(chat_input.document.end)
        return True

    # ------------------------------------------------- interrupt the model

    def _active_turn(self) -> "TurnState | None":
        """The TurnState of the session on screen, if it has a turn in flight.
        Interrupt and the visible spinner both target only this — a background
        turn in another session is never interruptible from here (decision 8)."""
        if self.active_session is None:
            return None
        return self._turns.get(self.active_session.session_id)

    def _can_interrupt(self) -> bool:
        """Only while the active session's own user turn is parked on the LLM —
        the one phase where telling the backend to stop makes sense, and the
        only case with a prompt to hand back."""
        ts = self._active_turn()
        return (
            ts is not None
            and ts.activity == LLM_WAIT_ACTIVITY
            and ts.user_text is not None
            and ts.interrupt_keep is not None
        )

    def _maybe_interrupt_llm(self) -> None:
        """Selecting the working indicator while the turn waits on the model
        offers to abort it. The confirm dialog can sit open long enough for the
        reply to land, so re-check that an interrupt is still possible when it
        returns before firing (callback form: a message handler is not a
        worker, so push_screen_wait is unavailable here)."""
        if not self._can_interrupt():
            return

        def resolved(confirmed: bool | None) -> None:
            if confirmed and self._can_interrupt():
                self._fire_interrupt()

        self.push_screen(
            ConfirmScreen("Interrupt the LLM and re-edit your last message?"),
            resolved,
        )

    def _fire_interrupt(self) -> None:
        """Capture what the interrupt needs and roll the turn back off the main
        loop — the message comes back to the entry for editing. Operates on the
        active session's turn only (decision 8)."""
        ts = self._active_turn()
        if ts is None or ts.interrupt_keep is None or ts.user_text is None:
            return
        self._interrupt_worker = self.run_worker(
            self._interrupt_turn(
                ts.session, ts.interrupt_keep, ts.user_text, ts.worker
            ),
            group="interrupt",
        )

    async def _interrupt_turn(self, session, keep, text, worker) -> None:
        """Abort the in-flight request, drop the aborted turn from the thread,
        and hand the message back to the entry for editing."""
        if worker is not None:
            worker.cancel()
            try:
                await worker.wait()  # let the cancellation unwind before we edit
            except (Exception, asyncio.CancelledError):
                pass
        self._turns.pop(session.session_id, None)
        self._refresh_session_row(session.session_id)  # clear the working marker
        self.hide_working()
        try:
            surviving = await rollback_thread(
                self.graph, session_id=session.session_id, keep=keep
            )
        except Exception as e:
            self.notify(f"Interrupt cleanup failed: {e}", severity="error")
            surviving = None
        if not self._is_active_session(session):
            # Switched away while the rollback ran: the message belongs to the
            # session it was typed in, so park it as that session's draft
            # rather than dropping it into whichever entry is on screen now.
            self._store_draft(session.session_id, text)
            self.notify(
                f"Interrupted “{session.title}” — the message is waiting there."
            )
            self.call_later(self.drain_work)
            return
        if surviving is not None:
            await self._set_chat_messages(surviving)
        found = self.query("#chat-input")
        if found:  # absent only if the screen is tearing down
            chat_input = found.first(ChatInput)
            chat_input.text = text
            chat_input.move_cursor(chat_input.document.end)
            self.focus_chat_input()
            self.notify("Interrupted — edit your message and send again.")
        self.call_later(self.drain_work)

    def _run_agent(
        self,
        session: Session,
        *,
        user_text: str | None = None,
        resume: Command | None = None,
        forced_skill: "Skill | None" = None,
    ):
        """Start a turn for one session; it stays that session's turn even if
        the user switches away while the model works. Everything the turn
        touches — tool context, transcript log — is captured here, not read
        from whatever session happens to be open when the reply lands.
        """
        # The turn reads its own session's profile snapshot — an approval may
        # resume it while a differently-profiled session is open on screen.
        # The snapshot is frozen per session (not reloaded per turn) so the
        # prompt prefix stays cacheable; approved writes refresh it.
        turn_memory = self._memory_snapshot(session.profile)
        if session.profile == self.profile:
            self.profile_memory = turn_memory
        turn_skills = load_skills(session.profile)
        # Recalled memory and the volatile date/time both ride on the API copy
        # of the user message — the model is warned in-context, the stored
        # transcript stays clean, and (unlike the system prompt) the tail is
        # where changing content belongs so the cacheable prefix survives.
        api_content = None
        if user_text is not None:
            # A "/<skill>" invocation: the model works from the request alone
            # (the "/<skill>" prefix stripped off) plus the skill's procedure on
            # the sidecar. Recall runs on the request too, not the command word.
            if forced_skill is not None:
                request = user_text[1:].partition(" ")[2].strip()
                directive = build_skill_directive(
                    forced_skill.name,
                    forced_skill.description,
                    forced_skill.body,
                )
            else:
                request = user_text
                directive = ""
            lines = self._recall_lines(request, turn_memory, session.profile)
            block = build_memory_context(
                lines,
                max_notes=self.settings.memory.rag_prefetch_count,
                max_chars=self.settings.memory.rag_prefetch_chars,
            )
            api_content = compose_api_content(
                request, block, environment_facts(), skill_directive=directive
            )
        log = open_log(self.settings, session)
        # Everything this turn owns lives in its own TurnState, keyed by
        # session, so a concurrent turn in another session never reads it.
        ts = TurnState(
            session=session,
            ctx=self._make_tool_ctx(session, log, skills=turn_skills),
            memory=turn_memory,
            skills=turn_skills,
            # A fresh user message can be interrupted and re-edited (§ interrupt);
            # a resume/event has no prompt to hand back, so it arms nothing.
            user_text=user_text,
            interrupt_keep=None,  # filled once we know the pre-turn count
        )
        self._turns[session.session_id] = ts
        # Light the sidebar in-flight marker — works whether or not this
        # session is on screen (decision 6).
        self._refresh_session_row(session.session_id)
        if self._is_active_session(session):
            # the UI's context (process list, registry) stays the turn's twin
            self._tool_ctx = ts.ctx
            self.show_working()
        ts.worker = self.run_worker(
            self._agent_turn(
                session,
                user_text=user_text,
                resume=resume,
                log=log,
                api_content=api_content,
            ),
            # Per-session group (not exclusive): a session's own new turn still
            # cannot double-run, but a turn in another session is untouched.
            group=f"turn-{session.session_id}",
        )
        return ts.worker

    def _chat_list(self) -> ListView | None:
        """The chat column, or None once the screen is gone.

        A queued turn can start — and finish — while the app is shutting
        down, and neither the spinner appearing nor disappearing is worth
        raising over at that point.
        """
        found = self.query("#chat-list")
        return found.first(ListView) if found else None

    def show_working(self, label: str | None = None) -> None:
        """Put the spinner after the last message: a reply is on its way.

        With no ``label``, seeded from the active session's own turn activity,
        so re-opening a busy session shows what that turn is doing, not a stale
        global. An explicit ``label`` names the step directly — used by silent
        backend calls (e.g. /conclude) that have no TurnState to read from."""
        chat_list = self._chat_list()
        if chat_list is None:
            return
        started = None
        if label is not None:
            activity = label
        else:
            ts = self._active_turn()
            activity = ts.activity if ts is not None else "working"
            started = ts.started if ts is not None else None
        if not chat_list.query(WorkingIndicator):
            chat_list.append(ChatItem(WorkingIndicator(activity, started=started)))
            chat_list.scroll_end(animate=False)

    @asynccontextmanager
    async def _backend_working(self, label: str, *, session: "Session | None"):
        """Feedback around a silent backend (LLM) call. Chat-bottom spinner when
        `session` is the open, turn-free session; otherwise the sidebar row glyph.
        Teardown is guaranteed, so an exception in the call can't strand it."""
        sid = session.session_id if session is not None else None
        use_chat = (
            session is not None
            and self._is_active_session(session)
            and sid not in self._turns
        )
        if use_chat:
            self.show_working(label)
        elif sid is not None:
            self._busy_sessions.add(sid)
            self._refresh_session_row(sid)
        try:
            yield
        finally:
            if use_chat:
                self.hide_working()
            elif sid is not None:
                self._busy_sessions.discard(sid)
                self._refresh_session_row(sid)

    def hide_working(self) -> None:
        chat_list = self._chat_list()
        if chat_list is None:
            return
        for item in list(chat_list.children):
            if item.query(WorkingIndicator):
                item.remove()

    def report_activity(self, session_id: str, activity: str) -> None:
        """What a session's turn is doing right now, for the spinner to say.

        Recorded on that turn's own TurnState (so it survives leaving and
        re-opening the session), but only the visible chat's spinner is
        updated — a background turn never touches a chat the user is not
        looking at (decision 7)."""
        ts = self._turns.get(session_id)
        if ts is not None:
            ts.activity = activity
        if (
            self.active_session is not None
            and self.active_session.session_id == session_id
        ):
            for indicator in self.query(WorkingIndicator):
                indicator.set_activity(activity)

    async def _agent_turn(
        self,
        session: Session,
        *,
        user_text: str | None,
        resume: Command | None,
        log: SessionLog | None,
        api_content: str | None = None,
    ) -> None:
        # The point to roll the thread back to if this turn is interrupted:
        # captured before run_turn appends the user message (§ interrupt).
        if user_text is not None:
            ts = self._turns.get(session.session_id)
            try:
                keep = await thread_message_count(
                    self.graph, session_id=session.session_id
                )
            except Exception:
                keep = None
            if ts is not None:
                ts.interrupt_keep = keep
        try:
            result = await run_turn(
                self.graph,
                session_id=session.session_id,
                user_text=user_text,
                resume=resume,
                api_content=api_content,
            )
        except Exception as e:
            self.hide_working()
            if log is not None:
                log.write("error", str(e))
            if self._is_active_session(session):
                await self._append_chat("error", f"Agent error: {e}")
            else:
                self._mark_session_updated(session)
            self.notify(str(e), severity="error")
            return
        finally:
            # Drop this session's turn wholesale — everything the old finally
            # block cleared lived on the TurnState and goes with it.
            self._turns.pop(session.session_id, None)
            # Clear the sidebar in-flight marker now the turn is gone (works
            # off-screen too; a pending decision, set just below, repaints its
            # own "!").
            self._refresh_session_row(session.session_id)
            # Whatever queued up behind this turn starts as soon as this
            # handler unwinds, rather than waiting for the next timer tick.
            self.call_later(self.drain_work)
        self.hide_working()
        await self._log_turn(result, log, session)
        if self._is_active_session(session):
            await self._set_chat_messages(
                result.messages, result.thinking, result.calls
            )
            await self.refresh_watchers()
        else:
            # The reply belongs to a session the user has left: never yank
            # them back — frame its row in the list instead.
            self._mark_session_updated(session)
        if result.interrupt is not None:
            # Parked on an approval (destructive op, or a gated execution in
            # manual mode): the turn's thread cannot move without an
            # answer. Record it as this session's pending decision — shown
            # inline if the session is open, or only as a sidebar "!" if the
            # user has switched away — never a modal over the other columns.
            self._awaiting_approval.add(session.session_id)
            await self._set_pending_decision(session, "approval", result.interrupt)
            return
        await self.maybe_title_session(session, result.messages, log=log)

    async def maybe_title_session(
        self, session: Session, messages: list[dict], *, log: SessionLog | None = None
    ) -> bool:
        """Name a fresh session after its first exchange, once.

        The opening message is only a placeholder — truncated mid-word, and
        wrong by the time the session has moved on. One attempt: a failure is
        not worth a second wait, and (t) is always there.
        """
        if session.session_id not in self._untitled:
            return False
        self._untitled.discard(session.session_id)
        title = await self._propose_title(messages, log=log, session=session)
        if title is None:
            return False
        self._rename_session(session, title, by="llm", log=log)
        return True

    async def _log_turn(
        self, result, log: SessionLog | None, session: Session
    ) -> None:
        """Write what this turn added into the turn's own transcript — not
        into whichever session is open when the reply lands.

        Only the tail is logged, so re-reading a session never duplicates it.
        The same tail feeds the episodic index (redesign Phase 2): user and
        assistant messages only — tool traffic and thinking would drown BM25
        in tool vocabulary.
        """
        entries = build_entries(
            result.messages, result.thinking, result.calls, start=result.first_new
        )
        if log is not None:
            for entry in entries:
                kind = LOG_KINDS.get(entry.kind, entry.kind)
                if entry.kind == THINKING:
                    kind = f"thinking ({entry.summary()})"
                log.write(kind, entry.text)
        turns = [
            (entry.kind, entry.text)
            for entry in entries
            if entry.kind in (USER_ENTRY, ASSISTANT_ENTRY)
        ]
        session_id = session.session_id
        profile = session.profile

        def _record(conn) -> None:
            # The user may delete the session while this waits in the DB
            # queue; recording then would leave ghost rows in the index.
            if SessionStore(conn).get(session_id) is None:
                return
            EpisodicStore(conn).record(
                session_id=session_id, profile=profile, entries=turns
            )

        try:
            # On the DB thread: the FTS triggers make this the heaviest
            # per-turn write, too slow for the loop on an NFS home.
            await self._db(_record)
        except Exception as e:  # recall is best-effort; never fail the turn
            if log is not None:
                log.write("error", f"episodic index write failed: {e}")

    def _log_write(self, kind: str, text: str) -> None:
        if self._log is not None:
            self._log.write(kind, text)

    def _on_approval(
        self, session: Session, approved: bool | None, reason: str = ""
    ) -> None:
        # Resume the turn on the thread it belongs to — the user may have
        # switched sessions since the prompt appeared.
        self._awaiting_approval.discard(session.session_id)
        self._run_agent(
            session,
            resume=Command(resume={"approved": bool(approved), "reason": reason}),
        )

    # -------------------------------------------------- inline decision prompt

    def _decision_bar(self) -> DecisionBar | None:
        """The inline decision panel, or None once the screen is gone."""
        found = self.query("#decision-bar")
        return found.first(DecisionBar) if found else None

    async def _set_pending_decision(
        self, session: Session, kind: str, payload
    ) -> None:
        """Record a decision this session is waiting on and surface it: inline
        if the session is open, otherwise only as the sidebar "!"."""
        self._pending_decision[session.session_id] = {
            "kind": kind,
            "payload": payload,
            "stage": "ask",  # y/n first; refusing opens the box for why
            "reason": "",
        }
        self._refresh_session_row(session.session_id)  # light the "!"
        if self._is_active_session(session):
            await self._sync_decision_bar()
            # Land on the prompt so its keys work at once — but only if the
            # user is already in the chat column; a decision must never yank
            # focus away from another column (or another session).
            if self.focused_column_id == "chat":
                self._focus_decision_bar()

    async def _sync_decision_bar(self) -> None:
        """Show the active session's pending decision in the inline bar, or
        hide the bar when it has none. The bar only ever shows the open
        session's decision — background ones live in ``_pending_decision`` and
        the sidebar "!" until their session is opened."""
        bar = self._decision_bar()
        if bar is None:
            return
        pending = None
        if self.active_session is not None:
            pending = self._pending_decision.get(self.active_session.session_id)
        if pending is None:
            await bar.clear_decision()
        else:
            await bar.show_approval(
                pending["payload"],
                stage=pending.get("stage", "ask"),
                reason=pending.get("reason", ""),
            )

    def _focus_decision_bar(self) -> None:
        """Focus the inline prompt so its keys reach it — the y/n bar, or the
        box once a refusal is being explained."""
        bar = self._decision_bar()
        if bar is None or not bar.display:
            return
        bar.focus_prompt()

    def _park_reason(self) -> None:
        """Keep a half-written refusal reason under the session being left.

        The same problem as the chat drafts: one bar serves every session, so
        without this the words typed about one session's script are gone the
        moment the user looks at another — and the decision they belong to is
        still sitting there waiting to be answered.
        """
        session = self.active_session
        if session is None:
            return
        pending = self._pending_decision.get(session.session_id)
        if pending is None or pending.get("stage") != "reason":
            return
        bar = self._decision_bar()
        if bar is None:  # the screen is tearing down: nothing to keep
            return
        pending["reason"] = bar.reason_text()

    def begin_decline(self, kind: str) -> None:
        """Move the open session's decision to its second half: the answer is
        "no", and the box now opening collects why.

        Nothing is sent yet — the turn stays parked, so the refusal and the
        reason reach the model together and the model is never told "no" twice.
        """
        session = self.active_session
        if session is None:
            return
        pending = self._pending_decision.get(session.session_id)
        if pending is None or pending["kind"] != kind:
            return  # stale keystroke: the decision changed or is already gone
        pending["stage"] = "reason"
        self.call_later(self._open_reason_box)

    async def _open_reason_box(self) -> None:
        """Re-render the prompt at its reason stage and land in the box — the
        user pressed a key to get here, so the cursor belongs there."""
        await self._sync_decision_bar()
        self._focus_decision_bar()

    def resolve_decision(self, kind: str, value, reason: str = "") -> None:
        """Answer the inline decision the chat column is showing (DecisionBar).

        Only the active session's decision is ever displayed, so that is the
        one resolved. Clear the pending state and the "!", hide the bar, then
        run the very same resume path the modal screens used to — kept
        synchronous so ``_run_agent``'s exclusive turn worker is unchanged.

        ``reason`` is what the user typed into the box when refusing; empty
        for an approval, and for a refusal they chose not to explain."""
        session = self.active_session
        if session is None:
            return
        pending = self._pending_decision.get(session.session_id)
        if pending is None or pending["kind"] != kind:
            return  # stale keystroke: the decision changed or is already gone
        self._pending_decision.pop(session.session_id, None)
        self._refresh_session_row(session.session_id)  # drop the "!"
        self.call_later(self._sync_decision_bar)  # hide the now-empty bar
        self.focus_chat_input()
        self._on_approval(session, value, reason)

    # ------------------------------------------------------------ agent modes

    def _mode_of(self, session: Session | None) -> str:
        """A session's interaction mode (§3.5); the configured default when
        the session never chose one (or there is no session)."""
        if session is None:
            return self.settings.agent.default_mode
        return session.mode or self.settings.agent.default_mode

    def _mode_for_turn(self, session_id: str) -> str:
        """The mode the graph obeys this round for the session running
        ``session_id``. Read fresh from the store so cycling the mode mid-turn
        applies to the very next round instead of a stale Session copy."""
        fresh = self.session_store.get(session_id)
        if fresh is not None:
            return self._mode_of(fresh)
        ts = self._turns.get(session_id)
        return self._mode_of(ts.session if ts is not None else None)

    def action_cycle_mode(self) -> None:
        session = self.active_session
        if session is None:
            return
        mode = next_mode(self._mode_of(session))
        session.mode = mode
        self.session_store.set_mode(session.session_id, mode)
        self._refresh_mode_bar()

    def _mode_bar(self) -> ModeBar | None:
        found = self.query("#mode-bar")
        return found.first(ModeBar) if found else None

    def _refresh_mode_bar(self) -> None:
        """The mode line above the entry: shown with a session, hidden without."""
        bar = self._mode_bar()
        if bar is None:
            return
        if self.active_session is None:
            bar.display = False
            return
        bar.display = True
        bar.set_mode(self._mode_of(self.active_session))

    # ---------------------------------------------------------------- memory

    def _handle_slash_command(self, text: str) -> None:
        command, _, rest = text[1:].partition(" ")
        rest = rest.strip()
        if command in {name for name, _ in COMMANDS} and self._conn is not None:
            # Count recognised commands so the autocomplete lists them by use.
            record_command_use(self._conn, command)
        if command == "memorize":
            if not rest:
                self.notify("Usage: /memorize <note>", severity="warning")
                return
            self.run_worker(self._memorize_worker(rest), exclusive=True)
        elif command == "conclude":
            if self.active_session is None:
                self.notify("No active session to conclude.", severity="warning")
                return
            self.run_worker(self._conclude_worker(), exclusive=True)
        elif command == "compact":
            if self.active_session is None:
                self.notify("No active session to compact.", severity="warning")
                return
            self.run_worker(
                self._compact_worker(self.active_session, rest), exclusive=True
            )
        elif command == "skill-creator":
            self.run_worker(self._skill_creator_worker(), exclusive=True)
        elif command == "skills-list":
            self._show_skills_list()
        elif command == "skill-remove":
            self.run_worker(self._skill_remove_worker(), exclusive=True)
        else:
            self.notify(f"Unknown command: /{command}", severity="warning")

    # --------------------------------------------------------------- context

    async def _compact_worker(self, session: Session, guidance: str) -> None:
        """/compact [instruction]: fold this conversation into a summary now.

        The automatic fold waits for the window to fill and keeps the recent
        turns verbatim, because it fires unasked. This one is asked for, so it
        folds everything and takes the text after the command as its brief —
        material to preserve, or the step the user is about to take, which is
        the same instruction from the summarizer's point of view: that is what
        the summary is being written *for*.

        The chat is not touched: the fold changes what the model receives, not
        what the user can scroll back to.
        """
        session_id = session.session_id
        if session_id in self._awaiting_approval:
            # The thread is parked on an interrupt; rewriting its state under
            # the pending decision is not something to do quietly.
            self.notify(
                f"“{session.title}” is waiting on an approval — answer that first.",
                severity="warning",
            )
            return
        try:
            async with self._backend_working("compacting context", session=session):
                folded = await compact_now(
                    self.graph,
                    session_id=session_id,
                    llm=self._labelled_llm("compact", session=session),
                    guidance=guidance,
                )
        except Exception as e:
            # Nothing was written: the thread is exactly as it was.
            self.notify(f"/compact failed: {e}", severity="error")
            return
        if folded is None:
            self.notify("Nothing new to compact in this conversation.")
            return
        note = f"Context compacted: {folded['folded']} messages folded into a summary."
        if guidance:
            note = f"{note} Asked to keep: {guidance}"
        # The backend's last token count measured the unfolded prompt, so it no
        # longer describes what the next turn will send: drop it and show what
        # the folded view actually costs.
        self._context_used.pop(session_id, None)
        if self._is_active_session(session):
            # ``_log_write`` writes to whichever session is open, so it belongs
            # under this check: the user may have switched away while the
            # summary was being written, and the record is worth nothing in
            # another session's transcript.
            self._log_write(
                "context compacted", f"{note}\n{folded['summary']['content']}"
            )
            # Shown in full, not just announced: the user asked for this fold,
            # possibly naming what it had to keep, and a small model does not
            # always keep it. What the agent remembers from here on is exactly
            # this text, so it is worth reading once.
            await self._append_chat(
                "notice", f"{note}\n\n{folded['summary']['content']}"
            )
            snapshot = await self.graph.aget_state(
                {"configurable": {"thread_id": session_id}}
            )
            self._show_context_estimate(snapshot.values or {})

    # ------------------------------------------------------------- skills (§5.1)

    async def _skill_creator_worker(self) -> None:
        """/skill-creator: collect a skill in a form and write it at the level
        the user chose — global (every profile), profile (the one on screen),
        or project (the working directory)."""
        from hpca.tui.skill_screens import SkillCreatorScreen

        result = await self.push_screen_wait(SkillCreatorScreen())
        if result is None:
            return
        skill, level = result
        project_root = Path.cwd()
        # Refuse only when this exact level already has that skill; the same
        # name may live at another level, where precedence keeps them distinct.
        target = skill_path(
            skill.name, self.profile, level=level, project_root=project_root
        )
        if target.exists():
            self.notify(
                f"A skill named “{skill.name}” already exists at the "
                f"{level} level.",
                severity="warning",
            )
            return
        write_skill(skill, self.profile, level=level, project_root=project_root)
        self._refresh_skills()
        where = {
            "global": "all profiles",
            "profile": f"profile “{self.profile}”",
            "project": "this project",
        }[level]
        self.notify(f"Added skill “{skill.name}” for {where}.")

    # The profile's own skills carry no tag — they are the removable, unsurprising
    # case; everything else says where it came from.
    SKILL_LEVEL_TAGS = {
        "project": "  (project)",
        "profile": "",
        "global": "  (global)",
        "builtin": "  (built-in)",
    }

    def _show_skills_list(self) -> None:
        """/skills-list: a read-only view of every skill the profile can see,
        marking which level each resolves to (project > profile > global >
        built-in)."""
        project_root = Path.cwd()
        visible = load_skills(self.profile, project_root=project_root)
        if not visible:
            self.notify(
                f"No skills for profile “{self.profile}”. Add one with "
                "/skill-creator.",
            )
            return
        lines = []
        for skill in visible:
            tag = self.SKILL_LEVEL_TAGS.get(skill.level, "")
            lines.append(f"• {skill.name}{tag}")
            if skill.description:
                lines.append(f"    {skill.description}")
        self.push_screen(
            InspectScreen(f"Skills · profile “{self.profile}”", "\n".join(lines))
        )

    async def _skill_remove_worker(self) -> None:
        """/skill-remove: pick one of the profile's own or project skills and
        delete it. Global (_shared) and legacy skills are not offered —
        removing one would change every other profile that sees it."""
        from hpca.tui.skill_screens import SkillPickerScreen

        project_root = Path.cwd()
        # Removable = the profile's own plus this project's own. On a name
        # collision the project skill shadows the profile one, matching load
        # precedence, so offer that one.
        by_name = {s.name: s for s in load_own_skills(self.profile)}
        by_name.update(
            {s.name: s for s in load_project_skills(project_root=project_root)}
        )
        removable = sorted(by_name.values(), key=lambda s: s.name)
        if not removable:
            self.notify(
                f"Profile “{self.profile}” has no skills of its own to remove.",
            )
            return
        skill = await self.push_screen_wait(SkillPickerScreen(removable))
        if skill is None:
            return
        confirmed = await self.push_screen_wait(
            ConfirmScreen(f"Remove skill “{skill.name}”?")
        )
        if not confirmed:
            return
        if delete_own_skill(skill, self.profile, project_root=project_root):
            self._refresh_skills()
            self.notify(f"Removed skill “{skill.name}”.")
        else:
            self.notify(f"Could not remove “{skill.name}”.", severity="warning")

    def _refresh_skills(self) -> None:
        """Make a skill change visible without a graph rebuild: the on-screen
        profile's list feeds the next turn's prompt. read_skill is registered
        from the start (HPCA ships skills), but a caller may have passed in its
        own registry, so top it up rather than assume."""
        self.skills = load_skills(self.profile, project_root=Path.cwd())
        if "read_skill" not in self._tools.names():
            add_skill_tools(self._tools)

    async def _memorize_worker(self, note: str) -> None:
        """/memorize <note>: the model turns the note plus the conversation so
        far into durable memory proposals; each needs the user's approval
        (§5.3 — a small model writes these, so review is essential)."""
        messages = (
            await self._session_messages(self.active_session)
            if self.active_session is not None
            else []
        )
        self.profile_memory = Profile.load(self.profile)  # merge, don't clobber
        try:
            async with self._backend_working(
                "forming memories", session=self.active_session
            ):
                proposals = await propose_memories(
                    self._labelled_llm("memorize", session=self.active_session),
                    messages,
                    system_prompt_memories=self.profile_memory.scope_text(
                        MemoryScope.SYSTEM_PROMPT
                    ),
                    guidance=note,
                )
        except Exception as e:
            self.notify(f"/memorize failed: {e}", severity="error")
            return
        if not proposals:
            self.notify("The model proposed no memories for that note.")
            return
        await self._review_proposals(proposals)

    async def _conclude_worker(self) -> None:
        """The one place memory is generated: a full self-review of the whole
        conversation (memories, struggle notes, skills) followed by the facts
        the agent flagged mid-session — all approved in one pass."""
        assert self.active_session is not None
        session = self.active_session
        messages = await self._session_messages(session)
        pending = self._pending_memory.get(session.session_id)
        if not messages and not pending:
            self.notify("Nothing to conclude yet.", severity="warning")
            return
        try:
            kept = await self.review_conversation(session, messages, span="whole")
        except Exception as e:
            self.notify(f"/conclude failed: {e}", severity="error")
            kept = 0
        kept += await self._drain_pending_memory(session)
        if kept == 0:
            self.notify("Nothing durable to keep from this conversation.")

    def _memory_write_blocked(
        self, scope: MemoryScope, text: str, memory: Profile | None = None
    ) -> bool:
        """Hard token budget on writes: a full system-prompt scope rejects new
        memories until the user condenses it. Injection never truncates; only
        growth is stopped. RAG is retrieved, not injected, so it has no budget.

        ``memory`` is the profile the caller is about to write to — pass the
        same object, or the budget gets checked against one state and the
        write lands in another.
        """
        if scope is not MemoryScope.SYSTEM_PROMPT:
            return False  # RAG is retrieved, not injected: no budget
        target = memory if memory is not None else self.profile_memory
        cap = self.settings.memory.system_prompt_token_cap
        if not target.would_exceed(text, cap=cap):
            return False
        self.notify(
            f"System-prompt memory is full "
            f"({target.usage_meter(cap)}) — memory NOT saved. "
            "Press ctrl+e to condense the profile, then retry.",
            severity="warning",
            timeout=12,
        )
        return True

    async def _review_proposals(self, proposals: list[MemoryProposal]) -> int:
        """One approval dialog per proposal; only approved ones are kept."""
        kept = 0
        for i, proposal in enumerate(proposals, start=1):
            approved = await self.push_screen_wait(
                MemoryProposalScreen(proposal, i, len(proposals))
            )
            if approved:
                if self._memory_write_blocked(proposal.scope, proposal.text):
                    continue
                self.profile_memory.add_memory(
                    proposal.text,
                    scope=proposal.scope,
                    backend=self._active_model(),
                    kind=proposal.kind,
                )
                kept += 1
        if kept:
            self.profile_memory.save()
            self._refresh_memory_snapshot(self.profile)
        self.notify(f"Kept {kept} of {len(proposals)} proposed memories.")
        self.check_memory_caps()
        return kept

    def _queue_memory_edits(
        self, session_id: str, operations: list[MemoryOp]
    ) -> str:
        """The `memory` tool's path: queue a flagged batch for review at the
        next /conclude. Nothing is written now — the agent flags, the user
        decides. Returns the tool result so the model knows it was noted."""
        if not operations:
            return "Nothing to flag."
        self._pending_memory.setdefault(session_id, []).extend(operations)
        count = sum(1 for _ in operations)
        return (
            f"Noted {count} memory change(s) — they will be reviewed together "
            "when the user runs /conclude. Nothing is saved yet."
        )

    async def _drain_pending_memory(self, session: Session) -> int:
        """Review the facts flagged this session, at /conclude. Returns how
        many batches were applied (0 or 1 — the batch is all-or-nothing)."""
        operations = self._pending_memory.pop(session.session_id, [])
        if not operations:
            return 0
        profile = session.profile
        loaded = Profile.load(profile)
        cap = self.settings.memory.system_prompt_token_cap
        try:
            result = apply_batch(
                loaded,
                operations,
                backend=self._active_model(),
                system_prompt_cap=cap,
            )
        except MemoryOpError as e:
            self.notify(f"Flagged memory not applied: {e}", severity="warning")
            return 0
        if not result.applied:
            return 0
        approved = await self.push_screen_wait(
            MemoryBatchScreen(operations, result.flagged)
        )
        if not approved:
            self.notify("Discarded the flagged memory changes.")
            return 0
        # The user may have edited this file by hand since it was loaded;
        # rewriting from a stale copy would silently discard those edits.
        path = Profile.path_for(profile)
        if path.exists() and drift_detected(loaded, path.read_text()):
            backup = path.with_suffix(f".bak.{int(time())}")
            backup.write_text(path.read_text())
            self._refresh_memory_snapshot(profile)
            self.notify(
                "Flagged memory not applied: the profile file changed on disk "
                f"since this session read it (backed up to {backup.name}).",
                severity="warning",
            )
            return 0
        result.profile.save()
        self._refresh_memory_snapshot(profile)
        if profile == self.profile:
            self.profile_memory = Profile.load(profile)
        self.check_memory_caps()
        return 1

    async def review_conversation(
        self, session: Session, messages: list[dict], *, span: str = "whole"
    ) -> int:
        """The /conclude self-review: propose what this conversation is worth
        keeping — memories, struggle notes, skill changes — each approved by
        the user. Returns how many proposals were kept."""
        memory = self._memory_snapshot(session.profile)
        skills = load_skills(session.profile)
        async with self._backend_working("reviewing conversation", session=session):
            proposals = await propose_reflections(
                self._labelled_llm("conclude", session=session),
                messages,
                system_prompt_memories=memory.scope_text(MemoryScope.SYSTEM_PROMPT),
                rag_memories=memory.scope_text(MemoryScope.RAG),
                skills=summarize_skills(skills),
                allow_new_skills=self.settings.memory.propose_new_skills,
                span=span,
            )
        if not proposals:
            return 0
        kept = 0
        for index, proposal in enumerate(proposals, start=1):
            approved = await self.push_screen_wait(
                ReflectionScreen(proposal, index, len(proposals))
            )
            if approved and self._apply_reflection(proposal, session.profile):
                kept += 1
        if kept:
            self.check_memory_caps()
        return kept

    def _apply_reflection(self, proposal: Reflection, profile: str) -> bool:
        """Persist one approved proposal; returns whether anything was written."""
        if proposal.kind in ("memory", "struggle"):
            text = proposal.memory_text()
            scope = proposal.target_scope()
            target = Profile.load(profile)  # merge, don't clobber
            if self._memory_write_blocked(scope, text, target):
                return False
            target.add_memory(
                text,
                scope=scope,
                backend=self._active_model(),
                kind=STRUGGLE_KIND if proposal.kind == "struggle" else "learning",
            )
            target.save()
            self._refresh_memory_snapshot(profile)
            if profile == self.profile:
                self.profile_memory = target
            return True
        return self._apply_skill_reflection(proposal, profile)

    def _apply_skill_reflection(self, proposal: Reflection, profile: str) -> bool:
        existing = {s.name: s for s in load_skills(profile)}
        if proposal.kind == "skill_patch":
            skill = existing.get(proposal.skill_name)
            if skill is None:
                self.notify(
                    f"No skill named “{proposal.skill_name}” to patch.",
                    severity="warning",
                )
                return False
            # Patches land in the profile's own copy, never in _shared/: one
            # profile's correction must not change another's procedure.
            skill.body = patched_body(skill, proposal.text)
            write_skill(skill, profile)
        else:
            if proposal.skill_name in existing:
                self.notify(
                    f"A skill named “{proposal.skill_name}” already exists.",
                    severity="warning",
                )
                return False
            write_skill(
                Skill(
                    name=proposal.skill_name,
                    description=" ".join(proposal.text.split())[:60],
                    triggers=proposal.keywords,
                    body=proposal.text,
                ),
                profile,
            )
        if profile == self.profile:
            self.skills = load_skills(profile)
        if "read_skill" not in self._tools.names():
            add_skill_tools(self._tools)  # the first skill enables the tool
        return True

    def run_curator_if_due(self) -> dict:
        """Age old tier-3 entries out, at most once every few days (P6).

        Runs at startup rather than on a timer: the app is idle then by
        definition, and this touches the same profile files a turn reads.
        """
        interval = self.settings.memory.curator_interval_days
        if interval <= 0 or not curator.due(interval_days=interval):
            return {}
        try:
            reports = curator.run(
                Profile.list_profiles(),
                stale_days=self.settings.memory.curator_stale_days,
                archive_days=self.settings.memory.curator_archive_days,
            )
        except Exception as e:
            self.notify(f"Memory curation skipped: {e}", severity="warning")
            return {}
        for name, report in reports.items():
            if report.changed:
                self._refresh_memory_snapshot(name)
                self.notify(
                    f"Memory curation ({name}): {report.summary()} — "
                    f"archived entries are in {name}.archive.md",
                    timeout=10,
                )
        return reports

    def check_memory_caps(self) -> bool:
        """Size warning; returns whether the system-prompt scope is over budget.

        It can only get over cap through hand edits (in-app writes are rejected
        at the cap), so the fix offered is the external editor."""
        cap = self.settings.memory.system_prompt_token_cap
        if not self.profile_memory.over_budget(cap):
            return False
        meter = self.profile_memory.usage_meter(cap)
        self.notify(
            f"System-prompt memory is over its budget ({meter}) — "
            "press ctrl+e to edit the profile externally.",
            severity="warning",
            timeout=12,
        )
        return True

    def action_edit_profile(self) -> None:
        """§6.4 (e): suspend the TUI, open the profile in the user's editor."""
        import subprocess

        self.profile_memory.save()  # ensure the file reflects current state
        editor = resolve_editor(self.settings.editor, os_environ)
        path = Profile.path_for(self.profile)
        try:
            with self.suspend():
                subprocess.call(editor + [str(path)])
        except Exception as e:
            self.notify(f"Cannot suspend for editing: {e}", severity="error")
            return
        self.profile_memory = Profile.load(self.profile)
        self._refresh_memory_snapshot(self.profile)
        if self.profile_memory.problems:
            self.notify(
                "Profile problems after edit: "
                + "; ".join(self.profile_memory.problems[:3]),
                severity="warning",
            )
        else:
            self.notify("Profile reloaded.")
        self.check_memory_caps()

    # -------------------------------------------------------------- sessions

    def _set_working_profile(self, profile: str) -> None:
        """The working profile is the active session's: its memories, its
        file under ctrl+e, its name in the top bar."""
        if profile == self.profile:
            return
        self.profile = profile
        self.profile_memory = Profile.load(profile)
        self._memory_snapshots[profile] = self.profile_memory
        self.skills = load_skills(profile)
        if self.profile_memory.problems:
            self.notify(
                "Profile file has problems: "
                + "; ".join(self.profile_memory.problems[:3]),
                severity="warning",
            )
        self._refresh_top_bar()

    def _make_tool_ctx(
        self,
        session: Session,
        log: SessionLog | None,
        *,
        skills: "list[Skill] | None" = None,
    ) -> ToolContext:
        """A tool context bound to one session and its transcript, so a turn
        keeps its own registry, runner and log however the UI moves on.

        ``skills`` is the running turn's frozen skill set; the on-screen
        session (no turn) falls back to ``self.skills``."""
        ctx = ToolContext(
            registry=PathRegistry(
                self._conn, profile=session.profile, session_id=session.session_id
            ),
            runner=ProcessRunner(
                self._conn,
                session_id=session.session_id,
                log_dir=app_dir() / "proc_logs",
            ),
            settings=self.settings,
            scripts_dir=app_dir() / "scripts",
            session_id=session.session_id,
            profile=session.profile,
            slurm=self.slurm,
            jobs=self.job_store,
            job_log_dir=app_dir() / "job_logs",
            watches=self.watch_store,
            llm=self._llm,
            trash=self.trash,
            symbols=self.symbol_index,
            rag=self.rag_store,
            embedder=self.embedder,
            episodic=self.episodic,
            skills=skills if skills is not None else self.skills,
            queue_memory_edits=lambda operations: self._queue_memory_edits(
                session.session_id, operations
            ),
        )
        if log is not None:
            ctx.llm = LoggedLLM(
                self._llm,
                log,
                label=lambda: f"subagent:{ctx.current_tool or '?'}",
            )
        return ctx

    def _park_draft(self) -> None:
        """Take the unsent text out of the entry and keep it under the session
        being left, so it is waiting there on the way back."""
        if self.active_session is None:
            return
        found = self.query("#chat-input")
        if not found:  # the screen is tearing down: nothing to keep
            return
        self._store_draft(self.active_session.session_id, found.first(ChatInput).text)

    def _store_draft(self, session_id: str, text: str) -> None:
        """Remember (or forget, when empty) one session's unsent text."""
        if text:
            self._drafts[session_id] = text
        else:
            self._drafts.pop(session_id, None)

    def _restore_draft(self, session: Session) -> None:
        """Put the session's own unsent text back in the entry, cursor behind
        it, so a message interrupted by a switch continues where it stopped.

        Dropped from the store as it goes on screen: what is in the entry is
        the live draft, and only sessions the user has left are parked here.
        """
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.text = self._drafts.pop(session.session_id, "")
        chat_input.move_cursor(chat_input.document.end)

    def _activate_session(self, session: Session) -> None:
        # Whatever is half-typed belongs to the session being left, not to the
        # one being opened (per-session drafts) — in the chat entry, and in the
        # box asking why that session's script was refused.
        self._park_draft()
        self._park_reason()
        # A session boundary is a deliberate refresh point (redesign Phase 1):
        # memories written by another session or instance are picked up here,
        # while WITHIN a session the frozen snapshot keeps the prompt prefix
        # byte-stable for the backend's prefix cache.
        self._refresh_memory_snapshot(session.profile)
        self.active_session = session
        self._refresh_session_log()
        self._restore_draft(session)
        self.query_one("#chat-input", ChatInput).display = True
        self._refresh_mode_bar()
        # The top bar and context meter follow the opened session's LLM.
        self._refresh_top_bar()
        self._refresh_model_line()
        self._refresh_context_bar()
        if self._log is not None:
            self._log.write(
                "session opened",
                f"{session.session_id} · profile {self.profile} · "
                f"model {self._active_model()}",
            )

    def _refresh_session_log(self) -> None:
        """(Re)open the active session's transcript and rebuild its tool
        context, so changed logging or llm settings take effect now."""
        if self.active_session is None:
            self._log = None
            self._tool_ctx = None
            return
        self._log = open_log(self.settings, self.active_session)
        self._tool_ctx = self._make_tool_ctx(self.active_session, self._log)

    def _labelled_llm(
        self,
        label: str,
        *,
        log: SessionLog | None = None,
        session: Session | None = None,
    ) -> Any:
        """The client for one of the app's own sub-agent calls (titling,
        \\conclude, struggle notes), logged under that name.

        Routed through the session's *own* backend — the same live client chat
        uses — not the bootstrap client, which may point at a backend that is
        not running once the session has been switched to another (§ per-session
        LLM). Falls back to the open session, then the bootstrap client."""
        client = self._client_for(
            session if session is not None else self.active_session
        )
        sink = log if log is not None else self._log
        if sink is None:
            return client
        return LoggedLLM(client, sink, label=lambda: f"subagent:{label}")

    def pick_profile_for_new_session(self) -> None:
        """Every new session starts by choosing its profile (or creating one),
        then the LLM it will talk to."""

        def chosen(profile: str | None) -> None:
            if profile is not None:
                self._pick_llm_for_new_session(profile)

        self.push_screen(ProfilePickerScreen(current=self.profile), chosen)

    def _pick_llm_for_new_session(self, profile: str) -> None:
        """Choose the LLM the new session uses. With no configured backends
        there is nothing to pick, so fall back to the bootstrap client."""
        backends = list(self.settings.backends)
        if not backends:
            self.run_worker(
                self.start_new_session(profile=profile), group="sessions"
            )
            return

        def chosen(backend: LLMBackend | None) -> None:
            if backend is None:
                return  # cancelled the LLM pick: no session is created
            self.run_worker(
                self.start_new_session(
                    profile=profile, backend=backend.model_dump_json()
                ),
                group="sessions",
            )

        self.push_screen(
            SwitchLLMScreen(
                backends,
                self.settings.is_active,
                title="LLM for the new session",
            ),
            chosen,
        )

    async def start_new_session(
        self, profile: str | None = None, *, backend: str = ""
    ) -> None:
        """Open an empty session under the given profile and LLM, and start
        typing.

        An untouched session is reused (retagged to the chosen profile and
        backend) rather than piling up empty rows when "(new session)" is
        entered repeatedly.
        """
        profile = profile or self.profile
        self._set_working_profile(profile)
        if self._is_untouched(self.active_session):
            changed = False
            if self.active_session.profile != profile:
                self.session_store.set_profile(
                    self.active_session.session_id, profile
                )
                self.active_session.profile = profile
                changed = True
            if backend and self.active_session.backend != backend:
                self.session_store.set_backend(
                    self.active_session.session_id, backend
                )
                self.active_session.backend = backend
                changed = True
            if changed:
                self._refresh_session_log()
                self._refresh_top_bar()
                self._refresh_model_line()
                self._refresh_context_bar()
                await self._reload_sessions()
        else:
            self._activate_session(
                self.session_store.create(
                    profile=profile,
                    title=UNTITLED_SESSION,
                    mode=self.settings.agent.default_mode,
                    backend=backend,
                )
            )
            await self._set_chat_messages([])
            bar = self._context_bar()
            if bar is not None:
                bar.reset()  # a fresh thread starts from an empty window
            # A decision the last session was showing is not this session's to
            # answer: hide it here, the way open_session does on its way in. A
            # fresh thread never has one of its own, so this only ever clears.
            await self._sync_decision_bar()
            await self._reload_sessions()
        self._focus_column("chat")

    def _is_untouched(self, session: Session | None) -> bool:
        return (
            session is not None
            and session.title == UNTITLED_SESSION
            and not self._chat_entries
        )

    def _name_session(self, title: str) -> None:
        """Provisional name from the opening message, until the model or the
        user writes a better one (§3 sessions column)."""
        assert self.active_session is not None
        self.session_store.rename(self.active_session.session_id, title)
        self.active_session.title = title
        self._untitled.add(self.active_session.session_id)

    def _rename_session(
        self,
        session: Session,
        title: str,
        *,
        by: str,
        log: SessionLog | None = None,
    ) -> None:
        self.session_store.rename(session.session_id, title)
        session.title = title
        self._untitled.discard(session.session_id)  # named; don't overwrite it
        if self._is_active_session(session):
            self.active_session.title = title
        sink = log if log is not None else (
            self._log if self._is_active_session(session) else None
        )
        if sink is not None:
            sink.write("session renamed", f"{title} (by {by})")
        self.run_worker(self._reload_sessions(), group="sessions")

    async def _session_messages(self, session: Session) -> list[dict]:
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        return (snapshot.values or {}).get("messages", [])

    def rename_selected_session(self) -> None:
        session = self._highlighted_session()
        if session is None:
            return

        def apply(title: str | None) -> None:
            if title:
                self._rename_session(session, title, by="user")

        self.push_screen(RenameScreen(session.title), apply)

    def retitle_selected_session(self) -> None:
        session = self._highlighted_session()
        if session is None:
            return
        self.run_worker(self._retitle_worker(session), group="title")

    async def _retitle_worker(self, session: Session) -> None:
        messages = await self._session_messages(session)
        if not messages:
            self.notify("Nothing to summarize yet.", severity="warning")
            return
        async with self._backend_working("writing a title", session=session):
            title = await self._propose_title(messages, session=session)
        if title is None:
            self.notify("The model could not write a title.", severity="error")
            return
        self._rename_session(session, title, by="llm")
        self.notify(f"Renamed to “{title}”")

    async def _propose_title(
        self,
        messages: list[dict],
        *,
        log: SessionLog | None = None,
        session: Session | None = None,
    ) -> str | None:
        try:
            return await propose_title(
                self._labelled_llm("title", log=log, session=session), messages
            )
        except Exception:
            return None  # naming is a nicety; never break a turn over it

    def _highlighted_session(self) -> Session | None:
        highlighted = self.query_one("#sessions-list", ListView).highlighted_child
        return getattr(highlighted, "data_session", None)

    def confirm_delete_session(self) -> None:
        session = self._highlighted_session()
        if session is None:
            return

        def verdict(confirmed: bool | None) -> None:
            if confirmed:
                self.run_worker(self._delete_session(session), group="sessions")

        self.push_screen(
            ConfirmScreen(
                f"Really delete “{session.title}”?\n"
                "The chat is dropped; its log on disk is kept."
            ),
            verdict,
        )

    async def _delete_session(self, session: Session) -> None:
        """Drop a chat thread: its row, its aliases, its checkpoints.

        The transcript on disk is deliberately untouched — it is the record
        the session existed at all.
        """
        self._log_write("session deleted", f"{session.title} (log kept)")
        if self._is_active_session(session):
            await self.close_session()
        self._untitled.discard(session.session_id)
        self._updated.discard(session.session_id)
        # A parked thread and its inline decision go with the session.
        self._awaiting_approval.discard(session.session_id)
        self._pending_decision.pop(session.session_id, None)
        # Stage-2 leftover: drop this session's stored context number, and any
        # live turn defensively, so nothing leaks past the delete.
        self._context_used.pop(session.session_id, None)
        self._turn_speed.pop(session.session_id, None)
        self._turns.pop(session.session_id, None)
        self._drafts.pop(session.session_id, None)  # unsent text goes too
        self.session_store.delete(session.session_id)
        # Patient-data environment: a deleted conversation must not resurface
        # through episodic search either.
        self.episodic.forget_session(session.session_id)
        # Its watches go too. They are session-scoped, so leaving them would
        # leave boxes no session can ever show while the pollers went on
        # stat-ing their files and asking squeue about their jobs.
        self.watch_store.forget_session(session.session_id)
        try:
            await self._checkpointer.adelete_thread(session.session_id)
        except Exception as e:  # the row is already gone; say so and move on
            self.notify(f"Chat history left behind: {e}", severity="warning")
        await self._reload_sessions()
        self.notify(f"Deleted “{session.title}”")

    def _show_context_estimate(self, values: dict) -> None:
        """How full a reopened session's context already is.

        Estimated from the checkpointed history, since nothing has been sent
        yet this run. Compaction is accounted for: what the model will
        receive is the folded view, not the whole transcript.
        """
        bar = self._context_bar()
        if bar is None:
            return
        # A measured number for this session (its own turn, foreground or
        # background, already reported one) always supersedes an estimate.
        session_id = (
            self.active_session.session_id if self.active_session else None
        )
        measured = self._context_used.get(session_id) if session_id else None
        if measured:
            bar.set_used(measured)
            return
        messages = list(values.get("messages", []))
        compacted = values.get("compacted")
        if compacted:
            messages = [compacted["summary"]] + messages[compacted["upto"] :]
        if not messages:
            bar.reset()
            return
        bar.set_estimate(compact.estimate_tokens(messages))

    async def close_session(self) -> None:
        """Leave the active session: empty chat, no entry, nothing to type in."""
        # Deliberately leaving drops the measured number, so reopening re-derives
        # the fill from the stored history (which reflects compaction and any
        # growth since); a running turn re-stores its count when it reports.
        if self.active_session is not None:
            self._context_used.pop(self.active_session.session_id, None)
        # The draft is not dropped with it — leaving a session is how you go
        # and look something up, and coming back finds the sentence intact.
        # Same for a half-written reason on a decision left unanswered.
        self._park_draft()
        self._park_reason()
        self.query_one("#chat-input", ChatInput).text = ""
        self.active_session = None
        self._tool_ctx = None
        self._log = None
        bar = self._context_bar()
        if bar is not None:
            bar.reset()
        await self._set_chat_messages([])
        self.query_one("#chat-input", ChatInput).display = False
        self._refresh_mode_bar()
        self._refresh_model_line()  # no session: hide the model line
        await self._sync_decision_bar()  # no session: nothing to decide inline
        self._focus_column("sessions")

    async def open_session(self, session: Session) -> None:
        # The list row's Session may predate a mode switch; the store is
        # current.
        session = self.session_store.get(session.session_id) or session
        self._set_working_profile(session.profile)
        self._activate_session(session)
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        values = snapshot.values or {}
        await self._set_chat_messages(
            values.get("messages", []),
            values.get("thinking", []),
            values.get("calls", []),
        )
        self._show_context_estimate(values)
        self._updated.discard(session.session_id)  # its news is now on screen
        for item in self.query_one("#sessions-list", ListView).children:
            row_session = getattr(item, "data_session", None)
            if row_session is not None and row_session.session_id == session.session_id:
                item.remove_class("session-updated")
        if session.session_id in self._turns:
            self.show_working()  # its turn is still in flight
        # Reveal (or hide) whatever decision this session was left waiting on.
        await self._sync_decision_bar()

    @on(ListView.Selected, "#sessions-list")
    async def _on_session_selected(self, event: ListView.Selected) -> None:
        session = getattr(event.item, "data_session", None)
        if session is None:
            self.pick_profile_for_new_session()
        else:
            await self.open_session(session)
            self._focus_column("chat")  # entering a session means typing in it

    async def _reload_sessions(self) -> None:
        # Fetched on the DB thread first: rebuilding the list happens after
        # every turn, and a slow (NFS) read here must not stall the loop.
        sessions = await self._db(lambda conn: SessionStore(conn).list_all())
        # The await yields; app shutdown may have torn the widget down since.
        found = self.query("#sessions-list")
        if not found:
            return
        sessions_list = found.first(ListView)
        await sessions_list.clear()
        new_item = ListItem(Label("(new session)"))
        new_item.data_session = None
        items = [new_item]
        for session in sessions:
            item = ListItem(Label(self._session_row_text(session)))
            item.data_session = session
            item.set_class(session.session_id in self._updated, "session-updated")
            item.set_class(
                session.session_id in self._pending_decision, "session-pending"
            )
            # A full rebuild mid-turn keeps the in-flight marker on live rows.
            item.set_class(session.session_id in self._turns, "session-working")
            items.append(item)
        sessions_list.extend(items)
        # A ListView filled after mount has no cursor, and without one Enter
        # does nothing; highlight the open session, else "(new session)".
        sessions_list.index = next(
            (
                index
                for index, item in enumerate(items)
                if self._is_active_session(item.data_session)
            ),
            0,
        )

    def _mark_session_updated(self, session: Session) -> None:
        """Frame the session's row: a reply landed while the user was away."""
        self._updated.add(session.session_id)
        for item in self.query_one("#sessions-list", ListView).children:
            row_session = getattr(item, "data_session", None)
            if row_session is not None and row_session.session_id == session.session_id:
                item.add_class("session-updated")

    def _session_row_text(self, session: Session) -> Content:
        """One sidebar row's label: the title, its profile tag, and a leading
        marker. A "!" when the session is waiting on a decision (deliverable 2),
        else the working glyph while a turn is in flight (stage 3 / decision 6),
        so a user working elsewhere sees a choice pending — or an off-screen
        turn still running — in another session. The decision mark takes
        precedence: it needs the user, whereas working is transient (in normal
        flow the two never coexist — a parked turn has left ``_turns``)."""
        tag = (
            f"  · {session.profile}" if session.profile != DEFAULT_PROFILE else ""
        )
        if session.session_id in self._pending_decision:
            mark = "! "
        elif (
            session.session_id in self._turns
            or session.session_id in self._busy_sessions
        ):
            mark = WORKING_MARK
        else:
            mark = ""
        return Content(f"{mark}{session.title}{tag}")

    def _refresh_session_row(self, session_id: str) -> None:
        """Repaint one row's label, pending class, and working class from live
        state — used when a decision appears/answers or a turn starts/ends,
        without a full list reload. Guarded: it now fires on every turn end,
        including while the screen is tearing down after a turn finished late."""
        found = self.query("#sessions-list")
        if not found:
            return
        for item in found.first(ListView).children:
            row_session = getattr(item, "data_session", None)
            if row_session is not None and row_session.session_id == session_id:
                label = item.query(Label)
                if label:
                    label.first(Label).update(self._session_row_text(row_session))
                item.set_class(
                    session_id in self._pending_decision, "session-pending"
                )
                item.set_class(
                    session_id in self._turns
                    or session_id in self._busy_sessions,
                    "session-working",
                )
                return

    def _is_active_session(self, session: Session | None) -> bool:
        return (
            session is not None
            and self.active_session is not None
            and session.session_id == self.active_session.session_id
        )

    # ------------------------------------------------------------- processes

    async def _db(self, fn):
        """Run ``fn(conn)`` on the DB thread; see DbIO for why not the loop."""
        assert self._dbio is not None
        return await self._dbio.run(fn)

    async def refresh_watchers(self) -> None:
        """Repaint the right column from the ``watches`` table.

        Guarded against overlap: a fetch outlasting the 2s tick (NFS) would
        let two calls interleave clear() and extend() and paint duplicates.
        Skipping is safe — the next tick repaints.
        """
        if self._refreshing_watchers:
            return
        self._refreshing_watchers = True
        try:
            await self._refresh_watchers_inner()
        finally:
            self._refreshing_watchers = False

    async def _refresh_watchers_inner(self) -> None:
        if self._dbio is None or self._dbio.closed:
            return
        session = self.active_session
        session_id = session.session_id if session is not None else None

        def _gather(conn):
            # A watch belongs to the session that registered it, so the column
            # describes the conversation being read and nothing else. With no
            # session open there is nothing of anyone's to show.
            if session_id is None:
                return []
            return WatchStore(conn).list(session_id=session_id)

        watches = await self._db(_gather)
        if session_id is not None and (
            self.active_session is None
            or self.active_session.session_id != session_id
        ):
            return  # switched away mid-fetch; the next tick repaints
        # The DB await above yields; shutdown may have torn the widget down.
        found = self.query("#watchers-list")
        if not found:
            return
        await self._paint_panel(found.first(ListView), self._panel_rows(watches))

    def _panel_rows(self, watches: list[Watch]) -> list[PanelRow]:
        """The whole right column, top to bottom: one box per watch.

        It used to carry the session's own run history underneath, under a
        "── this session ──" heading. That is gone. Every one of those calls is
        already in the chat log a column to the left, so the history was a
        second copy of something the user had just read — and it grew without
        bound while the boxes they had actually asked to be shown were pushed
        off the bottom of a short terminal.

        In ``watches`` order, which is the user's own — see
        ``WatchStore.move`` and ``action_move_watch``.
        """
        now = datetime.now(timezone.utc)
        return [
            PanelRow(
                key=f"w{watch.id}",
                text="\n".join(watch_lines(watch, now=now)),
                classes=f"watch {watch_class(watch)}",
                title=watch.title,
                watch=watch,
            )
            for watch in watches
        ]

    async def _paint_panel(self, panel: ListView, rows: list[PanelRow]) -> None:
        """Draw the column, rebuilding only when its shape actually changed.

        A watch box counts up ("last write 12s ago"), so the text changes on
        every tick while the set of rows does not. Clearing and re-extending
        for that would reset the highlight twice a second and make the column
        impossible to navigate — so identical keys means update in place, and
        a genuine change restores the cursor onto the row it was on.

        Something always ends up highlighted while the column has rows in it.
        A `ListView` starts with no selection and `clear()` returns it to none,
        and an unhighlighted column advertises none of its keys: `check_action`
        answers about the highlighted row, so with nothing highlighted the
        footer loses peek, unwatch and both moves, and the column reads as
        inert until the user happens to press ↓. That was visible on the first
        paint and every time the remembered row was gone — after dropping a
        box, and on any switch to a session whose boxes are different ones.
        """
        keys = [row.key for row in rows]
        if keys == self._panel_keys and len(panel.children) == len(rows):
            bodies = [getattr(item, "data_body", None) for item in panel.children]
            if all(body is not None for body in bodies):
                for item, body, row in zip(panel.children, bodies, rows):
                    # The payloads are re-attached, not just the text: the row
                    # stands for a watch whose state has moved on, and `d`
                    # must act on what the box currently says.
                    item.data_watch = row.watch
                    body.update(Content(row.text))
                    if body.classes != frozenset(row.classes.split()):
                        body.set_classes(row.classes)
                    body.border_title = row.title
                return
        selected = self._selected_key(panel)
        previous = panel.index
        self._panel_keys = keys
        await panel.clear()
        items = []
        for row in rows:
            body = Static(Content(row.text), classes=row.classes)
            body.border_title = row.title
            item = ListItem(body)
            item.data_body = body
            item.data_key = row.key
            item.data_watch = row.watch
            items.append(item)
        await panel.extend(items)
        if keys:
            if selected is not None and selected in keys:
                panel.index = keys.index(selected)
            else:
                # The row we were on is gone. Hold the position rather than the
                # row: dropping the third of five boxes should leave the cursor
                # on what is now third, not throw it back to the top. Clamped,
                # for the box that was last, and 0 when there was no cursor at
                # all — which is the first paint.
                panel.index = min(previous or 0, len(keys) - 1)
        # The footer caches what check_action last said, so a column that has
        # just gained or lost its highlight has to ask for it to be asked again.
        self.refresh_bindings()

    def _selected_key(self, panel: ListView) -> str | None:
        highlighted = panel.highlighted_child
        return getattr(highlighted, "data_key", None) if highlighted else None

    async def poll_jobs(self) -> None:
        """Background sacct poll (§5.4); notifies on state changes.

        poll_active's store halves run on the DB thread; only the sacct
        subprocess is awaited here.
        """
        assert self.slurm is not None
        if self._dbio is None or self._dbio.closed:
            return
        try:
            active = await self._db(lambda conn: JobStore(conn).active())
            if not active:
                return
            statuses = await self.slurm.status([j.job_id for j in active])
            changes = await self._db(
                lambda conn: apply_statuses(JobStore(conn), active, statuses)
            )
        except Exception as e:
            self.notify(f"Job polling failed: {e}", severity="warning")
            return
        for change in changes:
            self.notify(
                f"Job {change.job_id}: {change.old_state} → {change.new_state}"
            )
            if change.new_state in SLURM_TERMINAL_STATES and change.session_id:
                self._pending_work.append(PendingWork(
                    session_id=change.session_id,
                    kind="event",
                    text=(
                        f"[job {change.new_state.lower()}] cluster job "
                        f"{change.job_id} is now {change.new_state}. Check its "
                        "logs with triage_job and act on the result."
                    ),
                ))
        if changes:
            await self.drain_work()
            await self.refresh_watchers()

    # ---------------------------------------------------------- watch polls

    def _panel_profile(self) -> str:
        """The profile behind the right column. Kept for callers that ask
        about the profile rather than the conversation."""
        session = self.active_session
        return session.profile if session is not None else self.profile

    def _panel_session(self) -> str | None:
        """Whose watches the right column shows: the open session's, and none
        at all when no session is open.

        Watches were profile-scoped, which in practice meant every session
        showed every other session's boxes — sessions on one profile are the
        normal case — and the column stopped describing the conversation being
        read. Polling stays store-wide (see hpca.watches), so a session
        returned to shows a current clock rather than a frozen one.
        """
        session = self.active_session
        return session.session_id if session is not None else None

    async def poll_watched_logs(self) -> None:
        """Stat every watched log — where the "last write …" clock comes from.

        Stat and write happen together on the DB thread: on an NFS home the
        stat is the slow half, and a stalled filesystem must cost the panel a
        late repaint, never the event loop.
        """
        if self._dbio is None or self._dbio.closed:
            return

        def _poll(conn):
            # Store-wide, not the open session's: scoping decides what is
            # *shown*, never what stays true. A watch left behind in another
            # session must still be current when the user goes back to it.
            store = WatchStore(conn)
            logs = [w for w in store.list() if w.kind == KIND_LOG]
            return poll_log_watches(store, logs)

        try:
            changes = await self._db(_poll)
        except Exception as e:
            self.notify(f"Log watch failed: {e}", severity="warning")
            return
        for change in changes:
            # A vanished file is the only thing a log poll can report, and the
            # only one worth interrupting for. There used to be a "no new
            # output" toast as well; it fired whenever a log had simply not
            # been written to for a while, which is not an event — the box
            # already says when the last write was, and a job between log
            # lines is not news.
            if change.old_state and change.new_state == LOG_GONE:
                self.notify(
                    f"{change.watch.title}: the file is gone", severity="warning"
                )
        if changes:
            await self.refresh_watchers()

    async def poll_watched_jobs(self) -> None:
        """Refresh watched Slurm jobs from squeue, and finished ones from sacct.

        Only jobs that can still move are asked about: a box that already says
        COMPLETED costs nothing from here on.
        """
        if self.slurm is None or self._dbio is None or self._dbio.closed:
            return
        watches = await self._db(
            lambda conn: [
                w
                for w in WatchStore(conn).list()  # store-wide; see poll_watched_logs
                if w.kind == KIND_JOB and not is_settled(w)
            ]
        )
        if not watches:
            return
        job_ids = [w.target for w in watches]
        try:
            details = await self.slurm.job_details(job_ids)
            gone = [i for i in job_ids if i not in details]
            finished = await self.slurm.status(gone) if gone else {}
        except Exception as e:
            self.notify(f"Job watch failed: {e}", severity="warning")
            return
        changes = await self._db(
            lambda conn: apply_job_details(
                WatchStore(conn), watches, details, finished
            )
        )
        for change in changes:
            if change.old_state:
                self.notify(
                    f"{change.watch.title}: {change.old_state} → "
                    f"{change.new_state}"
                )
        if changes:
            await self.refresh_watchers()

    async def watch_processes(self) -> None:
        """Turn finished background subprocesses into agent-visible events.

        The sibling of poll_jobs for local work. refresh_watchers only
        repaints the sidebar, so before this nothing ever told the agent that
        the script it started had exited — it promised to check back and had
        no way to keep the promise.
        """
        if self._conn is None or self._dbio is None or self._dbio.closed:
            return
        try:
            changes = await self._db(poll_processes)
        except Exception as e:
            self.notify(f"Process watch failed: {e}", severity="warning")
            return
        for change in changes:
            finding = (
                analyse_process_failure(change)
                if change.state != "finished"
                else None
            )
            text = format_process_event(change, finding)
            if finding is not None and finding.tier == 2:
                # Tier 3: the keyword scan found candidates but cannot say
                # which one caused it. That judgment is worth a model call.
                text = await self._explain_candidates(change, finding, text)
            self._pending_work.append(
                PendingWork(session_id=change.session_id, text=text, kind="event")
            )
        await self.drain_work()

    async def _explain_candidates(self, change, finding, fallback: str) -> str:
        """Ask the explainer to pick the causing line, and learn from it."""
        if self._llm is None:
            return fallback
        try:
            explanation = await explain_process_failure(
                self._llm,
                name=change.name,
                exit_code=change.exit_code,
                candidates=finding.candidates,
            )
        except Exception as e:
            self.notify(f"Log explainer failed: {e}", severity="warning")
            return fallback
        if not explanation.conclusive:
            return fallback  # it said it could not tell; do not dress that up
        if explanation.proposed_signature is not None:
            self._offer_signature(explanation.proposed_signature)
        head = fallback.split("\n\n", 1)[0]
        return f"{head}\n\n{explanation.render()}\n\nlog:\n{finding.excerpt}"

    def _confirm_then(self, question: str, coro) -> None:
        """Put a yes/no to the user and run ``coro`` if they accept.

        The coroutine is built by the caller and closed on a refusal, so a
        declined question leaves nothing un-awaited behind.
        """

        def on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self.run_worker(coro)
            else:
                coro.close()

        self.push_screen(ConfirmScreen(question), on_confirm)

    def _offer_signature(self, proposed) -> None:
        """A tier-3 diagnosis means a signature was missing; offer to keep it."""
        signature = Signature(
            id=proposed.id,
            title=proposed.title,
            patterns=list(proposed.patterns),
            hint=proposed.hint,
        )

        async def save() -> None:
            try:
                path = append_user_signature(signature)
            except Exception as e:
                self.notify(f"Could not save signature: {e}", severity="error")
                return
            self.notify(f"Saved error signature {signature.id!r} to {path.name}")

        self._confirm_then(
            f"Remember this failure as {signature.id!r} "
            f"({signature.title}) so it is recognised next time?",
            save(),
        )

    async def drain_work(self) -> None:
        """Start the next waiting item for every session that is free to run.

        Turns on different sessions run concurrently, so this no longer bails
        when "anything is busy": it walks the queue and starts the first item
        for each session that is not already running a turn and not parked on
        an approval. Same-session serialisation is preserved — a session with
        a turn in flight (or one already started this pass) is skipped, so two
        turns never interleave checkpoint writes on one thread_id.
        """
        if self._shutting_down:
            return
        started: set[str] = set()
        # Snapshot: _run_agent mutates self._turns, and _deliver_event awaits;
        # iterate a copy and remove drained items by identity.
        for item in list(self._pending_work):
            if item not in self._pending_work:
                continue
            sid = item.session_id
            # Skip a session already running a turn, one parked on an approval,
            # or one we already started an item for earlier in this same pass.
            if (
                sid in self._turns
                or sid in started
                or sid in self._awaiting_approval
            ):
                continue
            self._pending_work.remove(item)
            started.add(sid)
            try:
                if item.kind == "user":
                    session = self.session_store.get(sid)
                    if session is not None:
                        if self._is_active_session(session):
                            await self._promote_queued_entry(item.text)
                        self._run_agent(
                            session,
                            user_text=item.text,
                            forced_skill=item.forced_skill,
                        )
                else:
                    await self._deliver_event(sid, item.text)
            except Exception as e:
                self.notify(f"Queued work failed: {e}", severity="error")

    async def _promote_queued_entry(self, text: str) -> None:
        """A queued message is starting: show it as sent rather than waiting."""
        for entry in self._chat_entries:
            if entry.kind == "queued" and entry.text == text:
                entry.kind = "user"
                break
        await self._rerender_chat()

    def queued_texts_for(self, session_id: str) -> list[str]:
        return [
            work.text
            for work in self._pending_work
            if work.kind == "user" and work.session_id == session_id
        ]

    async def _deliver_event(self, session_id: str, text: str) -> None:
        """React now if the session is open, otherwise leave it in the thread.

        Both paths are the same LangGraph primitive — new input on an existing
        thread_id. The difference is only whether the model runs immediately
        (one call, agent speaks unprompted) or the message simply waits in the
        checkpointed history for the next turn (free).
        """
        session = self.session_store.get(session_id)
        if session is None:
            await deliver_event(self.graph, session_id=session_id, text=text)
            return
        if self._is_active_session(session):
            await self._append_chat("event", text)
            self._run_agent(session, user_text=text)
        else:
            await deliver_event(self.graph, session_id=session_id, text=text)
            self._mark_session_updated(session)
            self.notify(f"{session.title}: background work finished")

    def _selected_watch(self) -> Watch | None:
        highlighted = self.query_one("#watchers-list", ListView).highlighted_child
        return getattr(highlighted, "data_watch", None)

    def peek_selected_watch(self) -> None:
        watch = self._selected_watch()
        if watch is not None:
            self.peek_watch(watch)

    def peek_watch(self, watch: Watch) -> None:
        """Flash what a watched thing is saying, and let it fade.

        The question a box provokes — did it print an error, or is it just
        slow? — is answered by the last few hundred characters, so this is a
        toast that expires rather than a screen to open and dismiss.
        """
        if watch.kind == KIND_LOG:
            self.notify(peek(watch.target), title=watch.title, timeout=PEEK_TIMEOUT)
        else:
            self.run_worker(self._peek_job(watch))

    async def _peek_job(self, watch: Watch) -> None:
        """A job box's peek: its state, plus its output when hpca knows where
        that goes.

        Only jobs hpca submitted itself have a known stdout path. A job the
        user sbatch'ed by hand does not, and the state line is then the whole
        answer — watching its log is a separate box.
        """
        state = " · ".join(part for part in watch_lines(watch) if part)
        row = None
        if self._dbio is not None and not self._dbio.closed:
            job_id = watch.target
            row = await self._db(lambda conn: JobStore(conn).get(job_id))
        if row is not None and row.sbatch_stdout_path:
            state += "\n" + peek(row.sbatch_stdout_path)
        self.notify(state, title=watch.title, timeout=PEEK_TIMEOUT)

    def drop_selected_watch(self) -> None:
        """Stop watching the highlighted box (``d``).

        Unconfirmed on purpose: nothing is deleted but the box — the log and
        the job are untouched — and the user's own reason for pressing it is
        usually that the run is over and the box has stopped saying anything.
        """
        watch = self._selected_watch()
        if watch is None:
            return
        self.run_worker(self._drop_watch(watch))

    async def _drop_watch(self, watch: Watch) -> None:
        removed = await self._db(lambda conn: WatchStore(conn).remove(watch.id))
        if removed:
            self.notify(f"Stopped watching {watch.title}")
        await self.refresh_watchers()

    def move_selected_watch(self, delta: int) -> None:
        """Carry the highlighted box one place up (``-1``) or down (``+1``).

        Silent at either end. Holding alt+↑ to bring a box to the top is the
        normal way to use this, and a warning toast on each of the last few
        presses would be noise about having arrived.
        """
        watch = self._selected_watch()
        if watch is None:
            return
        self.run_worker(self._move_watch(watch, delta))

    async def _move_watch(self, watch: Watch, delta: int) -> None:
        moved = await self._db(lambda conn: WatchStore(conn).move(watch.id, delta))
        if moved:
            # The repaint restores the cursor by row key, so it follows the
            # box rather than staying at the index — which is what makes a
            # held-down key walk one box up the column instead of swapping the
            # same pair back and forth.
            await self.refresh_watchers()

    # -------------------------------------------------------------- chat log

    def chat_log_texts(self) -> list[str]:
        return [entry.text for entry in self._chat_entries]

    async def _set_chat_messages(
        self,
        messages: list[dict],
        thinking: list[dict] | None = None,
        calls: list[dict] | None = None,
    ) -> None:
        # Tolerates the screen already being gone: a queued turn can start and
        # land its reply while the app shuts down, and the entries are still
        # worth keeping (see _chat_list).
        chat_list = self._chat_list()
        if chat_list is not None:
            await chat_list.clear()
        self._chat_entries = []
        # Fresh entries mean the old id()-keyed expansion state is stale (and
        # a recycled id could wrongly re-open a new box); start clean.
        self._thinking_expanded.clear()
        for entry in build_entries(messages, thinking or [], calls or []):
            self._add_chat_entry(entry)
        # Messages typed while a turn ran may not be in this copy of the graph,
        # and the rebuild would erase them from under the user. Two kinds: the
        # one whose turn has since started, and those still waiting behind it.
        if self.active_session is not None:
            session_id = self.active_session.session_id
            running = self._turns.get(session_id)
            if running is not None and running.user_text is not None:
                # interrupt_keep is the thread's length before this turn
                # appended its message, so a copy no longer than that predates
                # it — as the finished turn's own snapshot does when the queue
                # drains into a new turn before that reply is drawn. None means
                # the turn has not reached the graph at all yet.
                keep = running.interrupt_keep
                if keep is None or len(messages) <= keep:
                    self._add_chat_entry(
                        Entry(kind="user", text=running.user_text)
                    )
            for text in self.queued_texts_for(session_id):
                self._add_chat_entry(Entry(kind="queued", text=text))

    async def _rerender_chat(self) -> None:
        """Redraw the chat from the entries already held, without rebuilding
        them from the graph — used when only an entry's kind changed."""
        chat_list = self._chat_list()  # gone during shutdown; nothing to draw
        if chat_list is None:
            return
        await chat_list.clear()
        for entry in self._chat_entries:
            for item in self._entry_items(entry):
                chat_list.append(item)
        chat_list.scroll_end(animate=False)

    async def _append_chat(self, kind: str, text: str) -> None:
        self._add_chat_entry(Entry(kind=kind, text=text))

    def _add_chat_entry(self, entry: Entry) -> None:
        self._chat_entries.append(entry)
        chat_list = self._chat_list()
        if chat_list is None:
            return  # screen already gone (shutdown); the entry is still kept
        for item in self._entry_items(entry):
            chat_list.append(item)
        chat_list.scroll_end(animate=False)

    def _entry_items(self, entry: Entry) -> list[ChatItem]:
        """The ListView rows an entry renders as: one for most entries, but an
        expanded thinking box also yields a collapsed :class:`StepBox` row per
        part. Consulting ``_thinking_expanded`` keeps a box open across a
        re-render rather than snapping it shut."""
        if entry.kind != THINKING:
            item = ChatItem(self._entry_widget(entry))
            item.data_entry = entry  # what activating the row acts on
            return [item]
        state = self._thinking_expanded.setdefault(id(entry), _ThinkingExpansion())
        items = [ChatItem(ThinkingBox(entry, expanded=state.open))]
        if state.open:
            items += [
                ChatItem(StepBox(part, expanded=state.parts.get(i, False)))
                for i, part in enumerate(entry.parts)
            ]
        return items

    def _entry_widget(self, entry: Entry) -> Static:
        # Thinking entries are rendered by _entry_items (box + optional step
        # rows); everything else is a single framed message.
        widget = Static(Content(entry.text), classes=f"chat-{entry.kind}")
        widget.border_title = CHAT_TITLES.get(entry.kind, entry.kind)
        return widget

    # ----------------------------------------------------------- focus model

    def check_action(self, action: str, parameters) -> bool | None:
        """Context-sensitive availability of the global hotkeys.

        (m) manage llms and (q) quit only from the main screen's sessions
        column, where sessions are started and nothing is typed; (ctrl+l)
        switch llm only from the chat column, which is also the one place
        settings are not offered — its letter keys belong to the message being
        typed. Returning False also hides the binding from the footer.
        """
        on_main_screen = len(self.screen_stack) == 1
        on_sessions = on_main_screen and self.focused_column_id == "sessions"
        in_chat = on_main_screen and self.focused_column_id == "chat"
        if action in ("manage_llms", "confirm_quit", "manage_profiles"):
            return on_sessions
        if action == "switch_llm":
            # Switches the open session's LLM, so it needs one.
            return in_chat and self.active_session is not None
        if action == "cycle_mode":
            # Mode is a per-session dial; without a session there is nothing
            # to switch. Only from the chat column — on the sessions and
            # watchers columns (and on modals) shift+tab keeps moving focus.
            return in_chat and self.active_session is not None
        if action == "open_settings":
            return not in_chat
        if action == "command_palette":
            # Rebound from ctrl+p to bare "p" (COMMAND_PALETTE_BINDING). Only on
            # the sessions column, where no typing happens; elsewhere "p" must
            # reach the widget (the chat entry) as a letter.
            return on_sessions
        if action == "quit":
            # Textual's own ctrl+q. Quitting goes through (q) on the sessions
            # column, which confirms first — and ctrl+q belongs to zellij.
            return False
        return True

    @property
    def focused_column_id(self) -> str | None:
        """Id of the ColumnPanel containing the focused widget, if any."""
        node = self.focused
        while node is not None:
            if isinstance(node, ColumnPanel):
                return node.id
            node = node.parent
        return None

    def _focus_column(self, column_id: str) -> None:
        """Focus a column. The chat column lands on its inline decision prompt
        when one is waiting — the user brings the chat window into focus to
        answer there — otherwise on its entry, ready to type."""
        if column_id == "chat" and self.active_session is not None:
            bar = self._decision_bar()
            if bar is not None and bar.display:
                self._focus_decision_bar()
            else:
                self.focus_chat_input()
        else:
            self.query_one(f"#{column_id}-list", ListView).focus()

    def focus_chat_input(self) -> None:
        """Focus the chat entry with the cursor behind what was typed so far."""
        chat_input = self.query_one("#chat-input", ChatInput)
        if not chat_input.display:
            return  # no session: there is nothing to type into
        chat_input.focus()
        chat_input.move_cursor(chat_input.document.end)

    def browse_chat_messages(self) -> None:
        """Leave the entry for the message log, starting at the last message."""
        chat_list = self.query_one("#chat-list", ListView)
        if not len(chat_list):
            return
        chat_list.index = len(chat_list) - 1
        chat_list.focus()

    def action_focus_column(self, delta: int) -> None:
        current = self.focused_column_id
        index = COLUMN_IDS.index(current) if current in COLUMN_IDS else 0
        self._focus_column(COLUMN_IDS[(index + delta) % len(COLUMN_IDS)])

    @on(ListView.Selected, "#chat-list")
    async def _on_chat_list_selected(self, event: ListView.Selected) -> None:
        if event.item.query(WorkingIndicator):
            # Enter on the "LLM processing" line offers to abort the turn.
            self._maybe_interrupt_llm()
            return
        steps = list(event.item.query(StepBox))
        if steps:
            await self._toggle_step(event.item, steps[0])
            return
        boxes = list(event.item.query(ThinkingBox))
        if boxes:
            await self._toggle_thinking(event.item, boxes[0])
            return
        entry = getattr(event.item, "data_entry", None)
        if entry is not None and entry.kind in OWN_MESSAGE_KINDS:
            self.reuse_message(entry.text)
        else:
            # Anything else in the log: Enter just moves to the input.
            self.focus_chat_input()

    def reuse_message(self, text: str) -> None:
        """Put one of the user's own past messages back in the entry, to send
        again or edit into the next one — usually a command that needs a word
        changed, which is otherwise retyped from the screen.

        Added to whatever is already being written rather than replacing it, so
        activating a message can never lose a draft. It starts its own line,
        except after a draft the user left ending in whitespace — that space is
        how you say "continue here" (``rerun this: `` + the old command).
        """
        chat_input = self.query_one("#chat-input", ChatInput)
        draft = chat_input.text
        if draft and not draft[-1].isspace():
            draft += "\n"
        chat_input.text = draft + text
        self.focus_chat_input()  # focused, cursor behind the reused text

    async def _toggle_thinking(self, item: ChatItem, box: ThinkingBox) -> None:
        """Expand a thinking box into its parts (or fold them away again),
        inserting/removing the child rows right below it. The highlight is held
        on the box itself so the user keeps their place across the toggle."""
        chat_list = self.query_one("#chat-list", ListView)
        index = list(chat_list.children).index(item)
        state = self._thinking_expanded.setdefault(
            id(box.entry), _ThinkingExpansion()
        )
        state.open = not state.open
        box.set_expanded(state.open)
        if state.open:
            children = [
                ChatItem(StepBox(part, expanded=state.parts.get(i, False)))
                for i, part in enumerate(box.entry.parts)
            ]
            await chat_list.insert(index + 1, children)
        else:
            # The children are exactly the rows following the box; drop them.
            count = len(box.entry.parts)
            await chat_list.remove_items(range(index + 1, index + 1 + count))
        chat_list.index = index

    async def _toggle_step(self, item: ChatItem, box: StepBox) -> None:
        """Flip a single revealed part open or closed. Its state is recorded
        against the owning box so a later re-render restores it."""
        rows = list(self.query_one("#chat-list", ListView).children)
        position = rows.index(item)
        # Walk up to the thinking box this part belongs to; the offset between
        # them is the part's index.
        owner = position
        while owner > 0 and not rows[owner].query(ThinkingBox):
            owner -= 1
        part_index = position - owner - 1
        box.toggle()
        parent = rows[owner].query_one(ThinkingBox)
        state = self._thinking_expanded.setdefault(
            id(parent.entry), _ThinkingExpansion()
        )
        state.parts[part_index] = box.expanded

    # ---------------------------------------------------------- llm backends

    def action_manage_llms(self) -> None:
        self.push_screen(ManageLLMsScreen())

    def action_manage_profiles(self) -> None:
        def done(_: object) -> None:
            # a memory edited here may be any profile's; drop every frozen
            # snapshot so the next turn sees the edits
            self._memory_snapshots.clear()
            self.profile_memory = Profile.load(self.profile)
            self.run_worker(self._reload_sessions(), group="sessions")

        self.push_screen(ProfilesScreen(), done)

    def save_profile_memories(self, name: str, text: str) -> None:
        """Persist the raw memory text a user edited; report parse trouble but
        never lose their edits — the file is theirs to fix by hand (§6.4)."""
        profile = Profile.parse(text, name=name)
        profile.save()
        if profile.problems:
            self.notify(
                "Saved with problems: " + "; ".join(profile.problems[:3]),
                severity="warning",
            )
        else:
            self.notify(f"Saved memories for “{name}”.")
        self._refresh_memory_snapshot(name)
        if name == self.profile:
            self.profile_memory = profile

    def save_profile_archive(self, name: str, text: str) -> None:
        """Persist a hand-edited archive file. Emptying it removes the file —
        the archive is a plain appendix, not part of the loaded profile."""
        path = curator.archive_path(name)
        if text.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text if text.endswith("\n") else text + "\n")
            self.notify(f"Saved archive for “{name}”.")
        elif path.exists():
            path.unlink()
            self.notify(f"Cleared archive for “{name}”.")

    def save_skill_file(self, profile: str, name: str, text: str) -> None:
        """Persist a hand-edited skill file verbatim (front matter and body).
        The user owns the file; a parse problem is reported, never fatal."""
        path = skill_path(name, profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n")
        if profile == self.profile:
            self.skills = load_skills(profile)
        self.notify(f"Saved skill “{name}”.")

    def delete_profile_skill(self, profile: str, name: str) -> None:
        """Delete one of a profile's own skills (never shared/project)."""
        skill = next(
            (s for s in load_own_skills(profile) if s.name == name), None
        )
        if skill is None or not delete_own_skill(skill, profile):
            self.notify(f"No skill “{name}” to delete.", severity="warning")
            return
        if profile == self.profile:
            self.skills = load_skills(profile)
        self.notify(f"Deleted skill “{name}”.")

    def create_profile(self, name: str) -> str | None:
        """Make a blank profile; returns an error message, or None on success."""
        try:
            cleaned = Profile.validate_name(name)
        except ValueError as e:
            return str(e)
        Profile.create(cleaned)
        return None

    def duplicate_profile(self, source: str, name: str) -> str | None:
        """Fork a profile: same learnings, its own future.

        Copies the memories and the source's own skills, then the two are
        independent — which is the point, a shared base that specialises in
        different directions. Returns an error message, or None on success.
        """
        try:
            cleaned = Profile.validate_name(name)
        except ValueError as e:
            return str(e)
        if source not in Profile.list_profiles():
            return f"There is no profile called “{source}”."
        Profile.duplicate(source, cleaned)
        skills = copy_profile_skills(source, cleaned)
        self._refresh_memory_snapshot(cleaned)
        memories = len(Profile.load(cleaned).memories)
        self.notify(
            f"Copied “{source}” to “{cleaned}” "
            f"({memories} memor{'y' if memories == 1 else 'ies'}"
            + (f", {skills} skill{'' if skills == 1 else 's'}" if skills else "")
            + "). They diverge from here."
        )
        return None

    def profile_delete_blocker(self, name: str) -> str | None:
        """Why this profile cannot be deleted right now, or None if it can.

        A profile is in use while a turn on one of its sessions is in flight,
        or while any of its sessions has a live sub-process — deleting it then
        would strand running work under a gone profile.
        """
        if any(ts.session.profile == name for ts in self._turns.values()):
            return f"“{name}” has a reply in progress — wait for it to finish."
        session_ids = {
            session.session_id
            for session in self.session_store.list(profile=name)
        }
        if session_ids & running_session_ids(self._conn):
            return (
                f"“{name}” has running sub-process(es) — "
                "stop them before deleting it."
            )
        return None

    def delete_profile(self, name: str) -> None:
        moved = self.session_store.reassign_profile(name, "default")
        Profile.delete(name)
        # Its learnings go with it: memories, retrieval index, and the skills
        # it accumulated. Leaving orphaned skills behind would silently
        # resurrect them under a profile created with the same name later.
        delete_profile_skills(name)
        self._refresh_memory_snapshot(name)
        self.memory_index.forget_profile(name)
        if self.profile == name:  # unlikely, but keep the app coherent
            self.profile = "default"
            self.profile_memory = Profile.load("default")
            self.skills = load_skills("default")
        self.notify(
            f"Deleted “{name}”" + (f"; {moved} session(s) moved to default" if moved else "")
        )

    def action_switch_llm(self) -> None:
        """Change which LLM the OPEN session talks to (§ per-session LLM)."""
        if self.active_session is None:
            return
        if not self.settings.backends:
            self.notify(
                "No backends configured yet — press (m) to discover and add some.",
                severity="warning",
            )
            return

        def apply(backend: LLMBackend | None) -> None:
            if backend is not None:
                self.switch_backend(backend)

        self.push_screen(
            SwitchLLMScreen(
                list(self.settings.backends),
                self._marks_session_backend,
                title="Switch this session's LLM",
            ),
            apply,
        )

    def _marks_session_backend(self, backend: LLMBackend) -> bool:
        """Whether ``backend`` is the open session's current LLM (the ★)."""
        active = self._active_backend()
        if active is None:
            return self.settings.is_active(backend)
        return (
            active.base_url == backend.base_url and active.model == backend.model
        )

    def switch_backend(self, backend: LLMBackend) -> None:
        """Point the open session at a different configured backend."""
        if self.active_session is None:
            return
        # Refuse only if THIS session is mid-turn; a background turn in another
        # session does not block switching the open session's backend.
        if self.active_session.session_id in self._turns:
            self.notify(
                "The agent is mid-reply — switch backends once it finishes.",
                severity="warning",
            )
            return
        blob = backend.model_dump_json()
        self.session_store.set_backend(self.active_session.session_id, blob)
        self.active_session.backend = blob
        self._refresh_top_bar()
        self._refresh_model_line()
        self._refresh_context_bar()
        self._refresh_session_log()  # rebind the tool context to the new client
        self.notify(f"This session now uses {backend.model}")

    # ----------------------------------------------------------- auto-connect

    async def _auto_connect_cluster(self) -> None:
        """Discover live cluster vLLM endpoints and connect with no setup (§4).

        On the cluster (Slurm present) this reads the manifest dir, reconciles
        it against Slurm liveness + a live probe, then applies rule (c): one
        LLM auto-connects, several are announced for the picker, and the
        embeddings server (if up) is always wired to RAG. Off the cluster
        (no Slurm) it is a no-op — the manage-LLMs scan and tunnel template
        cover that path. Every step is best-effort: discovery must never take
        the app down or block the first turn — but every outcome, including
        a swallowed exception, leaves a trail in <app_dir>/autoconnect.log.
        """
        if self.slurm is None:
            return
        logger = _autoconnect_logger()
        try:
            endpoints = await discover_cluster_endpoints(
                self.settings.endpoints.dir_path(),
                self.slurm,
                api_keys=self.settings.llm_api_keys,
            )
        except Exception:
            logger.exception("auto-connect: discovery failed")
            return
        plan = plan_auto_connect(
            endpoints, preferred_models=self.settings.endpoints.preferred_models
        )
        logger.info(
            "auto-connect: %d LLM choice(s); connecting to %s; embeddings %s",
            len(plan.choices),
            plan.connect.model if plan.connect else "none",
            plan.embedding_base_url or "none",
        )
        if plan.embedding_base_url:
            self._wire_embedding(plan.embedding_base_url)
        if plan.connect is not None:
            self._auto_activate(plan.connect)
        elif plan.notice:
            self.notify(plan.notice)

    def _ensure_catalog(self, discovered: DiscoveredBackend) -> LLMBackend:
        """The catalog entry for a discovered endpoint, adding it if new."""
        for backend in self.settings.backends:
            if (
                backend.base_url == discovered.base_url
                and backend.model == discovered.model
            ):
                return backend
        entry = LLMBackend(
            model=discovered.model,
            base_url=discovered.base_url,
            api_key=discovered.api_key,
            max_model_len=discovered.max_model_len,
        )
        self.settings.backends.append(entry)
        self.settings.remember_llm_ports([entry.base_url])
        self.settings.remember_llm_key(entry.api_key)
        self.settings.save()
        return entry

    def _auto_activate(self, discovered: DiscoveredBackend) -> None:
        """Make a discovered LLM the active backend (adding it to the catalog
        first). A no-op when it is already active, so a restart against an
        unchanged endpoint causes no client churn."""
        entry = self._ensure_catalog(discovered)
        if self.settings.is_active(entry):
            return
        self.settings.activate_backend(entry)
        self.settings.save()
        self.reload_llm()
        self.notify(f"Auto-connected to {discovered.model}")

    def _wire_embedding(self, base_url: str) -> None:
        """Point RAG's embedder at a discovered embeddings server. Rebuilding
        ``self.embedder`` is enough: tool contexts read it when they are built,
        which for the auto-connect worker is before any turn runs."""
        if self.settings.rag.embedding_base_url == base_url:
            return
        self.settings.rag.embedding_base_url = base_url
        self.settings.save()
        old = getattr(self, "embedder", None)
        self.embedder = EmbeddingClient(
            base_url=base_url, model=self.settings.rag.embedding
        )
        if old is not None:
            self.run_worker(old.close(), group="embed-close")

    def reload_llm(self) -> None:
        """Rebuild the client so edited LLM settings apply to the next turn.

        An injected client belongs to whoever passed it in (tests, embedding
        hosts); only a client we built is ours to replace.
        """
        # A rebuild swaps the whole graph and bootstrap client, so it must wait
        # until NO turn is in flight anywhere — not just the active session.
        if self._turns:
            self.notify(
                "The agent is mid-reply — the new LLM settings apply "
                "after this turn.",
                severity="warning",
            )
            return
        if self._owns_llm:
            self._replace_llm()

    def _replace_llm(self) -> None:
        old_llm, owned = self._llm, self._owns_llm
        self._llm = LLMClient(self.settings.llm)
        self._owns_llm = True
        self._rebuild_graph()
        self._refresh_session_log()  # rebind the tool context to the new client
        self._refresh_model_line()  # the bootstrap model may have changed
        # A different model means a different window, and the token counts
        # measured against the old one no longer describe it — drop every
        # session's measurement so each re-measures on its next turn.
        self._discovered_window = None
        self._context_used.clear()
        bar = self._context_bar()
        if bar is not None:
            bar.reset()
        self._refresh_context_bar()
        self.run_worker(self._discover_context_window(), group="llm-probe")
        if owned and old_llm is not None:
            self.run_worker(old_llm.close(), group="llm-close")

    def _rebuild_graph(self) -> None:
        # Every per-session dependency is resolved by the invoked thread_id
        # (session_id), never from an app global, so two live turns on two
        # sessions each see their own client / context / mode / prompt.
        self.graph = build_graph(
            llm=self._llm_for_turn,
            tools=self._tools,
            checkpointer=self._checkpointer,
            ctx=self._ctx_for_turn,
            system_prompt_fn=self._render_system_prompt,
            max_retries=self.settings.llm.max_retries,
            max_tool_rounds=self.settings.llm.max_tool_rounds,
            # Wrapped so a late reassignment of report_activity / _on_usage
            # (tests, hot-swaps) is picked up, rather than freezing the bound
            # method at build time.
            on_activity=lambda sid, act: self.report_activity(sid, act),
            max_model_len=self._max_model_len_for,
            on_usage=lambda sid, usage: self._on_usage(sid, usage),
            mode_fn=self._mode_for_turn,
        )

    # ------------------------------------------- per-turn dependency resolvers

    def _turn_session(self, session_id: str) -> Session | None:
        """The session a turn belongs to, resolvable even after the user has
        switched away: its live TurnState's session, else the stored session."""
        ts = self._turns.get(session_id)
        if ts is not None:
            return ts.session
        return self.session_store.get(session_id)

    def _llm_for_turn(self, session_id: str) -> LLMClient:
        """The client for the session running ``session_id`` — its own backend's
        client, resolved off the thread_id the graph was invoked with."""
        return self._client_for(self._turn_session(session_id))

    def _ctx_for_turn(self, session_id: str) -> ToolContext | None:
        """The running turn's tool context, or the on-screen UI context for the
        active session with no turn in flight (tests, direct helpers)."""
        ts = self._turns.get(session_id)
        if ts is not None:
            return ts.ctx
        return self._tool_ctx

    def _max_model_len_for(self, session_id: str) -> int | None:
        """The context window of the session running ``session_id``, from its
        backend; the display fallback otherwise."""
        session = self._turn_session(session_id)
        backend = self._backend_of(session)
        if backend is not None:
            return backend.max_model_len
        if self._discovered_window is not None:
            return self._discovered_window
        for candidate in self.settings.backends:
            if self.settings.is_active(candidate):
                return candidate.max_model_len
        return None

    # --------------------------------------------------- per-session LLM

    def _backend_of(self, session: Session | None) -> LLMBackend | None:
        """The LLMBackend a session talks to, parsed from its stored JSON, or
        None to mean the bootstrap client."""
        if session is None or not session.backend:
            return None
        try:
            return LLMBackend.model_validate_json(session.backend)
        except Exception:
            return None

    def _client_for(self, session: Session | None) -> LLMClient:
        """The LLM client for a session: its own backend's client (built and
        cached on first use), or the bootstrap client when it has none."""
        backend = self._backend_of(session)
        if backend is None:
            return self._llm
        key = f"{backend.base_url}||{backend.model}"
        client = self._session_clients.get(key)
        if client is None:
            client = LLMClient(llm_settings_for(backend, self.settings.llm))
            self._session_clients[key] = client
        return client

    def _reads_session(self) -> Session | None:
        """Whose LLM the display/tagging reflects: always the session on screen.
        A background turn in another session never moves the top bar or meter."""
        return self.active_session

    def _model_of(self, session: Session | None) -> str:
        """The model name a specific session talks to, else the bootstrap."""
        backend = self._backend_of(session)
        return backend.model if backend is not None else self.settings.llm.model

    def _active_backend(self) -> LLMBackend | None:
        return self._backend_of(self._reads_session())

    def _active_model(self) -> str:
        """The model name to show/tag with: the on-screen session's, else the
        bootstrap model."""
        return self._model_of(self._reads_session())

    def _active_max_model_len(self) -> int | None:
        """The on-screen session's context window, from its backend (or the
        bootstrap discovery). Without either, compaction stays off — better
        than guessing a window and folding history needlessly.
        """
        backend = self._active_backend()
        if backend is not None:
            return backend.max_model_len
        if self._discovered_window is not None:
            return self._discovered_window
        for backend in self.settings.backends:
            if self.settings.is_active(backend):
                return backend.max_model_len
        return None

    def _on_usage(self, session_id: str, usage: dict) -> None:
        """The backend's token count for a session's just-made decision.

        prompt_tokens is what occupies the window; the completion is spent
        the moment it is generated. Reported per round, so a tool-heavy turn
        visibly fills the bar as it works. The count is stored per session
        ALWAYS — a background turn silently updates its own number so switching
        to it later shows the current fill — but the visible meter moves only
        when the reporting session is the one on screen (decision 9).

        The same report carries the turn's speed (completion tokens over the
        request's wall clock, measured by our client since OpenAI-style
        bodies hold no timing) — stored and displayed under the same
        per-session rules."""
        completion = usage.get("completion_tokens")
        seconds = usage.get("request_seconds")
        if completion and seconds:
            self._turn_speed[session_id] = completion / seconds
        prompt_tokens = usage.get("prompt_tokens")
        if not prompt_tokens:
            return
        self._context_used[session_id] = int(prompt_tokens)
        if (
            self.active_session is None
            or self.active_session.session_id != session_id
        ):
            return
        bar = self._context_bar()
        if bar is not None:
            bar.set_used(int(prompt_tokens))
            bar.set_speed(self._turn_speed.get(session_id))

    def _context_bar(self) -> ContextBar | None:
        found = self.query("#context-bar")
        return found.first(ContextBar) if found else None

    def _refresh_context_bar(self) -> None:
        bar = self._context_bar()
        if bar is None:
            return
        bar.set_window(self._active_max_model_len())
        # Bind the meter to the active session's stored number. Non-destructive
        # otherwise: callers that open/create a session follow with an explicit
        # estimate or reset, and a session showing only an estimate keeps it
        # when the async window probe lands.
        session_id = (
            self.active_session.session_id if self.active_session else None
        )
        used = self._context_used.get(session_id) if session_id else None
        if used:
            bar.set_used(used)
        # The speed always tracks the bound session (None clears): unlike the
        # fill it has no estimate/reset dance to preserve, and a stale rate
        # from the previously shown session would read as this one's.
        bar.set_speed(
            self._turn_speed.get(session_id) if session_id else None
        )

    async def _discover_context_window(self) -> None:
        """Ask the backend how big its window is, and remember it.

        This is why the meter needs no configuration: vLLM reports
        max_model_len per model, so switching from a 192k model to a 32k one
        moves the bar without anyone editing settings. The answer is written
        back to the catalog entry so compaction also benefits, and so the
        number survives a backend that is offline next time.
        """
        try:
            window = await self._llm.context_window()
        except Exception:
            window = None
        if not window:
            return
        self._discovered_window = window
        for backend in self.settings.backends:
            if self.settings.is_active(backend) and backend.max_model_len != window:
                backend.max_model_len = window
                self.settings.save()
                break
        self._refresh_context_bar()

    def action_confirm_quit(self) -> None:
        def verdict(confirmed: bool | None) -> None:
            if confirmed:
                self.exit()

        self.push_screen(ConfirmScreen("Really quit?"), verdict)

    # -------------------------------------------------------------- settings

    def action_open_settings(self) -> None:
        def apply(result: Settings | None) -> None:
            if result is not None:
                self.settings = result
                self.settings.save()
                self.reload_llm()  # so llm settings take effect without a restart
                self._refresh_session_log()
                self._refresh_top_bar()

        self.push_screen(SettingsScreen(self.settings), apply)

    def _refresh_top_bar(self) -> None:
        self.query_one(TopBar).update_info(profile=self.profile)

    def _model_line(self) -> ModelLine | None:
        found = self.query("#model-line")
        return found.first(ModelLine) if found else None

    def _refresh_model_line(self) -> None:
        """The model line atop the chat column: shown with a session and bound
        to the on-screen session's model, hidden without one (like the mode
        line and chat entry)."""
        line = self._model_line()
        if line is None:
            return
        if self.active_session is None:
            line.display = False
            return
        line.display = True
        line.set_model(self._active_model())
