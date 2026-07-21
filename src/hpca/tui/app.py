"""Main application: three-column layout, focus model, agent wiring (§3, §4)."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from os import environ as os_environ
from pathlib import Path
from time import monotonic, time
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.geometry import Offset
from textual.content import Content
from textual.message import Message
from textual.widgets import Footer, Label, ListItem, ListView, Static, TextArea

from hpca.agent import compact
from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.explainer import explain_process_failure
from hpca.agent.graph import (
    build_graph,
    deliver_event,
    rollback_thread,
    run_turn,
    thread_message_count,
)
from hpca.agent.job_tools import add_job_tools
from hpca.agent.tools import ToolRegistry
from hpca.clipboard import ClipboardManager, CopyResult
from hpca import curator
from hpca.config import LLMBackend, Settings, app_dir, llm_settings_for
from hpca.db import (
    checkpoints_db_path,
    command_use_counts,
    connect,
    init_db,
    record_command_use,
)
from hpca.jobs import JobRow, JobStore, poll_active
from hpca.agent.conclude import MemoryProposal, propose_memories
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.memory_context import (
    build_memory_context,
    compose_api_content,
    note_line,
    retrieved_line,
)
from hpca.agent.memory_tools import add_memory_tools
from hpca.agent.modes import add_plan_tool, kickoff_message, next_mode
from hpca.agent.prompts import environment_facts, orchestrator_system_prompt
from hpca.agent.reflect import Reflection, propose_reflections
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.titler import propose_title
from hpca.agent.struggle import (
    STRUGGLE_KIND,
    matching_struggles,
    turn_struggled,
)
from hpca.editor import resolve_editor
from hpca.embeddings import EmbeddingClient
from hpca.episodic import EpisodicStore
from hpca.llm import LLMClient
from hpca.logs import LoggedLLM, SessionLog, open_log
from hpca.rag import RagStore
from hpca.transcript import ASSISTANT as ASSISTANT_ENTRY
from hpca.transcript import THINKING, Entry, Step, build_entries
from hpca.transcript import USER as USER_ENTRY
from hpca.profiles import DEFAULT_PROFILE, Profile
from hpca.registry import PathRegistry
from hpca.runner import (
    ProcessRecord,
    ProcessRunner,
    analyse_process_failure,
    count_processes,
    describe,
    format_process_event,
    kill_unowned,
    list_processes,
    poll_processes,
    reconcile_orphans,
    running_session_ids,
)
from hpca.triage import Signature, append_user_signature
from hpca.sessions import Session, SessionStore
from hpca.skills import (
    Skill,
    any_skills,
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
from hpca.tui.approval_screen import ApprovalScreen
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.context_bar import ContextBar
from hpca.tui.inspect_screen import InspectScreen, format_job, format_process
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
from hpca.tui.plan_screen import PlanScreen
from hpca.tui.profiles_screen import ProfilePickerScreen, ProfilesScreen
from hpca.tui.rename_screen import RenameScreen
from hpca.tui.settings_screen import SettingsScreen
from hpca.tui.switch_llm import SwitchLLMScreen

COLUMN_IDS = ("sessions", "chat", "processes")
# Rows the processes column shows before summarising the rest. A long-lived
# session accumulates hundreds; the panel is a view, not an archive.
PROCESS_HISTORY_LIMIT = 60


def format_started(started_at: str) -> str:
    """When a process started, in the reader's own timezone.

    Stored UTC, shown local — the times are read next to a wall clock. The
    date is omitted for today, which is most of what the panel holds, and
    the narrow column has no room to spend on it.
    """
    if not started_at:
        return "  --  "
    try:
        moment = datetime.fromisoformat(started_at)
    except ValueError:
        return "  --  "
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    if moment.date() == datetime.now().date():
        return moment.strftime(" %H:%M")
    return moment.strftime("%m-%d %H:%M")
SESSION_TITLE_MAX = 40
UNTITLED_SESSION = "untitled"  # placeholder until the first message names it

# Interrupt-while-waiting-for-the-LLM (§ interrupt): the activity string the
# graph reports while a turn is parked on the model — the one phase the
# interrupt is armed in. The user aborts it by selecting the working indicator
# in the chat log and confirming (see _maybe_interrupt_llm), so the message can
# be re-edited and sent again.
LLM_WAIT_ACTIVITY = "LLM processing"
CHAT_TITLES = {
    "user": "you",
    "assistant": "agent",
    "error": "error",
    "event": "background",
    "queued": "queued",
    "recall": "recalled from memory",
}


@dataclass
class PendingWork:
    """One turn's worth of input waiting for the orchestrator to be free."""

    session_id: str
    text: str
    kind: str  # "user" — typed and waiting | "event" — background completion


# Chat commands ("/" or "\"): typing the prefix lists these above the entry.
COMMANDS = (
    ("memorize", "/memorize <note> — form memories from the note and this conversation"),
    ("conclude", "/conclude — propose memories from this conversation"),
    ("skill-creator", "/skill-creator — add a skill (name, description, body) to this profile"),
    ("skills-list", "/skills-list — list this profile's skills"),
    ("skill-remove", "/skill-remove — remove one of this profile's skills"),
)
LOG_KINDS = {
    "user": "user",
    "assistant": "agent",
    "error": "error",
    "event": "background",
    "recall": "recalled from memory",
}


class TopBar(Static):
    """Top bar: app name | profile | model | settings hint."""

    def __init__(self) -> None:
        super().__init__(id="top-bar")
        self._profile = ""
        self._model = ""

    def update_info(self, profile: str, model: str) -> None:
        self._profile = profile
        self._model = model
        self.update(self.render_text())

    def render_text(self) -> str:
        from hpca import __version__

        return (
            f" HPCA v{__version__} │ profile: {self._profile} │ "
            f"model: {self._model} │ (c) config"
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
    processes column, ↑ on the first line leaves to browse the message log.
    The draft is kept, so the user can step away mid-sentence and come back.
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
        if not select and self.cursor_at_first_line:
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
        super().__init__(classes="chat-step")
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
    thinking, or the tool in flight — and times that step, which is how you
    tell a slow answer from a lost one.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.08

    def __init__(self, activity: str = "working") -> None:
        super().__init__(classes="chat-working")
        self._frame = 0
        self._activity = activity
        self._started = monotonic()

    def on_mount(self) -> None:
        self._started = monotonic()
        self.set_interval(self.INTERVAL, self._advance)
        self._render_frame()

    @property
    def activity(self) -> str:
        return self._activity

    @property
    def elapsed(self) -> int:
        return int(monotonic() - self._started)

    def set_activity(self, activity: str) -> None:
        """Name the step now in flight; its clock starts over, so the number
        answers "is this step stuck?" rather than "how long since I asked?"."""
        if activity == self._activity:
            return
        self._activity = activity
        self._started = monotonic()
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
        self.update(Content(self._frame_text()))


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
        yield ContextBar(id="context-bar")
        yield ChatList(id="chat-list")
        menu = Static(id="command-menu")
        menu.display = False
        yield menu
        # The mode line sits directly above the entry, visible exactly when
        # the entry is: mode is a per-session property (§3.5).
        mode_bar = ModeBar(id="mode-bar")
        mode_bar.display = False
        yield mode_bar
        chat_input = ChatInput(placeholder="Message the agent…", id="chat-input")
        chat_input.display = False
        yield chat_input


class ProcessesList(ListView):
    """Right-column list; its hotkeys appear in the footer when focused."""

    BINDINGS = [
        Binding("enter", "inspect_process", "inspect"),
        Binding("k", "kill_process", "kill"),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        if action in ("inspect_process", "kill_process"):
            return self.highlighted_child is not None
        return True

    def action_inspect_process(self) -> None:
        self.app.inspect_selected_process()

    def action_kill_process(self) -> None:
        self.app.kill_selected_process()


class ProcessesPanel(ColumnPanel):
    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield ProcessesList(id="processes-list")


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
    #processes {
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
    /* Secondary rows in the processes column: the count of history not shown,
       and processes whose fate was never recorded because hpca exited first. */
    .proc-more, .proc-unknown { color: $text-muted; }
    .chat-working {
        color: $text-muted;
        padding: 0 1;
    }
    #command-menu {
        height: auto;
        border: round $accent;
        border-title-color: $accent;
        color: $text-muted;
        padding: 0 1;
    }
    /* A reply landed in a session the user has left: frame it, never
       force it open. Cleared when the session is opened. */
    #sessions-list > ListItem.session-updated {
        border: round $success;
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
            # Registered when ANY profile has a skill: the active profile
            # changes per session, but the tool registry does not.
            if any_skills():
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
        self._untitled: set[str] = set()  # sessions awaiting their first title
        self._updated: set[str] = set()  # replies that landed while switched away
        self._busy_turn: Session | None = None  # the one turn in flight, if any
        self._turn_worker = None  # handle to the in-flight turn, for cancelling
        self._interrupt_worker = None  # the rollback worker after a 3s hold
        # Interrupt bookkeeping for the in-flight turn: the message count to
        # roll back to, and the user's text to hand back for editing. Set only
        # for a fresh user message (not a resume/event), which is what "adjust
        # the last prompt" means.
        self._interrupt_keep: int | None = None
        self._interrupt_text: str | None = None
        # Work waiting for the orchestrator: messages the user typed while a
        # turn was running, and background completions reporting in. Exactly
        # one turn runs at a time — two on one thread_id would interleave
        # checkpoint writes — so anything arriving mid-turn waits here rather
        # than being refused. Drained in order, one item per pass.
        self._pending_work: list[PendingWork] = []
        # Sessions whose thread is parked on a destructive-op approval. Their
        # queued messages wait for the resume; other sessions are unaffected.
        self._awaiting_approval: set[str] = set()
        self._shutting_down = False
        self._turn_ctx: ToolContext | None = None  # that turn's tool context
        self._turn_memory: Profile | None = None  # that turn's profile memories
        self._turn_skills: list[Skill] | None = None  # that turn's skills
        # Frozen per-session memory views (redesign Phase 1): one snapshot per
        # profile, reused across turns so the system-prompt prefix stays
        # byte-stable for the backend's prefix cache. Refreshed on approved
        # writes and profile edits, never silently mid-session.
        self._memory_snapshots: dict[str, Profile] = {}
        # User turns since the last self-review, per session (redesign P4).
        self._turns_since_review: dict[str, int] = {}
        # Messages compaction dropped from the model's view, waiting to be
        # reviewed once the turn they were dropped during has finished (P6).
        self._evicted: dict[str, list[dict]] = {}
        # Context meter state: the window the backend reports, and the last
        # measured prompt size (per session — a different thread is a
        # different context).
        self._discovered_window: int | None = None
        self._context_used = 0
        # Panel labels by pid: describe() reads a script off disk, and the
        # script behind a finished process never changes.
        self._process_labels: dict[int, str] = {}
        self._activity = "working"
        self._conn = None
        self._saver_ctx = None

    def compose(self) -> ComposeResult:
        yield TopBar()
        with Horizontal(id="columns"):
            yield SessionsPanel("Sessions", id="sessions")
            yield ChatPanel("Chat", id="chat")
            yield ProcessesPanel("Processes", id="processes")
        yield Footer()

    async def on_mount(self) -> None:
        self.clipboard_manager = ClipboardManager(
            self.settings.clipboard, emit=self._emit_to_terminal
        )
        self._conn = connect()
        init_db(self._conn)
        # Processes the previous run was watching when it exited would claim
        # to be running forever; harmless while the panel was empty on
        # restart, a standing lie now that it shows history.
        reconcile_orphans(self._conn)
        self.session_store = SessionStore(self._conn)
        self.episodic = EpisodicStore(self._conn)
        self.memory_index = MemoryIndex(self._conn)
        self._saver_ctx = AsyncSqliteSaver.from_conn_string(str(checkpoints_db_path()))
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
        self.symbol_index = SymbolIndex(self._conn)
        self.rag_store = RagStore(app_dir() / "rag.db")
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
        self._refresh_context_bar()
        self.run_worker(self._discover_context_window(), group="llm-probe")
        await self._reload_sessions()
        self.run_curator_if_due()
        self.set_interval(2.0, self.refresh_processes)
        self.set_interval(2.0, self.watch_processes)
        if self.slurm is not None:
            self.set_interval(
                max(5, self.settings.cluster.job_poll_seconds), self.poll_jobs
            )
        self._focus_column("sessions")

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
        if self._conn is not None:
            self._conn.close()
        if getattr(self, "rag_store", None) is not None:
            self.rag_store.close()
        if getattr(self, "embedder", None) is not None:
            await self.embedder.close()
        if self._owns_llm and self._llm is not None:
            await self._llm.close()
        for client in self._session_clients.values():
            await client.close()

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

    def _render_system_prompt(self) -> str:
        """Per-call prompt assembly (§4.3): memories, skills, dynamic facts.

        A running turn reads its own session's profile memories, which may
        not be the profile on screen."""
        memory = (
            self._turn_memory
            if self._turn_memory is not None
            else self._memory_snapshot(self.profile)
        )
        skills = self._turn_skills if self._turn_skills is not None else self.skills
        caps = self.settings.memory
        backend = self._active_model()
        return orchestrator_system_prompt(
            tier1=memory.tier_prompt_text(1, active_backend=backend),
            tier2=memory.tier_prompt_text(2, active_backend=backend),
            tier1_meter=memory.usage_meter(1, caps.cap_chars(1)),
            tier2_meter=memory.usage_meter(2, caps.cap_chars(2)),
            skills=summarize_skills(skills),
            session_search="session_search" in self._tools.names(),
            memory_tool="memory" in self._tools.names(),
        )

    def _recall_lines(
        self, user_text: str, memory: Profile, profile: str
    ) -> list[str]:
        """What this request recalls: matching tier-2 struggle notes plus
        retrieved tier-3 memories (redesign Phase 5).

        Tier 3 is retrieved rather than injected wholesale, which is what lets
        it grow: situational memories cost context only on the turns they
        actually match.
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
                    limit=self.settings.memory.tier3_prefetch_count,
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
        if text.startswith(("\\", "/")):
            # Slash commands act on the UI and run their own exclusive
            # workers; they are not turns and are not queued.
            if self._busy_turn is not None:
                self.notify(
                    f"Still working in “{self._busy_turn.title}” — "
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
        queued = self._busy_turn is not None
        await self._append_chat("queued" if queued else "user", text)
        self._pending_work.append(
            PendingWork(
                session_id=self.active_session.session_id, text=text, kind="user"
            )
        )
        await self.drain_work()

    @on(TextArea.Changed, "#chat-input")
    def _on_draft_changed(self, event: TextArea.Changed) -> None:
        self._update_command_menu(event.text_area.text)

    def _update_command_menu(self, draft: str) -> None:
        """List the chat commands above the entry while one is being typed:
        substring match (so "/skill" finds every command with "skill" in it),
        most-used first, and ↑/↓ selectable (see ChatInput)."""
        menu = self.query_one("#command-menu", Static)
        stripped = draft.lstrip()
        if not stripped.startswith(("/", "\\")):
            menu.display = False
            self._command_matches = []
            return
        typed = stripped[1:].split(maxsplit=1)[0] if stripped[1:] else ""
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
        menu.border_title = "commands  (↑/↓ select · ⇥ complete)"
        self._render_command_menu()
        menu.display = True

    def _matching_commands(self, typed: str) -> list[tuple[str, str]]:
        """(name, usage) pairs whose name contains ``typed``, most-used first
        then in definition order. Empty ``typed`` matches everything."""
        needle = typed.lower()
        order = {name: i for i, (name, _) in enumerate(COMMANDS)}
        counts = self._command_counts()
        matches = [(n, u) for n, u in COMMANDS if needle in n.lower()]
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
        menu = self.query_one("#command-menu", Static)
        lines = [
            f"{'▶ ' if i == self._command_index else '  '}{usage}"
            for i, (_, usage) in enumerate(self._command_matches)
        ]
        menu.update(Content("\n".join(lines)))

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

    def _can_interrupt(self) -> bool:
        """Only while the active session's own user turn is parked on the LLM —
        the one phase where telling the backend to stop makes sense, and the
        only case with a prompt to hand back."""
        return (
            self._activity == LLM_WAIT_ACTIVITY
            and self._busy_turn is not None
            and self._interrupt_text is not None
            and self._interrupt_keep is not None
            and self._is_active_session(self._busy_turn)
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
        loop — the message comes back to the entry for editing."""
        session = self._busy_turn
        keep = self._interrupt_keep
        text = self._interrupt_text
        worker = self._turn_worker
        if session is None or keep is None or text is None:
            return
        self._interrupt_worker = self.run_worker(
            self._interrupt_turn(session, keep, text, worker), group="interrupt"
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
        self._busy_turn = None
        self.hide_working()
        try:
            surviving = await rollback_thread(
                self.graph, session_id=session.session_id, keep=keep
            )
        except Exception as e:
            self.notify(f"Interrupt cleanup failed: {e}", severity="error")
            surviving = None
        if self._is_active_session(session) and surviving is not None:
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
        self._turn_memory = self._memory_snapshot(session.profile)
        if session.profile == self.profile:
            self.profile_memory = self._turn_memory
        self._turn_skills = load_skills(session.profile)
        # Recalled memory and the volatile date/time both ride on the API copy
        # of the user message — the model is warned in-context, the stored
        # transcript stays clean, and (unlike the system prompt) the tail is
        # where changing content belongs so the cacheable prefix survives.
        api_content = None
        if user_text is not None:
            lines = self._recall_lines(user_text, self._turn_memory, session.profile)
            block = build_memory_context(
                lines,
                max_notes=self.settings.memory.tier3_prefetch_count,
                max_chars=self.settings.memory.tier3_prefetch_chars,
            )
            api_content = compose_api_content(
                user_text, block, environment_facts()
            )
        log = open_log(self.settings, session)
        self._busy_turn = session
        # A fresh user message can be interrupted and re-edited (§ interrupt);
        # a resume/event has no prompt to hand back, so it arms nothing.
        self._interrupt_text = user_text
        self._interrupt_keep = None  # filled once we know the pre-turn count
        self._turn_ctx = self._make_tool_ctx(session, log, memory=self._turn_memory)
        if self._is_active_session(session):
            # the UI's context (process list, registry) stays the turn's twin
            self._tool_ctx = self._turn_ctx
            self.show_working()
        self._turn_worker = self.run_worker(
            self._agent_turn(
                session,
                user_text=user_text,
                resume=resume,
                log=log,
                api_content=api_content,
            ),
            exclusive=True,
        )
        return self._turn_worker

    def _chat_list(self) -> ListView | None:
        """The chat column, or None once the screen is gone.

        A queued turn can start — and finish — while the app is shutting
        down, and neither the spinner appearing nor disappearing is worth
        raising over at that point.
        """
        found = self.query("#chat-list")
        return found.first(ListView) if found else None

    def show_working(self) -> None:
        """Put the spinner after the last message: a reply is on its way."""
        chat_list = self._chat_list()
        if chat_list is None:
            return
        if not chat_list.query(WorkingIndicator):
            chat_list.append(ChatItem(WorkingIndicator(self._activity)))
            chat_list.scroll_end(animate=False)

    def hide_working(self) -> None:
        chat_list = self._chat_list()
        if chat_list is None:
            return
        for item in list(chat_list.children):
            if item.query(WorkingIndicator):
                item.remove()

    def report_activity(self, activity: str) -> None:
        """What the graph is doing right now, for the spinner to say."""
        self._activity = activity  # survives leaving and re-opening the session
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
            try:
                self._interrupt_keep = await thread_message_count(
                    self.graph, session_id=session.session_id
                )
            except Exception:
                self._interrupt_keep = None
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
            self._busy_turn = None
            self._turn_worker = None
            self._interrupt_keep = None
            self._interrupt_text = None
            self._turn_ctx = None
            self._turn_memory = None
            self._turn_skills = None
            # Whatever queued up behind this turn starts as soon as this
            # handler unwinds, rather than waiting for the next timer tick.
            self.call_later(self.drain_work)
        self.hide_working()
        self._log_turn(result, log, session)
        if self._is_active_session(session):
            await self._set_chat_messages(result.messages, result.thinking)
            await self.refresh_processes()
        else:
            # The reply belongs to a session the user has left: never yank
            # them back — frame its row in the list instead.
            self._mark_session_updated(session)
        if result.interrupt is not None:
            # Parked on an approval (destructive op, or a gated execution in
            # manual/plan mode): the turn cannot move without an answer, so
            # ask even if another session is open.
            self._awaiting_approval.add(session.session_id)
            self.push_screen(
                ApprovalScreen(result.interrupt),
                lambda approved: self._on_approval(session, approved),
            )
            return
        if (
            result.plan
            and self._mode_of(session) == "plan"
            and self._is_active_session(session)
        ):
            # A plan-mode turn ended with a checklist: hand it to the user
            # to adjust and decide how to continue (§3.5). Never forced on a
            # session the user has switched away from — the plan waits in
            # the chat and the state.
            self.push_screen(
                PlanScreen(result.plan),
                lambda outcome: self._on_plan_decision(session, outcome),
            )
        if self._is_active_session(session):
            # Context dropped by compaction first: it is gone from the
            # model's view and this is the last chance to keep anything.
            evicted = self._evicted.pop(session.session_id, None)
            if evicted:
                await self.maybe_review(
                    session, evicted, forced=True, span="whole"
                )
            await self.maybe_review(session, result.messages)
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
        title = await self._propose_title(messages, log=log)
        if title is None:
            return False
        self._rename_session(session, title, by="llm", log=log)
        return True

    def _log_turn(self, result, log: SessionLog | None, session: Session) -> None:
        """Write what this turn added into the turn's own transcript — not
        into whichever session is open when the reply lands.

        Only the tail is logged, so re-reading a session never duplicates it.
        The same tail feeds the episodic index (redesign Phase 2): user and
        assistant messages only — tool traffic and thinking would drown BM25
        in tool vocabulary.
        """
        entries = build_entries(
            result.messages, result.thinking, start=result.first_new
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
        try:
            self.episodic.record(
                session_id=session.session_id,
                profile=session.profile,
                entries=turns,
            )
        except Exception as e:  # recall is best-effort; never fail the turn
            if log is not None:
                log.write("error", f"episodic index write failed: {e}")

    def _log_write(self, kind: str, text: str) -> None:
        if self._log is not None:
            self._log.write(kind, text)

    def _on_approval(self, session: Session, approved: bool | None) -> None:
        # Resume the turn on the thread it belongs to — the user may have
        # switched sessions while the approval dialog was up.
        self._awaiting_approval.discard(session.session_id)
        self._run_agent(session, resume=Command(resume={"approved": bool(approved)}))

    # ------------------------------------------------------------ agent modes

    def _mode_of(self, session: Session | None) -> str:
        """A session's interaction mode (§3.5); the configured default when
        the session never chose one (or there is no session)."""
        if session is None:
            return self.settings.agent.default_mode
        return session.mode or self.settings.agent.default_mode

    def _turn_mode(self) -> str:
        """The mode the graph obeys this round — for the turn in flight if
        there is one, else the open session. Read fresh from the store so
        cycling the mode mid-turn applies to the very next round instead of
        a stale Session copy."""
        session = (
            self._busy_turn if self._busy_turn is not None else self.active_session
        )
        if session is None:
            return self.settings.agent.default_mode
        fresh = self.session_store.get(session.session_id)
        return self._mode_of(fresh if fresh is not None else session)

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

    def _on_plan_decision(self, session: Session, outcome) -> None:
        """The user's verdict on a proposed plan (§3.5).

        ``None`` = keep planning — nothing changes, feedback is typed in
        chat. Otherwise switch the session to the chosen mode, store the
        (possibly edited) checklist in the thread state, and kick off
        execution as an event turn.
        """
        if not outcome:
            return
        mode, steps = outcome
        session.mode = mode
        self.session_store.set_mode(session.session_id, mode)
        if self._is_active_session(session):
            self.active_session.mode = mode
            self._refresh_mode_bar()
        self.run_worker(self._start_plan_execution(session, mode, steps))

    async def _start_plan_execution(
        self, session: Session, mode: str, steps: list[dict]
    ) -> None:
        # The user may have edited the checklist in the dialog; what they
        # approved is what the state must hold before execution starts.
        await self.graph.aupdate_state(
            {"configurable": {"thread_id": session.session_id}}, {"plan": steps}
        )
        self._pending_work.append(
            PendingWork(
                session_id=session.session_id,
                text=kickoff_message(mode),
                kind="event",
            )
        )
        await self.drain_work()

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
        elif command == "skill-creator":
            self.run_worker(self._skill_creator_worker(), exclusive=True)
        elif command == "skills-list":
            self._show_skills_list()
        elif command == "skill-remove":
            self.run_worker(self._skill_remove_worker(), exclusive=True)
        else:
            self.notify(f"Unknown command: /{command}", severity="warning")

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

    def _show_skills_list(self) -> None:
        """/skills-list: a read-only view of every skill the profile can see,
        marking which level each resolves to (project > profile > global)."""
        project_root = Path.cwd()
        visible = load_skills(self.profile, project_root=project_root)
        if not visible:
            self.notify(
                f"No skills for profile “{self.profile}”. Add one with "
                "/skill-creator.",
            )
            return
        project_names = {s.name for s in load_project_skills(project_root=project_root)}
        own_names = {s.name for s in load_own_skills(self.profile)}
        lines = []
        for skill in visible:
            if skill.name in project_names:
                tag = "  (project)"
            elif skill.name in own_names:
                tag = ""  # the profile's own — removable, no tag
            else:
                tag = "  (global)"
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
        profile's list feeds the next turn's prompt, and the first-ever skill
        enables the read_skill tool (the registry is shared, mutated in place)."""
        project_root = Path.cwd()
        self.skills = load_skills(self.profile, project_root=project_root)
        if (
            any_skills(project_root=project_root)
            and "read_skill" not in self._tools.names()
        ):
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
            proposals = await propose_memories(
                self._labelled_llm("memorize"),
                messages,
                tier1=self.profile_memory.tier_text(1),
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
        assert self.active_session is not None
        messages = await self._session_messages(self.active_session)
        if not messages:
            self.notify("Nothing to conclude yet.", severity="warning")
            return
        self.profile_memory = Profile.load(self.profile)
        try:
            proposals = await propose_memories(
                self._labelled_llm("conclude"),
                messages,
                tier1=self.profile_memory.tier_text(1),
            )
        except Exception as e:
            self.notify(f"/conclude failed: {e}", severity="error")
            return
        if not proposals:
            self.notify("The model proposed no memories for this conversation.")
            return
        await self._review_proposals(proposals)

    def _memory_write_blocked(
        self, tier: int, text: str, memory: Profile | None = None
    ) -> bool:
        """Hard char budget on writes (redesign Phase 1): a full tier rejects
        new memories until the user condenses it. Injection never truncates;
        only growth is stopped.

        ``memory`` is the profile the caller is about to write to — pass the
        same object, or the budget gets checked against one state and the
        write lands in another.
        """
        if tier not in (1, 2):
            return False  # tier 3 is retrieved, not injected: no budget
        target = memory if memory is not None else self.profile_memory
        cap = self.settings.memory.cap_chars(tier)
        if not target.would_exceed(tier, text, cap=cap):
            return False
        self.notify(
            f"Tier {tier} is full "
            f"({target.usage_meter(tier, cap)}) — memory NOT "
            "saved. Press ctrl+e to condense the profile, then retry.",
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
                if self._memory_write_blocked(proposal.tier, proposal.text):
                    continue
                self.profile_memory.add_memory(
                    proposal.text,
                    tier=proposal.tier,
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

    async def propose_memory_edits(
        self, operations: list[MemoryOp], *, profile: str
    ) -> str:
        """The `memory` tool's write path: validate, ask, apply (P3).

        Returns the tool result — what the user approved, or why the batch
        did not apply. The model is told the outcome plainly so it can react
        (condense and retry, or move on) instead of guessing.
        """
        loaded = Profile.load(profile)
        caps = {1: self.settings.memory.cap_chars(1), 2: self.settings.memory.cap_chars(2)}
        try:
            result = apply_batch(
                loaded,
                operations,
                backend=self._active_model(),
                caps=caps,
            )
        except MemoryOpError as e:
            return f"Memory unchanged: {e}"
        if not result.applied:
            return "Memory unchanged: " + "; ".join(result.skipped)
        approved = await self.push_screen_wait(
            MemoryBatchScreen(operations, result.flagged)
        )
        if not approved:
            return "Memory unchanged: the user rejected the proposed changes."
        # The user may have edited this file by hand since it was loaded;
        # rewriting from a stale copy would silently discard those edits.
        path = Profile.path_for(profile)
        if path.exists() and drift_detected(loaded, path.read_text()):
            backup = path.with_suffix(f".bak.{int(time())}")
            backup.write_text(path.read_text())
            self._refresh_memory_snapshot(profile)
            return (
                "Memory unchanged: the profile file changed on disk since "
                f"this session read it (backed up to {backup.name}). "
                "The edits were not applied."
            )
        result.profile.save()
        self._refresh_memory_snapshot(profile)
        if profile == self.profile:
            self.profile_memory = Profile.load(profile)
        self.notify(f"Memory updated ({len(result.applied)} change(s)).")
        self.check_memory_caps()
        report = "; ".join(result.applied)
        if result.skipped:
            report += " (skipped: " + "; ".join(result.skipped) + ")"
        return f"Saved to memory: {report}"

    def _review_due(self, session: Session, messages: list[dict]) -> bool:
        """Whether to look back at this stretch now (redesign Phase 4).

        Two triggers: a turn that visibly went wrong (§4.4 — reviewed at once,
        while the evidence is in context) and a plain counter, so learnings
        from conversations that went *fine* are captured too. The counter is
        what the old struggle-only heuristic was missing: a session where the
        user corrects a preference never trips a failure marker.
        """
        session_id = session.session_id
        count = self._turns_since_review.get(session_id, 0) + 1
        self._turns_since_review[session_id] = count
        if turn_struggled(messages):
            self._turns_since_review[session_id] = 0
            return True
        if count >= max(1, self.settings.memory.review_interval):
            self._turns_since_review[session_id] = 0
            return True
        return False

    async def maybe_review(
        self,
        session: Session,
        messages: list[dict],
        *,
        forced: bool = False,
        span: str = "recent",
    ) -> int:
        """Self-review: propose what this stretch is worth remembering.

        Runs after the reply is delivered, so it never competes with the
        user's turn. Best-effort throughout — a failed review is invisible.
        ``forced`` skips the cadence check, for the one case that cannot
        wait: context about to be discarded by compaction.
        """
        if forced:
            # A forced review covers this stretch as thoroughly as a due one,
            # so restart the cadence — otherwise the counter trips again a
            # turn later and re-proposes what was just reviewed.
            self._turns_since_review[session.session_id] = 0
        elif not self._review_due(session, messages):
            return 0
        memory = self._memory_snapshot(session.profile)
        skills = load_skills(session.profile)
        try:
            proposals = await propose_reflections(
                self._labelled_llm("review"),
                messages,
                tier1=memory.tier_text(1),
                tier2=memory.tier_text(2),
                tier3=memory.tier_text(3),
                skills=summarize_skills(skills),
                allow_new_skills=self.settings.memory.propose_new_skills,
                span=span,
            )
        except Exception:
            return 0  # reflection is best-effort; never disrupt the user
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
            self.notify(f"Self-review: kept {kept} of {len(proposals)}.")
            self.check_memory_caps()
        return kept

    def _apply_reflection(self, proposal: Reflection, profile: str) -> bool:
        """Persist one approved proposal; returns whether anything was written."""
        if proposal.kind in ("memory", "struggle"):
            text = proposal.memory_text()
            # Struggle notes go to tier 3: they are situational by nature and
            # were the main source of tier-2 bloat. Retrieval brings them
            # back when a request actually resembles the old one.
            tier = 3 if proposal.kind == "struggle" else proposal.tier
            target = Profile.load(profile)  # merge, don't clobber
            if self._memory_write_blocked(tier, text, target):
                return False
            target.add_memory(
                text,
                tier=tier,
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

    def check_memory_caps(self) -> list[int]:
        """§6.4 size warnings; returns the tiers currently over their cap.

        A tier can only get over cap through hand edits (in-app writes are
        rejected at the cap), so the fix offered is the external editor."""
        caps = self.settings.memory
        over = self.profile_memory.over_cap_tiers(
            tier1_cap=caps.cap_chars(1),
            tier2_cap=caps.cap_chars(2),
        )
        for tier in over:
            meter = self.profile_memory.usage_meter(tier, caps.cap_chars(tier))
            self.notify(
                f"Profile tier {tier} is over its budget ({meter}) — "
                "press ctrl+e to edit the profile externally.",
                severity="warning",
                timeout=12,
            )
        return over

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
        memory: Profile | None = None,
    ) -> ToolContext:
        """A tool context bound to one session and its transcript, so a turn
        keeps its own registry, runner and log however the UI moves on."""
        if memory is None:
            memory = self.profile_memory
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
            llm=self._llm,
            trash=self.trash,
            tier1_text=memory.tier_prompt_text(
                1, active_backend=self._active_model()
            ),
            symbols=self.symbol_index,
            rag=self.rag_store,
            embedder=self.embedder,
            episodic=self.episodic,
            skills=self._turn_skills if self._turn_skills is not None else self.skills,
            propose_memory_edits=lambda operations: self.propose_memory_edits(
                operations, profile=session.profile
            ),
        )
        if log is not None:
            ctx.llm = LoggedLLM(
                self._llm,
                log,
                label=lambda: f"subagent:{ctx.current_tool or '?'}",
            )
        return ctx

    def _activate_session(self, session: Session) -> None:
        # A session boundary is a deliberate refresh point (redesign Phase 1):
        # memories written by another session or instance are picked up here,
        # while WITHIN a session the frozen snapshot keeps the prompt prefix
        # byte-stable for the backend's prefix cache.
        self._refresh_memory_snapshot(session.profile)
        self.active_session = session
        self._refresh_session_log()
        self.query_one("#chat-input", ChatInput).display = True
        self._refresh_mode_bar()
        # The top bar and context meter follow the opened session's LLM.
        self._refresh_top_bar()
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

    def _labelled_llm(self, label: str, *, log: SessionLog | None = None) -> Any:
        """The client for one of the app's own sub-agent calls (titling,
        \\conclude, struggle notes), logged under that name."""
        sink = log if log is not None else self._log
        if sink is None:
            return self._llm
        return LoggedLLM(self._llm, sink, label=lambda: f"subagent:{label}")

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
            self._context_used = 0
            bar = self._context_bar()
            if bar is not None:
                bar.reset()  # a fresh thread starts from an empty window
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
        title = await self._propose_title(messages)
        if title is None:
            self.notify("The model could not write a title.", severity="error")
            return
        self._rename_session(session, title, by="llm")
        self.notify(f"Renamed to “{title}”")

    async def _propose_title(
        self, messages: list[dict], *, log: SessionLog | None = None
    ) -> str | None:
        try:
            return await propose_title(
                self._labelled_llm("title", log=log), messages
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
        self.session_store.delete(session.session_id)
        # Patient-data environment: a deleted conversation must not resurface
        # through episodic search either.
        self.episodic.forget_session(session.session_id)
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
        self._context_used = 0
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
        self.active_session = None
        self._tool_ctx = None
        self._log = None
        self._context_used = 0
        bar = self._context_bar()
        if bar is not None:
            bar.reset()
        await self._set_chat_messages([])
        self.query_one("#chat-input", ChatInput).display = False
        self._refresh_mode_bar()
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
            values.get("messages", []), values.get("thinking", [])
        )
        self._show_context_estimate(values)
        self._updated.discard(session.session_id)  # its news is now on screen
        for item in self.query_one("#sessions-list", ListView).children:
            row_session = getattr(item, "data_session", None)
            if row_session is not None and row_session.session_id == session.session_id:
                item.remove_class("session-updated")
        if (
            self._busy_turn is not None
            and self._busy_turn.session_id == session.session_id
        ):
            self.show_working()  # its turn is still in flight

    @on(ListView.Selected, "#sessions-list")
    async def _on_session_selected(self, event: ListView.Selected) -> None:
        session = getattr(event.item, "data_session", None)
        if session is None:
            self.pick_profile_for_new_session()
        else:
            await self.open_session(session)
            self._focus_column("chat")  # entering a session means typing in it

    async def _reload_sessions(self) -> None:
        sessions_list = self.query_one("#sessions-list", ListView)
        await sessions_list.clear()
        new_item = ListItem(Label("(new session)"))
        new_item.data_session = None
        items = [new_item]
        for session in self.session_store.list_all():
            tag = (
                f"  · {session.profile}"
                if session.profile != DEFAULT_PROFILE
                else ""
            )
            item = ListItem(Label(Content(f"{session.title}{tag}")))
            item.data_session = session
            item.set_class(session.session_id in self._updated, "session-updated")
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

    def _is_active_session(self, session: Session | None) -> bool:
        return (
            session is not None
            and self.active_session is not None
            and session.session_id == self.active_session.session_id
        )

    # ------------------------------------------------------------- processes

    async def refresh_processes(self) -> None:
        """Repaint the right column from the *table*, not the live runner.

        The runner only knows the processes it started, and a fresh one is
        built per turn, so reading it emptied the panel the moment a session
        was reopened. The table outlives all of that, which is what makes the
        history still be there after a restart.
        """
        session = self.active_session
        records: list[ProcessRecord] = []
        truncated = 0
        if session is not None and self._conn is not None:
            records = list_processes(
                self._conn, session_id=session.session_id, limit=PROCESS_HISTORY_LIMIT
            )
            if len(records) > PROCESS_HISTORY_LIMIT:
                records = records[:PROCESS_HISTORY_LIMIT]
                truncated = (
                    count_processes(self._conn, session_id=session.session_id)
                    - PROCESS_HISTORY_LIMIT
                )
        jobs = self.job_store.list(session_id=session.session_id) if session else []
        signature = [(r.pid, r.state) for r in records] + [
            (j.job_id, j.state) for j in jobs
        ]
        if signature == getattr(self, "_process_signature", None):
            return  # unchanged; avoid churn from the 2s timer
        self._process_signature = signature
        processes_list = self.query_one("#processes-list", ListView)
        await processes_list.clear()
        items = []
        for job in jobs:
            label = f"{job.state:<9} job {job.job_id} ({job.script_key})"
            item = ListItem(Static(Content(label), classes="proc-job"))
            item.data_job = job
            items.append(item)
        for record in records:
            label = (
                f"{format_started(record.started_at)} {record.state:<8} "
                f"{self._describe_process(record)} ({record.pid})"
            )
            item = ListItem(Static(Content(label), classes=f"proc-{record.state}"))
            item.data_record = record
            items.append(item)
        if truncated > 0:
            # Never let a cut list read as the whole history.
            items.append(
                ListItem(Static(Content(f"… {truncated} older"), classes="proc-more"))
            )
        processes_list.extend(items)

    def _describe_process(self, record: ProcessRecord) -> str:
        """Panel label for a process, cached by pid.

        ``describe`` reads the script from disk for throwaway run_bash names,
        and the panel repaints on every change; the content of a written
        script never changes, so once is enough.
        """
        cached = self._process_labels.get(record.pid)
        if cached is None:
            cached = describe(record)
            self._process_labels[record.pid] = cached
        return cached

    async def poll_jobs(self) -> None:
        """Background sacct poll (§5.4); notifies on state changes."""
        assert self.slurm is not None
        try:
            changes = await poll_active(self.slurm, self.job_store)
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
            await self.refresh_processes()

    async def watch_processes(self) -> None:
        """Turn finished background subprocesses into agent-visible events.

        The sibling of poll_jobs for local work. refresh_processes only
        repaints the sidebar, so before this nothing ever told the agent that
        the script it started had exited — it promised to check back and had
        no way to keep the promise.
        """
        if self._conn is None:
            return
        try:
            changes = poll_processes(self._conn)
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
                tier1=self.profile_memory.tier_text(1) if self.profile_memory else "",
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
        """Start the next waiting turn, if the orchestrator is free.

        One item per pass: concurrent ainvoke on a thread would interleave
        checkpoint writes, and draining a burst all at once would stack model
        calls anyway. Called when work arrives and again when a turn ends, so
        the queue does not sit waiting for the next timer tick.
        """
        if self._busy_turn is not None or self._shutting_down:
            return
        # A session parked on an approval has a thread mid-interrupt; its next
        # message must wait for the resume, but other sessions need not.
        index = next(
            (
                i
                for i, work in enumerate(self._pending_work)
                if work.session_id not in self._awaiting_approval
            ),
            None,
        )
        if index is None:
            return
        item = self._pending_work.pop(index)
        try:
            if item.kind == "user":
                session = self.session_store.get(item.session_id)
                if session is not None:
                    if self._is_active_session(session):
                        await self._promote_queued_entry(item.text)
                    self._run_agent(session, user_text=item.text)
            else:
                await self._deliver_event(item.session_id, item.text)
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

    def _selected_item(self) -> tuple[ProcessRecord | None, JobRow | None]:
        highlighted = self.query_one("#processes-list", ListView).highlighted_child
        return (
            getattr(highlighted, "data_record", None),
            getattr(highlighted, "data_job", None),
        )

    def inspect_selected_process(self) -> None:
        record, job = self._selected_item()
        if record is not None:
            self.push_screen(
                InspectScreen(f"Process: {record.name}", format_process(record))
            )
        elif job is not None:
            self.push_screen(InspectScreen(f"Job {job.job_id}", format_job(job)))

    def kill_selected_process(self) -> None:
        record, job = self._selected_item()
        if record is not None:
            if record.state != "running":
                self.notify(f"{record.name} is not running", severity="warning")
                return
            self._confirm_then(
                f"Kill process {record.name!r} (pid {record.pid})?",
                self._kill_process_and_refresh(record.pid),
            )
        elif job is not None:
            from hpca.slurm import TERMINAL_STATES

            if job.state in TERMINAL_STATES:
                self.notify(f"Job {job.job_id} is already {job.state}",
                            severity="warning")
                return
            self._confirm_then(
                f"Cancel cluster job {job.job_id} ({job.script_key})?",
                self._cancel_job_and_refresh(job.job_id),
            )

    def _confirm_then(self, question: str, coro) -> None:
        def on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self.run_worker(coro)
            else:
                coro.close()

        self.push_screen(ConfirmScreen(question), on_confirm)

    async def _kill_process_and_refresh(self, pid: int) -> None:
        """Kill by pid, whichever turn started it.

        The panel now shows the session's whole history, so the highlighted
        process may predate the current runner — which would have no monitor
        for it and raise. Prefer the owning runner when there is one, since
        its monitor records the outcome properly.
        """
        runner = self._tool_ctx.runner if self._tool_ctx else None
        if runner is not None and runner.owns(pid):
            await runner.kill(pid)
            await runner.wait(pid)
        elif self._conn is not None and self.active_session is not None:
            kill_unowned(
                self._conn, pid=pid, session_id=self.active_session.session_id
            )
        await self.refresh_processes()

    async def _cancel_job_and_refresh(self, job_id: str) -> None:
        assert self.slurm is not None
        try:
            await self.slurm.cancel(job_id)
        except Exception as e:
            self.notify(f"Cancel failed: {e}", severity="error")
            return
        self.job_store.mark(job_id, "CANCELLING")
        await self.refresh_processes()

    # -------------------------------------------------------------- chat log

    def chat_log_texts(self) -> list[str]:
        return [entry.text for entry in self._chat_entries]

    async def _set_chat_messages(
        self, messages: list[dict], thinking: list[dict] | None = None
    ) -> None:
        chat_list = self.query_one("#chat-list", ListView)
        await chat_list.clear()
        self._chat_entries = []
        # Fresh entries mean the old id()-keyed expansion state is stale (and
        # a recycled id could wrongly re-open a new box); start clean.
        self._thinking_expanded.clear()
        for entry in build_entries(messages, thinking or []):
            self._add_chat_entry(entry)
        # Messages typed while this turn ran are not in the graph yet, so the
        # rebuild would erase them from under the user.
        if self.active_session is not None:
            for text in self.queued_texts_for(self.active_session.session_id):
                self._add_chat_entry(Entry(kind="queued", text=text))

    async def _rerender_chat(self) -> None:
        """Redraw the chat from the entries already held, without rebuilding
        them from the graph — used when only an entry's kind changed."""
        chat_list = self.query_one("#chat-list", ListView)
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
            return [ChatItem(self._entry_widget(entry))]
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
            # processes columns (and on modals) shift+tab keeps moving focus.
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
        """Focus a column. The chat column lands on its entry, ready to type."""
        if column_id == "chat" and self.active_session is not None:
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
        else:
            # Enter on a message moves to the input (message actions later)
            self.focus_chat_input()

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
        if self._busy_turn is not None and self._busy_turn.profile == name:
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
        if self._busy_turn is not None:
            self.notify(
                "The agent is mid-reply — switch backends once it finishes.",
                severity="warning",
            )
            return
        if self.active_session is None:
            return
        blob = backend.model_dump_json()
        self.session_store.set_backend(self.active_session.session_id, blob)
        self.active_session.backend = blob
        self._refresh_top_bar()
        self._refresh_context_bar()
        self._refresh_session_log()  # rebind the tool context to the new client
        self.notify(f"This session now uses {backend.model}")

    def reload_llm(self) -> None:
        """Rebuild the client so edited LLM settings apply to the next turn.

        An injected client belongs to whoever passed it in (tests, embedding
        hosts); only a client we built is ours to replace.
        """
        if self._busy_turn is not None:
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
        # A different model means a different window, and the token count
        # measured against the old one no longer describes this one.
        self._discovered_window = None
        self._context_used = 0
        bar = self._context_bar()
        if bar is not None:
            bar.reset()
        self._refresh_context_bar()
        self.run_worker(self._discover_context_window(), group="llm-probe")
        if owned and old_llm is not None:
            self.run_worker(old_llm.close(), group="llm-close")

    def _rebuild_graph(self) -> None:
        self.graph = build_graph(
            # Per-session LLM: the turn resolves its client from whichever
            # session is running (busy turn), so no rebuild is needed on switch.
            llm=lambda: self._client_for(self._busy_turn or self.active_session),
            tools=self._tools,
            checkpointer=self._checkpointer,
            # a running turn keeps its own context however the UI moves on
            ctx=lambda: self._turn_ctx if self._turn_ctx is not None else self._tool_ctx,
            system_prompt_fn=self._render_system_prompt,
            max_retries=self.settings.llm.max_retries,
            max_tool_rounds=self.settings.llm.max_tool_rounds,
            on_activity=lambda activity: self.report_activity(activity),
            max_model_len=self._active_max_model_len,
            on_evict=self._extract_before_eviction,
            on_usage=self._on_usage,
            mode_fn=self._turn_mode,
        )

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
        """Whose LLM the display/tagging should reflect right now: the running
        turn if there is one, else the session on screen."""
        return self._busy_turn or self.active_session

    def _active_backend(self) -> LLMBackend | None:
        return self._backend_of(self._reads_session())

    def _active_model(self) -> str:
        """The model name to show/tag with: the current session's, else the
        bootstrap model."""
        backend = self._active_backend()
        return backend.model if backend is not None else self.settings.llm.model

    def _active_max_model_len(self) -> int | None:
        """The current session's context window, from its backend (or the
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

    def _on_usage(self, usage: dict) -> None:
        """The backend's token count for the decision just made.

        prompt_tokens is what occupies the window; the completion is spent
        the moment it is generated. Reported per round, so a tool-heavy turn
        visibly fills the bar as it works.
        """
        prompt_tokens = usage.get("prompt_tokens")
        if not prompt_tokens:
            return
        self._context_used = int(prompt_tokens)
        bar = self._context_bar()
        if bar is not None:
            bar.set_used(self._context_used)

    def _context_bar(self) -> ContextBar | None:
        found = self.query("#context-bar")
        return found.first(ContextBar) if found else None

    def _refresh_context_bar(self) -> None:
        bar = self._context_bar()
        if bar is None:
            return
        bar.set_window(self._active_max_model_len())
        if self._context_used:
            bar.set_used(self._context_used)

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

    async def _extract_before_eviction(self, messages: list[dict]) -> None:
        """Last look at context about to leave the model's view (Phase 6).

        Compaction is the one moment where something the agent learned can
        disappear without anyone deciding to drop it, so the review loop gets
        a chance at it first — but this runs *inside* the graph round, where
        putting a modal on screen would suspend the turn behind a dialog the
        user did not ask for. So the slice is only captured here; the review
        itself runs after the reply lands, like every other review.
        """
        session = self._busy_turn
        if session is None:
            return
        self._evicted.setdefault(session.session_id, []).extend(messages)

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
        self.query_one(TopBar).update_info(
            profile=self.profile, model=self._active_model()
        )
