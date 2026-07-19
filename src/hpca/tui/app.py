"""Main application: three-column layout, focus model, agent wiring (§3, §4)."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from os import environ as os_environ
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

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.explainer import explain_process_failure
from hpca.agent.graph import build_graph, deliver_event, run_turn
from hpca.agent.job_tools import add_job_tools
from hpca.agent.tools import ToolRegistry
from hpca.clipboard import ClipboardManager, CopyResult
from hpca.config import LLMBackend, Settings, app_dir
from hpca.db import checkpoints_db_path, connect, init_db
from hpca.jobs import JobRow, JobStore, poll_active
from hpca.agent.conclude import MemoryProposal, propose_memories
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.memory_context import (
    build_memory_context,
    compose_api_content,
    note_line,
)
from hpca.agent.memory_tools import add_memory_tools
from hpca.agent.prompts import orchestrator_system_prompt
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.titler import propose_title
from hpca.agent.struggle import (
    STRUGGLE_KIND,
    matching_struggles,
    propose_struggle_note,
    turn_struggled,
)
from hpca.editor import resolve_editor
from hpca.embeddings import EmbeddingClient
from hpca.episodic import EpisodicStore
from hpca.llm import LLMClient
from hpca.logs import LoggedLLM, SessionLog, open_log
from hpca.rag import RagStore
from hpca.transcript import ASSISTANT as ASSISTANT_ENTRY
from hpca.transcript import THINKING, Entry, build_entries
from hpca.transcript import USER as USER_ENTRY
from hpca.profiles import DEFAULT_PROFILE, Profile
from hpca.registry import PathRegistry
from hpca.runner import (
    ProcessRecord,
    ProcessRunner,
    analyse_process_failure,
    format_process_event,
    poll_processes,
    running_session_ids,
)
from hpca.triage import Signature, append_user_signature
from hpca.sessions import Session, SessionStore
from hpca.skills import Skill, any_skills, load_skills, summarize_skills
from hpca.slurm import TERMINAL_STATES as SLURM_TERMINAL_STATES
from hpca.slurm import SlurmClient
from hpca.symbols import SymbolIndex
from hpca.trash import TrashManager
from hpca.tui.approval_screen import ApprovalScreen
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen, format_job, format_process
from hpca.tui.manage_llms import ManageLLMsScreen
from hpca.memory_ops import (
    MemoryOp,
    MemoryOpError,
    apply_batch,
    drift_detected,
)
from hpca.tui.memory_screens import MemoryBatchScreen, MemoryProposalScreen
from hpca.tui.profiles_screen import ProfilePickerScreen, ProfilesScreen
from hpca.tui.rename_screen import RenameScreen
from hpca.tui.settings_screen import SettingsScreen
from hpca.tui.switch_llm import SwitchLLMScreen

COLUMN_IDS = ("sessions", "chat", "processes")
SESSION_TITLE_MAX = 40
UNTITLED_SESSION = "untitled"  # placeholder until the first message names it
CHAT_TITLES = {
    "user": "you",
    "assistant": "agent",
    "error": "error",
    "event": "background",
    "queued": "queued",
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
)
LOG_KINDS = {
    "user": "user",
    "assistant": "agent",
    "error": "error",
    "event": "background",
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
    to — folded into a single box. Collapsed by default; enter toggles it."""

    def __init__(self, entry: Entry) -> None:
        super().__init__(classes="chat-thinking")
        self.border_title = "thinking"
        self._entry = entry
        self._collapsed = True
        self._render_entry()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._render_entry()

    def _render_entry(self) -> None:
        marker = "▶" if self._collapsed else "▼"
        hint = "enter to expand" if self._collapsed else "enter to collapse"
        header = f"{marker} {self._entry.summary()}  ({hint})"
        body = "" if self._collapsed else "\n\n" + self._entry.text
        self.update(Content(header + body))


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

    def _render_frame(self) -> None:
        elapsed = f" {self.elapsed}s" if self.elapsed else ""
        self.update(
            Content(f"{self.FRAMES[self._frame]} {self._activity}…{elapsed}")
        )


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
        yield ChatList(id="chat-list")
        menu = Static(id="command-menu")
        menu.display = False
        yield menu
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

    BINDINGS = [
        Binding("left", "focus_column(-1)", "◀ column", show=False),
        Binding("right", "focus_column(1)", "column ▶", show=False),
        Binding("c", "open_settings", "config editor"),
        Binding("m", "manage_llms", "manage llms"),
        Binding("a", "manage_profiles", "profiles & learnings"),
        Binding("ctrl+l", "switch_llm", "switch llm"),
        Binding("ctrl+e", "edit_profile", "edit profile", show=False),
        # Not priority: (q) must reach the chat entry as a letter. Offered
        # only on the sessions column, where no typing happens (check_action).
        Binding("q", "confirm_quit", "quit"),
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
        self.active_session: Session | None = None
        self._tool_ctx: ToolContext | None = None
        self._chat_entries: list[Entry] = []
        self._log: SessionLog | None = None
        self._untitled: set[str] = set()  # sessions awaiting their first title
        self._updated: set[str] = set()  # replies that landed while switched away
        self._busy_turn: Session | None = None  # the one turn in flight, if any
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
        self.session_store = SessionStore(self._conn)
        self.episodic = EpisodicStore(self._conn)
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
        await self._reload_sessions()
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
        turns). See ``_memory_snapshots`` for why it is not reloaded per turn."""
        snapshot = self._memory_snapshots.get(profile)
        if snapshot is None:
            snapshot = Profile.load(profile)
            self._memory_snapshots[profile] = snapshot
        return snapshot

    def _refresh_memory_snapshot(self, profile: str) -> None:
        """Drop a profile's frozen view; the next turn reloads from disk.

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
        backend = self.settings.llm.model
        return orchestrator_system_prompt(
            tier1=memory.tier_prompt_text(1, active_backend=backend),
            tier2=memory.tier_prompt_text(2, active_backend=backend),
            tier1_meter=memory.usage_meter(1, caps.cap_chars(1)),
            tier2_meter=memory.usage_meter(2, caps.cap_chars(2)),
            skills=summarize_skills(skills),
            session_search="session_search" in self._tools.names(),
            memory_tool="memory" in self._tools.names(),
        )

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
        """List the chat commands above the entry while one is being typed,
        narrowed as the name grows — so /memorize is discoverable, not lore."""
        menu = self.query_one("#command-menu", Static)
        stripped = draft.lstrip()
        if not stripped.startswith(("/", "\\")):
            menu.display = False
            return
        typed = stripped[1:].split(maxsplit=1)[0] if stripped[1:] else ""
        matches = [usage for name, usage in COMMANDS if name.startswith(typed)]
        if not matches:  # unknown: show what exists rather than nothing
            matches = [usage for _, usage in COMMANDS]
        menu.border_title = "commands"
        menu.update(Content("\n".join(matches)))
        menu.display = True

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
        # Recalled struggle notes ride on the API copy of the user message —
        # the model is warned in-context, the stored transcript stays clean.
        api_content = None
        if user_text is not None:
            matches = matching_struggles(self._turn_memory.memories, user_text)
            if matches:
                block = build_memory_context([note_line(m) for m in matches])
                if block:
                    api_content = compose_api_content(user_text, block)
        log = open_log(self.settings, session)
        self._busy_turn = session
        self._turn_ctx = self._make_tool_ctx(session, log, memory=self._turn_memory)
        if self._is_active_session(session):
            # the UI's context (process list, registry) stays the turn's twin
            self._tool_ctx = self._turn_ctx
            self.show_working()
        return self.run_worker(
            self._agent_turn(
                session,
                user_text=user_text,
                resume=resume,
                log=log,
                api_content=api_content,
            ),
            exclusive=True,
        )

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
            # Parked on a destructive-op approval: the turn cannot move
            # without an answer, so ask even if another session is open.
            self._awaiting_approval.add(session.session_id)
            self.push_screen(
                ApprovalScreen(result.interrupt),
                lambda approved: self._on_approval(session, approved),
            )
            return
        if self._is_active_session(session):
            await self.maybe_propose_struggle_note(result.messages)
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

    # ---------------------------------------------------------------- memory

    def _handle_slash_command(self, text: str) -> None:
        command, _, rest = text[1:].partition(" ")
        rest = rest.strip()
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
        else:
            self.notify(f"Unknown command: /{command}", severity="warning")

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

    def _memory_write_blocked(self, tier: int, text: str) -> bool:
        """Hard char budget on writes (redesign Phase 1): a full tier rejects
        new memories until the user condenses it. Injection never truncates;
        only growth is stopped."""
        if tier not in (1, 2):
            return False
        cap = self.settings.memory.cap_chars(tier)
        if not self.profile_memory.would_exceed(tier, text, cap=cap):
            return False
        self.notify(
            f"Tier {tier} is full "
            f"({self.profile_memory.usage_meter(tier, cap)}) — memory NOT "
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
                    backend=self.settings.llm.model,
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
                backend=self.settings.llm.model,
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

    async def maybe_propose_struggle_note(self, messages: list[dict]) -> bool:
        """§4.4: after a bad turn, propose a struggle note for approval."""
        if not turn_struggled(messages):
            return False
        try:
            note = await propose_struggle_note(
                self._labelled_llm("struggle"), messages
            )
        except Exception:
            return False  # reflection is best-effort; never disrupt the user
        proposal = MemoryProposal(
            tier=2, kind=STRUGGLE_KIND, text=note.render()
        )
        approved = await self.push_screen_wait(
            MemoryProposalScreen(proposal, 1, 1)
        )
        if not approved:
            return False
        if self._memory_write_blocked(2, proposal.text):
            return False
        self.profile_memory.add_memory(
            proposal.text,
            tier=2,
            backend=self.settings.llm.model,
            kind=STRUGGLE_KIND,
        )
        self.profile_memory.save()
        self._refresh_memory_snapshot(self.profile)
        self.notify("Struggle note saved to the profile.")
        self.check_memory_caps()
        return True

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
                1, active_backend=self.settings.llm.model
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
        if self._log is not None:
            self._log.write(
                "session opened",
                f"{session.session_id} · profile {self.profile} · "
                f"model {self.settings.llm.model}",
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
        """Every new session starts by choosing its profile (or creating one)."""

        def chosen(profile: str | None) -> None:
            if profile is not None:
                self.run_worker(
                    self.start_new_session(profile=profile), group="sessions"
                )

        self.push_screen(ProfilePickerScreen(current=self.profile), chosen)

    async def start_new_session(self, profile: str | None = None) -> None:
        """Open an empty session under the given profile and start typing.

        An untouched session is reused (and retagged to the chosen profile)
        rather than piling up empty rows when "(new session)" is entered
        repeatedly.
        """
        profile = profile or self.profile
        self._set_working_profile(profile)
        if self._is_untouched(self.active_session):
            if self.active_session.profile != profile:
                self.session_store.set_profile(
                    self.active_session.session_id, profile
                )
                self.active_session.profile = profile
                self._refresh_session_log()
                await self._reload_sessions()
        else:
            self._activate_session(
                self.session_store.create(profile=profile, title=UNTITLED_SESSION)
            )
            await self._set_chat_messages([])
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

    async def close_session(self) -> None:
        """Leave the active session: empty chat, no entry, nothing to type in."""
        self.active_session = None
        self._tool_ctx = None
        self._log = None
        await self._set_chat_messages([])
        self.query_one("#chat-input", ChatInput).display = False
        self._focus_column("sessions")

    async def open_session(self, session: Session) -> None:
        self._set_working_profile(session.profile)
        self._activate_session(session)
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        values = snapshot.values or {}
        await self._set_chat_messages(
            values.get("messages", []), values.get("thinking", [])
        )
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
        records = self._tool_ctx.runner.list() if self._tool_ctx else []
        jobs = (
            self.job_store.list(session_id=self.active_session.session_id)
            if self.active_session
            else []
        )
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
            label = f"{record.state:<9} {record.name} ({record.pid})"
            item = ListItem(Static(Content(label), classes=f"proc-{record.state}"))
            item.data_record = record
            items.append(item)
        processes_list.extend(items)

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
        assert self._tool_ctx is not None
        await self._tool_ctx.runner.kill(pid)
        await self._tool_ctx.runner.wait(pid)
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
            chat_list.append(ChatItem(self._entry_widget(entry)))
        chat_list.scroll_end(animate=False)

    async def _append_chat(self, kind: str, text: str) -> None:
        self._add_chat_entry(Entry(kind=kind, text=text))

    def _add_chat_entry(self, entry: Entry) -> None:
        self._chat_entries.append(entry)
        chat_list = self._chat_list()
        if chat_list is None:
            return  # screen already gone (shutdown); the entry is still kept
        chat_list.append(ChatItem(self._entry_widget(entry)))
        chat_list.scroll_end(animate=False)

    def _entry_widget(self, entry: Entry) -> Static:
        if entry.kind == THINKING:
            return ThinkingBox(entry)
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
            return in_chat
        if action == "open_settings":
            return not in_chat
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
    def _on_chat_list_selected(self, event: ListView.Selected) -> None:
        boxes = list(event.item.query(ThinkingBox))
        if boxes:
            boxes[0].toggle()
        else:
            # Enter on a message moves to the input (message actions later)
            self.focus_chat_input()

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
        """Make a profile; returns an error message, or None on success."""
        try:
            cleaned = Profile.validate_name(name)
        except ValueError as e:
            return str(e)
        Profile.create(cleaned)
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
        self._refresh_memory_snapshot(name)
        if self.profile == name:  # unlikely, but keep the app coherent
            self.profile = "default"
            self.profile_memory = Profile.load("default")
            self.skills = load_skills("default")
        self.notify(
            f"Deleted “{name}”" + (f"; {moved} session(s) moved to default" if moved else "")
        )

    def action_switch_llm(self) -> None:
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
            SwitchLLMScreen(list(self.settings.backends), self.settings.is_active),
            apply,
        )

    def switch_backend(self, backend: LLMBackend) -> None:
        """Make a configured backend the active one, now and on next start."""
        if self._busy_turn is not None:
            self.notify(
                "The agent is mid-reply — switch backends once it finishes.",
                severity="warning",
            )
            return
        self.settings.activate_backend(backend)
        self.settings.save()
        self._replace_llm()
        self._refresh_top_bar()
        self.notify(f"Switched to {backend.model}")

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
        if owned and old_llm is not None:
            self.run_worker(old_llm.close(), group="llm-close")

    def _rebuild_graph(self) -> None:
        self.graph = build_graph(
            llm=self._llm,
            tools=self._tools,
            checkpointer=self._checkpointer,
            # a running turn keeps its own context however the UI moves on
            ctx=lambda: self._turn_ctx if self._turn_ctx is not None else self._tool_ctx,
            system_prompt_fn=self._render_system_prompt,
            max_retries=self.settings.llm.max_retries,
            max_tool_rounds=self.settings.llm.max_tool_rounds,
            on_activity=lambda activity: self.report_activity(activity),
        )

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
            profile=self.profile, model=self.settings.llm.model
        )
