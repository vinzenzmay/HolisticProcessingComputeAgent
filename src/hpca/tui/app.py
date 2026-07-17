"""Main application: three-column layout, focus model, agent wiring (§3, §4)."""

from __future__ import annotations

import shutil
from os import environ as os_environ
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Footer, Label, ListItem, ListView, Static, TextArea

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.graph import build_graph, run_turn
from hpca.agent.job_tools import add_job_tools
from hpca.agent.tools import ToolRegistry
from hpca.clipboard import ClipboardManager, CopyResult
from hpca.config import LLMBackend, Settings, app_dir
from hpca.db import checkpoints_db_path, connect, init_db
from hpca.jobs import JobRow, JobStore, poll_active
from hpca.agent.conclude import MemoryProposal, propose_memories
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.prompts import orchestrator_system_prompt
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.struggle import (
    STRUGGLE_KIND,
    matching_struggles,
    propose_struggle_note,
    turn_struggled,
)
from hpca.editor import resolve_editor
from hpca.embeddings import EmbeddingClient
from hpca.llm import LLMClient
from hpca.logs import LoggedLLM, SessionLog, open_log
from hpca.rag import RagStore
from hpca.transcript import THINKING, Entry, build_entries
from hpca.profiles import Profile
from hpca.registry import PathRegistry
from hpca.runner import ProcessRecord, ProcessRunner
from hpca.sessions import Session, SessionStore
from hpca.skills import load_skills, summarize_skills
from hpca.slurm import SlurmClient
from hpca.symbols import SymbolIndex
from hpca.trash import TrashManager
from hpca.tui.approval_screen import ApprovalScreen
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen, format_job, format_process
from hpca.tui.manage_llms import ManageLLMsScreen
from hpca.tui.memory_screens import MemoryProposalScreen, TierSelectScreen
from hpca.tui.settings_screen import SettingsScreen
from hpca.tui.switch_llm import SwitchLLMScreen

COLUMN_IDS = ("sessions", "chat", "processes")
SESSION_TITLE_MAX = 40
UNTITLED_SESSION = "untitled"  # placeholder until the first message names it
CHAT_TITLES = {"user": "you", "assistant": "agent", "error": "error"}
LOG_KINDS = {"user": "user", "assistant": "agent", "error": "error"}


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
            f"model: {self._model} │ (s) settings"
        )


class ColumnPanel(Vertical):
    """One of the three main columns; its ListView receives focus."""

    def __init__(self, title: str, *, id: str) -> None:
        super().__init__(id=id)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield ListView(id=f"{self.id}-list")


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
    """

    BINDINGS = [
        Binding("left", "focus_column(-1)", "◀ column", show=False),
        Binding("right", "focus_column(1)", "column ▶", show=False),
        Binding("s", "open_settings", "settings"),
        Binding("m", "manage_llms", "manage llms"),
        Binding("ctrl+l", "switch_llm", "switch llm"),
        Binding("ctrl+e", "edit_profile", "edit profile", show=False),
        Binding("ctrl+q", "confirm_quit", "quit", priority=True),
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
        self.skills = load_skills()
        if tools is not None:
            self._tools = tools
        else:
            self._tools = add_ask_docs(
                add_doc_tools(add_file_tools(default_tool_registry()))
            )
            if self.slurm is not None:
                add_job_tools(self._tools)
            if self.skills:
                add_skill_tools(self._tools)
        self.active_session: Session | None = None
        self._tool_ctx: ToolContext | None = None
        self._chat_entries: list[Entry] = []
        self._log: SessionLog | None = None
        self._conn = None
        self._saver_ctx = None

    def compose(self) -> ComposeResult:
        yield TopBar()
        with Horizontal(id="columns"):
            yield ColumnPanel("Sessions", id="sessions")
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
        self._saver_ctx = AsyncSqliteSaver.from_conn_string(str(checkpoints_db_path()))
        checkpointer = await self._saver_ctx.__aenter__()
        self._checkpointer = checkpointer  # kept for graph rebuilds on switch
        if self._llm is None:
            self._llm = LLMClient(self.settings.llm)
        self.profile_memory = Profile.load(self.profile)
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

    def _render_system_prompt(self) -> str:
        """Per-call prompt assembly (§4.3): memories, skills, dynamic facts."""
        return orchestrator_system_prompt(
            tier1=self.profile_memory.tier_text(1),
            tier2=self.profile_memory.tier_text(2),
            skills=summarize_skills(self.skills),
        )

    def warn_about_struggles(self, text: str) -> list:
        """§4.4: warn up front when a request matches a past struggle."""
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
        event.chat_input.text = ""
        if text.startswith("\\"):
            self._handle_slash_command(text)
            return
        if self.active_session is None:
            await self.start_new_session()
        if self.active_session.title == UNTITLED_SESSION:
            self._name_session(text[:SESSION_TITLE_MAX])
            await self._reload_sessions()
        await self._append_chat("user", text)
        self.warn_about_struggles(text)
        self._run_agent(user_text=text)

    def _run_agent(self, *, user_text: str | None = None, resume: Command | None = None):
        return self.run_worker(
            self._agent_turn(user_text=user_text, resume=resume), exclusive=True
        )

    async def _agent_turn(
        self, *, user_text: str | None, resume: Command | None
    ) -> None:
        assert self.active_session is not None
        try:
            result = await run_turn(
                self.graph,
                session_id=self.active_session.session_id,
                user_text=user_text,
                resume=resume,
            )
        except Exception as e:
            await self._append_chat("error", f"Agent error: {e}")
            self._log_write("error", str(e))
            self.notify(str(e), severity="error")
            return
        self._log_turn(result)
        await self._set_chat_messages(result.messages, result.thinking)
        await self.refresh_processes()
        if result.interrupt is not None:
            self.push_screen(ApprovalScreen(result.interrupt), self._on_approval)
            return
        await self.maybe_propose_struggle_note(result.messages)

    def _log_turn(self, result) -> None:
        """Write what this turn added: the message, its thinking, the answer.

        Only the tail is logged, so re-reading a session never duplicates it.
        """
        if self._log is None:
            return
        for entry in build_entries(
            result.messages, result.thinking, start=result.first_new
        ):
            kind = LOG_KINDS.get(entry.kind, entry.kind)
            if entry.kind == THINKING:
                kind = f"thinking ({entry.summary()})"
            self._log.write(kind, entry.text)

    def _log_write(self, kind: str, text: str) -> None:
        if self._log is not None:
            self._log.write(kind, text)

    def _on_approval(self, approved: bool | None) -> None:
        self._run_agent(resume=Command(resume={"approved": bool(approved)}))

    # ---------------------------------------------------------------- memory

    def _handle_slash_command(self, text: str) -> None:
        command, _, rest = text[1:].partition(" ")
        rest = rest.strip()
        if command == "memorize":
            if not rest:
                self.notify(r"Usage: \memorize <text>", severity="warning")
                return

            def on_tier(tier: int | None) -> None:
                if tier is None:
                    return
                self.profile_memory.add_memory(
                    rest, tier=tier, backend=self.settings.llm.model, kind="note"
                )
                self.profile_memory.save()
                self.notify(f"Memorized into tier {tier}.")
                self.check_memory_caps()

            self.push_screen(TierSelectScreen(rest), on_tier)
        elif command == "conclude":
            if self.active_session is None:
                self.notify("No active session to conclude.", severity="warning")
                return
            self.run_worker(self._conclude_worker(), exclusive=True)
        else:
            self.notify(f"Unknown command: \\{command}", severity="warning")

    async def _conclude_worker(self) -> None:
        assert self.active_session is not None
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": self.active_session.session_id}}
        )
        messages = (snapshot.values or {}).get("messages", [])
        if not messages:
            self.notify("Nothing to conclude yet.", severity="warning")
            return
        try:
            proposals = await propose_memories(
                self._llm, messages, tier1=self.profile_memory.tier_text(1)
            )
        except Exception as e:
            self.notify(f"\\conclude failed: {e}", severity="error")
            return
        if not proposals:
            self.notify("The model proposed no memories for this conversation.")
            return
        kept = 0
        for i, proposal in enumerate(proposals, start=1):
            approved = await self.push_screen_wait(
                MemoryProposalScreen(proposal, i, len(proposals))
            )
            if approved:
                self.profile_memory.add_memory(
                    proposal.text,
                    tier=proposal.tier,
                    backend=self.settings.llm.model,
                    kind=proposal.kind,
                )
                kept += 1
        if kept:
            self.profile_memory.save()
        self.notify(f"Kept {kept} of {len(proposals)} proposed memories.")
        self.check_memory_caps()

    async def maybe_propose_struggle_note(self, messages: list[dict]) -> bool:
        """§4.4: after a bad turn, propose a struggle note for approval."""
        if not turn_struggled(messages):
            return False
        try:
            note = await propose_struggle_note(self._llm, messages)
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
        self.profile_memory.add_memory(
            proposal.text,
            tier=2,
            backend=self.settings.llm.model,
            kind=STRUGGLE_KIND,
        )
        self.profile_memory.save()
        self.notify("Struggle note saved to the profile.")
        self.check_memory_caps()
        return True

    def check_memory_caps(self) -> list[int]:
        """§6.4 size warnings; returns the tiers currently over their cap."""
        over = self.profile_memory.over_cap_tiers(
            tier1_cap=self.settings.memory.tier1_token_cap,
            tier2_cap=self.settings.memory.tier2_token_cap,
        )
        for tier in over:
            self.notify(
                f"Profile tier {tier} is over its token cap "
                f"({self.profile_memory.tier_tokens(tier)} tokens) — "
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

    def _activate_session(self, session: Session) -> None:
        self.active_session = session
        self._tool_ctx = ToolContext(
            registry=PathRegistry(
                self._conn, profile=self.profile, session_id=session.session_id
            ),
            runner=ProcessRunner(
                self._conn,
                session_id=session.session_id,
                log_dir=app_dir() / "proc_logs",
            ),
            settings=self.settings,
            scripts_dir=app_dir() / "scripts",
            session_id=session.session_id,
            profile=self.profile,
            slurm=self.slurm,
            jobs=self.job_store,
            job_log_dir=app_dir() / "job_logs",
            llm=self._llm,
            trash=self.trash,
            tier1_text=self.profile_memory.tier_text(1),
            symbols=self.symbol_index,
            rag=self.rag_store,
            embedder=self.embedder,
            skills=self.skills,
        )
        self.query_one("#chat-input", ChatInput).display = True
        self._refresh_session_log()
        if self._log is not None:
            self._log.write(
                "session opened",
                f"{session.session_id} · profile {self.profile} · "
                f"model {self.settings.llm.model}",
            )

    def _refresh_session_log(self) -> None:
        """(Re)open the active session's transcript and re-wrap the client
        tools call, so switching logging in the settings takes effect now."""
        self._log = (
            open_log(self.settings, self.active_session)
            if self.active_session is not None
            else None
        )
        if self._tool_ctx is not None:
            self._tool_ctx.llm = self._subagent_llm()

    def _subagent_llm(self) -> Any:
        """The client tools use for their own model calls (§4.2), logged."""
        if self._log is None or self._tool_ctx is None:
            return self._llm
        context = self._tool_ctx
        return LoggedLLM(
            self._llm,
            self._log,
            label=lambda: f"subagent:{context.current_tool or '?'}",
        )

    async def start_new_session(self) -> None:
        """Open an empty session on the default backend and start typing in it.

        An untouched session is reused rather than piling up empty rows when
        "(new session)" is entered repeatedly.
        """
        if not self._is_untouched(self.active_session):
            self._activate_session(
                self.session_store.create(
                    profile=self.profile, title=UNTITLED_SESSION
                )
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
        """A session is named after its opening message (§3 sessions column)."""
        assert self.active_session is not None
        self.session_store.rename(self.active_session.session_id, title)
        self.active_session.title = title

    async def open_session(self, session: Session) -> None:
        self._activate_session(session)
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        values = snapshot.values or {}
        await self._set_chat_messages(
            values.get("messages", []), values.get("thinking", [])
        )

    @on(ListView.Selected, "#sessions-list")
    async def _on_session_selected(self, event: ListView.Selected) -> None:
        session = getattr(event.item, "data_session", None)
        if session is None:
            await self.start_new_session()
        else:
            await self.open_session(session)
            self._focus_column("chat")  # entering a session means typing in it

    async def _reload_sessions(self) -> None:
        sessions_list = self.query_one("#sessions-list", ListView)
        await sessions_list.clear()
        new_item = ListItem(Label("(new session)"))
        new_item.data_session = None
        items = [new_item]
        for session in self.session_store.list(profile=self.profile):
            item = ListItem(Label(Content(session.title)))
            item.data_session = session
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
        if changes:
            await self.refresh_processes()

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

    async def _append_chat(self, kind: str, text: str) -> None:
        self._add_chat_entry(Entry(kind=kind, text=text))

    def _add_chat_entry(self, entry: Entry) -> None:
        self._chat_entries.append(entry)
        chat_list = self.query_one("#chat-list", ListView)
        chat_list.append(ListItem(self._entry_widget(entry)))
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

        (m) manage llms only from the main screen's sessions column, where
        sessions are started; (ctrl+l) switch llm only from the chat column,
        which is also the one place settings are not offered — its letter keys
        belong to the message being typed. Returning False also hides the
        binding from the footer.
        """
        on_main_screen = len(self.screen_stack) == 1
        in_chat = on_main_screen and self.focused_column_id == "chat"
        if action == "manage_llms":
            return on_main_screen and self.focused_column_id == "sessions"
        if action == "switch_llm":
            return in_chat
        if action == "open_settings":
            return not in_chat
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
        self.settings.activate_backend(backend)
        self.settings.save()
        self._replace_llm()
        self._refresh_top_bar()
        self.notify(f"Switched to {backend.model}")

    def _reload_llm(self) -> None:
        """Rebuild the client so edited LLM settings apply to the next turn.

        An injected client belongs to whoever passed it in (tests, embedding
        hosts); only a client we built is ours to replace.
        """
        if self._owns_llm:
            self._replace_llm()

    def _replace_llm(self) -> None:
        old_llm, owned = self._llm, self._owns_llm
        self._llm = LLMClient(self.settings.llm)
        self._owns_llm = True
        self._rebuild_graph()
        if self._tool_ctx is not None:
            self._tool_ctx.llm = self._subagent_llm()
        if owned and old_llm is not None:
            self.run_worker(old_llm.close(), group="llm-close")

    def _rebuild_graph(self) -> None:
        self.graph = build_graph(
            llm=self._llm,
            tools=self._tools,
            checkpointer=self._checkpointer,
            ctx=lambda: self._tool_ctx,
            system_prompt_fn=self._render_system_prompt,
            max_retries=self.settings.llm.max_retries,
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
                self._reload_llm()  # so llm settings take effect without a restart
                self._refresh_session_log()
                self._refresh_top_bar()

        self.push_screen(SettingsScreen(self.settings), apply)

    def _refresh_top_bar(self) -> None:
        self.query_one(TopBar).update_info(
            profile=self.profile, model=self.settings.llm.model
        )
