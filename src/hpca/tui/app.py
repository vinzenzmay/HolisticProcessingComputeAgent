"""Main application: three-column layout, focus model, agent wiring (§3, §4)."""

from __future__ import annotations

import shutil
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.graph import build_graph, run_turn
from hpca.agent.job_tools import add_job_tools
from hpca.agent.tools import ToolRegistry
from hpca.clipboard import ClipboardManager, CopyResult
from hpca.config import Settings, app_dir
from hpca.db import checkpoints_db_path, connect, init_db
from hpca.jobs import JobRow, JobStore, poll_active
from hpca.llm import LLMClient
from hpca.registry import PathRegistry
from hpca.runner import ProcessRecord, ProcessRunner
from hpca.sessions import Session, SessionStore
from hpca.slurm import SlurmClient
from hpca.trash import TrashManager
from hpca.tui.approval_screen import ApprovalScreen
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen, format_job, format_process
from hpca.tui.settings_screen import SettingsScreen

COLUMN_IDS = ("sessions", "chat", "processes")
SESSION_TITLE_MAX = 40


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
        return f" HPCA │ profile: {self._profile} │ model: {self._model} │ (s) settings"


class ColumnPanel(Vertical):
    """One of the three main columns; its ListView receives focus."""

    def __init__(self, title: str, *, id: str) -> None:
        super().__init__(id=id)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield ListView(id=f"{self.id}-list")


class ChatPanel(ColumnPanel):
    """Center column: message log plus the chat input."""

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="column-title")
        yield ListView(id="chat-list")
        yield Input(placeholder="Message the agent…", id="chat-input")


class ProcessesList(ListView):
    """Right-column list; its hotkeys appear in the footer when focused."""

    BINDINGS = [
        Binding("i", "inspect_process", "inspect"),
        Binding("k", "kill_process", "kill"),
    ]

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
    #processes {
        width: 1fr;
        min-width: 24;
    }
    .chat-user {
        color: $text;
        text-style: bold;
    }
    .chat-assistant {
        color: $text;
    }
    .chat-tool {
        color: $text-muted;
    }
    .chat-error {
        color: $error;
    }
    """

    BINDINGS = [
        Binding("left", "focus_column(-1)", "◀ column", show=False),
        Binding("right", "focus_column(1)", "column ▶", show=False),
        Binding("s", "open_settings", "settings"),
        Binding("ctrl+q", "quit", "quit"),
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
        if tools is not None:
            self._tools = tools
        else:
            self._tools = add_file_tools(default_tool_registry())
            if self.slurm is not None:
                add_job_tools(self._tools)
        self.active_session: Session | None = None
        self._tool_ctx: ToolContext | None = None
        self._chat_entries: list[tuple[str, str]] = []
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
        if self._llm is None:
            self._llm = LLMClient(self.settings.llm)
        self.graph = build_graph(
            llm=self._llm,
            tools=self._tools,
            checkpointer=checkpointer,
            ctx=lambda: self._tool_ctx,
            max_retries=self.settings.llm.max_retries,
        )
        self.job_store = JobStore(self._conn)
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

    # ------------------------------------------------------------ chat/agent

    @on(Input.Submitted, "#chat-input")
    async def _on_chat_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if self.active_session is None:
            self._activate_session(
                self.session_store.create(
                    profile=self.profile, title=text[:SESSION_TITLE_MAX]
                )
            )
            await self._reload_sessions()
        await self._append_chat("user", text)
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
            self.notify(str(e), severity="error")
            return
        await self._set_chat_messages(result.messages)
        await self.refresh_processes()
        if result.interrupt is not None:
            self.push_screen(ApprovalScreen(result.interrupt), self._on_approval)

    def _on_approval(self, approved: bool | None) -> None:
        self._run_agent(resume=Command(resume={"approved": bool(approved)}))

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
        )

    async def start_new_session(self) -> None:
        """Clear the chat; a session row is created on the first message."""
        self.active_session = None
        self._tool_ctx = None
        await self._set_chat_messages([])

    async def open_session(self, session: Session) -> None:
        self._activate_session(session)
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        messages = (snapshot.values or {}).get("messages", [])
        await self._set_chat_messages(messages)

    @on(ListView.Selected, "#sessions-list")
    async def _on_session_selected(self, event: ListView.Selected) -> None:
        session = getattr(event.item, "data_session", None)
        if session is None:
            await self.start_new_session()
        else:
            await self.open_session(session)

    async def _reload_sessions(self) -> None:
        sessions_list = self.query_one("#sessions-list", ListView)
        await sessions_list.clear()
        new_item = ListItem(Label("(new session)"))
        new_item.data_session = None
        items = [new_item]
        for session in self.session_store.list(profile=self.profile):
            item = ListItem(Label(session.title))
            item.data_session = session
            items.append(item)
        sessions_list.extend(items)

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
        return [text for _, text in self._chat_entries]

    async def _set_chat_messages(self, messages: list[dict]) -> None:
        entries: list[tuple[str, str]] = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "system":
                continue
            if role == "assistant":
                entries.append(("assistant", content))
            elif content.startswith(("[tool result]", "[tool error]")):
                entries.append(("tool", content))
            else:
                entries.append(("user", content))
        chat_list = self.query_one("#chat-list", ListView)
        await chat_list.clear()
        self._chat_entries = []
        for kind, text in entries:
            await self._append_chat(kind, text)

    async def _append_chat(self, kind: str, text: str) -> None:
        self._chat_entries.append((kind, text))
        chat_list = self.query_one("#chat-list", ListView)
        prefix = {"user": "you", "assistant": "agent", "tool": "tool", "error": "!"}[
            kind
        ]
        item = ListItem(
            Static(Content(f"{prefix} ▏{text}"), classes=f"chat-{kind}")
        )
        chat_list.append(item)
        chat_list.scroll_end(animate=False)

    # ----------------------------------------------------------- focus model

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
        self.query_one(f"#{column_id}-list", ListView).focus()

    def action_focus_column(self, delta: int) -> None:
        current = self.focused_column_id
        index = COLUMN_IDS.index(current) if current in COLUMN_IDS else 0
        self._focus_column(COLUMN_IDS[(index + delta) % len(COLUMN_IDS)])

    @on(ListView.Selected, "#chat-list")
    def _on_chat_list_selected(self, event: ListView.Selected) -> None:
        # Enter in the chat column moves to the input (message actions later)
        self.query_one("#chat-input", Input).focus()

    # -------------------------------------------------------------- settings

    def action_open_settings(self) -> None:
        def apply(result: Settings | None) -> None:
            if result is not None:
                self.settings = result
                self.settings.save()
                self._refresh_top_bar()

        self.push_screen(SettingsScreen(self.settings), apply)

    def _refresh_top_bar(self) -> None:
        self.query_one(TopBar).update_info(
            profile=self.profile, model=self.settings.llm.model
        )
