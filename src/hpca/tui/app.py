"""Main application: three-column layout, focus model, agent wiring (§3, §4)."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from hpca.agent.graph import build_graph, run_turn
from hpca.agent.tools import ToolRegistry
from hpca.clipboard import ClipboardManager, CopyResult
from hpca.config import Settings
from hpca.db import connect, db_path, init_db
from hpca.llm import LLMClient
from hpca.sessions import Session, SessionStore
from hpca.tui.approval_screen import ApprovalScreen
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
        profile: str = "default",
    ) -> None:
        super().__init__()
        self.settings = settings or Settings.load()
        self.profile = profile
        self._llm = llm
        self._owns_llm = llm is None
        self._tools = tools if tools is not None else ToolRegistry()
        self.active_session: Session | None = None
        self._chat_entries: list[tuple[str, str]] = []
        self._conn = None
        self._saver_ctx = None

    def compose(self) -> ComposeResult:
        yield TopBar()
        with Horizontal(id="columns"):
            yield ColumnPanel("Sessions", id="sessions")
            yield ChatPanel("Chat", id="chat")
            yield ColumnPanel("Processes", id="processes")
        yield Footer()

    async def on_mount(self) -> None:
        self.clipboard_manager = ClipboardManager(
            self.settings.clipboard, emit=self._emit_to_terminal
        )
        self._conn = connect()
        init_db(self._conn)
        self.session_store = SessionStore(self._conn)
        self._saver_ctx = AsyncSqliteSaver.from_conn_string(str(db_path()))
        checkpointer = await self._saver_ctx.__aenter__()
        if self._llm is None:
            self._llm = LLMClient(self.settings.llm)
        self.graph = build_graph(
            llm=self._llm,
            tools=self._tools,
            checkpointer=checkpointer,
            max_retries=self.settings.llm.max_retries,
        )
        self._refresh_top_bar()
        await self._reload_sessions()
        self._focus_column("sessions")

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
            self.active_session = self.session_store.create(
                profile=self.profile, title=text[:SESSION_TITLE_MAX]
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
        if result.interrupt is not None:
            self.push_screen(ApprovalScreen(result.interrupt), self._on_approval)

    def _on_approval(self, approved: bool | None) -> None:
        self._run_agent(resume=Command(resume={"approved": bool(approved)}))

    # -------------------------------------------------------------- sessions

    async def start_new_session(self) -> None:
        """Clear the chat; a session row is created on the first message."""
        self.active_session = None
        await self._set_chat_messages([])

    async def open_session(self, session: Session) -> None:
        self.active_session = session
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
