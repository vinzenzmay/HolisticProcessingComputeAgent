"""Main application: three-column layout and arrow-key focus model (§3.2, §3.3)."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, ListView, Static

from hpca.clipboard import ClipboardManager, CopyResult
from hpca.config import Settings
from hpca.tui.settings_screen import SettingsScreen

COLUMN_IDS = ("sessions", "chat", "processes")


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
    #processes {
        width: 1fr;
        min-width: 24;
    }
    """

    BINDINGS = [
        Binding("left", "focus_column(-1)", "◀ column", show=False),
        Binding("right", "focus_column(1)", "column ▶", show=False),
        Binding("s", "open_settings", "settings"),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or Settings.load()

    def compose(self) -> ComposeResult:
        yield TopBar()
        with Horizontal(id="columns"):
            yield ColumnPanel("Sessions", id="sessions")
            yield ColumnPanel("Chat", id="chat")
            yield ColumnPanel("Processes", id="processes")
        yield Footer()

    def on_mount(self) -> None:
        self.clipboard_manager = ClipboardManager(
            self.settings.clipboard, emit=self._emit_to_terminal
        )
        self._refresh_top_bar()
        self._focus_column("sessions")

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

    def action_open_settings(self) -> None:
        def apply(result: Settings | None) -> None:
            if result is not None:
                self.settings = result
                self.settings.save()
                self._refresh_top_bar()

        self.push_screen(SettingsScreen(self.settings), apply)

    def _refresh_top_bar(self) -> None:
        self.query_one(TopBar).update_info(
            profile="default", model=self.settings.llm.model
        )
