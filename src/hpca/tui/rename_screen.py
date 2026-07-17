"""Name something small: rename a session, name a new profile."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static


class RenameScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "cancel", priority=True)]

    DEFAULT_CSS = """
    RenameScreen {
        align: center middle;
    }
    #rename-dialog {
        width: 60;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #rename-hint {
        color: $text-muted;
    }
    """

    def __init__(self, title: str, *, label: str = "Rename session") -> None:
        super().__init__()
        self._title = title
        self._label = label

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-dialog"):
            yield Static(self._label, id="rename-label")
            yield Input(value=self._title, id="rename-input")
            yield Static("(enter) save · (escape) cancel", id="rename-hint")

    def on_mount(self) -> None:
        name = self.query_one("#rename-input", Input)
        name.focus()
        name.cursor_position = len(name.value)  # edit the name, don't retype it

    @on(Input.Submitted, "#rename-input")
    def _on_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
