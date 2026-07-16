"""Small yes/no confirmation modal (kill, cancel, …)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "yes", "yes"),
        Binding("n", "no", "no"),
        Binding("escape", "no", "no", priority=True),
    ]

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 60;
        height: auto;
        border: heavy $warning;
        background: $surface;
        padding: 1 2;
    }
    #confirm-hint {
        color: $text-muted;
    }
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(Content(self._question), id="confirm-question")
            yield Static("(y) yes · (n) no", id="confirm-hint")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)
