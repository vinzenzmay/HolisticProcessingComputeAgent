"""What Enter on one of your own past messages offers (§ chat rewind).

Three ways to pick the conversation up from a message you sent: fork the
session from just before it (the original stays whole), roll this session
back to just before it (everything after is dropped), or simply copy the text
back into the entry — the old behavior, still on Enter so the reflex of
activating a message twice keeps doing what it always did.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

FORK = "fork"
ROLLBACK = "rollback"
COPY = "copy"

# The dialog quotes the message being rewound from so the user can check they
# grabbed the right one — a preview, not the transcript; the log has the rest.
PREVIEW_CHARS = 300


class RewindScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "cancel", priority=True),
        Binding("f", f"choose('{FORK}')", "fork"),
        Binding("r", f"choose('{ROLLBACK}')", "roll back"),
        Binding("c,enter", f"choose('{COPY}')", "copy"),
    ]

    DEFAULT_CSS = """
    RewindScreen {
        align: center middle;
    }
    #rewind-dialog {
        width: 70;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #rewind-message {
        color: $text-muted;
        margin: 1 0;
        max-height: 6;
    }
    #rewind-hint {
        color: $text-muted;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        preview = self._message[:PREVIEW_CHARS]
        if preview != self._message:
            preview += " …"
        with Vertical(id="rewind-dialog"):
            yield Static("This message again:", id="rewind-question")
            yield Static(Content(preview), id="rewind-message")
            yield Static(
                "(f) fork the session from here\n"
                "(r) roll this conversation back to here\n"
                "(c / enter) copy it into the entry",
                id="rewind-options",
            )
            yield Static("(esc) cancel", id="rewind-hint")

    def action_choose(self, choice: str) -> None:
        self.dismiss(choice)

    def action_cancel(self) -> None:
        self.dismiss(None)
