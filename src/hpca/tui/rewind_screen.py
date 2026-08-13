"""What Enter on one of your own messages offers.

For a message already in the thread (§ chat rewind), three ways to pick the
conversation up from it: fork the session from just before it (the original
stays whole), roll this session back to just before it (everything after is
dropped), or simply copy the text back into the entry — the old behavior,
still on Enter so the reflex of activating a message twice keeps doing what it
always did.

For one still queued behind a running turn, the same copy plus the thing a
turn in flight already had: cancel it. See :class:`QueuedScreen`.
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
UNQUEUE = "unqueue"

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


class QueuedScreen(ModalScreen[str | None]):
    """Enter on a message typed ahead of a running turn.

    A queued message is the one kind of message the user can still take back:
    it has not reached the model, so cancelling it costs nothing and needs no
    thread surgery — the reason a turn in flight offers an interrupt but a sent
    message only offers a rewind. Cancelling lands where the interrupt lands,
    with the text back in the entry to edit and send again.

    Escape leaves it queued, and Enter still copies, so activating a message
    twice does here what it does on the rewind dialog.
    """

    BINDINGS = [
        Binding("escape", "cancel", "cancel", priority=True),
        Binding("x", f"choose('{UNQUEUE}')", "cancel it"),
        Binding("c,enter", f"choose('{COPY}')", "copy"),
    ]

    DEFAULT_CSS = """
    QueuedScreen {
        align: center middle;
    }
    #queued-dialog {
        width: 70;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #queued-message {
        color: $text-muted;
        margin: 1 0;
        max-height: 6;
    }
    #queued-hint {
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
        with Vertical(id="queued-dialog"):
            yield Static("This message is still waiting to run:", id="queued-question")
            yield Static(Content(preview), id="queued-message")
            yield Static(
                "(x) cancel it — the text comes back to the entry\n"
                "(c / enter) copy it into the entry, leave it queued",
                id="queued-options",
            )
            yield Static("(esc) leave it queued", id="queued-hint")

    def action_choose(self, choice: str) -> None:
        self.dismiss(choice)

    def action_cancel(self) -> None:
        self.dismiss(None)
