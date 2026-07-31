"""A read-only modal for text too long to be a toast.

It was built to inspect a process or a cluster job from the right column, and
the two formatters that rendered them lived here. The column holds only watch
boxes now, and a watch answers itself with a peek, so both formatters went with
the rows they served — see git history if that view is ever wanted back. What
is left is the screen itself, which the profile's skill listing still opens.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static


class InspectScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close", priority=True)]

    DEFAULT_CSS = """
    InspectScreen {
        align: center middle;
    }
    #inspect-dialog {
        width: 90%;
        height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #inspect-title {
        height: 1;
        text-style: bold;
    }
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def body_text(self) -> str:
        return self._body

    def compose(self) -> ComposeResult:
        with Vertical(id="inspect-dialog"):
            yield Static(self._title, id="inspect-title")
            with VerticalScroll():
                yield Static(Content(self._body), id="inspect-body")

    def action_close(self) -> None:
        self.dismiss(None)
