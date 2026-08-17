"""The ``/thinking`` chooser: how hard this session's model reasons.

Same shape as ``SwitchLLMScreen`` — a modal over a four-item ListView, escape
cancels, enter picks — because it is the same kind of decision: a per-session
property with a handful of named values and one currently in force. The four
levels and what they mean are hpca.thinking's business, not this screen's.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from hpca.thinking import (
    EFFORT_HINTS,
    EFFORTS,
    XHIGH_INLINE_WARNING,
    normalize_effort,
)


class ThinkingScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "cancel", priority=True)]

    DEFAULT_CSS = """
    ThinkingScreen { align: center middle; }
    #thinking-dialog {
        width: 76;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #thinking-title { height: 1; text-style: bold; }
    #thinking-hint { color: $text-muted; }
    .thinking-warn { color: $warning; }
    """

    def __init__(self, current: str | None = None) -> None:
        super().__init__()
        self._current = normalize_effort(current)

    def compose(self) -> ComposeResult:
        with Vertical(id="thinking-dialog"):
            yield Static("Thinking effort for this session", id="thinking-title")
            yield ListView(id="thinking-list")
            yield Static("(esc) cancel · (enter) choose", id="thinking-hint")

    def on_mount(self) -> None:
        effort_list = self.query_one("#thinking-list", ListView)
        for effort in EFFORTS:
            star = " ★" if effort == self._current else ""
            # The warning rides on the option itself, not only in the toast
            # afterwards: this is the moment the choice is made, and xhigh is
            # the one level whose cost a user cannot guess from its name — it
            # is currently a level that loses turns, not merely a slow one.
            warn = f"  {XHIGH_INLINE_WARNING}" if effort == "xhigh" else ""
            item = ListItem(
                Label(
                    Content(f"{effort}{star}{warn}\n  {EFFORT_HINTS[effort]}"),
                    classes="thinking-warn" if effort == "xhigh" else "",
                )
            )
            item.data_effort = effort
            effort_list.append(item)
        effort_list.index = EFFORTS.index(self._current)
        effort_list.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(getattr(event.item, "data_effort", None))

    def action_cancel(self) -> None:
        self.dismiss(None)
