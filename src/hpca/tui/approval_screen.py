"""HITL confirmation modal for destructive operations (§5.3).

Shows the exact operation the agent wants to perform; dismisses with a bool.
"""

from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static


class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "approve", "approve"),
        Binding("n", "deny", "deny"),
        Binding("escape", "deny", "deny", priority=True),
    ]

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        border: heavy $error;
        background: $surface;
        padding: 1 2;
    }
    #approval-title {
        text-style: bold;
        color: $error;
    }
    #approval-hint {
        color: $text-muted;
    }
    """

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self._payload = payload

    def details_text(self) -> str:
        return (
            f"Tool: {self._payload.get('tool')}\n"
            f"Arguments: {json.dumps(self._payload.get('arguments'), indent=2)}\n"
            f"{self._payload.get('description', '')}"
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static("Destructive operation — approve?", id="approval-title")
            yield Static(Content(self.details_text()), id="approval-details")
            yield Static("(y) approve · (n) deny", id="approval-hint")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
