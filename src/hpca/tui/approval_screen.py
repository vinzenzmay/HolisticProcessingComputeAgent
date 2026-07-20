"""HITL confirmation modal for gated operations (§5.3, §3.5).

Two kinds of gate share this screen, told apart by ``payload["kind"]``:

* ``destructive`` — the always-on gate for destructive operations.
* ``execution`` — manual/plan mode showing a script or command before it
  runs; the user decides on the actual script text, not on a JSON blob.

Dismisses with a bool either way; the graph resumes with it.
"""

from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
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
    ApprovalScreen.execution #approval-dialog {
        border: heavy $warning;
    }
    #approval-title {
        text-style: bold;
        color: $error;
    }
    ApprovalScreen.execution #approval-title {
        color: $warning;
    }
    #approval-script {
        height: auto;
        max-height: 16;
        border: round $panel;
        padding: 0 1;
        margin: 1 0;
    }
    #approval-hint {
        color: $text-muted;
    }
    """

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self._payload = payload
        if self.kind == "execution":
            self.add_class("execution")

    @property
    def kind(self) -> str:
        return self._payload.get("kind", "destructive")

    def title_text(self) -> str:
        if self.kind == "execution":
            return f"Run this — {self._payload.get('tool')}?"
        return "Destructive operation — approve?"

    def hint_text(self) -> str:
        if self.kind == "execution":
            return "(y) run script · (n) skip script"
        return "(y) approve · (n) deny"

    def details_text(self) -> str:
        text = f"Tool: {self._payload.get('tool')}"
        # With a script shown below, the raw arguments would repeat it as a
        # JSON blob; without one they are all there is to judge the call by.
        if not self._payload.get("script"):
            text += (
                f"\nArguments: {json.dumps(self._payload.get('arguments'), indent=2)}"
            )
        description = self._payload.get("description", "")
        if description:
            text += f"\n{description}"
        details = self._payload.get("details")
        if details:  # resolved real paths (§5.3)
            text += f"\n\n{details}"
        return text

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static(self.title_text(), id="approval-title")
            yield Static(Content(self.details_text()), id="approval-details")
            script = self._payload.get("script")
            if script:
                with VerticalScroll(id="approval-script"):
                    yield Static(Content(script))
            yield Static(self.hint_text(), id="approval-hint")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
