"""Plan-mode handoff modal (§3.5).

Shown when a plan-mode turn ends with a checklist: the user reads the plan,
may edit it (it is a plain ``[ ]``/``[x]`` checklist in a text area), and
decides how to continue — execute on auto, execute step-by-step under manual
approval, or keep planning. Dismisses with ``(mode, steps)`` or ``None`` for
"keep planning"; the transition itself is the app's job.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea

from hpca.agent.modes import parse_checklist, render_checklist

PlanDecision = tuple[str, list[dict]] | None


class PlanScreen(ModalScreen[PlanDecision]):
    BINDINGS = [
        Binding("ctrl+r", "execute_auto", "run on auto", priority=True),
        Binding("ctrl+s", "execute_manual", "run step-by-step", priority=True),
        Binding("escape", "keep", "keep planning", priority=True),
    ]

    DEFAULT_CSS = """
    PlanScreen {
        align: center middle;
    }
    #plan-dialog {
        width: 76;
        height: auto;
        max-height: 85%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #plan-title {
        text-style: bold;
        color: $accent;
    }
    #plan-editor {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }
    #plan-hint {
        color: $text-muted;
    }
    """

    def __init__(self, steps: list[dict]) -> None:
        super().__init__()
        self._steps = steps

    def compose(self) -> ComposeResult:
        with Vertical(id="plan-dialog"):
            yield Static("The agent proposes this plan", id="plan-title")
            yield TextArea(render_checklist(self._steps), id="plan-editor")
            yield Static(
                "Edit freely ([x] marks a step done), then:\n"
                "ctrl+r  execute on auto · ctrl+s  execute step-by-step "
                "(each script asks first) · esc  keep planning",
                id="plan-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#plan-editor", TextArea).focus()

    def _edited_steps(self) -> list[dict]:
        text = self.query_one("#plan-editor", TextArea).text
        steps = parse_checklist(text)
        # An emptied-out checklist is not a plan; keep what the agent wrote
        # rather than executing nothing.
        return steps or self._steps

    def action_execute_auto(self) -> None:
        self.dismiss(("auto", self._edited_steps()))

    def action_execute_manual(self) -> None:
        self.dismiss(("manual", self._edited_steps()))

    def action_keep(self) -> None:
        self.dismiss(None)
