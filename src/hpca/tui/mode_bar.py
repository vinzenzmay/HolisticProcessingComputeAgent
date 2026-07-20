"""The one-line agent-mode indicator right above the chat entry (§3.5).

Same pattern as the ContextBar: a single-height Static the app feeds. Only
visible while a session is open — mode is a per-session property.
"""

from __future__ import annotations

from textual.widgets import Static

MODE_HINTS = {
    "manual": "scripts run only with your approval",
    "auto": "works until the task is done",
    "full-auto": "asks for nothing, destructive ops included",
    "plan": "drafts a checklist, executes nothing",
}
# ctrl+m is carriage return in most terminals (it arrives as Enter), so
# shift+tab is the binding that always works; ctrl+m is bound too for
# terminals whose keyboard protocol can tell them apart.
SWITCH_HINT = "shift+tab to switch"


class ModeBar(Static):
    DEFAULT_CSS = """
    ModeBar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    ModeBar.mode-manual {
        color: $warning;
    }
    ModeBar.mode-auto {
        color: $success;
    }
    ModeBar.mode-full-auto {
        color: $error;
    }
    ModeBar.mode-plan {
        color: $accent;
    }
    """

    mode: str = ""

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        for known in MODE_HINTS:
            self.remove_class(f"mode-{known}")
        self.add_class(f"mode-{mode}")
        hint = MODE_HINTS.get(mode, "")
        label = mode.replace("-", " ")  # "full-auto" reads as "full auto"
        self.update(f"mode: {label} — {hint} · {SWITCH_HINT}")
