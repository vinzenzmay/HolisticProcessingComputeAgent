"""Modal settings editor (§3.3): edit the settings JSON, validated on save.

The screen validates and returns a new ``Settings`` object via ``dismiss``;
persisting it is the app's job.
"""

from __future__ import annotations

import json

from pydantic import ValidationError
from textual.app import ComposeResult
from textual.content import Content
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, TextArea

from hpca.config import Settings


class SettingsScreen(ModalScreen[Settings | None]):
    BINDINGS = [
        Binding("escape", "cancel", "cancel", priority=True),
        Binding("ctrl+s", "save", "save", priority=True),
    ]

    DEFAULT_CSS = """
    SettingsScreen {
        align: center middle;
    }
    #settings-dialog {
        width: 80;
        height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #settings-title {
        height: 1;
        text-style: bold;
    }
    #settings-editor {
        height: 1fr;
    }
    #settings-error {
        height: auto;
        max-height: 6;
        color: $error;
    }
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Static("Settings — ctrl+s save · esc cancel", id="settings-title")
            yield TextArea(
                self._settings.model_dump_json(indent=2), id="settings-editor"
            )
            yield Static("", id="settings-error")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        text = self.query_one(TextArea).text
        error = self.query_one("#settings-error", Static)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            error.update(Content(f"Malformed JSON: {e}"))
            return
        try:
            new_settings = Settings.model_validate(data)
        except ValidationError as e:
            error.update(Content(f"Invalid settings: {e}"))
            return
        self.dismiss(new_settings)
