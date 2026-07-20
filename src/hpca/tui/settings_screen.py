"""Modal config editor (§3.3): edit the settings JSON, validated on save.

Called the "config editor" in the UI — it is a raw JSON editor over the whole
settings file, not a friendly settings menu, and the old name misled. The
screen validates and returns a new ``Settings`` object via ``dismiss``;
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
    # Save is resolved on escape ("Keep changes? y/n"), mirroring the profile
    # memory editor — there is no ctrl+s (a reserved hotkey: terminal XOFF).
    BINDINGS = [Binding("escape", "close", "back", priority=True)]

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
        self._original = settings.model_dump_json(indent=2)

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Static(
                "Config editor — esc: back (asks to keep changes)",
                id="settings-title",
            )
            yield TextArea(self._original, id="settings-editor")
            yield Static("", id="settings-error")

    def on_mount(self) -> None:
        self.query_one("#settings-editor", TextArea).focus()

    def _validate(self, text: str) -> Settings | None:
        """Parse+validate the edited JSON, showing the reason on failure."""
        error = self.query_one("#settings-error", Static)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            error.update(Content(f"Malformed JSON: {e}"))
            return None
        try:
            return Settings.model_validate(data)
        except ValidationError as e:
            error.update(Content(f"Invalid settings: {e}"))
            return None

    def action_close(self) -> None:
        text = self.query_one("#settings-editor", TextArea).text
        if text == self._original:
            self.dismiss(None)  # nothing changed; no need to ask
            return
        new_settings = self._validate(text)
        if new_settings is None:
            return  # invalid: keep editing rather than lose the work
        from hpca.tui.confirm_screen import ConfirmScreen

        def verdict(keep: bool | None) -> None:
            self.dismiss(new_settings if keep else None)

        self.app.push_screen(ConfirmScreen("Keep changes?"), verdict)
