"""The config editor (§4.3 item 26): raw JSON over the whole settings file.

Deliberately not a friendly settings menu — that is what the Textual screen's
docstring insisted on, and the reason is that the settings model grows a field
whenever anything does, and a hand-built form is the copy of it that falls
behind. The file is the interface.

Three properties come from `EditorOverlay` and one is this screen's own:
escape asks to keep changes, no edits means no question, the work stays on
screen when it cannot be kept — and *what* cannot be kept is decided by a
validator handed in from outside. Syntax is checked here, because JSON is what
the box holds; whether the parsed object is usable *settings* is a question
only `hpca.config` can answer, and this module has never heard of it (the
layering rule, §3.1 — `client.py` injects `validate`).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from hpca.ui.overlays.base import EditorOverlay


class ConfigOverlay(EditorOverlay):
    """`c`: the settings file, as text, refusing to close while it is broken.

    ``validate`` takes the edited text and answers "" or the reason it cannot
    be saved. It is only ever asked about text that already parses, so it
    never has to repeat the JSON check.
    """

    title = "config editor"

    def __init__(
        self,
        text: str = "",
        *,
        validate: Callable[[str], str] | None = None,
        awaiting=None,
    ) -> None:
        super().__init__(text, awaiting=awaiting)
        self._validate = validate

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("↑↓←→", "move"),
            ("^u", "clear"),
            ("esc", "back — asks to keep changes"),
        ]

    def refuse(self) -> str:
        """Why this text is not a settings file, or "".

        Reported on the rule and the screen stays open, which is the whole
        point of validating on the way out: a config editor that closed on
        broken JSON would throw away the edit that broke it.
        """
        try:
            json.loads(self.editor.text())
        except json.JSONDecodeError as e:
            return f"invalid JSON: line {e.lineno} — {e.msg}"
        if self._validate is None:
            return ""
        return self._validate(self.editor.text())
