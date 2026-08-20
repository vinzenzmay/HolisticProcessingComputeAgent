"""The config editor: raw JSON over the whole settings file."""

from __future__ import annotations

import json

from hpca.ui.ansi import BOLD, CYAN, RESET, rule
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS
from hpca.ui.overlays.base import Overlay


class ConfigOverlay(Overlay):
    """Config editor (c): raw JSON over the whole settings file, validated on
    save — deliberately not a friendly settings menu, which is what the real
    screen's docstring insists on."""

    title = "config editor"

    def __init__(self, text: str) -> None:
        self.editor = Editor(text)
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        return [
            ("^s", "validate & save"),
            ("↑↓←→", "move"),
            ("^u", "clear"),
            ("esc", "back"),
        ]

    def paste(self, text: str) -> None:
        self.editor.insert_text(text)

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + rule(self.title, width, self.note) + RESET]
        out += self.editor.render(width, height - 1, focused=True, numbers=True)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        if key == "esc":
            return False
        if key == "ctrl-s":
            try:
                json.loads(self.editor.text())
            except json.JSONDecodeError as e:
                self.note = f"invalid: line {e.lineno} — {e.msg}"
            else:
                self.note = "saved"
            return True
        if key == "enter" or key in NEWLINE_KEYS:
            self.editor.newline()
        else:
            self.editor.handle(key)
        return True
