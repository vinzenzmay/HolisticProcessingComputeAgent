"""Profiles & learnings: the list, and one profile's memories in an editor."""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, RESET, rule
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS
from hpca.ui.overlays.base import Overlay
from hpca.ui.pane import Item, Pane


class ProfilesOverlay(Overlay):
    """Profiles & learnings (a): the list, and one profile's memories open in
    a plain editor — the two states the Textual screen has."""

    title = "profiles & learnings"

    def __init__(self, profiles: list[Item], learnings: dict[str, str]) -> None:
        self.pane = Pane("profiles", profiles)
        self.learnings = learnings
        self.editor: Editor | None = None
        self.editing = ""
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        if self.editor is not None:
            return [("^s", "keep"), ("esc", "discard"), ("↑↓←→", "move")]
        return [
            ("↑↓", "move"),
            ("→←", "open"),
            ("enter", "edit learnings"),
            ("c", "copy"),
            ("d", "delete"),
            ("esc", "back"),
        ]

    def paste(self, text: str) -> None:
        if self.editor is not None:
            self.editor.insert_text(text)

    def render(self, width: int, height: int) -> list[str]:
        if self.editor is not None:
            head = rule(f"learnings · {self.editing}", width, self.note)
            out = [BOLD + CYAN + head + RESET]
            out += self.editor.render(width, height - 1, focused=True, numbers=True)
            return out[:height]
        out = [BOLD + CYAN + rule(self.title, width, self.note) + RESET]
        out += self.pane.render(width, height - 1, focused=True)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        inner = max(8, width - 2)
        if self.editor is not None:
            if key == "esc":
                self.editor = None
                self.note = "discarded"
            elif key == "ctrl-s":
                self.learnings[self.editing] = self.editor.text()
                self.editor = None
                self.note = "kept"
            elif key == "enter" or key in NEWLINE_KEYS:
                self.editor.newline()
            else:
                self.editor.handle(key)
            return True
        if key == "esc":
            return False
        view = max(1, height - 2)
        if key == "up":
            self.pane.move(-1, view, inner)
        elif key == "down":
            self.pane.move(1, view, inner)
        elif key == "right":
            if not self.pane.expand(inner) and self.pane.is_open(
                self.pane.current(inner)
            ):
                self.pane.move(1, view, inner)
        elif key == "left":
            self.pane.collapse(inner)
        elif key == "shift-right":
            self.pane.expand_all(inner)
        elif key == "shift-left":
            self.pane.collapse_all(inner)
        elif key == "enter":
            item = self.pane.current(inner)
            name = self.pane.items[item].head.split("  ")[0].strip("▸▾ ")
            self.editing = name
            self.editor = Editor(self.learnings.get(name, "(nothing learned yet)\n"))
            self.note = ""
        elif key == "c":
            self.note = "copied under a new name"
        elif key == "d":
            self.note = "deleted (never the default, never one in use)"
        return True
