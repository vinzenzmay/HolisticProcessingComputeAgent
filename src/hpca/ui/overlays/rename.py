"""Name something small: rename a session (§4.3 item 11).

One line of `Editor`, prefilled and with the cursor at the end — the point of
the screen is that a name is *edited* rather than retyped, which is what
`tui/rename_screen.py`'s `cursor_position = len(value)` was for.

An empty name is refused here rather than sent: a nameless row is a row the
user cannot find again, and the core refuses it too (`_rename_session`). It is
refused by keeping the screen open with a word about why, because the other
way to spell "refused" — closing and quietly changing nothing — looks exactly
like the save having worked.
"""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, pad, rule
from hpca.ui.editor import Editor
from hpca.ui.overlays.base import Overlay

EMPTY_REFUSAL = "a session needs a name"


class RenameOverlay(Overlay):
    """`r` on a session row: the name, editable, in a box.

    Carries the session it was opened on, for the reason every screen in this
    package that decides something does: a rename that closes after the user
    has switched sessions must not rename the new one.
    """

    title = "rename session"

    def __init__(self, title: str = "", session_id: str = "") -> None:
        self.editor = Editor()
        self.editor.set_text(title)  # cursor left at the end of it
        self.session_id = session_id
        self.was = title
        # What was decided; empty means the old name stands, which is what
        # escape leaves behind and what an unanswered screen leaves behind.
        self.name = ""
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        return [("enter", "save"), ("esc", "keep the old name")]

    def paste(self, text: str) -> None:
        self.editor.insert_text(text)

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + rule(self.title, width, self.note) + RESET]
        out += self.editor.render(width, max(1, height - 2), focused=True)
        out.append(DIM + pad(f"  was “{self.was}”", width) + RESET)
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        if key in ("esc", "quit"):
            return False
        if key == "enter":
            name = self.editor.text().strip()
            if not name:
                self.note = EMPTY_REFUSAL
                return True
            self.name = name
            return False
        self.editor.handle(key)
        return True
