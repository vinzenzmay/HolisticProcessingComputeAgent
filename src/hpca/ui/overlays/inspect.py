"""The read-only window (§4.3 item 31): text too long to be a toast.

`/skills-list` opens it, and so does the tunnel recipe manage-LLMs offers when
a scan finds nothing — text that has to be *retyped into a shell*, which is
the whole reason it is a window that waits for escape rather than a toast that
goes away on its own. With the mouse released the terminal's own selection
copies it, which is the argument for leaving the mouse released (§4.1 item 6).

Nothing is editable and nothing is chosen: the only keys are the ones that
scroll, and escape.
"""

from __future__ import annotations

from hpca.ui.ansi import RESET, fold, pad
from hpca.ui.overlays.base import BACK_KEYS, Overlay


class InspectOverlay(Overlay):
    """A block of text, scrolling, until escape.

    Unlike the key list — which closes on *any* key, because a list you opened
    by accident should not need the one key you were looking it up to find —
    this one stays: it is usually holding something the user is copying out of
    it, and losing that to a stray keystroke costs the whole errand.
    """

    def __init__(self, body: str = "", *, title: str = "", accent: str = "") -> None:
        super().__init__()
        self.text = body
        self.accent = accent
        self.offset = 0
        if title:
            self.title = title

    def keymap(self) -> list[tuple[str, str]]:
        return [("↑↓", "scroll"), ("pgup pgdn", "page"), ("esc", "close")]

    def lines(self, width: int) -> list[str]:
        """The text, wrapped to the box rather than cut off at its edge.

        Wrapped because what lands here is a recipe or a listing, and a
        truncated `ssh -L` line is a line nobody can use.
        """
        out: list[str] = []
        for paragraph in self.text.split("\n"):
            out += fold(paragraph, max(8, width - 4)) or [""]
        return out

    def body(self, width: int, height: int) -> list[str]:
        lines = self.lines(width)
        rows = max(1, height)
        self.offset = max(0, min(self.offset, max(0, len(lines) - rows)))
        shown = lines[self.offset : self.offset + rows]
        painted = [self.accent + pad(f"  {x}", width) + RESET for x in shown]
        return painted + [" " * width] * max(0, rows - len(painted))

    def render(self, width: int, height: int) -> list[str]:
        # How far down a long listing you are, on the rule, since there is no
        # cursor row to say it and no scrollbar to draw one.
        lines = self.lines(width)
        rows = max(1, height - 1)
        if len(lines) > rows:
            self.note = f"{min(self.offset + rows, len(lines))}/{len(lines)}"
        else:
            self.note = ""
        return super().render(width, height)

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            return False
        page = max(1, height - 2)
        step = {"up": -1, "down": 1, "pgup": -page, "pgdn": page}
        if key in step:
            self.offset += step[key]
        elif key == "home":
            self.offset = 0
        elif key == "end":
            self.offset = 10**9
        return True
