"""The rewind screen: fork, roll back, or copy one of your own messages."""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, fold, pad, rule
from hpca.ui.overlays.base import Overlay

FORK, ROLLBACK, COPY = "fork", "rollback", "copy"

# The dialog quotes the message being rewound from so you can check you grabbed
# the right one — a preview, not the transcript; the log has the rest. Same
# figure as hpca.tui.rewind_screen.
PREVIEW_CHARS = 300
PREVIEW_LINES = 6  # what the Textual dialog's max-height comes to


class RewindOverlay(Overlay):
    """Enter on one of your own messages in the log (§ chat rewind).

    Three ways to pick the conversation up from it, the same three
    RewindScreen offers and on the same keys: fork the session from just
    before it (the original stays whole), roll this conversation back to just
    before it (everything after is dropped), or copy the text into the message
    box — the old behaviour, still on Enter, so the reflex of activating a
    message twice keeps doing what it always did.

    The choice is read back by the caller rather than acted on here, for the
    reason app.py captures the session before pushing the screen: what happens
    belongs to the conversation the choice was made in.
    """

    title = "this message again"

    OPTIONS = [
        ("f", "fork the session from here — the original stays whole"),
        ("r", "roll this conversation back to here"),
        ("c / enter", "copy it into the message box"),
        ("esc", "cancel"),
    ]

    def __init__(self, message: str, index: int) -> None:
        self.message = message
        self.index = index
        self.choice = ""

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + rule(self.title, width) + RESET]
        preview = self.message[:PREVIEW_CHARS]
        if preview != self.message:
            preview += " …"
        quoted: list[str] = []
        for paragraph in preview.split("\n"):
            quoted += fold(paragraph, max(8, width - 6))
        for line in quoted[:PREVIEW_LINES]:
            out.append(DIM + pad(f"    {line}", width) + RESET)
        if len(quoted) > PREVIEW_LINES:
            out.append(DIM + pad("    …", width) + RESET)
        out.append(" " * width)
        for key, label in self.OPTIONS:
            row = pad(f"      {key:<12}{label}", width)
            out.append(row[:6] + CYAN + row[6:18] + RESET + row[18:])
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        picked = {"f": FORK, "r": ROLLBACK, "c": COPY, "enter": COPY}
        if key in picked:
            self.choice = picked[key]
            return False
        if key in ("esc", "quit"):
            return False
        return True  # a modal ignores what it has no answer for

    def footer(self) -> list[tuple[str, str]]:
        return [
            ("f", "fork"),
            ("r", "roll back"),
            ("c / enter", "copy"),
            ("esc", "cancel"),
        ]
