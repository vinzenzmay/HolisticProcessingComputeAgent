"""The rewind screen: fork, roll back, or copy one of your own messages."""

from __future__ import annotations

from hpca.ui.ansi import DIM, RESET, fold, pad
from hpca.ui.overlays.base import BACK_KEYS, Overlay, options

FORK, ROLLBACK, COPY = "fork", "rollback", "copy"

# The dialog quotes the message being rewound from so you can check you grabbed
# the right one — a preview, not the transcript; the log has the rest. Same
# figure as hpca.tui.rewind_screen.
PREVIEW_CHARS = 300
PREVIEW_LINES = 6  # what the Textual dialog's max-height comes to


def quoted(message: str, width: int) -> list[str]:
    """A few lines of what is being decided about, cut and marked as cut.

    Shared by the two screens that ask about one message the user wrote, and
    by the memory dialogs, which ask about one paragraph the agent wrote.
    """
    preview = message[:PREVIEW_CHARS]
    if preview != message:
        preview += " …"
    lines: list[str] = []
    for paragraph in preview.split("\n"):
        lines += fold(paragraph, max(8, width - 6))
    out = [DIM + pad(f"    {line}", width) + RESET for line in lines[:PREVIEW_LINES]]
    if len(lines) > PREVIEW_LINES:
        out.append(DIM + pad("    …", width) + RESET)
    return out


class ChoiceDialog(Overlay):
    """One subject, a handful of lettered ways out, and no list to scroll.

    The shape `RewindOverlay` and `QueuedOverlay` already shared: quote the
    thing, print the options, and answer exactly the keys that are printed. A
    key it has no answer for leaves it open, because a modal that closed on
    any keystroke would lose the decision to a stray arrow.
    """

    OPTIONS: list[tuple[str, str]] = []
    # key -> what the caller reads off ``choice`` afterwards.
    PICKS: dict[str, str] = {}

    def __init__(self, message: str, seq: int = 0, session_id: str = "") -> None:
        super().__init__()
        self.message = message
        # The row's core-assigned name (`protocol.Entry.seq`), not its position
        # in the log: the cut is decided here and carried out later, and a turn
        # appending in between moves every position after this one.
        self.seq = seq
        # And the conversation it was decided in, since a decision that closes
        # after the user has switched sessions must not act on the new one.
        self.session_id = session_id
        self.choice = ""

    def body(self, width: int, height: int) -> list[str]:
        return quoted(self.message, width) + [" " * width] + options(
            self.OPTIONS, width
        )

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in self.PICKS:
            self.choice = self.PICKS[key]
            return False
        return key not in BACK_KEYS  # a modal ignores what it has no answer for


class RewindOverlay(ChoiceDialog):
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
    PICKS = {"f": FORK, "r": ROLLBACK, "c": COPY, "enter": COPY}

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("f", "fork"),
            ("r", "roll back"),
            ("c / enter", "copy"),
            ("esc", "cancel"),
        ]
