"""The rewind screen: fork or roll back from one of your own messages."""

from __future__ import annotations

from hpca.ui import theme
from hpca.ui.ansi import RESET, fold, pad
from hpca.ui.overlays.base import BACK_KEYS, Overlay, options

FORK, ROLLBACK = "fork", "rollback"

# The dialog quotes the message being rewound from so you can check you grabbed
# the right one — a preview, not the transcript; the log has the rest. Same
# figure as hpca.tui.rewind_screen.
PREVIEW_CHARS = 300
PREVIEW_LINES = 6  # what the Textual dialog's max-height comes to

# Said under both choices, since both do it: the message is not only cut away,
# it is handed back to the message box to be sent again or edited first.
BACK_TO_THE_BOX = "either way the message comes back to the box"


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
    out = [theme.faint + pad(f"    {line}", width) + RESET for line in lines[:PREVIEW_LINES]]
    if len(lines) > PREVIEW_LINES:
        out.append(theme.faint + pad("    …", width) + RESET)
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

    Two ways to pick the conversation up from it: fork the session from just
    before it (the original stays whole), or roll this conversation back to
    just before it (everything after is dropped). Copying the text used to be
    the third, and is now `c` on the chat row itself (`_copy_row`) — one key,
    without a dialog in front of it, on every row rather than only your own.

    Enter is the fork, because Enter is what opened this screen and the reflex
    of activating a message twice must not land on the cut that drops rows: a
    fork loses nothing, so it is the only one of the two that is safe under a
    key pressed by habit. Escape is still how you leave without either.

    Both cuts end the conversation just before this message, and the reason
    to make one is almost always to say it differently — so either choice
    hands the message back to the box (`RowUI._rewind`, and `adopt` for the
    fork, whose box belongs to a session that does not exist yet). The line
    under the options says so, because a message reappearing where the user
    is about to type is the kind of help that is alarming unannounced.

    The choice is read back by the caller rather than acted on here, for the
    reason app.py captures the session before pushing the screen: what happens
    belongs to the conversation the choice was made in.
    """

    title = "this message again"

    OPTIONS = [
        ("f / enter", "fork the session from here — the original stays whole"),
        ("r", "roll this conversation back to here"),
        ("esc", "cancel"),
    ]
    PICKS = {"f": FORK, "enter": FORK, "r": ROLLBACK}

    def body(self, width: int, height: int) -> list[str]:
        return super().body(width, height) + [
            " " * width,
            theme.faint + pad(f"      {BACK_TO_THE_BOX}", width) + RESET,
        ]

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("f / enter", "fork"),
            ("r", "roll back"),
            ("esc", "cancel"),
        ]
