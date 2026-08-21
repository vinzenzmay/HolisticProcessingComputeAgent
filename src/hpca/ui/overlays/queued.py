"""The queued-message dialog (§4.3 item 34): cancel it, or copy it."""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, fold, pad, rule
from hpca.ui.overlays.base import Overlay
from hpca.ui.overlays.rewind import COPY, PREVIEW_CHARS, PREVIEW_LINES

UNQUEUE = "unqueue"


class QueuedOverlay(Overlay):
    """Enter on a message typed ahead of a running turn.

    A queued message is the one kind of message the user can still take back:
    it has not reached the model, so cancelling it costs nothing and needs no
    thread surgery — the reason a turn in flight offers an interrupt and a
    sent message only offers a rewind. Cancelling lands where the interrupt
    lands, with the text back in the entry to edit and send again.

    Escape leaves it queued, and Enter still copies, so activating a message
    twice does here what it does on the rewind dialog. Same keys and same
    copy as `tui/rewind_screen.py`'s QueuedScreen, which this replaces.

    What leaves here is the row's ``seq`` and the session it was queued in,
    never its position: the turn ahead can finish while this dialog sits open,
    and the queue behind it then shifts by one (`protocol.TurnUnqueue`).
    """

    title = "still waiting to run"

    OPTIONS = [
        ("x", "cancel it — the text comes back to the entry"),
        ("c / enter", "copy it into the entry, leave it queued"),
        ("esc", "leave it queued"),
    ]

    def __init__(self, message: str, seq: int = 0, session_id: str = "") -> None:
        self.message = message
        self.seq = seq
        self.session_id = session_id
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
        picked = {"x": UNQUEUE, "c": COPY, "enter": COPY}
        if key in picked:
            self.choice = picked[key]
            return False
        if key in ("esc", "quit"):
            return False  # left queued: no choice, so nothing is sent
        return True

    def footer(self) -> list[tuple[str, str]]:
        return [
            ("x", "cancel it"),
            ("c / enter", "copy"),
            ("esc", "leave it queued"),
        ]
