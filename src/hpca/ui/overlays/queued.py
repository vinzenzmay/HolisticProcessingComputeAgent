"""The queued-message dialog (§4.3 item 34): cancel it, or copy it."""

from __future__ import annotations

from hpca.ui.overlays.rewind import ChoiceDialog

UNQUEUE = "unqueue"
# The rewind dialog dropped its copy — `c` on the chat row does that now — so
# the choice lives here, with the one screen that still offers it.
COPY = "copy"


class QueuedOverlay(ChoiceDialog):
    """Enter on a message typed ahead of a running turn.

    A queued message is the one kind of message the user can still take back:
    it has not reached the model, so cancelling it costs nothing and needs no
    thread surgery — the reason a turn in flight offers an interrupt and a
    sent message only offers a rewind. Cancelling lands where the interrupt
    lands, with the text back in the entry to edit and send again.

    Escape leaves it queued, and Enter still copies it into the entry: a
    queued message has no cut to offer — there is nothing behind it to roll
    back to — so copy is the only thing the reflex of activating a message
    twice can mean here. Same keys and same copy as `tui/rewind_screen.py`'s
    QueuedScreen, which this replaces.

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
    PICKS = {"x": UNQUEUE, "c": COPY, "enter": COPY}

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("x", "cancel it"),
            ("c / enter", "copy"),
            ("esc", "leave it queued"),
        ]
