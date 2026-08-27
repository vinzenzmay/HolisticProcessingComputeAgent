"""`/thinking` (§4.3 item 30): how hard this session's model reasons.

The four levels, what each does to a turn, and a star on the one in force.
Which levels exist and what they mean is `hpca.thinking`'s business and is
imported rather than copied — a second list of them here would be the copy that
disagrees with the server, and the whole reason each level carries a hint is
that a name is not enough to tell what it costs.

Per session, like the mode and the backend, and for a sharper reason than
either: two levels differ from the very first token of the prompt, so changing
one mid-conversation throws away that conversation's whole prefix cache.
"""

from __future__ import annotations

from hpca.ui.overlays.base import ListOverlay
from hpca.ui.pane import Item
from hpca.ui.state import SetThinking
from hpca.thinking import EFFORT_HINTS, EFFORTS, normalize_effort

NO_SESSION = "no session open — the level belongs to a conversation"


def effort_item(effort: str, current: str) -> Item:
    """One level: its name, the star if it is the one in force, and the hint.

    The hint rides on the option itself rather than on a toast afterwards,
    because this is the moment the choice is made. Every level gets the same
    treatment and none is flagged — the list orders them by how much they
    think, and the hint is what says what that buys.
    """
    star = " ★" if effort == current else ""
    return Item(
        head=f"{effort + star:<10}{EFFORT_HINTS[effort]}",
        kind="effort",
        text=effort,
        key=effort,
    )


class ThinkingOverlay(ListOverlay):
    """The chooser. Escape leaves the level alone; enter sets this session's.

    The cursor starts on the level already in force, so the common case of
    opening it to *see* the level costs one keystroke to leave and cannot
    change anything by accident.
    """

    title = "thinking effort for this session"
    name = "levels"

    def __init__(self, current: str = "", session_id: str = "") -> None:
        self.current = normalize_effort(current)
        super().__init__(effort_item(x, self.current) for x in EFFORTS)
        self.session_id = session_id
        self.effort = ""
        self.pane.show(self.current)

    def keymap(self) -> list[tuple[str, str]]:
        return [("↑↓", "move"), ("enter", "choose"), ("esc", "leave it alone")]

    def chose(self, item: Item | None) -> bool:
        if item is None:
            return False
        self.effort = item.text
        # Sent from here rather than read back by the caller: the level belongs
        # to the session this screen was opened on, and a `/thinking` answered
        # after the user switched conversations must not move the new one.
        self.send(SetThinking(self.session_id, item.text))
        return False
