"""The new-session flow: pick a profile, then — if there is a choice — an LLM.

Two stages in one screen rather than two screens, because they are one
decision with one way out: escaping either half creates nothing, which is what
`tui/app.py`'s `pick_profile_for_new_session` arranged with a chain of
callbacks that each returned None. Here the caller reads ``chosen``
afterwards, and there is one flag to get wrong instead of two.

The rows are handed in rather than fetched, for the reason every screen in
this package is: an overlay draws what it was given and has never heard of a
socket. What a *backend* is stays opaque all the way through — `Item.text`
carries whatever string the core wants back, and nothing here looks inside it.
"""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, pad, rule
from hpca.ui.overlays.base import Overlay
from hpca.ui.pane import Item, Pane

PROFILE, BACKEND = "profile", "backend"

# What each stage is called on its rule, and the line under it. The second
# says *why* it is being asked: "which LLM" is only ever asked when a catalog
# exists, so a user who has just configured their first two backends is seeing
# the question for the first time.
STAGE_TITLES = {
    PROFILE: "new session · which profile",
    BACKEND: "new session · which llm",
}
STAGE_HINTS = {
    PROFILE: "its memories and skills come with the conversation",
    BACKEND: "this conversation stays pinned to it",
}


def choice(name: str, detail: str = "", *, value: str = "") -> Item:
    """One row of either list: what is drawn, and what goes on the wire.

    ``value`` defaults to the name because a profile *is* its own name; a
    backend is not, which is why the two are separate fields rather than one.
    """
    return Item(
        head=f"{name:<20}{detail}".rstrip(),
        text=value or name,
        key=value or name,
    )


class NewSessionOverlay(Overlay):
    """Enter on the `(new session)` row (§4.3 item 14).

    With no backends configured there is no second stage at all — the core
    then talks to the bootstrap client — because a picker holding one thing
    that cannot be declined is a keypress that asks for nothing.
    """

    def __init__(
        self, profiles: list[Item], backends: list[Item] | None = None
    ) -> None:
        self.pane = Pane(PROFILE, list(profiles))
        self._backends = list(backends or [])
        self.stage = PROFILE
        self.profile = ""
        self.backend = ""
        # Whether anything was decided. Escape at *either* stage leaves this
        # false, which is the "cancelling either creates nothing" rule with
        # one place to get it wrong instead of two.
        self.chosen = False

    @property
    def title(self) -> str:
        return STAGE_TITLES[self.stage]

    def footer(self) -> list[tuple[str, str]]:
        return [
            ("↑↓", "move"),
            ("enter", "choose"),
            ("esc", "cancel — nothing is created"),
        ]

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + rule(self.title, width) + RESET]
        out.append(DIM + pad(f"  {STAGE_HINTS[self.stage]}", width) + RESET)
        out += self.pane.render(width, max(1, height - 2), focused=True)
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        inner = max(8, width - 2)
        view = max(1, height - 3)
        if key in ("esc", "quit"):
            return False
        if key == "up":
            self.pane.move(-1, view, inner)
        elif key == "down":
            self.pane.move(1, view, inner)
        elif key == "pgup":
            self.pane.move(-view, view, inner)
        elif key == "pgdn":
            self.pane.move(view, view, inner)
        elif key == "home":
            self.pane.move(-(10**9), view, inner)
        elif key == "end":
            self.pane.move(10**9, view, inner)
        elif key == "enter":
            return self._pick(inner)
        return True

    def _pick(self, inner: int) -> bool:
        """Take the row under the cursor, and either move on or finish."""
        index = self.pane.current(inner)
        if index < 0:
            # A list with nothing in it. Closing with nothing chosen is the
            # honest answer: the alternative is a screen with no way out.
            return False
        picked = self.pane.items[index].text
        if self.stage == PROFILE:
            self.profile = picked
            if not self._backends:
                self.chosen = True
                return False
            self.stage = BACKEND
            self.pane = Pane(BACKEND, self._backends)
            return True
        self.backend = picked
        self.chosen = True
        return False
