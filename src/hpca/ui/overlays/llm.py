"""Manage LLMs: the discovered endpoints and the configured catalog."""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, RESET, rule
from hpca.ui.overlays.base import Overlay
from hpca.ui.pane import Item, Pane


class LlmOverlay(Overlay):
    """Manage LLMs (m): discovered endpoints above, configured catalog below.

    Stacked rather than side by side, like the main view. Columns cost this
    screen more than they cost anywhere else: an endpoint line is a URL and a
    model name and a context size, which is most of eighty characters before
    the catalog gets any, and halving the width truncated all of it. Rows give
    each list the whole width and cost only vertical space, which is the one
    thing a list can scroll.

    ^↑/^↓ move between the two — the same keys as the main view, so the habit
    transfers — and the footer offers "add" only on discovered and
    "remove"/"make default" only on configured, which is what the Textual
    version does by hanging bindings off each panel.
    """

    title = "manage llms"

    def __init__(self, discovered: list[Item], configured: list[Item]) -> None:
        self.panes = [Pane("discovered", discovered), Pane("configured", configured)]
        self.side = 0
        self.note = ""

    def footer(self) -> list[tuple[str, str]]:
        keys = [("^↑^↓", "row"), ("↑↓", "move"), ("→←", "open")]
        if self.side == 0:
            keys.append(("enter", "add to catalog"))
            keys.append(("s", "rescan"))
        else:
            keys.append(("enter", "make default"))
            keys.append(("d", "remove"))
        return keys + [("esc", "back")]

    def _heights(self, height: int, width: int) -> list[int]:
        """Half each, but a short list only takes what it has.

        Same rule as the main view: whatever the top does not need goes to the
        bottom, so three discovered endpoints do not hold half the screen open
        above a catalog that has to scroll.
        """
        avail = max(4, height - 1)
        inner = max(8, width - 2)
        top = max(2, min(1 + len(self.panes[0].flat(inner)), avail // 2))
        bottom = avail - top
        if bottom < 2:
            top, bottom = avail - 2, 2
        return [top, bottom]

    def render(self, width: int, height: int) -> list[str]:
        out = [BOLD + CYAN + rule(self.title, width, self.note) + RESET]
        for index, pane_h in enumerate(self._heights(height, width)):
            out += self.panes[index].render(width, pane_h, focused=self.side == index)
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        pane = self.panes[self.side]
        inner = max(8, width - 2)
        view = max(1, self._heights(height, width)[self.side] - 1)
        if key == "esc":
            return False
        if key in ("ctrl-up", "ctrl-down", "tab", "shift-tab", "left", "right"):
            self.side = 1 - self.side
        elif key == "up":
            pane.move(-1, view, inner)
        elif key == "down":
            pane.move(1, view, inner)
        elif key == "pgup":
            pane.move(-view, view, inner)
        elif key == "pgdn":
            pane.move(view, view, inner)
        elif key == "right":
            if not pane.expand(inner) and pane.current(inner) in pane.expanded:
                pane.move(1, view, inner)
        elif key == "left":
            pane.collapse(inner)
        elif key == "shift-right":
            pane.expand_all(inner)
        elif key == "shift-left":
            pane.collapse_all(inner)
        elif key == "enter":
            self.note = (
                "added to the catalog" if self.side == 0 else "made the default"
            )
        elif key == "d" and self.side == 1:
            self.note = "removed (confirm in the real screen)"
        elif key == "s" and self.side == 0:
            self.note = "rescanned: 3 endpoints"
        return True
