"""Manage LLMs (`m`, §4.3 item 28): what was found, and what is configured.

Two lists, stacked rather than side by side like the Textual screen's columns.
Columns cost this screen more than they cost anywhere else: an endpoint line is
a URL and a model name and a context size, which is most of eighty characters
before the catalog gets any, and halving the width truncated all of it. Rows
give each list the whole width and cost only vertical space, which is the one
thing a list can scroll.

`^↑`/`^↓` move between them — the same keys as the main view, so the habit
transfers — and the footer offers "add" only on discovered and the catalog
keys only on configured, which is what the Textual version did by hanging
bindings off each panel widget.

**Two things this screen cannot do yet, and does not pretend to.** Discovery
is not implemented anywhere: `llm.catalog` carries a `discovered` flag per
entry and nothing ever sets it, because no localhost port scan and no cluster
manifest read runs in the core. So the top panel is empty and says why, and
there is no rescan key that would only ever redraw the same rows. Removing a
catalog entry has no command either — `backend.set` only sets — so `r` is
absent rather than present and inert.
"""

from __future__ import annotations

from collections.abc import Iterable

from hpca.ui.overlays.backendform import BackendFormOverlay
from hpca.ui.overlays.backends import backend_item, row_key
from hpca.ui.overlays.base import BACK_KEYS, Overlay
from hpca.ui.pane import Item, Pane
from hpca.ui.state import BackendInfo

DISCOVERED, CONFIGURED = "discovered", "configured"

NOTHING_FOUND = "nothing discovered — no scan runs yet; a on this panel adds one by hand"
NOTHING_CONFIGURED = "no backends configured — press a to add one"
# Enter on a configured row is deliberately inert: the catalog is a list of
# what may be used, and which one a *conversation* uses is `ctrl+l`'s question.
ALREADY_IN = "already in the catalog — ctrl+l picks one for a session"
FORM_TAG = "form"


class LlmOverlay(Overlay):
    """Discovered endpoints above, the configured catalog below."""

    title = "manage llms"

    def __init__(
        self,
        catalog: Iterable[BackendInfo] | None = None,
        configured: Iterable[BackendInfo] | None = None,
    ) -> None:
        super().__init__()
        # One list with a flag on the wire (`protocol.LLMEntry.discovered`),
        # two panels here — because "found" and "configured" are the two
        # answers to different questions and the keys on them differ.
        entries = list(catalog or []) + list(configured or [])
        self.discovered = [x for x in entries if x.discovered]
        self.configured = [x for x in entries if not x.discovered]
        self.panes = [
            Pane(DISCOVERED, self._rows(self.discovered, NOTHING_FOUND, star=False)),
            Pane(CONFIGURED, self._rows(self.configured, NOTHING_CONFIGURED)),
        ]
        self.side = 0

    @staticmethod
    def _rows(
        backends: list[BackendInfo], empty: str, *, star: bool = True
    ) -> list[Item]:
        if not backends:
            return [Item(head=empty, kind="empty")]
        return [backend_item(x, star=star) for x in backends]

    def refresh(self) -> None:
        self.panes[0].replace(
            self._rows(self.discovered, NOTHING_FOUND, star=False)
        )
        self.panes[1].replace(self._rows(self.configured, NOTHING_CONFIGURED))

    def catalog_changed(self, catalog: list[BackendInfo]) -> None:
        """A fresh `llm.catalog` while this screen is open.

        Restated rather than left alone, because the probes arrive in a second
        frame: a panel that only refreshed on the next open would draw "not
        probed" for the whole time the answer was already in.
        """
        entries = list(catalog)
        self.discovered = [x for x in entries if x.discovered]
        self.configured = [x for x in entries if not x.discovered]
        self.refresh()

    def keymap(self) -> list[tuple[str, str]]:
        keys = [("^↑^↓", "panel"), ("↑↓", "move"), ("→←", "open")]
        if self.side == 0:
            keys += [("enter", "add to catalog")]
        keys += [("a", "add by hand")]
        return keys + [("esc", "back")]

    # ------------------------------------------------------------- the frame

    def heights(self, body_h: int, width: int) -> list[int]:
        """Half of the body each, but a short list only takes what it has.

        Same rule as the main view: whatever the top does not need goes to the
        bottom, so three discovered endpoints do not hold half the screen open
        above a catalog that has to scroll. Measured in *body* rows, which is
        what the frame has already taken its rule and its question out of.
        """
        avail = max(4, body_h)
        inner = max(8, width - 2)
        top = max(2, min(1 + len(self.panes[0].flat(inner)), avail // 2))
        bottom = avail - top
        if bottom < 2:
            top, bottom = avail - 2, 2
        return [top, bottom]

    def body(self, width: int, height: int) -> list[str]:
        out: list[str] = []
        for index, pane_h in enumerate(self.heights(height, width)):
            out += self.panes[index].render(
                width, pane_h, focused=self.side == index
            )
        return out

    # -------------------------------------------------------------- the keys

    def pane(self) -> Pane:
        return self.panes[self.side]

    def picked(self, width: int) -> BackendInfo | None:
        pane = self.pane()
        at = pane.current(max(8, width - 2))
        if not (0 <= at < len(pane.items)):
            return None
        item = pane.items[at]
        source = self.discovered if self.side == 0 else self.configured
        return next((x for x in source if row_key(x) == item.key), None)

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            return False
        pane = self.pane()
        inner = max(8, width - 2)
        body_h = height - 1 - len(self.question_rows(1))
        view = max(1, self.heights(body_h, width)[self.side] - 1)
        if key in ("ctrl-up", "ctrl-down", "tab", "shift-tab"):
            self.side = 1 - self.side
            return True
        steps = {"up": -1, "down": 1, "pgup": -view, "pgdn": view}
        if key in steps:
            pane.move(steps[key], view, inner)
        elif key == "home":
            pane.move(-(10**9), view, inner)
        elif key == "end":
            pane.move(10**9, view, inner)
        elif key == "right":
            if not pane.expand(inner) and pane.is_open(pane.current(inner)):
                pane.move(1, view, inner)
        elif key == "left":
            pane.collapse(inner)
        elif key == "shift-right":
            pane.expand_all(inner)
        elif key == "shift-left":
            pane.collapse_all(inner)
        elif key == "enter":
            return self._enter(width)
        elif key == "a":
            # The manual form, from either panel: with no scan running it is
            # the only way a backend gets into the catalog at all, and hiding
            # it behind a panel that is always empty would hide the screen's
            # one working action.
            return self.open(BackendFormOverlay(), FORM_TAG)
        return True

    def _enter(self, width: int) -> bool:
        if self.side == 1:
            self.note = ALREADY_IN
            return True
        found = self.picked(width)
        if found is None:
            return True
        # Prefilled and with the URL locked: the endpoint is a fact about the
        # row that was picked, and letting it be edited would make the form
        # describe a different server than the one chosen. A locked endpoint
        # opens on the key, which is the one thing the scan could not read.
        return self.open(
            BackendFormOverlay(
                base_url=found.base_url,
                model="" if found.needs_key else found.model,
                context=found.context,
                locked_url=True,
            ),
            FORM_TAG,
        )

    def child_closed(self, child) -> None:
        if child.tag != FORM_TAG or child.backend is None:
            return
        # The command went out from the form; what is left is to say so here,
        # since nothing on the wire will repaint this list (see the module
        # docstring). Shown before the core answers, and the next catalog the
        # UI is handed is what makes it true or takes it back.
        added = child.info()
        if not any(row_key(x) == row_key(added) for x in self.configured):
            self.configured.append(added)
        self.refresh()
        self.note = f"added {added.label}"
