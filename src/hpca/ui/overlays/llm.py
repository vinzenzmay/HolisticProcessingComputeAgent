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

**The scan runs from here, and it fills as it goes.** Opening the screen sends
`backend.scan`, which is two searches the core cannot merge — a localhost port
sweep for the SSH-tunnelled backends, and the cluster's manifest dir for the
ones living on a compute node's own IP that no localhost sweep can see. The
answer is a sequence: every hit restates `llm.catalog` with the new row
flagged `discovered`, and `backend.scanned` closes the run with the one thing
this side cannot work out — what an *empty* scan meant. Nothing here blocks on
any of it, which is what keeps the screen closable during a sweep of tens of
thousands of ports; the rows simply arrive, and a rescan clears the stale ones
on its first frame because the core forgets what it found before it starts.

`s` rescans — `f5` in the Textual screen, and this UI's key decoder has no
function keys — and `r` removes a configured entry, after asking. Both were
absent while the commands were, and the docstring that said so outlived them:
`backend.scan`, `backend.probe` and `backend.remove` all landed in the core in
one milestone, and this screen went on saying discovery "is not implemented
anywhere" for the rest of it.
"""

from __future__ import annotations

from collections.abc import Iterable

from hpca.ui.overlays.backendform import BackendFormOverlay
from hpca.ui.overlays.backends import KEY_REQUIRED, backend_item, row_key
from hpca.ui.overlays.base import BACK_KEYS, Overlay
from hpca.ui.pane import Item, Pane
from hpca.ui.state import BackendInfo, RemoveBackend, ScanBackends

DISCOVERED, CONFIGURED = "discovered", "configured"

SCANNING = "scanning for endpoints… (the sweep is tens of thousands of ports)"
NOTHING_FOUND = "nothing discovered — s rescans; a on this panel adds one by hand"
NOTHING_CONFIGURED = "no backends configured — press a to add one"
# Enter on a configured row is deliberately inert: the catalog is a list of
# what may be used, and which one a *conversation* uses is `ctrl+l`'s question.
ALREADY_IN = "already in the catalog — ctrl+l picks one for a session"
# What `r` asks before it drops one. Named in the question, because two vLLMs
# serving one model on two nodes differ only by their label.
REMOVE_BACKEND = "Remove “{label}” from the catalog?"
NOT_REMOVABLE = "a discovered endpoint is not in the catalog — nothing to remove"
FORM_TAG = "form"


class LlmOverlay(Overlay):
    """Discovered endpoints above, the configured catalog below."""

    title = "manage llms"
    # The recipe an empty scan comes back with belongs on this screen: it is
    # the answer to the question this screen asked (`RowUI.window`).
    welcomes_window = True

    def __init__(
        self,
        catalog: Iterable[BackendInfo] | None = None,
        configured: Iterable[BackendInfo] | None = None,
    ) -> None:
        super().__init__()
        # One list with a flag on the wire (`protocol.LLMEntry.discovered`),
        # two panels here — because "found" and "configured" are the two
        # answers to different questions and the keys on them differ.
        self.discovered: list[BackendInfo] = []
        self.configured: list[BackendInfo] = []
        # Whether a sweep is out. Only the empty panel's text and the rule's
        # note read it: everything else about this screen behaves identically
        # while one runs, which is the point.
        self.scanning = False
        self._split(list(catalog or []) + list(configured or []))
        self.panes = [
            Pane(DISCOVERED, self._rows(self.discovered, self._empty(), star=False)),
            Pane(CONFIGURED, self._rows(self.configured, NOTHING_CONFIGURED)),
        ]
        self.side = 0
        # The label `r` is asking about, since the answer arrives a keypress
        # later and the list can be repainted by a catalog in between.
        self._removing = ""

    def opened(self) -> None:
        """Ask for a scan the moment the screen is on the stack.

        On open rather than on a key, because the panel exists to answer "what
        is out there that I have not configured", and a screen that showed
        nothing until the user found the rescan key would answer it with
        "nothing". `Overlay.opened` and not `__init__`: the intent has
        somewhere to go only once `RowUI.push` has wired it.
        """
        self.scan()

    def scan(self) -> None:
        """Say what is happening, then ask — in that order.

        The demo's core answers in the same breath as the send, and a note
        written afterwards would overwrite the verdict with "scanning…" on a
        scan that had already finished.
        """
        self.scanning = True
        self.note = "scanning…"
        self.refresh()
        self.send(ScanBackends())

    # ------------------------------------------------------------- the lists

    def _split(self, entries: list[BackendInfo]) -> None:
        """One catalog into two panels, and the dedup that goes with it.

        A discovered row for an endpoint that is *already configured* is the
        same server under two names: the core emits both (`backends.catalog`
        builds the discovered rows without excluding the configured ones), and
        drawing both is the duplicate the Textual screen kept a `_superseded`
        guard for. The configured row is the one that survives — it is the one
        with the key on it, and the one a command may name.
        """
        self.configured = [x for x in entries if not x.discovered]
        known = {x.base_url for x in self.configured if x.base_url}
        self.discovered = [
            x for x in entries if x.discovered and x.base_url not in known
        ]

    def _empty(self) -> str:
        return SCANNING if self.scanning else NOTHING_FOUND

    @staticmethod
    def _rows(
        backends: list[BackendInfo], empty: str, *, star: bool = True
    ) -> list[Item]:
        if not backends:
            return [Item(head=empty, kind="empty")]
        return [backend_item(x, star=star) for x in backends]

    def refresh(self) -> None:
        self.panes[0].replace(
            self._rows(self.discovered, self._empty(), star=False)
        )
        self.panes[1].replace(self._rows(self.configured, NOTHING_CONFIGURED))

    def catalog_changed(self, catalog: list[BackendInfo]) -> None:
        """A fresh `llm.catalog` while this screen is open.

        Restated rather than left alone, and this is what makes the scan fill
        incrementally: every hit is a whole catalog with one more row in it,
        and the panel is a function of the last one that arrived. The probes
        land the same way — a panel that only refreshed on the next open would
        draw "not probed" for the whole time the answer was already in.
        """
        self._split(list(catalog))
        self.refresh()

    def scanned(self, found: int, cluster: int) -> None:
        """`backend.scanned`: the sweep is over, and how much it turned up.

        Only the status line. What an empty scan *meant* is two texts the core
        mints and `RowUI.scanned` places — a toast for the passing remark, a
        window for the tunnel recipe — because one is read and dismissed and
        the other is retyped into a shell.
        """
        self.scanning = False
        where = f" ({cluster} from the cluster)" if cluster else ""
        self.note = f"scan finished: {found} endpoint(s) found{where} · s rescans"
        self.refresh()

    def keymap(self) -> list[tuple[str, str]]:
        keys = [("^↑^↓", "panel"), ("↑↓", "move"), ("→←", "open")]
        if self.side == 0:
            keys += [("enter", "add to catalog")]
        else:
            keys += [("r", "remove")]
        keys += [("a", "add by hand"), ("s", "rescan")]
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
        elif key == "s":
            # The old screen's f5, on a letter: this UI's decoder has no
            # function keys, and a binding that could never arrive would be
            # exactly the "present and inert" this screen used to argue against.
            self.scan()
        elif key == "r":
            return self._remove(width)
        elif key == "a":
            # The manual form, from either panel: it is how a backend that no
            # sweep can see gets into the catalog, and it is the only thing to
            # do on a discovered panel that has come back empty.
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
        #
        # The model comes across unless it is the sentinel — a cluster
        # manifest names the model of an endpoint the sweep could only get a
        # 401 out of, and dropping it because the row needs a key would make
        # the user retype something we were told.
        return self.open(
            BackendFormOverlay(
                base_url=found.base_url,
                model="" if found.model == KEY_REQUIRED else found.model,
                context=found.context,
                locked_url=True,
            ),
            FORM_TAG,
        )

    def _remove(self, width: int) -> bool:
        """`r`: drop a configured entry, after asking.

        Inert on the discovered panel, which is what the Textual screen's
        binding was: a discovered row is not in the settings file, so there is
        nothing there to remove and the key would be lying about what it did.
        """
        if self.side == 0:
            self.note = NOT_REMOVABLE
            return True
        entry = self.picked(width)
        if entry is None:
            return True
        self._removing = entry.label
        self.ask(REMOVE_BACKEND.format(label=entry.label))
        return True

    def answered(self, question: str, yes: bool) -> bool:
        label, self._removing = self._removing, ""
        if not yes or not label:
            self.note = "kept it"
            return True
        self.send(RemoveBackend(label))
        # Off the screen now and true when the next `llm.catalog` says so —
        # the same shown-before-it-is-true bargain the mode bar makes, and the
        # core does restate the catalog after a remove.
        self.configured = [x for x in self.configured if x.label != label]
        self.refresh()
        self.note = f"removed {label}"
        return True

    def child_closed(self, child) -> None:
        if child.tag != FORM_TAG or child.backend is None:
            return
        # The command went out from the form; what is left is to say so here,
        # since the catalog the core restates arrives a round trip later.
        # Shown before the core answers, and the next catalog the UI is handed
        # is what makes it true or takes it back.
        added = child.info()
        if not any(row_key(x) == row_key(added) for x in self.configured):
            self.configured.append(added)
        self.refresh()
        self.note = f"added {added.label}"
