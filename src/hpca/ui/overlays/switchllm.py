"""`ctrl+l` (§4.3 item 29): which LLM answers in *this* conversation.

Per session, not per app: two conversations can be on two different servers,
which is the whole reason this key exists — and the reason the star here means
"the one this session is on" rather than "the global default", which is what
the same star means one screen over on manage-LLMs.

Escape changes nothing, which is the claim worth being careful about: a picker
whose cursor movement already switched the backend would change the session by
being looked at.

**One thing it cannot do.** `backend.set` is the only command that points a
session at a backend and it carries a whole `LLMBackend` blob, api key
included — and the catalog deliberately never puts a key on the wire
(`protocol.LLMEntry`). So a key-locked entry cannot be switched to from here
without inventing a key, and this screen says so instead of sending a blob
that would build a client the endpoint answers 401 to. Closing it properly
wants a by-label form of `backend.set`, which is not this milestone's to add.
"""

from __future__ import annotations

from collections.abc import Iterable

from hpca.ui.overlays.backends import backend_item, row_key
from hpca.ui.overlays.base import ListOverlay
from hpca.ui.pane import Item
from hpca.ui.state import BackendInfo, SetBackend

EMPTY = "no backends configured — add one on the manage-llms screen (m)"
NEEDS_KEY = (
    "that one needs an api key, and the catalog never carries keys — "
    "add it again on the manage-llms screen (m) to store one"
)


class SwitchLlmOverlay(ListOverlay):
    """The configured catalog, with `★` on the one this session is using."""

    title = "switch llm for this session"
    name = "backends"

    def __init__(
        self,
        backends: Iterable[BackendInfo] | None = None,
        *,
        session_id: str = "",
        current: str = "",
    ) -> None:
        # ``current`` is the *session's* model name, as `session.rows` gave it
        # (`protocol.SessionRow.model`) — a label, because that is all the
        # sidebar carries and all a star needs.
        self.backends = list(backends or [])
        self.current = current
        super().__init__(self._rows())
        self.session_id = session_id
        self.chosen: BackendInfo | None = None
        for info in self.backends:
            if self._is_current(info):
                self.pane.show(row_key(info))
                break

    def _is_current(self, info: BackendInfo) -> bool:
        return bool(self.current) and self.current in (info.label, info.model)

    def _rows(self) -> list[Item]:
        if not self.backends:
            return [Item(head=EMPTY, kind="empty")]
        rows = []
        for info in self.backends:
            # The star is re-aimed rather than taken from `active`: on this
            # screen it answers "which one is this conversation on", and the
            # global default is a different question.
            here = BackendInfo(**{**info.__dict__, "active": self._is_current(info)})
            rows.append(backend_item(here))
        return rows

    def catalog_changed(self, catalog: list[BackendInfo]) -> None:
        """A fresh `llm.catalog` while the picker is open — usually the probes.

        Only the configured half: this screen picks what a conversation talks
        to, and a discovered endpoint is not something a session can be pinned
        to until it has been added.
        """
        self.backends = [x for x in catalog if not x.discovered]
        self.replace(self._rows())

    def keymap(self) -> list[tuple[str, str]]:
        return [("↑↓", "move"), ("enter", "use it here"), ("esc", "change nothing")]

    def picked(self, key: str) -> BackendInfo | None:
        return next((x for x in self.backends if row_key(x) == key), None)

    def chose(self, item: Item | None) -> bool:
        if item is None or item.kind != "backend":
            return False
        picked = self.picked(item.key)
        if picked is None:  # pragma: no cover - the rows are built from the list
            return False
        if picked.needs_key:
            self.note = NEEDS_KEY
            return True
        self.chosen = picked
        # Named session, so this is the per-session half of `backend.set` and
        # not the global one — the two are told apart by exactly that. The blob
        # is rebuilt from what the catalog carries, which for a keyless entry
        # is everything `config.LLMBackend` needs.
        blob: dict = {"model": picked.model, "base_url": picked.base_url}
        if picked.context:
            blob["max_model_len"] = picked.context
        self.send(SetBackend(blob, self.session_id))
        return False
