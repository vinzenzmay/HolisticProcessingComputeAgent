"""The memory review (§4.3 item 32): proposal, batch, reflection.

Three Textual modals with one shape between them — read a thing the agent
wants to keep, answer y or n — and one wire answer: `memory.resolve` carries a
list of verdicts *positionally*, against the order the proposals were offered
in. So this is one screen that walks the offer rather than three screens that
each dismiss with a bool, and the position it is at is the index into that
list. A short list rejects the rest (`protocol.MemoryResolve`), which is what
escaping half-way through means and why escaping is an answer here.

The batch is the one that is not per-item. A batch is often a trade — remove
two stale entries to make room for one new one — and approving half of it
leaves memory in a state nobody chose, so the whole offer is one question when
the core sends it as one. That distinction is on the wire as the length of the
list: one proposal answered by one verdict, several answered by several.
"""

from __future__ import annotations

from collections.abc import Sequence

from hpca.ui import theme
from hpca.ui.ansi import BOLD, RESET, pad
from hpca.ui.overlays.base import BACK_KEYS, Overlay, options
from hpca.ui.overlays.rewind import quoted
from hpca.ui.state import Proposal, ResolveMemory

# The keys the three Textual screens answered on, kept so the reflex transfers.
CHOICES = [
    ("y", "keep it"),
    ("n", "discard it"),
    ("esc", "discard this and everything after it"),
]

NOTHING = "nothing to review"


class MemoryReviewOverlay(Overlay):
    """One offer at a time, and one `memory.resolve` at the end of it.

    Nothing is sent until the walk finishes, because the answer is the whole
    list: a verdict sent per item would be several commands the core would
    have to reassemble into the batch it is holding.
    """

    title = "keep this?"

    def __init__(
        self, proposals: Sequence[Proposal] = (), session_id: str = ""
    ) -> None:
        super().__init__()
        self.proposals = list(proposals)
        self.session_id = session_id
        self.at = 0
        self.verdicts: list[bool] = []

    # ------------------------------------------------------------- the frame

    @property
    def current(self) -> Proposal | None:
        return self.proposals[self.at] if self.at < len(self.proposals) else None

    def heading(self) -> str:
        if not self.proposals:
            return self.title
        return f"{self.title} · {self.at + 1}/{len(self.proposals)}"

    def body(self, width: int, height: int) -> list[str]:
        proposal = self.current
        if proposal is None:
            return [theme.faint + pad(f"  {NOTHING}", width) + RESET]
        # `scope` and `kind` are open strings on the wire (`protocol.Proposal`
        # mirrors `agent.conclude`), so they are shown rather than translated —
        # a kind this screen has never heard of still has to be readable.
        said = " · ".join(x for x in (proposal.scope, proposal.kind) if x)
        rows = [BOLD + pad(f"  {said}", width) + RESET] if said else []
        rows += quoted(proposal.text, width)
        rows.append(" " * width)
        rows += options(CHOICES, width)
        kept = sum(self.verdicts)
        if self.verdicts:
            rows.append(
                theme.faint
                + pad(
                    f"      so far: {kept} kept, "
                    f"{len(self.verdicts) - kept} discarded",
                    width,
                )
                + RESET
            )
        return rows

    def keymap(self) -> list[tuple[str, str]]:
        return [("y", "keep"), ("n", "discard"), ("esc", "discard the rest")]

    # -------------------------------------------------------------- the keys

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            # Answered short. The core rejects whatever was not answered for,
            # so leaving early is a decision rather than a way out of one.
            return self._finish()
        if key not in ("y", "n"):
            return True
        self.verdicts.append(key == "y")
        self.at += 1
        if self.at >= len(self.proposals):
            return self._finish()
        return True

    def _finish(self) -> bool:
        # Sent even when nothing was answered: an empty list is the honest
        # "none of them", and a review closed without a word would leave the
        # core holding proposals nobody will ever come back to.
        self.send(ResolveMemory(self.session_id, tuple(self.verdicts)))
        return False
