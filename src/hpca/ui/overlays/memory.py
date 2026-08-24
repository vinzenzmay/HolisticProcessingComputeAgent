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

The proposal itself scrolls, and is drawn whole. It used to be `quoted()` —
the 300-character preview the rewind dialog shows of the message being cut —
and that is the right shape *there*, where the quote only has to identify a
message the user wrote seconds ago and the log still holds the rest. Here
there is no rest: what is on this screen is the entire text of a memory that
will be read back in a month, the decision is about its wording, and
`/conclude` routinely proposes paragraphs longer than that preview. Every one
of them arrived with a "…" on the end, which reads as a model that cut its own
memory off rather than as a screen that cut the reading of it. Same reasoning
as `compact`, and the same scrolling.
"""

from __future__ import annotations

from collections.abc import Sequence

from hpca.ui import theme
from hpca.ui.ansi import BOLD, RESET, fold, pad
from hpca.ui.overlays.base import BACK_KEYS, Overlay, framed, options
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
        # How far down the proposal on screen. Reset per proposal rather than
        # kept for the screen: answering one and landing half-way down the
        # next would hide the beginning of a memory nobody has read a word of.
        self.offset = 0

    # ------------------------------------------------------------- the frame

    @property
    def current(self) -> Proposal | None:
        return self.proposals[self.at] if self.at < len(self.proposals) else None

    def heading(self) -> str:
        if not self.proposals:
            return self.title
        return f"{self.title} · {self.at + 1}/{len(self.proposals)}"

    def lines(self, width: int) -> list[str]:
        """The proposal, folded to the box and never joined across its own
        lines — a struggle note's `keywords:` line is a separate statement
        from the note above it, and wrapping the two into one paragraph would
        read as one sentence."""
        proposal = self.current
        if proposal is None:
            return []
        out: list[str] = []
        for paragraph in proposal.text.split("\n"):
            out += fold(paragraph, max(8, width - 6)) or [""]
        return out

    def head(self, width: int) -> list[str]:
        # `scope` and `kind` are open strings on the wire (`protocol.Proposal`
        # mirrors `agent.conclude`), so they are shown rather than translated —
        # a kind this screen has never heard of still has to be readable.
        proposal = self.current
        if proposal is None:
            return []
        said = " · ".join(x for x in (proposal.scope, proposal.kind) if x)
        return [BOLD + pad(f"  {said}", width) + RESET] if said else []

    def tail(self, width: int) -> list[str]:
        rows = [" " * width] + options(CHOICES, width)
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

    def view(self, width: int, height: int) -> int:
        """Rows the proposal itself gets. What scrolls has to agree with what
        was drawn, or page-down moves by a different amount than it showed."""
        return max(1, height - len(self.head(width)) - len(self.tail(width)))

    def body(self, width: int, height: int) -> list[str]:
        if self.current is None:
            return [theme.faint + pad(f"  {NOTHING}", width) + RESET]
        head, tail = self.head(width), self.tail(width)
        rows = max(1, height - len(head) - len(tail))
        lines = self.lines(width)
        self.offset = max(0, min(self.offset, max(0, len(lines) - rows)))
        shown = lines[self.offset : self.offset + rows]
        painted = [
            theme.faint + pad(f"    {line}", width) + RESET for line in shown
        ]
        painted += [" " * width] * max(0, rows - len(painted))
        return head + painted + tail

    def render(self, width: int, height: int) -> list[str]:
        # How far down a long proposal you are, on the rule: there is no cursor
        # row to say it, and a memory long enough to scroll is exactly the one
        # whose ending has to be read before it is kept.
        tail = self.question_rows(width)
        inner = max(1, height - 1 - len(tail))
        note = self.note
        lines = self.lines(width)
        rows = self.view(width, inner)
        if not note and len(lines) > rows:
            note = f"{min(self.offset + rows, len(lines))}/{len(lines)}"
        return framed(
            self.heading(),
            self.body(width, inner),
            width,
            height,
            note=note,
            tail=tail,
        )

    def keymap(self) -> list[tuple[str, str]]:
        return [
            ("y", "keep"),
            ("n", "discard"),
            ("↑↓", "scroll"),
            ("esc", "discard the rest"),
        ]

    # -------------------------------------------------------------- the keys

    def keys(self, key: str, width: int, height: int) -> bool:
        if key in BACK_KEYS:
            # Answered short. The core rejects whatever was not answered for,
            # so leaving early is a decision rather than a way out of one.
            return self._finish()
        if key not in ("y", "n"):
            return self._scroll(key, self.view(width, height))
        self.verdicts.append(key == "y")
        self.at += 1
        self.offset = 0
        if self.at >= len(self.proposals):
            return self._finish()
        return True

    def _scroll(self, key: str, view: int) -> bool:
        steps = {"up": -1, "down": 1, "pgup": -view, "pgdn": view}
        if key in steps:
            self.offset = max(0, self.offset + steps[key])
        elif key == "home":
            self.offset = 0
        elif key == "end":
            # Clamped against the real length when it is next drawn, which is
            # the only place the number of folded lines is known.
            self.offset = 10**9
        return True

    def _finish(self) -> bool:
        # Sent even when nothing was answered: an empty list is the honest
        # "none of them", and a review closed without a word would leave the
        # core holding proposals nobody will ever come back to.
        self.send(ResolveMemory(self.session_id, tuple(self.verdicts)))
        return False
