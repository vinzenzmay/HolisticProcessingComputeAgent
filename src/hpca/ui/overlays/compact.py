"""The compaction review: read the summary before it becomes the history.

A fold is the one thing this UI asks the core for that cannot be taken back.
`/compact` used to write the summary and then *show* it — a toast the user read
after the conversation behind it had already stopped reaching the model — and
the thing they most often had to say about it ("it stops mid-sentence", "it
lost the sbatch flags") was the thing there was no longer anywhere to say. So
the summary arrives here first, and nothing lands until it is accepted.

Three answers, because a summary is not a yes/no:

* **accept** — fold it in. The core writes it and the meter re-derives.
* **again** — say what is wrong with it in a line, and the core writes another
  one with that sentence in front of the summarizer. This is the whole reason
  the screen exists, so the comment box is on this screen rather than in a
  child: a retry is one gesture, and a second screen to type into would be two.
* **discard** — throw the summary away and leave the conversation as it was.

Escape is none of them. It closes the screen and answers nothing, and the core
goes on holding the offer, so `/compact` brings the same summary back rather
than paying for a new one. That is what makes escape safe on a screen that
cost a generation to open.

The summary itself scrolls, for the reason `inspect` does: a guided summary can
be several thousand characters, and a screen that showed the first fifteen
lines of the thing it is asking about would be the truncation this whole
exchange exists to let the user complain about.
"""

from __future__ import annotations

from hpca.ui import theme
from hpca.ui.ansi import BOLD, RESET, fold, pad
from hpca.ui.editor import Editor
from hpca.ui.keys import NEWLINE_KEYS
from hpca.ui.overlays.base import BACK_KEYS, Overlay, framed, options
from hpca.ui.state import CompactProposal

ACCEPT, RETRY, DISCARD = "accept", "retry", "discard"

# The keys, and what they are called where they are drawn.
CHOICES = [
    ("enter / a", "fold it in"),
    ("r", "again, with a comment"),
    ("d", "discard it"),
    ("esc", "decide later — /compact reopens it"),
]

# What the screen says when the core warned that the text is cut. The wording
# names the fix, because "truncated" alone leaves the user to work out that the
# retry comment is where the complaint goes.
CUT_WARNING = "this summary is cut off — press r and say “finish it”"

# The prompt over the comment box.
COMMENT_HINT = "what should the summary do differently?"
# Refused rather than sent: a retry with nothing said is the same generation
# again, and the model has no way to know it was turned down.
COMMENT_REFUSAL = "say what to change, or esc to go back"

NOTHING = "no summary to review"

# Rows the comment box gets when it is open. Two, because the comment is a
# sentence and not a name — "you cut it off, and keep the sbatch flags" is
# already more than one line of a narrow terminal.
COMMENT_ROWS = 2


class CompactReviewOverlay(Overlay):
    """One summary, and the three things that can happen to it.

    Carries the session it was opened for, like every screen in this package
    that decides something: a review answered after the user switched
    conversations must not fold the one they switched to.
    """

    title = "compact this conversation?"

    def __init__(
        self, proposal: CompactProposal | None = None, session_id: str = ""
    ) -> None:
        super().__init__()
        self.proposal = proposal or CompactProposal()
        self.session_id = session_id
        # What was decided, read by `RowUI._closed` — which is also what marks
        # the context meter stale, and that is why the verdict is read there
        # rather than sent from here: accepting is the moment the number on
        # the bar stops describing the prompt.
        self.action = ""
        self.comment = ""
        self.offset = 0
        # Whether the box at the bottom has the keys. The one modal state this
        # screen has, and it is a small one: escape leaves the box without
        # leaving the screen, so a comment started by accident costs nothing.
        self.commenting = False
        self.editor = Editor(wrap=True)

    # ------------------------------------------------------------- the frame

    def heading(self) -> str:
        if self.proposal.attempt > 1:
            return f"{self.title} · attempt {self.proposal.attempt}"
        return self.title

    def lines(self, width: int) -> list[str]:
        """The summary, folded to the box and never joined across its own
        lines — the prefix and the user's instruction under it are separate
        statements, and wrapping them into one paragraph would read as one."""
        out: list[str] = []
        for paragraph in self.proposal.summary.split("\n"):
            out += fold(paragraph, max(8, width - 4)) or [""]
        return out

    def head(self, width: int) -> list[str]:
        """What is being decided, above the text of it."""
        rows = [
            BOLD
            + pad(
                f"  {self.proposal.folded} messages fold into this summary",
                width,
            )
            + RESET
        ]
        if self.proposal.guidance:
            rows.append(
                theme.faint
                + pad(f"  asked to keep: {self.proposal.guidance}", width)
                + RESET
            )
        if self.proposal.truncated:
            rows.append(theme.warn + pad(f"  {CUT_WARNING}", width) + RESET)
        rows.append(" " * width)
        return rows

    def tail(self, width: int) -> list[str]:
        if self.commenting:
            return [
                " " * width,
                theme.faint + pad(f"  {COMMENT_HINT}", width) + RESET,
            ] + self.editor.render(width, COMMENT_ROWS, focused=True)
        return [" " * width] + options(CHOICES, width)

    def view(self, width: int, height: int) -> int:
        """Rows the summary itself gets. What scrolls has to agree with what
        was drawn, or page-down moves by a different amount than it showed."""
        return max(1, height - len(self.head(width)) - len(self.tail(width)))

    def body(self, width: int, height: int) -> list[str]:
        if not self.proposal.summary:
            return [theme.faint + pad(f"  {NOTHING}", width) + RESET] + [
                " " * width
            ] * max(0, height - 1)
        head, tail = self.head(width), self.tail(width)
        rows = max(1, height - len(head) - len(tail))
        lines = self.lines(width)
        self.offset = max(0, min(self.offset, max(0, len(lines) - rows)))
        shown = lines[self.offset : self.offset + rows]
        painted = [pad(f"  {x}", width) for x in shown]
        painted += [" " * width] * max(0, rows - len(painted))
        return head + painted + tail

    def render(self, width: int, height: int) -> list[str]:
        # How far down a long summary you are, on the rule: there is no cursor
        # row to say it, and "is there more of this below?" is precisely the
        # question a user reading a summary for truncation is asking.
        tail = self.question_rows(width)
        inner = max(1, height - 1 - len(tail))
        note = self.note
        lines = self.lines(width)
        rows = self.view(width, inner)
        if not note and self.proposal.summary and len(lines) > rows:
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
        if self.commenting:
            return [("enter", "ask again"), ("esc", "back to the summary")]
        return [
            ("enter", "accept"),
            ("r", "again"),
            ("d", "discard"),
            ("↑↓", "scroll"),
            ("esc", "later"),
        ]

    # -------------------------------------------------------------- the keys

    def keys(self, key: str, width: int, height: int) -> bool:
        if self.commenting:
            return self._comment_key(key)
        if key in BACK_KEYS:
            return False  # answered nothing; the core keeps the offer
        if not self.proposal.summary:
            return True
        if key in ("enter", "a"):
            return self._answer(ACCEPT)
        if key == "d":
            return self._answer(DISCARD)
        if key == "r":
            self.commenting = True
            self.note = ""
            return True
        return self._scroll(key, self.view(width, height))

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

    def _comment_key(self, key: str) -> bool:
        if key in BACK_KEYS:
            # Out of the box, not out of the screen: the summary is still
            # there to be accepted, and a comment abandoned half-way is not a
            # decision about it.
            self.commenting = False
            self.note = ""
            return True
        if key == "enter":
            comment = self.editor.text().strip()
            if not comment:
                self.note = COMMENT_REFUSAL
                return True
            self.comment = comment
            return self._answer(RETRY)
        if key in NEWLINE_KEYS:
            self.editor.newline()
            return True
        self.editor.handle(key)
        self.note = ""
        return True

    def paste(self, text: str) -> None:
        """A block the terminal handed over whole. Only the comment box has
        anywhere to put it; the summary is somebody else's text."""
        if self.commenting:
            self.editor.insert_text(text)

    def _answer(self, action: str) -> bool:
        self.action = action
        return False

