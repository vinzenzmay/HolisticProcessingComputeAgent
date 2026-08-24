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
  the review exists, so the comment box is *here* rather than in a child
  screen: a retry is one gesture, and a second screen to type into would be
  two.
* **discard** — throw the summary away and leave the conversation as it was.

Escape is none of them. It takes the prompt off the screen and answers
nothing, and the core goes on holding the offer, so `/compact` brings the same
summary back rather than paying for a new one. That is what makes escape safe
on a prompt that cost a generation to open — and it is also why escape here
gives the message box back, unlike the approval next door: a summary holds no
turn hostage, so a conversation with one waiting in it is a conversation that
can still be typed into.

**Inline, in the session's own column, and not a modal** — the property this
shares with the approval prompt (`ui/approval.py`), for the same reason and by
the same mechanism: it stands in the message box's slot at the foot of the
chat column, so the conversation it summarizes is still on screen behind it,
and a summary waiting in one conversation does not cover another one the user
switched to. It arrives a whole model call after the keystroke that asked for
it, which is long enough for the user to be reading something else entirely;
a screen taking the terminal at that moment would be asking about a
conversation that is not the one in front of them. What a background session's
summary does instead is mark its sidebar row and toast once — opening that
session is what puts the prompt up.

The summary itself scrolls, for the reason `inspect` does: a guided summary
can be several thousand characters, and a prompt that showed the first six
lines of the thing it is asking about would be the truncation this whole
exchange exists to let the user complain about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hpca.ui import theme
from hpca.ui.ansi import BOLD, PULSE_PERIOD, RESET, fold, pad, pulse, rule
from hpca.ui.editor import Editor

# The three verdicts, spelled as `state.ResolveCompact` sends them.
ACCEPT, RETRY, DISCARD = "accept", "retry", "discard"

# The two halves of the prompt, the same split the approval has: the summary
# and its three keys, then the summary with a box under it saying what to do
# differently. Naming them apart is what lets the letters be verdicts in one
# half and letters in the other.
ASK, COMMENT = "ask", "comment"

# What the prompt says when the core warned that the text is cut. The wording
# names the fix, because "truncated" alone leaves the user to work out that the
# retry comment is where the complaint goes.
CUT_WARNING = "this summary is cut off — press r and say “finish it”"

# The three keys and the way out, on one line — the approval's `(y) approve ·
# (n) deny` row, with a third answer and a fourth thing that is not one.
HINT = "(enter) fold it in · (r) again · (d) discard · (esc) later"

# The prompt over the comment box.
COMMENT_HINT = "what should the summary do differently?"
# Refused rather than sent: a retry with nothing said is the same generation
# again, and the model has no way to know it was turned down.
COMMENT_REFUSAL = "say what to change, or esc to go back"

# Rows the comment box gets when it is open. Two, because the comment is a
# sentence and not a name — "you cut it off, and keep the sbatch flags" is
# already more than one line of a narrow terminal.
COMMENT_ROWS = 2

# The least this prompt can be drawn in: the rule, the heading, one line of
# summary and the keys. Below that there is nothing left to drop that was not
# the question itself or the way to answer it.
MIN_ROWS = 4


@dataclass
class CompactProposal:
    """A summary `/compact` wrote, waiting to be accepted (`compact.proposed`).

    Held on the session, for the reason `state.Offer` is: it arrives a model
    call after the keystroke, and by then the user may be reading another
    conversation. Answering it against the wrong one would fold the wrong
    history.

    ``summary`` is the message as the core would store it, because that is what
    the user is being asked to accept — not a preview of it. ``truncated`` is
    the core's warning that the text is cut, which is the one thing a reader
    cannot tell for themselves.
    """

    summary: str = ""
    folded: int = 0
    guidance: str = ""
    attempt: int = 1
    truncated: bool = False


@dataclass
class CompactReview:
    """One offered summary, and what the user has done about it so far.

    The `approval.Decision` shape, and for the same reasons. ``proposal`` is
    the core's; ``stage``, ``comment`` and ``offset`` are the UI's, and they
    live on the `SessionState` so that switching conversations parks them —
    the half-typed complaint about one summary is still there on the way back,
    the same bargain §4.4 struck for the half-typed refusal reason.

    ``standing`` is whether the prompt is in the message box's slot right now.
    Escape clears it and answers nothing: the offer is still the session's,
    the sidebar still says so, and `/compact` stands it up again.

    ``window`` is how many rows of summary the last frame had room for, kept
    because two things need the same answer and are asked at different times:
    the rule draws "42/180" before the window has been worked out, and page-up
    has to move by what was actually shown or it scrolls past what was read.
    """

    proposal: CompactProposal | None = None
    stage: str = ASK
    comment: Editor = field(default_factory=lambda: Editor(wrap=True))
    offset: int = 0
    standing: bool = True
    window: int = 1

    @property
    def asking(self) -> bool:
        return self.stage == ASK

    @property
    def up(self) -> bool:
        """Whether this is what the entry band is drawing."""
        return self.proposal is not None and self.standing

    def again(self) -> None:
        """Open the box that says what to do differently. Nothing is sent."""
        self.stage = COMMENT

    def back(self) -> None:
        """Out of the box, not out of the prompt: the summary is still there
        to be accepted, and a comment abandoned half-way is not a decision
        about it."""
        self.stage = ASK

    def comment_text(self) -> str:
        return self.comment.text().strip()

    def scroll(self, key: str) -> bool:
        """The summary under the arrows. True if the key was one of them."""
        view = max(1, self.window)
        steps = {"up": -1, "down": 1, "pgup": -view, "pgdn": view}
        if key in steps:
            self.offset = max(0, self.offset + steps[key])
        elif key == "home":
            self.offset = 0
        elif key == "end":
            # Clamped against the real length when it is next drawn, which is
            # the only place the number of folded lines is known.
            self.offset = 10**9
        else:
            return False
        return True


# ------------------------------------------------------------- what it says


def heading(proposal: CompactProposal) -> str:
    """How many messages this folds, and which try this is.

    The attempt is named only from the second one on: "attempt 1" over a first
    summary would be answering a question nobody has asked yet.
    """
    text = f"{proposal.folded} messages fold into this summary"
    return f"{text} · attempt {proposal.attempt}" if proposal.attempt > 1 else text


def summary_lines(review: CompactReview, width: int) -> list[str]:
    """The summary, folded to the column and never joined across its own lines
    — the "[earlier in this session]" prefix and the paragraphs under it are
    separate statements, and wrapping them into one would read as one."""
    proposal = review.proposal
    if proposal is None:
        return []
    out: list[str] = []
    for paragraph in proposal.summary.split("\n"):
        out += fold(paragraph, max(8, width - 6)) or [""]
    return out


def _position(review: CompactReview, width: int) -> str:
    """How far down a long summary the reader is, on the rule.

    There is no cursor row here to say it, and "is there more of this below?"
    is precisely the question a user reading a summary for truncation is
    asking. Nothing while the whole thing fits, which is what a short
    conversation's fold looks like.
    """
    total = len(summary_lines(review, width))
    if total <= review.window:
        return ""
    return f"{min(review.offset + review.window, total)}/{total}"


def _head(review: CompactReview, width: int, focused: bool) -> list[tuple[str, str]]:
    """The rule and what is being decided, above the text of it."""
    proposal = review.proposal
    if proposal is None:
        return []
    rows = [
        (
            BOLD + theme.chrome if focused else theme.faint,
            rule("compact", width, _position(review, width)),
        ),
        (BOLD, f"  {heading(proposal)}"),
    ]
    if proposal.guidance:
        rows.append((theme.faint, f"  asked to keep: {proposal.guidance}"))
    if proposal.truncated:
        rows.append((theme.warn, f"  {CUT_WARNING}"))
    return rows


def _tail(review: CompactReview, style: str) -> list[tuple[str, str]]:
    """The keys, or the line over the box that has taken them."""
    if review.asking:
        return [(style, f"  {HINT}")]
    return [(theme.faint, f"  {COMMENT_HINT}")]


def review_height(review: CompactReview, width: int, cap: int) -> int:
    """How tall the prompt would like to be, within what it may have.

    Everything it has to say, and then as much of the summary as is left over
    — capped, because the conversation this is a summary *of* has to stay on
    screen behind it, which is the whole reason this is not a screen.
    """
    if not review.up:
        return 0
    wants = (
        len(_head(review, width, False))
        + len(_tail(review, ""))
        + len(summary_lines(review, width))
        + (0 if review.asking else COMMENT_ROWS)
    )
    return max(MIN_ROWS, min(cap, wants))


def render_review(
    review: CompactReview,
    width: int,
    height: int,
    *,
    focused: bool,
    now: float | None = None,
    period: float = PULSE_PERIOD,
) -> list[str]:
    """The prompt, in exactly ``height`` rows of exactly ``width`` cells.

    What gives when the room is short is the summary, and only the summary:
    the rule, the heading, the warning and the keys are the question and the
    ways to answer it, while a summary that scrolls has already said there is
    more of it than fits. So the text keeps whatever is left after those, down
    to a single line.

    ``now`` is the clock the keys line pulses on and None is "do not" — the
    arrangement the approval prompt has, for the same reason: how tall this is
    must not depend on what time it is, so the measurement passes None. Only
    the keys breathe, and only while they are the keys: at the comment stage
    the cursor is already in a box, and two things asking for the eye at once
    is neither of them getting it.
    """
    if review.proposal is None:
        return [" " * width] * height
    style = pulse(now, period) if review.asking and now is not None else theme.faint
    head = _head(review, width, focused)
    tail = _tail(review, style)
    box = 0 if review.asking else COMMENT_ROWS
    box = max(0, min(box, height - len(head) - len(tail) - 1))
    review.window = max(1, height - len(head) - len(tail) - box)
    lines = summary_lines(review, width)
    review.offset = max(0, min(review.offset, max(0, len(lines) - review.window)))
    # Built again now that the window and the offset are settled: the rule
    # carries "42/180", and the first pass worked it out from the window of
    # the frame before this one.
    head = _head(review, width, focused)
    shown = lines[review.offset : review.offset + review.window]
    # A gutter rather than a box: what is in here is the text being decided
    # on, and it has to be told apart at a glance from the sentence above it —
    # the same device the approval draws a script with.
    rows = head + [(theme.faint, f"  │ {line}") for line in shown]
    rows += [("", "")] * max(0, review.window - len(shown))
    rows += tail
    out = [f"{sgr}{pad(text, width)}{RESET}" for sgr, text in rows[:height]]
    if box:
        body = review.comment.render(max(4, width - 2), box, focused=focused)
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((theme.warn if focused else theme.faint) + marker + RESET + line)
    while len(out) < height:
        out.append(" " * width)
    return out[:height]
