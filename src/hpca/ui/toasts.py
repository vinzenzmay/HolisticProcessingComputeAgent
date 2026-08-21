"""Toasts: what the core said, drawn over the frame (§4.3 item 35).

A `notify` carries text nobody sanitised — LLM output, an exception string, a
memory excerpt, the tail of a log — and the Textual app learned what that costs
the hard way: a stray ``[`` was parsed as markup, `MarkupError` came out of the
compositor, and the whole app went down. Its answer was ``markup=False`` on
every call; ours cannot be, because this renderer has no markup to turn off.

What is left is the same risk in the two forms this UI can actually be hurt by,
and the rule is that **a toast must never break the frame**:

* **Control characters.** An escape sequence in the body would be *drawn* by
  the terminal — moving the cursor, changing the colour, switching the
  character set — while `cell_width` counted it as zero cells. That is the one
  combination a differential repaint cannot survive, because every later row is
  addressed absolutely against a screen that has since moved. `ansi.safe`
  takes them all out, tabs included, before anything is measured.
* **Width.** A hundred-kilobyte line and a hundred-line body are both ordinary
  things for a tool result to contain. Nothing here grows with the payload: the
  body is sliced to what could possibly be drawn *before* it is folded, the
  fold is capped, and every row goes through `pad`, so the block is exactly
  ``width`` cells wide and at most ``MAX_ROWS`` tall whatever it carries.

Drawn over the frame rather than taking rows from it, the way
`RowUI._over_confirm` draws the yes/no: a toast comes and goes on a timer, and
a layout that changed under one would move the conversation up and down while
the user reads it.
"""

from __future__ import annotations

from collections.abc import Sequence

from hpca.ui.ansi import BOLD, DIM, RED, RESET, YELLOW, fold, pad, rule, safe
from hpca.ui.state import Toast

# How long one stays up when the core does not say. Longer for the ones the
# user is meant to actually read, which is what severity is for.
TIMEOUTS = {"information": 5.0, "warning": 8.0, "error": 10.0}

# The severity colours, and the word on the rule.
STYLES = {"information": DIM, "warning": YELLOW, "error": RED}

# At most three at once and at most eight rows between them: a toast covers the
# top of the sessions column, and a stack that could cover the column entirely
# would be a modal nobody asked for.
MAX_TOASTS = 3
MAX_ROWS = 8
# Rows of body one toast may spend before it is clipped. The rest is a line
# saying so, because a toast is a notification and not a window — the footer
# note and the session log are where a wall of text is read.
MAX_BODY = 3

# What a clipped body ends on.
CLIPPED = "… more in the log"


def expires(toast: Toast) -> float:
    """When this one stops being shown, on the same clock it was raised on."""
    timeout = toast.timeout
    if timeout is None:
        timeout = TIMEOUTS.get(toast.severity, TIMEOUTS["information"])
    return toast.at + max(0.0, float(timeout))


def live(toasts: Sequence[Toast], now: float) -> list[Toast]:
    """The ones still up, newest last, at most `MAX_TOASTS` of them."""
    return [x for x in toasts if expires(x) > now][-MAX_TOASTS:]


def next_wake(toasts: Sequence[Toast], now: float) -> float | None:
    """Seconds until the block changes on its own, or None.

    A toast is the third thing in this UI that changes with no keypress behind
    it (after the armed escape and the spinner), so it books its own repaint
    through `RowUI.next_wake` rather than being given a timer of its own.
    """
    due = [expires(x) - now for x in toasts if expires(x) > now]
    # A hair past the moment it expires, for `next_wake`'s reason: waking
    # exactly on the boundary would redraw the same frame and have nothing
    # left to schedule, and the toast would stick.
    return min(due) + 0.01 if due else None


def one(toast: Toast, width: int, height: int) -> list[str]:
    """One toast, at most ``height`` rows of exactly ``width`` cells."""
    if height < 1 or width < 4:
        return []
    style = STYLES.get(toast.severity, DIM)
    out = [style + rule(toast.severity, width) + RESET]
    body = max(4, width - 4)
    room = height - 1
    title = safe(toast.title).replace("\n", " ").strip()
    if title and room > 0:
        out.append(BOLD + pad("  " + title, width) + RESET)
        room -= 1
    lines: list[str] = []
    clipped = False
    for logical in safe(toast.text).split("\n"):
        # Sliced before it is folded: a single 100 KB line folds into a
        # thousand rows of which three are ever drawn, and paying for the other
        # nine hundred and ninety-seven once per frame is how a toast stops
        # being a crash and becomes a performance bug instead.
        piece = logical[: (MAX_BODY + 1) * body]
        clipped = clipped or len(piece) < len(logical)
        lines += fold(piece, body) or [""]
        if len(lines) > MAX_BODY:
            clipped = True
            break
    keep = min(MAX_BODY, room - 1 if clipped else room)
    clipped = clipped or len(lines) > keep
    for line in lines[: max(0, keep)]:
        out.append(style + pad("  " + line, width) + RESET)
    if clipped and len(out) < height + 1:
        out.append(DIM + pad(f"  {CLIPPED}", width) + RESET)
    return out[:height]


def render(
    toasts: Sequence[Toast], now: float, width: int, height: int
) -> list[str]:
    """The whole stack, oldest first, within ``height`` rows.

    Rows are handed out newest first, so the toast that has just arrived is
    never the one dropped for want of room — it is the one being read.
    """
    showing = live(toasts, now)
    if not showing or height < 1:
        return []
    room = min(height, MAX_ROWS)
    blocks: list[list[str]] = []
    for toast in reversed(showing):
        if room < 2:
            break
        block = one(toast, width, min(room, 2 + MAX_BODY))
        if not block:
            break
        blocks.insert(0, block)
        room -= len(block)
    return [row for block in blocks for row in block]
