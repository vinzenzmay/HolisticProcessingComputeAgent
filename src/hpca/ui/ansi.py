"""Escape sequences, cell widths, and the string helpers built on them.

Everything that draws goes through ``pad``: every line is built plain, padded
to an exact width, and only then wrapped in SGR. Keeping that order in one
place is what stops an escape sequence from ever being counted as width or cut
in half by a truncation.

The width being counted is *terminal cells*, not characters. An emoji or a CJK
ideograph occupies two of them, a combining mark none, and a zero-width joiner
sequence is several code points drawn as one glyph. Counting with ``len()``
shifts the rest of the line by however far the count was wrong, and because the
screen is repainted differentially — only the rows that changed are rewritten —
a row drawn one cell short stays wrong until something else happens to touch
it. So this is a correctness property of the repaint, not a cosmetic one.

``unicodedata`` is enough for it and is in the standard library, so there is no
``wcwidth`` dependency. Two deliberate limits, both shared with ``wcwidth``:

* East-asian *ambiguous* characters count as one. Every rule, marker and arrow
  this UI draws (``─ ▌ ▸ ▾ ● ○ … ↑``) is ambiguous-width, so widening
  them would be wrong in the overwhelmingly common case of a non-CJK
  locale.
* An emoji ZWJ sequence is counted per emoji rather than per glyph, because
  what a terminal actually does with ``👩‍💻`` varies by terminal. Nothing is
  ever *split* inside one, which is the half that matters: a dangling joiner
  leaves the terminal waiting for a glyph that never arrives.
"""

from __future__ import annotations

import math
import unicodedata
from functools import lru_cache

ESC = "\x1b"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
REVERSE = f"{ESC}[7m"
# The colours are not here any more. They are settings (`config.PaletteSettings`)
# resolved by `ui.theme`, which is what lets a user build their own palette —
# and they could not stay constants to do it: twenty-one modules said
# `from hpca.ui.ansi import CYAN`, which binds the string at import, so a
# palette that changes while the app runs has to be *looked up* when the row is
# drawn. What is left in this module is the part that has no colour in it: the
# attributes below, the width arithmetic, and `pad`.
#
# The role names those constants became: CYAN is `theme.chrome`, GREEN is
# `theme.ok`, YELLOW `theme.warn`, RED `theme.danger`, WHITE `theme.agent`,
# AMBER `theme.user`, and the DIM attribute is `theme.faint` — a real grey, on
# the argument that what DIM looked like was the terminal theme's opinion and
# differed from one to the next on the same screen. BLUE is simply gone; it had
# no call site at all.

# ------------------------------------------------------------- the pulse

# The one colour on the screen that is a function of the clock rather than of
# what is on the row. The decision prompt's answer line breathes between the
# colour prose is written in and the colour the chrome is, because a parked
# turn is a turn nobody is driving: the "!" in the sidebar says a *background*
# session is waiting, and this says the session in front of you is — a thing a
# static dim line failed to say, since it looks exactly like the key hints
# under every other row.
#
# The ramp is walked rather than the two ends being swapped, because a hard
# switch between two colours is a blink, and a blinking line is read as broken
# rather than as waiting. It is built by `ui.theme.ramp` now that both ends are
# settings, and the care that used to be described here lives with it: an
# index walk is interpolated in *level* space rather than in RGB, because the
# cube's levels are unevenly spaced and quantising an even RGB walk rounds the
# three channels at different points — which drops a grey step into the middle
# of a walk that should never have left its hue.

# How long one breath takes, and how often the frame it is on has to be drawn
# again. The period is a setting now (`config.DisplaySettings`, arriving as
# `protocol.DisplaySettings.decision_pulse_seconds`) and this is what a caller
# that was handed none uses; it is also the floor of what the interval below
# can resolve, so the two are read together. It was 2.4s, on the argument that
# a slow breath is not a flicker — 1.0 because in use the line is read as a
# *prompt* waiting for an answer, and a two-and-a-half-second cycle is slow
# enough that a glance at the screen catches it standing still.
#
# The interval is the spinner's 0.1 for the spinner's reason (`ui/state.py`):
# it is the idle cost of having a decision on screen, and this is the cheapest
# rate that still moves. The sine is fastest through the middle of its sweep,
# where ten frames a second skips a step of the seven — which is the part of a
# gradient nobody can follow anyway; the ends, where it lingers, get every one.
PULSE_PERIOD = 1.0
PULSE_INTERVAL = 0.1


def pulse(now: float, period: float = PULSE_PERIOD) -> str:
    """The answer line's colour at this instant, and at no other.

    A pure function of the clock, exactly as the spinner's glyph is
    (`Turn.frame`), and for the same two reasons: a UI that repaints only when
    something changed can work out when this one next will (`PULSE_INTERVAL`),
    and a test can pin the clock and get the colour back rather than watching
    for a change it has no way to time.

    ``period`` is how long one breath takes. A value that cannot be divided by
    falls back to the module's own rather than raising: the settings model is
    what refuses a period of zero (`config.DisplaySettings` — ``gt=0``, one
    line beside the editor that typed it), and by the time a number has
    crossed the wire it is being divided by inside a repaint, where the only
    thing an exception can do is take the frame down with the terminal in raw
    mode. Belt and braces, and the braces are the ones the user can read.
    """
    from hpca.ui import theme  # deferred: `theme` imports this module

    if period <= 0:
        period = PULSE_PERIOD
    ramp = theme.pulse
    phase = (math.sin(now * math.tau / period) + 1) / 2
    return ramp[min(len(ramp) - 1, int(phase * len(ramp)))]


# Joiners ask for the glyph after them, so a slice must never end on one.
JOINERS = "‍‌"

# The UI's own furniture, every character of it one cell wide.
#
# This exists because the ASCII fast path below missed almost everything it
# was written for. Nearly every row this UI draws carries a rule, a marker or
# an arrow — `── chat ──`, `▸ ● qwen`, `↑↓ line` — and one non-ASCII character
# is enough to drop the whole row onto the per-character path. The fast path
# was therefore fastest on exactly the rows the UI does not have.
#
# Correctness is not taken on trust: `char_width` remains the authority and
# `test_ui_ansi.py` asserts that it returns 1 for every character in here.
# A glyph added to the UI and not to this set is merely slower; a glyph added
# to this set that is not one cell wide fails the suite.
ONE_CELL_GLYPHS = (
    "─│└▌▏█"          # rules, gutters, the cursor block and the meter
    "▸▾●○★•"          # row markers and the backend dots
    "→←↑↓⇧⇥⌫⏎⟳↺⇔"     # the key hints
    "…—–·›“”§✓✗⚠≤÷"   # typography and status marks
    # The halfwidth katakana the spinner and the quit screen are drawn from.
    # Halfwidth is the whole reason they can be in here: `ア` is two cells and
    # `ｱ` is one, which is also why `rain` picked this block (see its GLYPHS).
    + "".join(chr(code) for code in range(0xFF66, 0xFF9E))
)

# Printable ASCII plus the above. `frozenset.issuperset` walks the string in C,
# which is what makes checking cheaper than measuring.
_ONE_CELL = frozenset(map(chr, range(0x20, 0x7F))) | frozenset(ONE_CELL_GLYPHS)


# --------------------------------------------------------------- cell widths


@lru_cache(maxsize=4096)
def char_width(ch: str) -> int:
    """Terminal cells one character occupies: 0, 1 or 2.

    Cached because the hot path is a per-frame re-measure of the visible rows,
    and the alphabet a conversation is written in is tiny however long it is.
    """
    if ch < "\x7f":  # the overwhelmingly common case, and the cheapest
        return 1 if ch >= " " else 0
    if unicodedata.combining(ch):
        return 0
    if unicodedata.category(ch) in ("Cc", "Cf", "Mn", "Me"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def cell_width(text: str) -> int:
    """How many terminal cells ``text`` occupies.

    Only ever called on *plain* text: control characters count as zero here,
    which is true of a combining mark and a lie about a tab, so anything that
    could carry one — a paste, an entry body — is cleaned before it is drawn.

    The fast path is not an optimisation in the usual "nice to have" sense.
    Measuring cells rather than characters made this function 222x more
    expensive than the ``len()`` it replaced, and it runs on every visible row
    of every frame — enough to move frame time by an order of magnitude. Two
    C-level scans settle the common case: text that is ASCII and printable
    occupies one cell per character by definition, and almost all of it is.
    """
    if text.isascii() and text.isprintable():
        return len(text)
    if _ONE_CELL.issuperset(text):
        return len(text)
    return sum(map(char_width, text))


def fit_index(text: str, start: int, cells: int) -> int:
    """The index one past the longest run from ``start`` fitting ``cells``.

    The slicing primitive the rest of the module is built on. A character whose
    second cell would fall outside the budget is left out entirely, so the run
    is *at most* ``cells`` wide and never ends half way through a glyph. Zero
    width characters cost nothing and so always come along with the character
    they belong to.

    Same fast path as `cell_width`, and for the same reason — this is what
    `pad` measures with, so it runs once per drawn row. One cell per character
    means the answer is arithmetic, and a slice cannot land inside a glyph
    when every glyph is one cell wide.

    **The fast path is taken on a window, never on the whole string, and that
    is load-bearing rather than tidy.** No answer can depend on a character
    further than ``cells`` past ``start`` — a run that wide is already full —
    so at most that many need looking at. Scanning the whole string instead
    makes this O(length) where the loop below is O(cells): the loop stops the
    moment the budget is used. A single chat entry can be a megabyte and is
    measured once per frame it is visible in, so the difference is the
    difference between 0.06 ms and 6 ms a frame. `tests/test_ui_perf.py`
    holds the case that proves it.
    """
    room = max(0, cells)
    # One character past the budget, and that one is what makes this correct.
    # Zero-width characters follow the character they modify and come along
    # free, so a window cut exactly at the budget would drop a combining mark
    # that belongs inside it. Including the next character means any such mark
    # is *in* the window, where it fails the all-one-cell test and sends the
    # answer to the loop below. When the window does pass, the character past
    # the budget is known to be one cell wide, so it is known not to come.
    window = text[start : start + room + 1]
    if (window.isascii() and window.isprintable()) or _ONE_CELL.issuperset(window):
        return min(len(text), start + room)
    used, at, n = 0, start, len(text)
    while at < n:
        step = char_width(text[at])
        if used + step > cells:
            break
        used += step
        at += 1
    return at


def cut(text: str, cells: int) -> str:
    """The longest prefix of ``text`` that fits ``cells``, glyphs kept whole."""
    at = fit_index(text, 0, cells)
    while at and text[at - 1] in JOINERS:
        at -= 2 if at > 1 else 1  # the joiner and what it was joining to
    return text[:at]


# ------------------------------------------------------------ string helpers


def pad(text: str, width: int) -> str:
    """Exactly ``width`` terminal cells — truncated with an ellipsis, or padded.

    Every line is built plain and padded *before* any SGR is wrapped around it,
    so a highlight covers the full row and no escape sequence is ever cut in
    half by the truncation.

    Truncation cuts on a character boundary and then pads the remainder, so a
    two-cell character that straddles the limit is dropped rather than halved:
    the answer is a well-defined ``width`` cells, of which the last may be a
    space, instead of a partial glyph whose real width is the terminal's guess.
    """
    if width <= 0:
        return ""
    # Measured with ``fit_index`` rather than ``cell_width`` so the cost is the
    # width of the row and not the length of the string: a single chat entry
    # can be a megabyte, and it is padded once per frame it is visible in.
    if fit_index(text, 0, width) >= len(text):
        return text + " " * (width - cell_width(text))
    if width == 1:  # no room for both a character and the ellipsis
        head = cut(text, 1)
        return head + " " * (1 - cell_width(head))
    head = cut(text, width - 1)
    return head + "…" + " " * (width - 1 - cell_width(head))


# What a clipped preview says about itself.
#
# Deliberately not the "…" `pad` truncates with, though both mean text was
# cut. That one is an accident of the terminal being this wide; this one is a
# fact about the row — there is more of this message, and → shows it. Two
# different things, and only the second has a gesture attached to it, so they
# are not allowed to look the same.
CLIP = " [...]"


def clip(text: str, cells: int) -> str:
    """``text`` in at most ``cells`` cells, marked if anything was cut.

    Measured with `fit_index` rather than `cell_width` so the cost is the
    width of the row and not the length of the string: the text handed to this
    is a whole chat message, which can be a megabyte, and it is clipped once
    per frame it is visible in.
    """
    if fit_index(text, 0, cells) >= len(text):
        return text
    room = cells - cell_width(CLIP)
    if room < 1:  # a pane too narrow to say both; the words win
        return cut(text, cells)
    # `rstrip` so the mark reads as one space after the last word rather than
    # as two after a cut that happened to land on one.
    return cut(text, room).rstrip() + CLIP


# The least space that may stand between two columns of a row. One is a word
# break; two is a gutter, and a gutter is what says the name on the left and
# the count on the right are two different facts rather than one long phrase.
GAP = 2


def column(text: str, width: int, second: str = "", gap: int = GAP) -> str:
    """``text`` in a column ``width`` cells wide, then ``second`` after it.

    What `f"{name:<20}{detail}"` was doing, with the two things that spelling
    gets wrong. It counts *characters*, so a name with a two-cell glyph in it
    pushes the second column a cell right of everyone else's; and it pads to
    exactly the width, so a name that is already that long has its detail
    written straight onto the end of it — `svirlpool validation3 sessions`,
    which is the bug this exists to make unspellable. The column is a minimum
    here, never a maximum: a name is not truncated to keep an alignment, it
    simply takes the room it needs and the gutter follows it.
    """
    if not second:
        return text
    return f"{text}{' ' * max(gap, width - cell_width(text))}{second}"


def fold(text: str, width: int) -> list[str]:
    """``text`` broken into lines of at most ``width`` cells.

    What ``textwrap.wrap`` was doing before cells were counted, minus its
    reflowing of whitespace: the pieces concatenate back to the original, which
    is the same property the message box needs of ``wrap_spans`` and for the
    same reason. Tabs are expanded because the terminal, not this code, decides
    how wide one is, and a row whose width the terminal decides is a row this
    UI cannot pad exactly.
    """
    text = text.expandtabs(4)
    return [text[a:b] for a, b in wrap_spans(text, width)]


def wrap_spans(text: str, width: int) -> list[tuple[int, int]]:
    """Where one logical line breaks to fit ``width`` cells, as ``(start, end)``.

    Broken at a space where there is one and mid-run where there is not. The
    spans are *character* indices and partition the line exactly — nothing is
    dropped, not even the space that caused the break — because the cursor is
    addressed by column, and a swallowed character would leave a column with
    nowhere to stand. Only the budget is measured in cells; the indices stay
    characters so that the partition is exact and every column keeps a slot.

    A row can therefore come out a cell short of full: if the next character is
    two cells wide and one cell is left, it goes to the following row and the
    cell stays empty. Splitting it would be the alternative, and a half glyph
    has no defined width at all.
    """
    if width < 1:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    at, n = 0, len(text)
    while at < n:
        end = fit_index(text, at, width)
        if end >= n:
            spans.append((at, n))
            break
        # ``end`` is exclusive, so this looks at exactly the characters that
        # fit — the space in the last cell included.
        cut_at = text.rfind(" ", at, end)
        spans.append((at, cut_at + 1 if cut_at > at else end))
        at = spans[-1][1]
    if not spans:
        return [(0, 0)]
    start, stop = spans[-1]
    if cell_width(text[start:stop]) >= width:
        spans.append((n, n))  # a full last line still needs a cursor slot
    return spans


def rule(label: str, width: int, right: str = "") -> str:
    left = f"── {label} "
    tail = f"{right} ──" if right else "──"
    gap = max(1, width - cell_width(left) - cell_width(tail))
    return pad(f"{left}{'─' * gap}{tail}", width)


def reverse(text: str, ranges: list[tuple[int, int]]) -> str:
    """``text`` with those *character* ranges highlighted, nothing else moved.

    Character indices rather than cell columns on purpose: a highlight follows
    the characters it was asked for, so marking the ideograph under the cursor
    covers both of its cells without the caller having to know that it has two.
    Every caller works in the same character indices the editor's columns are.

    The one thing cells decide here is where a range may *begin* and *end*: a
    combining mark is drawn on top of the character before it, so a boundary
    between the two would paint the accent over an unhighlighted base. Both
    ends are nudged outwards past any zero-width character so a glyph is always
    highlighted whole.
    """
    spans = sorted((a, b) for a, b in ranges if b > a)
    if not spans:
        return text
    out: list[str] = []
    at = 0
    for start, end in spans:
        start, end = max(start, at), min(end, len(text))
        while 0 < start < len(text) and char_width(text[start]) == 0:
            start -= 1  # do not start on a mark: take the base it sits on
        while end < len(text) and char_width(text[end]) == 0:
            end += 1  # do not end before one: take it with its base
        if end <= start:
            continue
        out.append(text[at:start])
        out.append(REVERSE + text[start:end] + RESET)
        at = end
    out.append(text[at:])
    return "".join(out)


def _footer_note(note: str, width: int) -> str:
    """The note as it will be drawn, cut to what a row can hold.

    Cut rather than allowed to run past the edge: every row is padded to an
    exact number of cells, so one over-long `notify` would otherwise shift the
    differential repaint by however far it overflowed. Three cells go to the
    leading space and the two after the note.
    """
    return cut(note, max(0, width - 3)) if note else ""


def _footer_fill(
    pairs: list[tuple[str, str]], width: int, used: int, max_rows: int
) -> list[list[tuple[str, str]]]:
    """``pairs`` dealt into rows of ``width``, the first row starting ``used``
    cells in. The loop `footer_wrap` is two calls to."""
    rows: list[list[tuple[str, str]]] = [[]]
    for pair in pairs:
        piece = cell_width(pair[0]) + 1 + cell_width(pair[1])
        extra = piece + (2 if rows[-1] else 0)
        if used + extra <= width - 1:
            rows[-1].append(pair)
            used += extra
        elif len(rows) >= max_rows:
            break  # no rows left to give: the rest fall off the end
        elif 1 + piece <= width - 1:
            rows.append([pair])
            used = 1 + piece
    return rows


def footer_wrap(
    pairs: list[tuple[str, str]], width: int, note: str = "", max_rows: int = 1
) -> list[list[tuple[str, str]]]:
    """Which ``key label`` pairs land on which footer row at this width.

    Split out of `footer_lines` because the frame has to know how tall the
    footer is *before* it can decide how many rows are left for the panes, and
    asking that must not mean building the styled strings a second time from a
    second set of inputs — one function answers both questions, and they cannot
    disagree (`RowUI._avail`).

    A pair is never broken across rows: it is a key and the word for what the
    key does, and half of that is not a hint. One so wide that no row could
    hold it whole is dropped on its own and the rest carry on, rather than
    everything after it going with it.

    An empty last row is the note's: see `footer_lines` for why it gets one.
    """
    note = _footer_note(note, width)
    used = 1 + (cell_width(note) + 2 if note else 0)
    rows = _footer_fill(pairs, width, used, max_rows)
    if not note or len(rows) == 1:
        return rows
    return _footer_fill(pairs, width, 1, max_rows - 1) + [[]]


def footer_lines(
    pairs: list[tuple[str, str]],
    width: int,
    note: str = "",
    style: str = "",
    max_rows: int = 1,
) -> list[str]:
    """The ``key label`` pairs, keys bright and labels dim, over as many rows
    as they need — up to ``max_rows``.

    Wrapping rather than truncating, because a hint that is not on the screen
    is a hint nobody has: on a narrow terminal the pairs that used to fall off
    the right-hand end were exactly the ones a user was least likely to know
    already. So the row fills, the next one starts, and the caller takes the
    rows off the panes' share (`RowUI._avail`) — which is why ``max_rows``
    exists at all. Under that cap the old policy is what is left: whole pairs
    fall off the end, since a footer that has eaten the conversation is worse
    than a hint ``?`` will still list in full.

    The note shares the row while there is only one, exactly as it always has.
    Once the hints need more than one, it takes the bottom line for itself: a
    note is a sentence rather than a hint, it arrives and expires while the
    keys sit still, and the bottom line of the screen is where a reader already
    looks for one — under a stack of key rows is not where the eye would find
    it. Costing a row is the price of that, and only while a note is up.
    """
    from hpca.ui import theme  # deferred: `theme` imports this module

    note = _footer_note(note, width)
    style = style or theme.warn
    rows = footer_wrap(pairs, width, note, max_rows)
    at = len(rows) - 1 if note and len(rows) > 1 else 0
    key_style, label_style = theme.chrome, theme.faint
    out: list[str] = []
    for index, row in enumerate(rows):
        head = f"{style}{note}{RESET}  " if note and index == at else ""
        used = 1 + (cell_width(note) + 2 if head else 0)
        styled: list[str] = []
        for key, label in row:
            used += cell_width(key) + 1 + cell_width(label) + (2 if styled else 0)
            styled.append(f"{key_style}{key}{RESET} {label_style}{label}{RESET}")
        out.append(" " + head + "  ".join(styled) + " " * max(0, width - used))
    return out


def safe(text: str) -> str:
    """Arbitrary text as something this UI can draw and measure.

    The one function every unsanitised string goes through — a pasted block, a
    `notify` carrying LLM output, a skill description off somebody's disk —
    and the reason it is here rather than next to any one of them: the risk it
    answers is a property of *drawing*, not of where the text came from.

    Line endings are normalised because a CR is a line break in the source and
    not a carriage return to obey. Tabs become spaces because how wide a tab is
    is the terminal's decision, and a row whose width the terminal decides is a
    row this UI cannot pad exactly. Everything else in the control range goes,
    escape sequences included: `cell_width` counts a control character as zero
    cells and the terminal draws it as *something* — moves the cursor, changes
    the colour, redefines the character set — which is the one combination that
    corrupts a differential repaint. The Textual UI met the same class of bug
    from the other end and answered it with ``markup=False``; there is no
    markup here, so width and control characters are what is left.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    return "".join(ch for ch in text if ch == "\n" or (ch >= " " and ch != "\x7f"))
