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

import unicodedata
from functools import lru_cache

ESC = "\x1b"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
REVERSE = f"{ESC}[7m"
CYAN = f"{ESC}[38;5;44m"
GREEN = f"{ESC}[38;5;71m"
YELLOW = f"{ESC}[38;5;179m"
RED = f"{ESC}[38;5;167m"
BLUE = f"{ESC}[38;5;68m"

# Joiners ask for the glyph after them, so a slice must never end on one.
JOINERS = "‍‌"


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
    """
    if text.isascii() and text.isprintable():
        return min(len(text), start + max(0, cells))
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


def footer_line(
    pairs: list[tuple[str, str]], width: int, note: str = "", style: str = YELLOW
) -> str:
    """As many ``key label`` pairs as fit, keys bright and labels dim.

    Truncation is by whole pairs rather than by characters: half a hint is
    worse than one hint fewer, and ``?`` opens the full list anyway — which is
    the honest answer to "show *all* the hotkeys" on an 80-column terminal.
    """
    plain: list[str] = []
    styled: list[str] = []
    used = 1
    if note:
        # Cut rather than allowed to run past the edge: every row is padded to
        # an exact number of cells, so one over-long `notify` would otherwise
        # shift the differential repaint by however far it overflowed. Three
        # cells go to the leading space and the two after the note.
        note = cut(note, max(0, width - 3))
        used += cell_width(note) + 2
    for key, label in pairs:
        piece = f"{key} {label}"
        extra = cell_width(piece) + (2 if plain else 0)
        if used + extra > width - 1:
            break
        plain.append(piece)
        styled.append(f"{CYAN}{key}{RESET} {DIM}{label}{RESET}")
        used += extra
    head = f"{style}{note}{RESET}  " if note else ""
    return " " + head + "  ".join(styled) + " " * max(0, width - used)


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
