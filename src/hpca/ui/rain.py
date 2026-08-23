"""Falling glyphs, for the one screen that has nothing else on it.

A confirmation clears the frame it was asked over (`app.RowUI._over_confirm`),
which leaves a black screen with three lines in the middle of it. This fills
the black — and the reason it is worth the code is not only that it is fun: a
terminal that goes blank is a terminal that might have died, and a screen that
is visibly *animating* behind the question is one that is unmistakably alive
and waiting for an answer.

Everything here is a pure function of ``(width, height, now)``. That is not
tidiness for its own sake: `render` may be called more than once for the same
instant, the repaint is differential, and a drop whose position depended on
how many frames had been drawn would fall at a speed set by how busy the
machine was. Nothing here holds state, and no two frames at the same ``now``
can differ.

Costs are paid only while a question is on screen, which is seconds at a time
— and every cell of it is one column wide, because a frame whose rows are not
exactly ``width`` cells is a frame the repaint tears (see `ansi.cell_width`).
"""

from __future__ import annotations

from hpca.ui import theme
from hpca.ui.ansi import BOLD, RESET

# Halfwidth katakana, which are the glyphs the effect is remembered for, plus
# digits for the flicker. Halfwidth deliberately: `ア` (U+30A2) is two cells
# wide and `ｱ` (U+FF71) is one, so the full-width block would put a column of
# the frame half a cell out of step with every row it fell through.
KATAKANA = tuple(chr(code) for code in range(0xFF66, 0xFF9E))
GLYPHS = KATAKANA + tuple("0123456789")

# Rows a second. The spread is the whole illusion — one speed reads as a
# curtain being lowered rather than as rain — and the range is bounded below
# by "does it look stopped" and above by how far a drop moves between two
# repaints: at the lowest frame rate the settings allow, 30 rows a second is a
# drop jumping three rows at a time, which reads as a dotted line rather than
# as a streak.
MIN_SPEED, MAX_SPEED = 4.0, 18.0

# How long a drop's tail is, in rows.
MIN_TRAIL, MAX_TRAIL = 5, 16

# Dark rows added to a column's cycle after the drop has fallen off the
# bottom, as a multiple of the height. This is what makes the field sparse:
# without it every column rains at once, which is a wall of green and reads as
# a broken terminal rather than as an effect.
#
# It is also the dial that decides what this costs. Every lit cell carries an
# escape sequence, so the bytes on the wire are very nearly linear in how many
# of them there are, and a third of the columns falling at once is both the
# better picture and about 50 KiB/s down the link rather than 150.
DARK = 2.6

# How often the glyphs in place are swapped for others, in hertz. The drops
# fall and the characters *also* churn; the second one is most of what makes
# it look like the code from the film rather than like snow.
CHURN = 9.0

# Frames a second, and the default for `config.DisplaySettings.quit_rain_fps`.
# The loop repaints when something happens, so anything that moves by the
# clock alone has to book its own frame (`app.RowUI.next_wake`) — this is how
# often that one asks.
#
# Sixty, because falling is the one thing this draws and ten a second reads as
# stepping. What that costs is worth writing down, since it is not the linear
# thing it looks like: `screen.paint` sends only the rows that *changed*, and
# what changes per second is set by how fast the drops fall and how fast the
# glyphs churn, not by how often the field is sampled. Measured over a 120x40
# screen, one second of it:
#
#     fps      wire     rows/s     cpu
#      10   88 KiB/s      399    3.9 ms/s
#      30  223 KiB/s      990    9.4 ms/s
#      60  313 KiB/s     1379   18.9 ms/s
#     120  379 KiB/s     1662   37.7 ms/s
#
# So six times the frame rate is three and a half times the bytes, and the cpu
# never matters. Locally 313 KiB/s is nothing; down a tunnel to a login node
# it is enough to be felt, which is what the setting is for — and why the
# lower end of it is genuinely usable rather than a token.
FPS = 60

# The head is the bright one, then a few full-strength rows, then the tail
# fades. Three tiers rather than a gradient because a 256-colour ramp of green
# costs a distinct SGR sequence per row of every drop, and at this size the
# difference is not visible against the difference in what it costs to send.
#
# Each one opens with a RESET so that it says the whole state rather than a
# change to it: the row is written as runs, and a `theme.ok` following the head
# would otherwise inherit the head's BOLD — and the eye reads bold green as a
# second head, which puts two heads in one drop.
# Read when the field is drawn rather than bound here: the palette is a
# setting now, and a module constant is exactly the staleness `ui.theme`
# exists to prevent — it would hold whichever colours were current at import
# and go on raining them after the settings editor had said otherwise.
def _tiers() -> tuple[str, str, str]:
    """Head, body and tail, in the palette in force now."""
    return (
        RESET + BOLD + theme.agent,
        RESET + theme.ok,
        RESET + theme.faint + theme.ok,
    )


BODY_ROWS = 3


def _mix(n: int) -> int:
    """A 32-bit integer hash. Deterministic across processes, unlike ``hash``.

    Which is the point: `hash` of a str is salted per interpreter, so a frame
    built with it would differ between two runs of the same test — and, worse,
    between two front-ends drawing the same instant.
    """
    n &= 0xFFFFFFFF
    n = (n ^ 61) ^ (n >> 16)
    n = (n + (n << 3)) & 0xFFFFFFFF
    n ^= n >> 4
    n = (n * 0x27D4EB2D) & 0xFFFFFFFF
    return n ^ (n >> 15)


def _unit(n: int) -> float:
    """``_mix`` as a float in [0, 1)."""
    return _mix(n) / 4294967296.0


def rain(width: int, height: int, now: float) -> list[str]:
    """A frame of it: ``height`` rows of exactly ``width`` cells.

    ``now`` is a clock in seconds; any origin will do, since every column's
    phase is its own. A width or height of nothing is nothing, which is the
    answer a terminal too small for the question has already given.
    """
    if width <= 0 or height <= 0:
        return []
    # Two planes: what is at each cell, and what colour it is. Kept apart so
    # the row can be written as runs of one style — a cleared screen of rain
    # is the most escape-dense frame this UI ever draws, and paying for a
    # sequence per *cell* rather than per run costs about a third again.
    cells = [[" "] * width for _ in range(height)]
    styles: list[list[str]] = [[""] * width for _ in range(height)]
    # Once for the frame, not once per drop: the palette cannot change while
    # one frame is being built, and this is the hottest loop the UI has.
    head_style, body_style, tail_style = _tiers()
    tick = int(now * CHURN)
    for x in range(width):
        speed = MIN_SPEED + _unit(x * 0x9E3779B1) * (MAX_SPEED - MIN_SPEED)
        trail = MIN_TRAIL + int(_unit(x * 0x85EBCA77 + 1) * (MAX_TRAIL - MIN_TRAIL))
        # The cycle a column repeats on: down the screen, off the bottom with
        # its tail behind it, then dark for a while.
        cycle = height + trail + DARK * height
        phase = _unit(x * 0xC2B2AE35 + 2) * cycle
        head = (now * speed + phase) % cycle
        for depth in range(trail):
            y = int(head) - depth
            if not 0 <= y < height:
                continue
            cells[y][x] = GLYPHS[
                _mix(x * 0x27D4EB2D + y * 0x165667B1 + tick) % len(GLYPHS)
            ]
            if depth == 0:
                styles[y][x] = head_style
            elif depth <= BODY_ROWS:
                styles[y][x] = body_style
            else:
                styles[y][x] = tail_style
    return [_row(cells[y], styles[y]) for y in range(height)]


def _row(glyphs: list[str], styles: list[str]) -> str:
    """One row, with a style written only where it changes.

    A space needs no colour of its own — these are foreground colours, and an
    unpainted space and a green one are the same picture — so the runs are
    allowed to reach across the gaps between drops, and only a glyph whose
    style differs from the last one written pays for a sequence. The row ends
    reset, because a row that leaves a colour open is a colour that runs into
    whatever the next row turns out to be.
    """
    out: list[str] = []
    current = ""
    for glyph, style in zip(glyphs, styles):
        if style and style != current:
            out.append(style)
            current = style
        out.append(glyph)
    if current:
        out.append(RESET)
    return "".join(out)


# --------------------------------------------------------------- the spinner

# The same effect, four cells wide, for the working row.
#
# A one-character spinner says "something is happening" and nothing else; this
# screen already has a vocabulary for waiting, and it is the one behind the
# quit question. So the working row gets a drop of it: four columns, the head
# oscillating between the walls instead of falling, and the trail behind it —
# behind in the direction it *came from*, which is what makes the bounce read
# as a bounce rather than as two unrelated slides.
#
# It does not rain. A column that fell off the bottom would need somewhere to
# fall to, and the row it is on is one line high; the oscillation is what
# gives the same glyphs the same restlessness in the space there is.
SPINNER_WIDTH = 4

# Where the head is on each frame of the cycle. Six frames rather than eight:
# the walls are not held for a second frame, because a drop that pauses at the
# end of its travel reads as a stutter in the animation rather than as a turn.
SPINNER_STEPS = (0, 1, 2, 3, 2, 1)

def spinner_trail() -> tuple[str, ...]:
    """What the trail costs in brightness, by how far behind the head it is.

    Four tiers where `rain` has three, and the reason the argument there does
    not apply is arithmetic: a full screen of drops pays a distinct SGR
    sequence per row of every one of them, and this pays at most four per
    frame. So the decay can be a real ramp, which over four cells is the whole
    of what makes the trail read as fading rather than as a block of one colour
    with a bright end.

    It is `palette.spinner` and not a tier of the signal colours, because a
    spinner says *busy* and the signal three say how things are — a green that
    also means busy is a green that no longer means well. Shorter than four is
    a shorter trail rather than an error, and one entry is a plain blinking
    cell, which is a legitimate thing to ask a spinner to be.

    Each opens with a RESET so it states the whole style rather than a change
    to it, for the reason `_tiers` gives: these are written as runs, and a
    colour that inherited the previous cell's BOLD would read as a second head.
    """
    return tuple(RESET + colour for colour in theme.spinner)


def spinner(frame: int) -> tuple[str, tuple[str, ...]]:
    """One frame of it: ``SPINNER_WIDTH`` glyphs, and the style of each.

    Returned apart, and that is not a convenience. Every line this UI draws is
    built plain and measured before any SGR is wrapped around it
    (`ansi.pad`, `pane.Item`), so a spinner that arrived already coloured
    would be a working row measured with its escapes in it. The glyphs go into
    the line; the styles are put on afterwards, over the padded row.

    Katakana only, where the field mixes digits in: the digits are the
    flicker in a screenful of them, and in four cells a `7` is not a flicker,
    it is a number the eye tries to read off the working row.

    ``frame`` is a count, not a clock — `state.Turn` derives it from the same
    tick that decides when the row is next drawn, so the glyphs churn only on
    frames that were going to be painted anyway.
    """
    step = frame % len(SPINNER_STEPS)
    head = SPINNER_STEPS[step]
    # Where it came from, which is the side the trail is on. Reading the
    # previous step rather than working the direction out from the frame index
    # is what makes the two turning points fall out for free: at a wall the
    # drop has just arrived from the one place it can have come from.
    came_from = SPINNER_STEPS[step - 1]
    direction = 1 if head >= came_from else -1
    glyphs = [" "] * SPINNER_WIDTH
    styles = [""] * SPINNER_WIDTH
    for depth, style in enumerate(spinner_trail()):
        x = head - direction * depth
        if not 0 <= x < SPINNER_WIDTH:
            continue
        glyphs[x] = KATAKANA[
            _mix(x * 0x27D4EB2D + frame * 0x165667B1) % len(KATAKANA)
        ]
        styles[x] = style
    return "".join(glyphs), tuple(styles)
