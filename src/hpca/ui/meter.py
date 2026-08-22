"""The context meter's two pure functions: how full the window is, and how bad.

Copied verbatim (bar glyphs, thresholds and formatting) from
`hpca/tui/context_bar.py`, whose module docstring is the argument for the
feature and is worth keeping to hand:

    HPCA is meant to run on whatever a site can host locally, and a 32k model
    fills up fast — a few tool results and a long log excerpt will do it. When
    it does, quality degrades in ways that look like the model getting stupid
    rather than the model running out of room, so the fill level has to be
    visible *before* that happens, not explained afterwards.

Copied out of `tui/` rather than imported, back when that package still
existed: these two functions were the only part of it with no Textual in them.
Copied rather than reimplemented because they already have tests
(`tests/test_context_bar.py`)
and rewriting a formatter from its output is how a bar ends up one cell wider
on a full window than on an empty one.

What is *not* here is the widget: the state it held — measured or estimated,
the speed, the thinking level — lives on `state.Context`, because in this UI a
meter is a string on a line rather than a thing that owns its own repaint.
"""

from __future__ import annotations

BAR_CELLS = 28
FULL, EMPTY = "█", "─"

# Below `warn` the bar is unremarkable; past it compaction is approaching (the
# graph folds history at 70%); past `danger` the next tool result may not fit.
WARN_FRACTION = 0.70
DANGER_FRACTION = 0.90


def render_bar(
    used: int, window: int | None, *, cells: int = BAR_CELLS, estimated: bool = False
) -> str:
    # "~" marks a figure derived from character counts rather than measured by
    # the backend, so a number that is only roughly right never looks exact.
    mark = "~" if estimated else ""
    if not window:
        return f"context: {mark}{used:,} tokens used · window unknown"
    fraction = min(1.0, used / window)
    if cells <= 0:
        # The narrow-terminal form: the numbers survive, the picture goes. The
        # bar is the first thing to drop because it is the redundant half —
        # it draws the percentage that is written next to it.
        return f"context {mark}{used:,} / {window:,} ({fraction * 100:.0f}%)"
    filled = round(fraction * cells)
    bar = FULL * filled + EMPTY * (cells - filled)
    return f"context [{bar}] {mark}{used:,} / {window:,} ({fraction * 100:.0f}%)"


def severity(used: int, window: int | None) -> str:
    if not window:
        return "unknown"
    fraction = used / window
    if fraction >= DANGER_FRACTION:
        return "danger"
    if fraction >= WARN_FRACTION:
        return "warn"
    return "ok"
