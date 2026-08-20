"""Helpers the `hpca.ui` tests share.

`RowUI` is synchronous and does no I/O, so the whole harness is "construct it,
feed keys through `handle()`, read frames back from `render()`". These four
helpers are everything on top of that: strip the SGR so a frame can be searched
as plain text, drive the escape-stop clock by hand instead of sleeping through
its window, and find a row worth aiming at.
"""

from __future__ import annotations

import re

from hpca.ui.app import CHAT, RowUI

# Every style is an SGR sequence, so one pattern strips a frame back to what a
# terminal would actually show — which is what the width assertions count.
SGR = re.compile(r"\x1b\[[0-9;]*m")


def plain(line: str) -> str:
    return SGR.sub("", line)


def frame(ui: RowUI, width: int, height: int) -> list[str]:
    """One rendered frame, styles removed."""
    return [plain(x) for x in ui.render(width, height)]


def widths(lines: list[str]) -> set[int]:
    """The distinct visible widths in a frame — `{width}` if it is well formed."""
    return {len(plain(x)) for x in lines}


def clocked(ui: RowUI) -> RowUI:
    """Drive the escape window by hand instead of sleeping through it."""
    ui._now = 100.0
    ui.clock = lambda: ui._now
    return ui


def on_own_message(ui: RowUI, nth: int = 4) -> int:
    """Put the chat cursor on the nth message the user wrote — not the first,
    so that a fork or a rollback actually has a conversation to cut."""
    ui.focus = CHAT
    ui.chat.cursor = 0
    seen = 0
    while True:
        index = ui.chat.current(118)
        if ui.chat.items[index].kind == "user":
            seen += 1
            if seen == nth:
                assert index > 0, "the cut has to be mid-conversation to mean anything"
                return index
        was = ui.chat.cursor
        ui.chat.move(1, 20, 118)
        if ui.chat.cursor == was:
            raise AssertionError("ran out of chat before finding one")
