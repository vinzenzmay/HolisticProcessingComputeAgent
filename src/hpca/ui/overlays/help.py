"""The full key list, since the footer can only ever show the ones that fit."""

from __future__ import annotations

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, pad, rule
from hpca.ui.overlays.base import Overlay


class HelpOverlay(Overlay):
    """Every key, since the footer can only ever show the ones that fit."""

    title = "keys"

    SECTIONS = [
        (
            "anywhere",
            [
                ("^↑ ^↓", "move between rows (tab / shift-tab outside the chat)"),
                ("esc esc", "stop the agent, from any row — one esc does nothing"),
                ("m", "manage llms"),
                ("a", "profiles & learnings"),
                ("c", "config editor"),
                ("^l", "switch llm for this session"),
                ("?", "this list"),
                ("q", "quit"),
            ],
        ),
        (
            "in any row",
            [
                ("↑ ↓", "move one line"),
                ("pgup pgdn", "move one screen"),
                ("home end", "first / last line"),
                ("→", "open the entry under the cursor, or step into it"),
                ("←", "close it again"),
                ("shift-→ ←", "open or close every entry in the row"),
            ],
        ),
        (
            "sessions row",
            [
                ("enter", "open it and go straight to the message box"),
                ("r", "rename"),
                ("t", "ask the llm for a title"),
                ("d", "delete"),
            ],
        ),
        (
            "chat row",
            [
                ("i", "go to the message box"),
                ("shift-tab", "cycle this session's agent mode"),
                ("enter", "on the working row: stop the turn"),
                ("→ →", "open a turn into its steps, then a step into its result"),
                ("enter", "on one of your own messages: fork, roll back, copy"),
                ("enter", "on anything else: go to the message box"),
            ],
        ),
        (
            "message box",
            [
                ("enter", "send"),
                ("⇧enter", "new line (alt-enter and ^j too)"),
                ("↑ ↓", "move one screen line (long text wraps)"),
                ("^← ^→", "jump a word (alt-← alt-→ also)"),
                ("shift-← →", "mark text; shift-^← ^→ by the word"),
                ("shift-↑ ↓", "mark whole lines; shift-home end to an end"),
                ("^⌫", "delete the word before the cursor"),
                ("^del", "delete the word after it"),
                ("⌫ del", "delete the marked text, if any"),
                ("^u", "clear"),
                ("^↑", "back to the chat — escape is the stop gesture, not an exit"),
            ],
        ),
        (
            "watchers row",
            [
                ("enter", "peek at the log"),
                ("d", "unwatch"),
                ("alt-↑ alt-↓", "reorder"),
            ],
        ),
    ]

    def __init__(self) -> None:
        self.offset = 0

    def _body(self, width: int) -> list[str]:
        out = []
        for name, keys in self.SECTIONS:
            out.append(DIM + pad(f"  {name}", width) + RESET)
            for key, label in keys:
                # Padded plain and coloured afterwards by column, never by
                # adding the escape lengths to the width — that arithmetic is
                # exactly the kind that leaves a row one cell short.
                row = pad(f"      {key:<12}{label}", width)
                out.append(row[:6] + CYAN + row[6:18] + RESET + row[18:])
            out.append(" " * width)
        return out

    def render(self, width: int, height: int) -> list[str]:
        body = self._body(width)
        rows = max(1, height - 1)
        self.offset = max(0, min(self.offset, max(0, len(body) - rows)))
        more = len(body) > rows
        tail = f"{self.offset + rows}/{len(body)}" if more else ""
        out = [BOLD + CYAN + rule("keys", width, tail) + RESET]
        out += body[self.offset : self.offset + rows]
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def handle(self, key: str, width: int, height: int) -> bool:
        # Scrolls, because the list outgrew a short terminal. Anything else
        # closes it: a list you opened by accident should not need the one key
        # you were looking it up to find.
        step = {"up": -1, "down": 1, "pgup": -(height - 2), "pgdn": height - 2}
        if key in step:
            self.offset += step[key]
            return True
        if key in ("home", "end"):
            self.offset = 0 if key == "home" else 10**9
            return True
        return False

    def footer(self) -> list[tuple[str, str]]:
        return [("↑↓", "scroll"), ("any key", "back")]
