"""The full key list, since the footer can only ever show the ones that fit."""

from __future__ import annotations

from hpca.ui import theme
from hpca.ui.ansi import BOLD, RESET, pad, rule
from hpca.ui.overlays.base import Overlay, keyed


class HelpOverlay(Overlay):
    """Every key, since the footer can only ever show the ones that fit."""

    title = "keys"

    SECTIONS = [
        (
            "anywhere",
            [
                ("^↑ ^↓", "move between rows — tab cycles on too, from"),
                ("", "every row and from the message box"),
                ("esc esc", "stop the agent, from any row — one esc does nothing"),
                ("m", "manage llms — sessions row only"),
                ("a", "profiles & learnings — sessions row only"),
                ("c", "config editor — anywhere but the chat row"),
                ("^l", "switch llm for this session"),
                ("/thinking", "how hard this session's model reasons"),
                ("?", "this list — from one of the three rows, where"),
                ("", "it is not a character somebody is typing"),
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
                ("enter", "on (new session): pick a profile, then start one"),
                ("r", "rename"),
                ("t", "ask the llm for a title"),
                ("d", "delete (asks first; the log on disk is kept)"),
                ("alt-↑ alt-↓", "reorder"),
            ],
        ),
        (
            "chat row",
            [
                ("c", "copy the row under the cursor to the clipboard"),
                ("enter", "on the working row: stop the turn (asks first)"),
                ("→ →", "open a turn into its steps, then a step into its result"),
                ("enter", "on one of your own messages: fork, or roll back to it"),
                ("enter", "on a queued message: cancel it, or copy it"),
                ("enter", "on anything else: go to the message box"),
            ],
        ),
        (
            "the decision prompt",
            [
                ("y", "approve — run it"),
                ("n", "refuse, and say what should be different"),
                ("esc", "refuse without saying why"),
                ("enter", "in the box: send the reason with the refusal"),
                ("⇧enter", "in the box: new line"),
                ("^↑", "leave it unanswered — the half-typed reason waits"),
            ],
        ),
        (
            "a yes/no question",
            [
                ("y", "yes"),
                ("n / esc", "no — escape is an answer here, not a way past"),
            ],
        ),
        (
            "message box",
            [
                ("enter", "send"),
                ("tab", "cycle on to the watchers row"),
                ("shift-tab", "cycle this session's agent mode — only here"),
                ("⇧enter", "new line (alt-enter and ^j too)"),
                ("↑ ↓", "move one screen line; from the top/bottom line, your"),
                ("", "own past messages — the draft comes back with ↓"),
                ("^← ^→", "jump a word (alt-← alt-→ also)"),
                ("shift-← →", "mark text; shift-^← ^→ by the word"),
                ("shift-↑ ↓", "mark whole lines; shift-home end to an end"),
                ("^⌫", "delete the word before the cursor"),
                ("^del", "delete the word after it"),
                ("⌫ del", "delete the marked text, if any"),
                ("^u", "clear"),
                ("^z ^y", "undo / redo — a send starts the undo again"),
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
        super().__init__()
        self.offset = 0

    def _body(self, width: int) -> list[str]:
        out = []
        for name, keys in self.SECTIONS:
            out.append(theme.faint + pad(f"  {name}", width) + RESET)
            for key, label in keys:
                out.append(keyed(f"      {key:<12}{label}", width))
            out.append(" " * width)
        return out

    def render(self, width: int, height: int) -> list[str]:
        body = self._body(width)
        rows = max(1, height - 1)
        self.offset = max(0, min(self.offset, max(0, len(body) - rows)))
        more = len(body) > rows
        tail = f"{self.offset + rows}/{len(body)}" if more else ""
        out = [BOLD + theme.chrome + rule("keys", width, tail) + RESET]
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
