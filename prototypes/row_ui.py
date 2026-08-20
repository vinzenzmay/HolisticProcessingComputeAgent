#!/usr/bin/env python3
"""A runnable sketch of the row-oriented UI, drawn without Textual.

    pixi run -e dev python prototypes/row_ui.py
    pixi run -e dev python prototypes/row_ui.py --chat 2000 --sessions 40

Nothing here imports hpca and nothing here opens a database: the point is to
try the *shape* — a header and three stacked rows, ctrl-arrow to move between
them, arrows line by line inside one, ``e`` to open an entry — before any of it
is wired to real data. The content is synthetic and deliberately over-long,
because the question this exists to answer is what the UI feels like once a
turn has produced hundreds of steps.

The rendering model is pi-tui's, which is the reason for the exercise. pi-tui
is TypeScript, so what is ported here is the architecture, not the library:

* a component renders to ``list[str]`` at a known width. There is no style
  cascade, no auto-height measurement and no arrange pass, so there is no
  O(conversation) traversal to accidentally trigger — which is the whole of
  what makes the Textual version stutter;
* only the *visible* slice of a pane is ever turned into lines, so frame time
  is flat in the number of entries. The header prints it: that number staying
  put while you scroll a 2000-entry chat is the claim being tested;
* the screen is repainted differentially — lines that did not change are not
  written — inside a synchronised-output pair so the terminal shows each frame
  atomically instead of tearing.

What is deliberately missing: real data, mouse support, text selection, an
input box, modal screens, colour theming. Those are the arguments *against*
leaving Textual, and this prototype is not the place to pretend they are
solved.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import textwrap
import time
import tty
from dataclasses import dataclass, field

# Cosmetic only — kept literal so this file stands alone and can be copied out
# of the repo to try in another terminal.
VERSION = "0.25.0-proto"

# --------------------------------------------------------------------- ansi

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


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _pad(text: str, width: int) -> str:
    """Exactly ``width`` visible characters — truncated with an ellipsis, or
    padded out.

    Every line is built plain and padded *before* any SGR is wrapped around it,
    so a highlight covers the full row and no escape sequence is ever cut in
    half by the truncation.
    """
    if width <= 0:
        return ""
    if len(text) > width:
        return text[: width - 1] + "…" if width > 1 else text[:width]
    return text + " " * (width - len(text))


class Screen:
    """Raw terminal, alternate screen, differential repaint."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._saved: list | None = None
        self._prev: list[str] = []

    def __enter__(self) -> "Screen":
        self._saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        # ?1049h alternate screen, ?25l hide the cursor, ?7l autowrap off.
        # Autowrap matters: every line is padded to the full width and painted
        # by absolute cursor address, and a character landing in the last cell
        # of the last row would otherwise leave a wrap pending that scrolls the
        # whole frame as soon as the next one is written.
        _write(f"{ESC}[?1049h{ESC}[?25l{ESC}[?7l{ESC}[2J")
        return self

    def __exit__(self, *exc) -> None:
        _write(f"{ESC}[?7h{ESC}[?25h{ESC}[?1049l")
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def paint(self, lines: list[str], *, full: bool = False) -> None:
        out = [f"{ESC}[?2026h"]  # begin synchronised update
        if full:
            out.append(f"{ESC}[2J")
            self._prev = []
        for row, line in enumerate(lines):
            if not full and row < len(self._prev) and self._prev[row] == line:
                continue
            out.append(f"{ESC}[{row + 1};1H{ESC}[2K{line}")
        out.append(f"{ESC}[?2026l")
        _write("".join(out))
        self._prev = list(lines)


# ---------------------------------------------------------------------- keys

KEYS = {
    f"{ESC}[A": "up",
    f"{ESC}OA": "up",
    f"{ESC}[B": "down",
    f"{ESC}OB": "down",
    f"{ESC}[1;5A": "ctrl-up",
    f"{ESC}[1;5B": "ctrl-down",
    f"{ESC}[5~": "pgup",
    f"{ESC}[6~": "pgdn",
    f"{ESC}[H": "home",
    f"{ESC}[F": "end",
    f"{ESC}[1~": "home",
    f"{ESC}[4~": "end",
    f"{ESC}[Z": "shift-tab",
    "\t": "tab",
    "\r": "enter",
    "\n": "enter",
    "\x03": "quit",
    "\x04": "quit",
}


def decode(data: bytes) -> list[str]:
    """One read into a list of key names, longest escape sequence first."""
    text = data.decode("utf-8", "replace")
    keys: list[str] = []
    at = 0
    while at < len(text):
        for size in (6, 5, 4, 3, 2):
            chunk = text[at : at + size]
            if chunk in KEYS:
                keys.append(KEYS[chunk])
                at += size
                break
        else:
            keys.append(KEYS.get(text[at], text[at]))
            at += 1
    return keys


# --------------------------------------------------------------------- panes


@dataclass
class Item:
    """One entry: a single line, plus the body it opens into."""

    head: str
    body: list[str] = field(default_factory=list)
    accent: str = ""


class Pane:
    """One of the three navigable rows.

    Holds its own cursor line, scroll offset and set of open entries, which is
    what makes "each row remembers where you were" fall out rather than need
    arranging: leaving a pane changes nothing about it.
    """

    def __init__(self, name: str, items: list[Item]) -> None:
        self.name = name
        self.items = items
        self.expanded: set[int] = set()
        self.cursor = 0  # index into the flattened line list
        self.offset = 0  # first visible flattened line
        self._flat: list[tuple[int, str, bool]] | None = None
        self._flat_width = -1

    # ------------------------------------------------------------- content

    def flat(self, width: int) -> list[tuple[int, str, bool]]:
        """``(item index, text, is head line)`` for every line the pane shows.

        Cached, and rebuilt only when an entry opens or closes or the width
        changes — never per keystroke. Scrolling slices this list, which is why
        moving the cursor costs the same at twenty entries as at two thousand.
        """
        if self._flat is not None and self._flat_width == width:
            return self._flat
        lines: list[tuple[int, str, bool]] = []
        for index, item in enumerate(self.items):
            marker = ("▾" if index in self.expanded else "▸") if item.body else " "
            lines.append((index, f"{marker} {item.head}", True))
            if index in self.expanded:
                for raw in item.body:
                    for piece in textwrap.wrap(raw, max(8, width - 4)) or [""]:
                        lines.append((index, f"    {piece}", False))
        self._flat = lines
        self._flat_width = width
        return lines

    def invalidate(self) -> None:
        self._flat = None

    # ----------------------------------------------------------- navigation

    def _current(self, width: int) -> int:
        lines = self.flat(width)
        if not lines:
            return -1
        return lines[min(self.cursor, len(lines) - 1)][0]

    def _go_to(self, item: int, width: int) -> None:
        for row, (owner, _, _) in enumerate(self.flat(width)):
            if owner == item:
                self.cursor = row
                return

    def _scroll_into_view(self, view_h: int, total: int) -> None:
        view_h = max(1, view_h)
        self.cursor = max(0, min(self.cursor, max(0, total - 1)))
        self.offset = max(0, min(self.offset, max(0, total - view_h)))
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + view_h:
            self.offset = self.cursor - view_h + 1

    def move(self, delta: int, view_h: int, width: int) -> None:
        total = len(self.flat(width))
        if not total:
            return
        self.cursor = max(0, min(total - 1, self.cursor + delta))
        self._scroll_into_view(view_h, total)

    def toggle(self, width: int) -> None:
        item = self._current(width)
        if item < 0 or not self.items[item].body:
            return
        self.expanded.symmetric_difference_update({item})
        self.invalidate()
        # Land back on the entry's own first line: opening one twelve lines
        # long and being left in the middle of it reads as a jump.
        self._go_to(item, width)

    def toggle_all(self, width: int) -> None:
        item = self._current(width)
        if self.expanded:
            self.expanded.clear()
        else:
            self.expanded = {i for i, e in enumerate(self.items) if e.body}
        self.invalidate()
        if item >= 0:
            self._go_to(item, width)

    # -------------------------------------------------------------- drawing

    def render(self, width: int, height: int, *, focused: bool) -> list[str]:
        """The pane as exactly ``height`` lines: a title, then the body."""
        inner = max(8, width - 2)  # two columns go to the gutter
        lines = self.flat(inner)
        body_h = max(1, height - 1)
        self._scroll_into_view(body_h, len(lines))
        current = lines[self.cursor][0] if lines else -1
        out = [self._title(width, len(lines), focused)]
        for row in range(self.offset, self.offset + body_h):
            if row >= len(lines):
                out.append(" " * width)
                continue
            owner, text, is_head = lines[row]
            gutter = "▌ " if owner == current else "  "
            painted = _pad(gutter + text, width)
            if row == self.cursor:
                # The unfocused pane still shows where it was left, dimmed —
                # that is the "memory" being visible rather than merely kept.
                painted = (REVERSE if focused else DIM + REVERSE) + painted + RESET
            elif not is_head:
                painted = DIM + painted + RESET
            elif self.items[owner].accent:
                painted = self.items[owner].accent + painted + RESET
            out.append(painted)
        return out

    def _title(self, width: int, total: int, focused: bool) -> str:
        right = f"line {min(self.cursor + 1, total)}/{total}"
        if self.expanded:
            right += f" · {len(self.expanded)} open"
        left = f"── {self.name} "
        gap = max(1, width - len(left) - len(right) - 3)
        text = _pad(f"{left}{'─' * gap}{right} ──", width)
        return (BOLD + CYAN + text + RESET) if focused else (DIM + text + RESET)


# ----------------------------------------------------------------------- app


class RowUI:
    HEADER = 1
    MIN_CHAT = 4

    def __init__(self, panes: list[Pane], *, profile: str) -> None:
        self.panes = panes  # top to bottom: sessions, chat, watchers
        self.focus = 1  # the chat, which is where a session starts
        self.profile = profile
        self.frame_ms = 0.0

    def _heights(self, height: int, width: int) -> list[int]:
        """How the three rows split the screen.

        A quarter each for sessions and watchers and the rest to the chat — but
        only as much of a quarter as the pane actually has to show, so a short
        session list or an empty watcher row costs nothing instead of holding
        a quarter of the screen open. What is left over always goes to the
        chat, which is the row that can use it.
        """
        avail = max(6, height - self.HEADER)
        quarter = max(2, avail // 4)
        inner = max(8, width - 2)
        top = max(2, min(1 + len(self.panes[0].flat(inner)), quarter))
        bottom = max(2, min(1 + len(self.panes[2].flat(inner)), quarter))
        while avail - top - bottom < self.MIN_CHAT and (top > 2 or bottom > 2):
            if top >= bottom and top > 2:
                top -= 1
            elif bottom > 2:
                bottom -= 1
            else:
                break
        middle = avail - top - bottom
        if middle < 1:  # a terminal too short for the design at all
            top = bottom = max(1, (avail - 1) // 3)
            middle = avail - top - bottom
        return [top, middle, bottom]

    def render(self, width: int, height: int) -> list[str]:
        out = [self._header(width)]
        for index, (pane, pane_h) in enumerate(
            zip(self.panes, self._heights(height, width))
        ):
            out.extend(pane.render(width, pane_h, focused=index == self.focus))
        while len(out) < height:
            out.append(" " * width)
        return out[:height]

    def _header(self, width: int) -> str:
        left = f" HPCA {VERSION}  ·  {self.profile}"
        right = (
            f"{self.frame_ms:5.2f}ms  ·  ↑↓ scroll   ^↑ ^↓ row   "
            "e open   a open all   q quit "
        )
        gap = width - len(left) - len(right)
        text = left + " " * gap + right if gap > 0 else left
        return REVERSE + _pad(text, width) + RESET

    def handle(self, key: str, width: int, height: int) -> bool:
        """Act on one key; False means quit."""
        if key in ("q", "quit"):
            return False
        inner = max(8, width - 2)
        view_h = max(1, self._heights(height, width)[self.focus] - 1)
        pane = self.panes[self.focus]
        if key in ("ctrl-down", "tab"):
            self.focus = (self.focus + 1) % len(self.panes)
        elif key in ("ctrl-up", "shift-tab"):
            self.focus = (self.focus - 1) % len(self.panes)
        elif key == "up":
            pane.move(-1, view_h, inner)
        elif key == "down":
            pane.move(1, view_h, inner)
        elif key == "pgup":
            pane.move(-view_h, view_h, inner)
        elif key == "pgdn":
            pane.move(view_h, view_h, inner)
        elif key == "home":
            pane.move(-(10**9), view_h, inner)
        elif key == "end":
            pane.move(10**9, view_h, inner)
        elif key in ("e", "E", "enter"):
            pane.toggle(inner)
        elif key in ("a", "A"):
            pane.toggle_all(inner)
        return True


# ------------------------------------------------------------- sample content

TASKS = [
    "annotate the cohort BAMs with sniffles",
    "why did the snakemake run stall at rule merge_vcf",
    "write the methods section for the SV paper",
    "check GPU utilisation on the last training job",
    "rebuild the reference index on scratch",
    "compare coverage between the two batches",
]

SNIPPETS = [
    "/scratch/proj/cohort/run3/annotation.tsv (412 lines)",
    "wrote 38 lines, backup kept in trash",
    "exit 0 in 2.4s",
    "no match — falling back to a wider search",
    "/home/mayv_c/projects/sv-paper/methods.md",
    "12 files, 3.2 GB",
]

REPLIES = [
    "The merge rule stalled because two shards wrote to the same temp path.",
    "Coverage is 31x in batch A and 28x in batch B; the gap is one flowcell.",
    "I have written the methods section — it is 42 lines, have a look.",
    "The job is still queued behind a reservation; nothing is wrong with it.",
]


def sample_sessions(count: int) -> list[Item]:
    return [
        Item(
            head=f"{TASKS[i % len(TASKS)][:42]:<44}{2 + i * 7:>4}m ago   qwen3-27b",
            body=[
                f"session 9f3c{i:04x} · profile hpc · mode agent",
                f"{4 + (i * 7) % 60} messages · {(i * 13) % 90}% of context used",
                f"last: {REPLIES[i % len(REPLIES)]}",
            ],
            accent=GREEN if i == 0 else "",
        )
        for i in range(count)
    ]


def sample_chat(count: int) -> list[Item]:
    tools = ["read_file", "edit_file", "create_file", "run_bash", "list_dir"]
    items: list[Item] = []
    for i in range(count):
        slot = i % 3
        if slot == 0:
            items.append(
                Item(head=f"you   {TASKS[i % len(TASKS)]}", accent=BLUE)
            )
        elif slot == 1:
            steps = 3 + (i * 5) % 18
            names = [tools[(i + k) % len(tools)] for k in range(steps)]
            items.append(
                Item(
                    head=(
                        f"      {steps} steps · " + " → ".join(names[:3]) + " …"
                    ),
                    body=[
                        f"{name:<14}{SNIPPETS[(i + k) % len(SNIPPETS)]}"
                        for k, name in enumerate(names)
                    ],
                )
            )
        else:
            reply = REPLIES[i % len(REPLIES)]
            items.append(
                Item(
                    head=f"hpca  {reply}",
                    body=[
                        reply,
                        "The detail is in the log at "
                        "/scratch/proj/cohort/run3/logs/merge_vcf.log, and the "
                        "two shards are listed at the bottom of it. I have not "
                        "changed anything yet — say the word and I will patch "
                        "the rule to use a per-shard temp path.",
                    ],
                    accent=YELLOW,
                )
            )
    return items


def sample_watchers(count: int) -> list[Item]:
    states = [
        ("RUNNING", GREEN),
        ("PENDING", ""),
        ("COMPLETED", ""),
        ("FAILED", RED),
    ]
    out = []
    for i in range(count):
        state, accent = states[i % len(states)]
        out.append(
            Item(
                head=(
                    f"{'job ' + str(4821000 + i):<16}{state:<11}"
                    f"last write {3 + i * 11}s ago"
                ),
                body=[
                    f"/scratch/proj/cohort/run3/logs/step{i}.log",
                    "[12:41:07] merging shard 3 of 8",
                    "[12:41:44] merging shard 4 of 8",
                ],
                accent=accent,
            )
        )
    return out


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Row-oriented HPCA UI prototype (no Textual)."
    )
    parser.add_argument("--chat", type=int, default=400, help="chat entries")
    parser.add_argument("--sessions", type=int, default=14)
    parser.add_argument("--watchers", type=int, default=5)
    args = parser.parse_args(argv)

    if not sys.stdin.isatty():
        print("row_ui needs a terminal.", file=sys.stderr)
        return 2

    ui = RowUI(
        [
            Pane("sessions", sample_sessions(args.sessions)),
            Pane("chat", sample_chat(args.chat)),
            Pane("watchers", sample_watchers(args.watchers)),
        ],
        profile="hpc",
    )
    # Open on the newest chat entry, as the real app does.
    ui.panes[1].cursor = 10**9

    with Screen() as screen:
        size = (0, 0)
        while True:
            width, height = os.get_terminal_size()
            resized = (width, height) != size
            if resized:
                size = (width, height)
                for pane in ui.panes:
                    pane.invalidate()
            started = time.perf_counter()
            screen.paint(ui.render(width, height), full=resized)
            # Shown on the next frame rather than this one: measuring the frame
            # that reports the measurement would need two passes to say
            # anything true.
            ui.frame_ms = (time.perf_counter() - started) * 1000
            # The timeout is also how fast a resize is noticed; SIGWINCH would
            # be tidier and is not worth a handler in a prototype.
            if not select.select([screen.fd], [], [], 0.5)[0]:
                continue
            data = os.read(screen.fd, 4096)
            if not data:
                return 0
            for key in decode(data):
                if not ui.handle(key, width, height):
                    return 0


if __name__ == "__main__":
    raise SystemExit(main())
