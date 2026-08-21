"""Terminal setup and the read/render loop.

    pixi run -e dev python -m hpca.ui.run --demo

The only entry point that touches a terminal, so everything above it stays
testable without one.
"""

from __future__ import annotations

import argparse
import codecs
import os
import select
import sys
import time

from hpca.ui.keys import PASTE_START, decode
from hpca.ui.screen import Resizes, Screen

# How long a held-back escape waits for the rest of its sequence before it is
# taken to be the escape key. A pressed escape and the first byte of an arrow
# key are the same byte; what tells them apart is that the terminal already has
# the rest of the arrow buffered and delivers it within microseconds, while a
# person's next keystroke does not. The classic vim/readline figure.
ESC_TIMEOUT = 0.05
# How often the loop wakes with nothing to do. Not a resize poll any more —
# SIGWINCH wakes it — but the escape-stop hint expires on a timer and has to be
# painted away.
IDLE_POLL = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Row-oriented HPCA UI (no Textual)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="draw the UI over synthetic content, with no backend",
    )
    parser.add_argument("--chat", type=int, default=400, help="chat entries")
    parser.add_argument("--sessions", type=int, default=14)
    parser.add_argument("--watchers", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.demo:
        # The client and the state layer are real (M2); what is not wired up
        # yet is an `AgentService` to put on the other end of the connection
        # (M3). Saying so beats drawing an empty UI.
        print("hpca.ui.run only knows --demo so far.", file=sys.stderr)
        return 2

    if not sys.stdin.isatty():
        print("the row UI needs a terminal.", file=sys.stderr)
        return 2

    from hpca.ui.demo import build

    ui = build(args.chat, args.sessions, args.watchers)
    with Screen() as screen, Resizes() as resizes:
        # What decode() could not name yet, carried to the front of the next
        # read: an escape sequence can straddle a read boundary, and a paste
        # routinely spans dozens of them.
        pending = ""
        # A *character* can straddle one too, and one plain decode() per read
        # would turn the halves of a pasted ideograph into two replacement
        # characters. The incremental decoder holds the partial bytes instead.
        chars = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            # A resize invalidates the diff baseline outright — it was taken at
            # the old geometry — so the frame after one is painted whole.
            resized = resizes.taken()
            width, height = os.get_terminal_size()
            if resized:
                ui.invalidate()
            started = time.perf_counter()
            screen.paint(ui.render(width, height), full=resized)
            # Shown on the next frame rather than this one: measuring the frame
            # that reports the measurement would need two passes to say
            # anything true.
            ui.frame_ms = (time.perf_counter() - started) * 1000
            # A held-back escape is only resolved by *nothing* following it, so
            # the wait shortens to the escape timeout while one is pending. An
            # unfinished paste is not on that clock: it is held until its end
            # marker whatever happens, because flushing it as keystrokes is the
            # bug bracketing exists to prevent.
            waiting = bool(pending) and not pending.startswith(PASTE_START)
            timeout = ESC_TIMEOUT if waiting else IDLE_POLL
            ready = select.select([screen.fd, resizes.fd], [], [], timeout)[0]
            if screen.fd not in ready:
                if waiting:
                    keys, pending = decode(pending, final=True)
                else:
                    keys = []
            else:
                data = os.read(screen.fd, 65536)
                if not data:
                    return 0
                keys, pending = decode(pending + chars.decode(data))
            for key in keys:
                if not ui.handle(key, width, height):
                    return 0


if __name__ == "__main__":
    raise SystemExit(main())
