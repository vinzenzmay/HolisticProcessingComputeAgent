"""Terminal setup and the read/render loop.

    pixi run -e dev python -m hpca.ui.run --demo

The only entry point that touches a terminal, so everything above it stays
testable without one.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import time

from hpca.ui.keys import decode
from hpca.ui.screen import Screen


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
        # Nothing but the demo is wired up yet; the real client arrives with
        # specs-ui-replacement.md M2/M3. Saying so beats drawing an empty UI.
        print("hpca.ui.run only knows --demo so far.", file=sys.stderr)
        return 2

    if not sys.stdin.isatty():
        print("the row UI needs a terminal.", file=sys.stderr)
        return 2

    from hpca.ui.demo import build

    ui = build(args.chat, args.sessions, args.watchers)
    with Screen() as screen:
        size = (0, 0)
        while True:
            width, height = os.get_terminal_size()
            resized = (width, height) != size
            if resized:
                size = (width, height)
                ui.invalidate()
            started = time.perf_counter()
            screen.paint(ui.render(width, height), full=resized)
            # Shown on the next frame rather than this one: measuring the frame
            # that reports the measurement would need two passes to say
            # anything true.
            ui.frame_ms = (time.perf_counter() - started) * 1000
            # The timeout is also how fast a resize is noticed; SIGWINCH would
            # be tidier and is not worth a handler yet.
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
