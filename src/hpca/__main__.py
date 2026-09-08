"""Entry point: ``hpca`` console script / ``python -m hpca``.

The one rule this file has is that **argv is parsed before the UI is
imported**. specs/specs-ui-replacement.md §4.2 item 10 records why: the same shape is
what a ``--serve`` core will need (M11), and a core process that imported the
UI on its way to deciding it was not one would pay for a terminal driver and
every screen in the package to run a graph nobody is watching. The import is
therefore inside :func:`main`, not at the top.

``-p/--print`` is the first thing to take that rule up on: it runs one turn
against the same runtime with no terminal at all (`hpca.headless`), so it must
not drag `hpca.ui` in on the way past. Which is also why the tty check below
sits *after* the branch — it exists because raw mode on a pipe hangs, and a
headless run never enters raw mode. A pipe is its normal input.

This used to carry two front-ends and a ``--new-ui`` flag to pick between them.
The Textual one is gone as of v0.26.0 and the flag with it; the reasoning, and
the measurements behind it, are in specs/specs-ui-baseline.md.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    from hpca import __version__

    parser = argparse.ArgumentParser(
        prog="hpca",
        description="Terminal AI agent for HPC Slurm clusters.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="the profile to work under (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    headless = parser.add_argument_group(
        "headless (-p)", "one turn, no terminal — for scripts and other agents"
    )
    from hpca.headless import add_arguments

    add_arguments(headless)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.prompt is not None:
        from hpca.headless import main as headless_main

        return headless_main(args)

    import asyncio

    from hpca.ui.boot import start

    # Checked before anything is drawn: the UI puts the terminal into raw mode
    # and the alternate screen, and doing that to a pipe produces a hang rather
    # than an error.
    if not sys.stdin.isatty():
        print("hpca needs a terminal, or -p to run one turn without one.", file=sys.stderr)
        return 2
    return asyncio.run(start(profile=args.profile))


if __name__ == "__main__":
    raise SystemExit(main())
