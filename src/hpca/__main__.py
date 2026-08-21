"""Entry point: ``hpca`` console script / ``python -m hpca``.

Two front-ends live here for the length of the port. Textual is still the
default — the row UI is opt-in behind ``--new-ui`` until M9 swaps them
(specs-ui-replacement.md §6) — and the one rule this file has is that **argv is
parsed before either of them is imported**. §4.2 item 10 records why: the same
shape is what a ``--serve`` core will need, and a core process that imported
the UI on its way to deciding it was not one would pay for Textual, a terminal
driver and every widget in the package to run a graph nobody is watching. The
imports are therefore inside the branches, not at the top.
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
        "--new-ui",
        action="store_true",
        help="run the row-oriented UI instead of the Textual one",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="the profile to work under (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.new_ui:
        import asyncio

        from hpca.ui.boot import start

        if not sys.stdin.isatty():
            print("the row UI needs a terminal.", file=sys.stderr)
            return 2
        return asyncio.run(start(profile=args.profile))

    from hpca.tui.app import HpcaApp

    HpcaApp(profile=args.profile).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
