"""Wiring the runtime to the terminal, and taking it down in order.

`run.py` owns the terminal and `client.py` owns the protocol; `Core`
(`hpca.core.boot`) owns the runtime. This is the module that joins the three,
and since v0.37.0 that is *all* it is: the runtime moved out when a second
front-end needed it, and what stayed behind is everything with an opinion
about a screen — the manage-LLMs screen a dead backend opens, the loop-lag
probe that measures a UI's event loop, and `start()` itself.

**Why in-process, and why that is not a shortcut.** `specs/specs-ui-replacement.md`
§2 puts the subprocess split (`--serve`) last on purpose: it buys crash
isolation, not correctness. `InProcessConnection.pair()` is the same
`Connection` surface the socket presents, down to close ending the peer's
iteration, so the day the core moves out of the *process* only `Core.start`
changes.

The shutdown order lives with the class that owns it; `tests/test_ui_boot.py`
asserts it of `hpca.core.boot`.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

from hpca.core.boot import (
    DB_SYNC_DONE_MESSAGE,
    DB_SYNC_INTERRUPT_MESSAGE,
    DB_SYNC_WAIT_MESSAGE,
    Core,
    _say,
    file_logger,
    logs_to_file,
)
from hpca.transport import InProcessConnection

logger = logging.getLogger("hpca.ui.boot")

# Re-exported rather than moved: `hpca.ui.boot` is what the UI suite and
# `ui/run.py` import these from, and a runtime that lives elsewhere is not a
# reason to make every caller say so.
__all__ = [
    "DB_SYNC_DONE_MESSAGE",
    "DB_SYNC_INTERRUPT_MESSAGE",
    "DB_SYNC_WAIT_MESSAGE",
    "Core",
    "file_logger",
    "quiet_terminal",
    "start",
]


def quiet_terminal():
    """`logs_to_file` under the name and file the TUI has always used.

    Kept as its own name because the reason a *terminal* needs it is specific:
    the UI holds the terminal in raw mode, so a stray record does not merely
    clutter the output, it lands inside a frame and stays there.
    """
    return logs_to_file("ui.log")


def _open_llm_screen(ui) -> None:
    """Manage-LLMs, opened because startup found nothing answering.

    The same screen `m` opens and drawn from the same catalog, so what the
    user gets is a screen they can also reach on their own rather than a modal
    that only exists at startup. The core has already said *why* — the warning
    it emits arrives as a toast — and that warning is also what repaints the
    frame: it is on the wire before this runs, so the client applies it, wakes
    the loop, and the screen this opened is drawn with the explanation over it.

    Not over an open screen. A probe takes a couple of seconds, and by then
    the user may have opened something of their own or be on their way out;
    the notification alone is better than a screen landing under their hands.
    """
    from hpca.ui.overlays import LlmOverlay

    if ui.overlay is not None:
        return
    ui.overlay = LlmOverlay(ui.catalog)


def _lag_probe(ui):
    """Event-loop scheduling delay, measured (specs/specs-core-process.md §8).

    The instrument for the measurement that justifies this whole
    architecture: the case for moving the agent into its own process is that
    synchronous work on the UI's loop makes it stutter, and that has to be a
    number taken before and after — the Textual app took the "before", and
    this is what takes the "after" against the same yardstick.

    Off unless `$HPCA_LOOPLAG` is set, and a disabled probe starts no task and
    touches nothing, which is what makes it safe to leave wired in
    permanently. The label hook is the app's own word for what it is doing, so
    a spike in the log names the step that caused it.
    """
    from hpca.looplag import LoopLagProbe

    return LoopLagProbe(
        enabled=bool(os.environ.get("HPCA_LOOPLAG")),
        label=ui.activity_label,
    )


def _write_lag_report(probe) -> None:
    """Leave the run's block in `<app_dir>/looplag.log`, if it measured one.

    Here rather than in `ui/run.py` because this is the module that knows what
    an app dir is — the loop is measured where it runs and the answer is
    written down where answers live. A disabled probe writes nothing and says
    so, so this needs no guard of its own; the `$HPCA_LOOPLAG` value rides
    along as the block's note, which is what makes two blocks in one file
    tellable apart ("baseline main", "row ui").
    """
    from hpca.config import app_dir

    with contextlib.suppress(Exception):
        probe.write_report(
            app_dir() / "looplag.log", note=os.environ.get("HPCA_LOOPLAG", "")
        )


async def start(
    *,
    settings=None,
    profile: str = "default",
    llm: Any = None,
    tools=None,
    slurm=None,
    app_dir: Path | None = None,
    screen=None,
    say=_say,
) -> int:
    """The whole `hpca` run: build, draw, shut down.

    The nesting is the shutdown order. `Screen` restores the terminal on the
    way out of its `with`, including out of a traceback, and the core is
    stopped *after* that — so the wait message lands on a terminal the user can
    read rather than on the alternate screen a moment before it disappears.
    """
    from hpca.ui.app import RowUI
    from hpca.ui.client import UIClient
    from hpca.ui.run import drive

    ui_end, core_end = InProcessConnection.pair()
    ui = RowUI()
    core = await Core.start(
        settings=settings,
        profile=profile,
        llm=llm,
        tools=tools,
        slurm=slurm,
        app_dir=app_dir,
        wire=core_end,
        # The one thing the core cannot do about a dead backend: put the
        # screen that fixes it in front of the user (`specs/specs-auto-connect.md`,
        # and `tui/app.py`'s `_ensure_backend_connected` before it).
        on_no_backend=lambda: _open_llm_screen(ui),
    )
    client = UIClient(ui, ui_end)
    probe = _lag_probe(ui)
    try:
        core.run()
        with quiet_terminal():
            return await drive(
                ui,
                client=client,
                conn=ui_end,
                screen=screen,
                # Started and stopped by the loop it measures; the report is
                # written below, where the app dir is known.
                probe=probe,
                # The one settings section the UI side needs: which tier the
                # clipboard uses. Handed over from here because this is the
                # module that reads the settings file — `ui/run.py` owns the
                # terminal and `ui/client.py` owns the wire, and neither of
                # them reads core state any more.
                clipboard=core.clipboard,
            )
    finally:
        # First, and outside the terminal's restoration: a run that ended in a
        # traceback is exactly the one whose lag block is worth having, and
        # writing it costs nothing that the shutdown below needs.
        _write_lag_report(probe)
        with contextlib.suppress(Exception):
            await ui_end.close()
        await core.stop(say=say)
