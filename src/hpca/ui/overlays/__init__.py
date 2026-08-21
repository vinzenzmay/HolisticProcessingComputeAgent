"""The screens that draw over the rows, one module each.

An overlay is an independent ``render``/``handle`` pair: it is handed the width
and height it may use, it answers keys, and returning False from ``handle`` is
how it closes. ``app`` reads whatever it decided out of the object afterwards,
so nothing here reaches back into the rows.
"""

from __future__ import annotations

from hpca.ui.overlays.base import Overlay
from hpca.ui.overlays.config import ConfigOverlay
from hpca.ui.overlays.help import HelpOverlay
from hpca.ui.overlays.llm import LlmOverlay
from hpca.ui.overlays.profiles import ProfilesOverlay
from hpca.ui.overlays.queued import UNQUEUE, QueuedOverlay
from hpca.ui.overlays.rewind import (
    COPY,
    FORK,
    PREVIEW_CHARS,
    PREVIEW_LINES,
    ROLLBACK,
    RewindOverlay,
)

__all__ = [
    "COPY",
    "FORK",
    "PREVIEW_CHARS",
    "PREVIEW_LINES",
    "ROLLBACK",
    "UNQUEUE",
    "ConfigOverlay",
    "HelpOverlay",
    "LlmOverlay",
    "Overlay",
    "ProfilesOverlay",
    "QueuedOverlay",
    "RewindOverlay",
]
