"""The context meter above the chat: how full the model's window is.

HPCA is meant to run on whatever a site can host locally, and a 32k model
fills up fast — a few tool results and a long log excerpt will do it. When it
does, quality degrades in ways that look like the model getting stupid rather
than the model running out of room, so the fill level has to be visible
*before* that happens, not explained afterwards.

The number is the backend's own ``prompt_tokens`` from the last decision, not
an estimate: tokenizers differ per model and a chars/4 guess is wrong by
enough to matter at the top of a small window. Before the first reply of a
session there is nothing measured yet, so the bar says so rather than
implying a precise zero.
"""

from __future__ import annotations

from textual.content import Content
from textual.widgets import Static

BAR_CELLS = 28
FULL, EMPTY = "█", "─"

# Below `warn` the bar is unremarkable; past it compaction is approaching (the
# graph folds history at 70%); past `danger` the next tool result may not fit.
WARN_FRACTION = 0.70
DANGER_FRACTION = 0.90


def render_bar(
    used: int, window: int | None, *, cells: int = BAR_CELLS, estimated: bool = False
) -> str:
    # "~" marks a figure derived from character counts rather than measured by
    # the backend, so a number that is only roughly right never looks exact.
    mark = "~" if estimated else ""
    if not window:
        return f"context: {mark}{used:,} tokens used · window unknown"
    fraction = min(1.0, used / window)
    filled = round(fraction * cells)
    bar = FULL * filled + EMPTY * (cells - filled)
    return (
        f"context [{bar}] {mark}{used:,} / {window:,} ({fraction * 100:.0f}%)"
    )


def severity(used: int, window: int | None) -> str:
    if not window:
        return "unknown"
    fraction = used / window
    if fraction >= DANGER_FRACTION:
        return "danger"
    if fraction >= WARN_FRACTION:
        return "warn"
    return "ok"


class ModelLine(Static):
    """The model in use for the session on screen, shown at the very top of
    the chat column so it reads as a property of *this* session rather than a
    global app setting (it sits directly above the context meter).

    Session-specific: the app repaints it on session switch and whenever the
    active session's backend changes. Blank/hidden when no session is open,
    matching the chat input and mode line.
    """

    DEFAULT_CSS = """
    ModelLine {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $boost;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._text = ""

    @property
    def text(self) -> str:
        """The line as displayed. Textual keeps rendered content private, so
        the widget reports its own state rather than tests reading internals."""
        return self._text

    def set_model(self, model: str) -> None:
        self._text = f"model: {model}"
        self.update(Content(self._text))


class ContextBar(Static):
    """One line at the top of the chat column, updated after every model call."""

    DEFAULT_CSS = """
    ContextBar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $boost;
    }
    ContextBar.context-warn { color: $warning; }
    ContextBar.context-danger { color: $error; text-style: bold; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._used = 0
        self._window: int | None = None
        self._measured = False
        self._estimated = False
        self._text = ""

    @property
    def text(self) -> str:
        """The line as displayed. Textual keeps rendered content private, so
        the widget reports its own state rather than tests reading internals."""
        return self._text

    def set_window(self, window: int | None) -> None:
        self._window = window
        self._refresh()

    def set_used(self, used: int) -> None:
        self._used = used
        self._measured = True
        self._estimated = False
        self._refresh()

    def set_estimate(self, used: int) -> None:
        """A character-derived figure for a session with no reply yet.

        Reopening a long session should show it is nearly full *before* the
        next message is sent, not after the reply that overflows it. A
        measured number always supersedes this.
        """
        self._used = used
        self._measured = True
        self._estimated = True
        self._refresh()

    def reset(self) -> None:
        """A different session's context is a different number; showing the
        previous one until the next reply would be a lie."""
        self._used = 0
        self._measured = False
        self._estimated = False
        self._refresh()

    def _refresh(self) -> None:
        if not self._measured:
            window = f"{self._window:,}" if self._window else "unknown"
            self._text = f"context: window {window} · no reply yet"
            level = "ok"
        else:
            self._text = render_bar(
                self._used, self._window, estimated=self._estimated
            )
            level = severity(self._used, self._window)
        self.update(Content(self._text))
        self.set_class(level == "warn", "context-warn")
        self.set_class(level == "danger", "context-danger")
