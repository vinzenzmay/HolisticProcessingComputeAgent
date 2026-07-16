"""Process inspection modal: status plus log tails (§3.3 `(i)` action)."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from hpca.runner import ProcessRecord

TAIL_LINES = 40


def _tail(path: Path, lines: int = TAIL_LINES) -> str:
    try:
        content = path.read_text(errors="replace")
    except OSError as e:
        return f"(could not read {path}: {e})"
    tail = content.splitlines()[-lines:]
    return "\n".join(tail) if tail else "(empty)"


class InspectScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "close", priority=True)]

    DEFAULT_CSS = """
    InspectScreen {
        align: center middle;
    }
    #inspect-dialog {
        width: 90%;
        height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #inspect-title {
        height: 1;
        text-style: bold;
    }
    """

    def __init__(self, record: ProcessRecord) -> None:
        super().__init__()
        self._record = record

    def body_text(self) -> str:
        r = self._record
        parts = [
            f"pid {r.pid} · {r.state}"
            + (f" (exit {r.exit_code})" if r.exit_code is not None else ""),
            f"cmd: {r.cmd}",
            f"started: {r.started_at}",
        ]
        if r.exit_info:
            parts.append(f"info: {r.exit_info}")
        parts += [
            "",
            f"── stdout tail ({r.stdout_path}) ──",
            _tail(r.stdout_path),
            "",
            f"── stderr tail ({r.stderr_path}) ──",
            _tail(r.stderr_path),
        ]
        return "\n".join(parts)

    def compose(self) -> ComposeResult:
        with Vertical(id="inspect-dialog"):
            yield Static(f"Process: {self._record.name}", id="inspect-title")
            with VerticalScroll():
                yield Static(Content(self.body_text()), id="inspect-body")

    def action_close(self) -> None:
        self.dismiss(None)
