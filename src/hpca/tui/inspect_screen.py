"""Inspection modal for processes and jobs (§3.3 `(i)` action)."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from hpca.jobs import JobRow
from hpca.runner import ProcessRecord, script_path_for

TAIL_LINES = 40
# The script comes first in the view, so it must not push the output off the
# top of a long scroll. Agent-written scripts are short; a long one is
# usually generated, and its head is the informative part.
SCRIPT_LINES = 80


def _tail(path: Path, lines: int = TAIL_LINES) -> str:
    try:
        content = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"(could not read {path}: {e})"
    tail = content.splitlines()[-lines:]
    return "\n".join(tail) if tail else "(empty)"


def _script_section(record: ProcessRecord) -> list[str]:
    """The script that produced this output, when the process ran one.

    Output without the script behind it is half the story: the usual question
    after reading an error is what exactly was run, and that otherwise means
    hunting through the chat log for the create_script call.
    """
    path = script_path_for(record.cmd)
    if path is None:
        return []
    try:
        content = Path(path).read_text(errors="replace")
    except OSError as e:
        return ["", f"── script ({path}) ──", f"(could not read: {e})"]
    lines = content.splitlines()
    body = "\n".join(lines[:SCRIPT_LINES]) if lines else "(empty)"
    if len(lines) > SCRIPT_LINES:
        body += f"\n… {len(lines) - SCRIPT_LINES} more lines in {path}"
    return ["", f"── script ({path}) ──", body]


def format_process(record: ProcessRecord) -> str:
    parts = [
        f"pid {record.pid} · {record.state}"
        + (f" (exit {record.exit_code})" if record.exit_code is not None else ""),
        f"cmd: {record.cmd}",
        f"started: {record.started_at}",
    ]
    if record.exit_info:
        parts.append(f"info: {record.exit_info}")
    parts += _script_section(record)
    parts += [
        "",
        f"── stdout tail ({record.stdout_path}) ──",
        _tail(record.stdout_path),
        "",
        f"── stderr tail ({record.stderr_path}) ──",
        _tail(record.stderr_path),
    ]
    return "\n".join(parts)


def format_job(job: JobRow) -> str:
    parts = [
        f"job {job.job_id} · {job.state}",
        f"script: {job.script_key} ({job.kind})",
        f"submitted: {job.submit_time}",
    ]
    if job.last_checked:
        parts.append(f"last checked: {job.last_checked}")
    if job.exit_info:
        parts.append(f"exit: {job.exit_info}")
    parts += [
        "",
        f"── job stdout tail ({job.sbatch_stdout_path}) ──",
        _tail(Path(job.sbatch_stdout_path)),
        "",
        f"── job stderr tail ({job.sbatch_stderr_path}) ──",
        _tail(Path(job.sbatch_stderr_path)),
    ]
    return "\n".join(parts)


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

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def body_text(self) -> str:
        return self._body

    def compose(self) -> ComposeResult:
        with Vertical(id="inspect-dialog"):
            yield Static(self._title, id="inspect-title")
            with VerticalScroll():
                yield Static(Content(self._body), id="inspect-body")

    def action_close(self) -> None:
        self.dismiss(None)
