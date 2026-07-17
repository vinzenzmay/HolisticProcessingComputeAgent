"""Plain-text session transcripts (§7 ``logging`` settings).

One file per session, written where hpca was started (``./hpca-logs`` by
default) so transcripts land next to the analysis they belong to. The format
is deliberately dumb — a timestamped header line, the text, a blank line — so
a later reader can grep it and a later script can split it on the headers.

Sub-agent traffic (the model calls tools make on their own, §4.2) goes through
``LoggedLLM``, a proxy around the client the tool context hands out.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from hpca.config import Settings
from hpca.llm import Message
from hpca.sessions import Session

DEFAULT_LOG_DIR = "hpca-logs"


def log_dir(settings: Settings) -> Path:
    return Path(settings.logging.dir or DEFAULT_LOG_DIR)


def log_path(directory: Path, session: Session) -> Path:
    """Stable per-session filename: when it started, and which session it is.

    Sessions are stored in UTC but the entries inside are stamped in local
    time, so the name is localised too — a file called 10-40 whose first line
    reads 12:40 is a trap for whoever reads these later.
    """
    try:
        started = datetime.fromisoformat(session.created_at).astimezone()
        stamp = started.strftime("%Y-%m-%dT%H-%M-%S")
    except ValueError:
        stamp = session.created_at[:19].replace(":", "-")
    return directory / f"{stamp}--{session.session_id[:8]}.log"


def open_log(settings: Settings, session: Session) -> "SessionLog | None":
    """The session's log, or None when logging is switched off."""
    if not settings.logging.enabled:
        return None
    return SessionLog(log_path(log_dir(settings), session))


class SessionLog:
    def __init__(self, path: Path, *, now: Callable[[], datetime] | None = None) -> None:
        self.path = path
        self._now = now or datetime.now

    def write(self, kind: str, text: str) -> None:
        """Append one timestamped block. Never raises: a failing log must not
        take down the session it is only observing."""
        stamp = self._now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] {kind}\n{text.rstrip()}\n\n")
        except OSError:
            pass


def render_messages(messages: list[Message]) -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )


class LoggedLLM:
    """Records a tool's own model calls — its sub-agents — in the session log.

    Wraps only the client handed to tools via the tool context, so the
    orchestrator's own decisions (already logged as thinking) are not doubled.
    """

    def __init__(
        self,
        llm: Any,
        log: SessionLog,
        *,
        label: Callable[[], str] | None = None,
    ) -> None:
        self._llm = llm
        self._log = log
        self._label = label or (lambda: "subagent")

    async def chat(self, messages: list[Message], **kwargs: Any) -> Any:
        label = self._label()
        self._log.write(f"{label} query", render_messages(messages))
        response = await self._llm.chat(messages, **kwargs)
        self._log.write(f"{label} reply", response.content or "")
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)
