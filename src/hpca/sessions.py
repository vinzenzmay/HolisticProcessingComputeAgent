"""Session store (§5.4): rows in the sessions table, one per chat thread.

The left TUI column is a view over these rows; ``checkpoint_ref`` is the
LangGraph thread id (currently identical to ``session_id``).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Session:
    session_id: str
    profile: str
    title: str
    created_at: str
    checkpoint_ref: str
    # Interaction mode (§3.5): manual | auto | full-auto | plan;
    # "" = configured default.
    mode: str = ""
    # The LLM this session talks to, as the JSON of an LLMBackend (model,
    # base_url, api_key, …). Chosen when the session is created; "" means fall
    # back to the app's bootstrap client. Stored on the session so the choice
    # survives even if that backend is later removed from the catalog.
    backend: str = ""
    # Thinking effort (hpca.thinking): off | low | medium | xhigh;
    # "" = configured default. Per session for the same reason as ``mode``,
    # plus one of its own: a level change re-writes the prompt's first tokens,
    # so it invalidates the session's prefix KV cache and is not something to
    # churn.
    thinking: str = ""
    # When something last happened here, ISO-8601 UTC — see `touch`.
    last_active: str = ""


class SessionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(
        self,
        *,
        profile: str,
        title: str = "untitled",
        mode: str = "",
        backend: str = "",
        thinking: str = "",
    ) -> Session:
        now = datetime.now(timezone.utc).isoformat()
        session = Session(
            session_id=str(uuid.uuid4()),
            profile=profile,
            title=title,
            created_at=now,
            checkpoint_ref="",
            mode=mode,
            backend=backend,
            thinking=thinking,
            # Being made is the first thing that happens in a conversation, so
            # a session with no turns in it yet reads as new rather than as
            # never having happened.
            last_active=now,
        )
        session.checkpoint_ref = session.session_id
        self._conn.execute(
            "INSERT INTO sessions (session_id, profile, title, created_at, "
            "checkpoint_ref, mode, backend, thinking, last_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.session_id,
                session.profile,
                session.title,
                session.created_at,
                session.checkpoint_ref,
                session.mode,
                session.backend,
                session.thinking,
                session.last_active,
            ),
        )
        self._conn.commit()
        return session

    def touch(self, session_id: str, when: str = "") -> None:
        """Something happened in this conversation, just now.

        Called where a turn is submitted and again where one is recorded, so
        the sidebar says "a minute ago" for a turn still running rather than
        only once it lands. Both, not one: a submit alone would leave a
        five-minute turn looking untouched for its whole length, and a record
        alone would leave it looking untouched until it finished.

        ``when`` is for a caller that already has the instant; otherwise now.
        """
        self._conn.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (when or datetime.now(timezone.utc).isoformat(), session_id),
        )
        self._conn.commit()

    def set_mode(self, session_id: str, mode: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET mode = ? WHERE session_id = ?", (mode, session_id)
        )
        self._conn.commit()

    def set_thinking(self, session_id: str, thinking: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET thinking = ? WHERE session_id = ?",
            (thinking, session_id),
        )
        self._conn.commit()

    def set_backend(self, session_id: str, backend: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET backend = ? WHERE session_id = ?",
            (backend, session_id),
        )
        self._conn.commit()

    def list(self, *, profile: str) -> list[Session]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE profile = ? "
            "ORDER BY created_at DESC, rowid DESC",
            (profile,),
        ).fetchall()
        return [self._to_session(row) for row in rows]

    def list_all(self) -> list[Session]:
        """Every session, newest first. The sessions column shows them all:
        each session carries its own profile, so filtering by one would hide
        the rest whenever a differently-profiled session is open."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        return [self._to_session(row) for row in rows]

    def set_profile(self, session_id: str, profile: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET profile = ? WHERE session_id = ?",
            (profile, session_id),
        )
        self._conn.commit()

    def get(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._to_session(row) if row else None

    def rename(self, session_id: str, title: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id)
        )
        self._conn.commit()

    def reassign_profile(self, from_profile: str, to_profile: str) -> int:
        """Move every session from one profile to another; returns how many.

        Used when a profile is deleted: its sessions fall back to the default
        rather than pointing at a profile file that no longer exists.
        """
        cursor = self._conn.execute(
            "UPDATE sessions SET profile = ? WHERE profile = ?",
            (to_profile, from_profile),
        )
        self._conn.commit()
        return cursor.rowcount

    def delete(self, session_id: str) -> None:
        """Forget a chat thread.

        Job and process rows stay: they record work that outlives the
        conversation about it — a cluster job runs on whether or not the chat
        it was submitted from still exists. The plain-text log stays too; it
        is the durable record (see hpca.logs). LangGraph's checkpoints are the
        caller's to drop, since only it holds the checkpointer.
        """
        self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()

    @staticmethod
    def _to_session(row: sqlite3.Row) -> Session:
        return Session(
            session_id=row["session_id"],
            profile=row["profile"],
            title=row["title"],
            created_at=row["created_at"],
            checkpoint_ref=row["checkpoint_ref"],
            mode=row["mode"] if "mode" in row.keys() else "",
            backend=row["backend"] if "backend" in row.keys() else "",
            thinking=row["thinking"] if "thinking" in row.keys() else "",
            last_active=(
                row["last_active"] if "last_active" in row.keys() else ""
            ),
        )
