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


class SessionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, *, profile: str, title: str = "untitled") -> Session:
        session = Session(
            session_id=str(uuid.uuid4()),
            profile=profile,
            title=title,
            created_at=datetime.now(timezone.utc).isoformat(),
            checkpoint_ref="",
        )
        session.checkpoint_ref = session.session_id
        self._conn.execute(
            "INSERT INTO sessions (session_id, profile, title, created_at, "
            "checkpoint_ref) VALUES (?, ?, ?, ?, ?)",
            (
                session.session_id,
                session.profile,
                session.title,
                session.created_at,
                session.checkpoint_ref,
            ),
        )
        self._conn.commit()
        return session

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
        """Forget a chat thread and the path aliases it named.

        Job and process rows stay: they record work that outlives the
        conversation about it — a cluster job runs on whether or not the chat
        it was submitted from still exists. The plain-text log stays too; it
        is the durable record (see hpca.logs). LangGraph's checkpoints are the
        caller's to drop, since only it holds the checkpointer.
        """
        self._conn.execute(
            "DELETE FROM path_registry WHERE session_id = ?", (session_id,)
        )
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
        )
