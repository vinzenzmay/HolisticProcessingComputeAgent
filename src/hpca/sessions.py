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
    # Where the row sits in the sidebar, 1 at the top. Assigned on insert and
    # only ever rewritten by ``SessionStore.move``; see there for why it is
    # dense, and why a new session is given the top rather than the bottom.
    position: int = 0


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
            # The top of the sidebar; see below.
            position=1,
        )
        session.checkpoint_ref = session.session_id
        # A new conversation belongs at the top, which is where newest-first
        # always put it and where the front-end that opens it expects to find
        # it. That is the opposite of a new watch box, which lands at the
        # bottom of its column — the two lists are read in opposite
        # directions, so "where a new one appears" is opposite too.
        #
        # Making room by pushing everyone down one, rather than handing out
        # ``MIN(position) - 1``: a decreasing counter walks into 0, which is
        # the "never assigned" sentinel the migration in db.py keys on, and a
        # session that collided with it would be re-seeded to somewhere else
        # entirely on the next start. Rewriting every row costs one statement
        # over the few dozen rows a sidebar holds, and only when a
        # conversation is created, which is a human-paced event.
        self._conn.execute("UPDATE sessions SET position = position + 1")
        self._conn.execute(
            "INSERT INTO sessions (session_id, profile, title, created_at, "
            "checkpoint_ref, mode, backend, thinking, last_active, position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                session.position,
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

    # Sidebar order: the user's arrangement first, newest-first to break ties.
    # The fallback is what a database whose rows all still carry position 0
    # sorts by, so the column reads exactly as it did before `position`
    # existed even if the backfill in db.py never ran — and it is the pair the
    # backfill itself reproduces, so nothing jumps on the release that
    # introduces the column.
    _ORDER = "ORDER BY position, created_at DESC, rowid DESC"

    def list(self, *, profile: str) -> list[Session]:
        rows = self._conn.execute(
            f"SELECT * FROM sessions WHERE profile = ? {self._ORDER}",
            (profile,),
        ).fetchall()
        return [self._to_session(row) for row in rows]

    def list_all(self) -> list[Session]:
        """Every session in sidebar order. The sessions column shows them all:
        each session carries its own profile, so filtering by one would hide
        the rest whenever a differently-profiled session is open.

        Newest first until the user rearranges it with alt+↑/alt+↓, after
        which it is whatever they arranged — see `move`.
        """
        rows = self._conn.execute(f"SELECT * FROM sessions {self._ORDER}").fetchall()
        return [self._to_session(row) for row in rows]

    def counts_by_profile(self) -> dict[str, int]:
        """How many conversations each profile has, for the profiles screen.

        Counted in sqlite rather than by listing and grouping in Python: the
        caller wants one number per profile and this is drawn on a screen, not
        walked. Profiles with no session are simply absent — the caller knows
        the profile names, this only knows the ones that were used.
        """
        rows = self._conn.execute(
            "SELECT profile, COUNT(*) AS n FROM sessions GROUP BY profile"
        ).fetchall()
        return {row["profile"]: row["n"] for row in rows}

    def move(self, session_id: str, delta: int) -> bool:
        """Shift a row one step up (``-1``) or down (``+1``) in the sidebar.

        The same gesture as the watch column's, and deliberately the same
        shape: a swap with the neighbour rather than an absolute slot, because
        alt+↑ pressed twice is how a row walks past two others and there is no
        way to say "third from the top".

        Over the whole sidebar rather than one profile's rows, because the
        whole sidebar is what is drawn (`list_all`): scoping the swap to a
        profile would let a row jump over the differently-profiled rows
        between it and its neighbour, landing somewhere the user did not aim.

        Renumbering the list rather than swapping two numbers, for the reason
        `WatchStore.move` gives: rows that predate the column all share
        position 0, and swapping two zeroes changes nothing at all. Numbering
        from 1 also keeps 0 meaning "never assigned", which the backfill in
        db.py keys on.

        Deleting a session leaves a hole in the numbers and that is all it
        does — the order of what is left is unchanged, and the next move
        closes the gaps anyway.

        Returns whether anything moved: at the top or the bottom there is no
        neighbour to trade with, and that is an ordinary outcome of holding
        the key down, not a failure worth a message.
        """
        rows = self.list_all()
        index = next(
            (i for i, row in enumerate(rows) if row.session_id == session_id),
            None,
        )
        if index is None:
            return False
        target = index + delta
        if not 0 <= target < len(rows):
            return False
        rows[index], rows[target] = rows[target], rows[index]
        self._conn.executemany(
            "UPDATE sessions SET position = ? WHERE session_id = ?",
            [
                (position, row.session_id)
                for position, row in enumerate(rows, 1)
            ],
        )
        self._conn.commit()
        return True

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
            position=row["position"] if "position" in row.keys() else 0,
        )
