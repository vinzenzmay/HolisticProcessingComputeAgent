"""Episodic memory (redesign Phase 2): every session, searchable.

User and assistant messages — never tool traffic or thinking, which would
drown BM25 in tool vocabulary — are persisted per turn and indexed with FTS5.
The ``session_search`` tool recalls them at zero LLM cost. A search hit is
reported Hermes-style: the *bookends* of the session (its opening request and
its closing answer) plus the matching snippet, so the model can reconstruct
goal → match → resolution without paying for the whole transcript.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

SNIPPET_TOKENS = 12
GOAL_CHARS = 200
RESOLUTION_CHARS = 200
WINDOW_RADIUS = 5
MESSAGE_CHARS = 300  # per message in read/scroll mode


@dataclass
class Hit:
    session_id: str
    profile: str
    title: str
    created_at: str
    snippet: str
    turn_no: int
    goal: str  # first user message of the session
    resolution: str  # last assistant message of the session


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())  # search output stays one line per field
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Words that carry no signal but, under an OR match, pull in every session.
STOP_WORDS = frozenset(
    "the a an and or but for with from into how what when where which who why "
    "this that these those there here can could would should did does do is "
    "are was were be been being it its my our your me you we they them him "
    "her his get got have has had need needs want wants please help about "
    "again also just now then than too very".split()
)


def _fts_query(query: str) -> str:
    """Quoted terms joined with OR.

    Every term is quoted because user text must never be parsed as FTS5
    syntax. They are joined with OR, not the implicit AND: a natural request
    ("how much memory does STAR need") shares only its rare words with the
    session that answered it, and requiring every word means never matching.
    Stop words and very short tokens are dropped, since under OR they would
    match everything; bm25 ranking then puts the sessions sharing the rare
    terms on top. A query with no distinctive term left returns nothing
    rather than falling back to the stop words themselves — matching on
    "how" would rank arbitrary sessions above nothing at all, which reads as
    a real recall.
    """
    terms = [
        term
        for term in (word.strip('"').lower() for word in query.split())
        if len(term) > 2 and term not in STOP_WORDS
    ]
    return " OR ".join(f'"{term}"' for term in terms)


class EpisodicStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def fts_available(self) -> bool:
        try:
            self._conn.execute("SELECT rowid FROM messages_fts LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False

    # -------------------------------------------------------------- writing

    def record(
        self, *, session_id: str, profile: str, entries: list[tuple[str, str]]
    ) -> None:
        """Persist one turn's (role, text) pairs in conversation order."""
        if not entries:
            return
        row = self._conn.execute(
            "SELECT COALESCE(MAX(turn_no), 0) AS n FROM messages "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_no = row["n"]
        for role, text in entries:
            turn_no += 1
            self._conn.execute(
                "INSERT INTO messages (session_id, profile, turn_no, role, "
                "content) VALUES (?, ?, ?, ?, ?)",
                (session_id, profile, turn_no, role, text),
            )
        self._conn.commit()

    def forget_session(self, session_id: str) -> None:
        """Deleting a session forgets its transcript here too — these are
        patient-data environments; a deleted conversation must not resurface
        through search."""
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.commit()

    # ------------------------------------------------------------ searching

    def search(
        self,
        query: str,
        *,
        profile: str | None = None,
        exclude_session_id: str = "",
        limit: int = 5,
    ) -> list[Hit]:
        """BM25 top hits, one per session (the best match of each).

        ``profile=None`` searches across profiles (config-gated at the tool).
        """
        if not self.fts_available:
            return []
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT m.session_id, m.profile, m.turn_no, "
            "  s.title, s.created_at, "
            "  snippet(messages_fts, 0, '>>', '<<', '…', ?) AS snip, "
            "  bm25(messages_fts) AS rank "
            "FROM messages_fts "
            "JOIN messages m ON m.rowid = messages_fts.rowid "
            "LEFT JOIN sessions s ON s.session_id = m.session_id "
            "WHERE messages_fts MATCH ? AND m.session_id != ?"
        )
        params: list = [SNIPPET_TOKENS, fts_query, exclude_session_id]
        if profile is not None:
            sql += " AND m.profile = ?"
            params.append(profile)
        sql += " ORDER BY rank LIMIT 100"
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []  # malformed MATCH despite quoting: no hits, not a crash
        hits: list[Hit] = []
        seen: set[str] = set()
        for row in rows:
            if row["session_id"] in seen:
                continue
            seen.add(row["session_id"])
            goal, resolution = self._bookends(row["session_id"])
            hits.append(
                Hit(
                    session_id=row["session_id"],
                    profile=row["profile"],
                    title=row["title"] or "untitled",
                    created_at=(row["created_at"] or "")[:10],
                    snippet=_clip(row["snip"], 200),
                    turn_no=row["turn_no"],
                    goal=goal,
                    resolution=resolution,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _bookends(self, session_id: str) -> tuple[str, str]:
        first = self._conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
            "ORDER BY turn_no LIMIT 1",
            (session_id,),
        ).fetchone()
        last = self._conn.execute(
            "SELECT content FROM messages WHERE session_id = ? "
            "AND role = 'assistant' ORDER BY turn_no DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return (
            _clip(first["content"], GOAL_CHARS) if first else "",
            _clip(last["content"], RESOLUTION_CHARS) if last else "",
        )

    def window(
        self,
        session_id: str,
        *,
        around: int | None = None,
        radius: int = WINDOW_RADIUS,
        profile: str | None = None,
    ) -> list[sqlite3.Row]:
        """Messages around a turn (or the session tail when no anchor)."""
        sql = "SELECT turn_no, role, content FROM messages WHERE session_id = ?"
        params: list = [session_id]
        if profile is not None:
            sql += " AND profile = ?"
            params.append(profile)
        if around is not None:
            sql += " AND turn_no BETWEEN ? AND ? ORDER BY turn_no"
            params += [around - radius, around + radius]
        else:
            sql += " ORDER BY turn_no DESC LIMIT ?"
            params.append(2 * radius + 1)
        rows = self._conn.execute(sql, params).fetchall()
        return list(rows) if around is not None else list(reversed(rows))
