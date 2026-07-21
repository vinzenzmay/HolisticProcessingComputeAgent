"""RAG retrieval index.

The RAG scope holds memories too situational for the system prompt — struggle
notes, backend-specific workarounds, one-topic learnings. They are not injected
wholesale; the ones matching the current request are retrieved and fenced into
the turn, so the scope can grow without costing context on every turn.

The markdown profile file stays the source of truth (users edit it by hand);
this index is derived from it and rebuilt whenever the file changes. Search
is BM25 with two pragmatic adjustments Hermes' experience argues for over
embeddings: recent memories and memories learned on the *active* backend
rank higher, because both are more likely to still apply.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from hpca.builtin_memory import BUILTIN_MEMORIES, BUILTIN_PROFILE
from hpca.profiles import Memory, MemoryScope, Profile

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_memories (
    profile TEXT NOT NULL,
    scope TEXT NOT NULL,
    kind TEXT,
    backend TEXT,
    created TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_memories ON profile_memories(profile, scope);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS profile_memories_fts USING fts5(
    text, content='profile_memories', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS profile_memories_ai
AFTER INSERT ON profile_memories BEGIN
    INSERT INTO profile_memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS profile_memories_ad
AFTER DELETE ON profile_memories BEGIN
    INSERT INTO profile_memories_fts(profile_memories_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
"""

# Recency and backend bonuses are subtracted from the BM25 score (lower is
# better in sqlite's bm25()). Deliberately small: they break ties between
# comparable matches rather than overriding relevance.
RECENT_DAYS = 60
RECENCY_BONUS = 0.5
BACKEND_BONUS = 0.5


@dataclass
class Retrieved:
    text: str
    kind: str
    backend: str
    created: str
    score: float


def _fts_query(query: str) -> str:
    terms = [term.replace('"', "") for term in query.split() if len(term) > 2]
    return " OR ".join(f'"{term}"' for term in terms)


def _age_days(created: str) -> int | None:
    try:
        return (date.today() - date.fromisoformat(created)).days
    except (ValueError, TypeError):
        return None


class MemoryIndex:
    """Search index over tier-3 memories, derived from the profile files."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(SCHEMA)
        try:
            conn.executescript(FTS_SCHEMA)
            self.available = True
        except sqlite3.OperationalError:
            self.available = False  # sqlite without FTS5
        conn.commit()
        self._seed_builtins()

    def _insert(self, profile: str, memory: Memory) -> None:
        self._conn.execute(
            "INSERT INTO profile_memories (profile, scope, kind, backend, "
            "created, text) VALUES (?, ?, ?, ?, ?, ?)",
            (profile, memory.scope.value, memory.kind, memory.backend,
             memory.created, memory.text),
        )

    def _seed_builtins(self) -> None:
        """Index HPCA's shipped memories under the reserved profile so any
        agent can recall them (see :mod:`hpca.builtin_memory`)."""
        self._conn.execute(
            "DELETE FROM profile_memories WHERE profile = ?", (BUILTIN_PROFILE,)
        )
        for memory in BUILTIN_MEMORIES:
            self._insert(BUILTIN_PROFILE, memory)
        self._conn.commit()

    def reindex(self, profile: Profile) -> int:
        """Rebuild one profile's RAG index from its parsed memories."""
        self._conn.execute(
            "DELETE FROM profile_memories WHERE profile = ?", (profile.name,)
        )
        memories = [m for m in profile.memories if m.scope is MemoryScope.RAG]
        for memory in memories:
            self._insert(profile.name, memory)
        self._conn.commit()
        return len(memories)

    def search(
        self,
        query: str,
        *,
        profile: str,
        active_backend: str = "",
        limit: int = 3,
    ) -> list[Retrieved]:
        if not self.available:
            return []
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        try:
            rows = self._conn.execute(
                "SELECT m.text, m.kind, m.backend, m.created, "
                "  bm25(profile_memories_fts) AS rank "
                "FROM profile_memories_fts "
                "JOIN profile_memories m ON m.rowid = profile_memories_fts.rowid "
                "WHERE profile_memories_fts MATCH ? AND m.profile IN (?, ?) "
                "ORDER BY rank LIMIT 50",
                (fts_query, profile, BUILTIN_PROFILE),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        scored = []
        for row in rows:
            score = row["rank"]
            age = _age_days(row["created"] or "")
            if age is not None and age <= RECENT_DAYS:
                score -= RECENCY_BONUS
            if active_backend and row["backend"] == active_backend:
                score -= BACKEND_BONUS
            scored.append(
                Retrieved(
                    text=row["text"],
                    kind=row["kind"] or "",
                    backend=row["backend"] or "",
                    created=row["created"] or "",
                    score=score,
                )
            )
        scored.sort(key=lambda hit: hit.score)
        return scored[:limit]

    def forget_profile(self, name: str) -> None:
        self._conn.execute("DELETE FROM profile_memories WHERE profile = ?", (name,))
        self._conn.commit()


def demote(profile: Profile, memories: list[Memory]) -> int:
    """Move memories to RAG: they stop costing context on every turn but stay
    retrievable. This is what a full system-prompt scope offers instead of
    deletion."""
    moved = 0
    for memory in memories:
        if memory.scope is not MemoryScope.RAG:
            memory.scope = MemoryScope.RAG
            moved += 1
    return moved
