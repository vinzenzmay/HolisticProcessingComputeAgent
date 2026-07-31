"""The application sqlite database (§5.4): jobs, sessions, path registry, processes.

One database at ``<app_dir>/hpca.db`` in WAL mode. LangGraph checkpoints live
in the same file (their tables are managed by the langgraph sqlite saver);
``sessions.checkpoint_ref`` stores the thread id.
"""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

from hpca.config import app_dir

T = TypeVar("T")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    session_id TEXT,
    profile TEXT,
    submit_time TEXT,
    state TEXT,
    script_key TEXT,
    sbatch_stdout_path TEXT,
    sbatch_stderr_path TEXT,
    snakemake_log_path TEXT,
    last_checked TEXT,
    exit_info TEXT
);
CREATE TABLE IF NOT EXISTS job_logs (
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    rule_or_step TEXT,
    log_path TEXT NOT NULL,
    tool_name TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    title TEXT,
    created_at TEXT,
    checkpoint_ref TEXT,
    -- Interaction mode (§3.5): manual | auto | full-auto | plan. Empty string means
    -- "use the configured default", so changing the default in settings
    -- reaches sessions that were never explicitly switched.
    mode TEXT NOT NULL DEFAULT '',
    -- The LLM this session uses, as an LLMBackend JSON blob; '' falls back to
    -- the app's bootstrap client. Chosen at session creation.
    backend TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS path_registry (
    profile TEXT NOT NULL,
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (profile, session_id, key)
);
CREATE TABLE IF NOT EXISTS symbols (
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    parent TEXT,
    signature TEXT,
    params TEXT,
    source TEXT NOT NULL,
    lineno INTEGER,
    doc TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    turn_no INTEGER NOT NULL,
    role TEXT NOT NULL,      -- user | assistant; tool traffic is not indexed
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn_no);
CREATE TABLE IF NOT EXISTS processes (
    pid INTEGER,
    session_id TEXT,
    name TEXT,
    cmd TEXT,
    state TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    started_at TEXT,
    exit_code INTEGER,
    exit_info TEXT,
    -- 1 once the agent has been told this process reached a terminal state.
    -- Persisted rather than kept in memory so a completion that happens while
    -- the TUI is closed is still delivered on the next start (§5.4).
    notified INTEGER NOT NULL DEFAULT 0,
    -- 1 only for start_background_script. run_bash hands its result back as
    -- the tool result the agent is already reading, so announcing it again
    -- would tell it the same thing twice.
    background INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS watches (
    -- Things the user asked to keep an eye on: a log file, a Slurm job id.
    -- Bound to a profile rather than a session — a watch describes the
    -- machine, not the conversation. See hpca.watches.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',   -- provenance: who registered it
    kind TEXT NOT NULL,                    -- 'log' | 'job'
    target TEXT NOT NULL,                  -- absolute path | Slurm job id
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT,
    state TEXT NOT NULL DEFAULT '',
    -- The rendered halves of the panel box, filled by the poller.
    head TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    -- When the watched thing last moved (log mtime, job state change), not
    -- when a poll noticed: the panel counts up from this.
    changed_at TEXT NOT NULL DEFAULT '',
    checked_at TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_watches_target
    ON watches(profile, kind, target);
CREATE TABLE IF NOT EXISTS command_usage (
    -- How often each slash command has been run, so the autocomplete menu can
    -- list the most-used first.
    name TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after the first release. sqlite has no "ADD COLUMN IF NOT
# EXISTS", and CREATE TABLE IF NOT EXISTS silently leaves an existing table
# alone, so an additive migration is the only way an old database gains them.
ADDED_COLUMNS = [
    ("processes", "notified", "INTEGER NOT NULL DEFAULT 0"),
    ("processes", "background", "INTEGER NOT NULL DEFAULT 0"),
    ("sessions", "mode", "TEXT NOT NULL DEFAULT ''"),
    ("sessions", "backend", "TEXT NOT NULL DEFAULT ''"),
]

# Episodic search index (redesign Phase 2): an external-content FTS5 table
# over messages, kept in sync by triggers. Separate from SCHEMA because FTS5
# is a compile-time sqlite option; a build without it still gets a working
# app, just with `session_search` reporting itself unavailable.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


def db_path() -> Path:
    """Where hpca.db is *kept*. While the app runs it may be opened from a
    node-local copy instead — see hpca.dbcache."""
    return app_dir() / "hpca.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class DbIO:
    """Sqlite work on one dedicated thread, keeping the event loop responsive.

    On a cluster node ``$HOME`` is typically NFS, where every sqlite call is
    network round-trips plus a remote fsync — hundreds of milliseconds each,
    seconds under lock contention. Run on the event loop (as the TUI's 2s
    poll timers did), that blocks keystroke handling and freezes the UI.

    The worker owns its own connection, opened lazily *inside* the thread, so
    it is never touched from the loop thread and the two logical transactions
    of loop-side and worker-side callers can never interleave on one
    connection. Cross-connection visibility is safe because every store
    method commits, and writer collisions wait out sqlite's 5s busy timeout
    instead of failing.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hpca-db"
        )
        self._conn: sqlite3.Connection | None = None
        self._closed = False

    def _connection(self) -> sqlite3.Connection:
        # Worker thread only.
        if self._conn is None:
            self._conn = connect(self._path)
        return self._conn

    @property
    def closed(self) -> bool:
        return self._closed

    async def run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn(conn)`` on the worker thread and await its result."""
        if self._closed:
            raise RuntimeError("DbIO is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: fn(self._connection())
        )

    async def close(self) -> None:
        """Close the worker's connection after queued work has drained."""
        if self._closed:
            return
        self._closed = True

        def _close() -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        await asyncio.get_running_loop().run_in_executor(self._executor, _close)
        self._executor.shutdown(wait=False)


def record_command_use(conn: sqlite3.Connection, name: str) -> None:
    """Bump a slash command's run count (upsert), for the frequency sort."""
    conn.execute(
        "INSERT INTO command_usage (name, count) VALUES (?, 1) "
        "ON CONFLICT(name) DO UPDATE SET count = count + 1",
        (name,),
    )
    conn.commit()


def command_use_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Every command's run count, name -> count."""
    return {
        row["name"]: row["count"]
        for row in conn.execute("SELECT name, count FROM command_usage")
    }


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, column, decl in ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    try:
        conn.executescript(FTS_SCHEMA)
    except sqlite3.OperationalError:
        pass  # sqlite built without FTS5; episodic search degrades gracefully
    conn.commit()
