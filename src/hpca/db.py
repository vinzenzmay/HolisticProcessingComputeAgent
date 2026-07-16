"""The application sqlite database (§5.4): jobs, sessions, path registry, processes.

One database at ``<app_dir>/hpca.db`` in WAL mode. LangGraph checkpoints live
in the same file (their tables are managed by the langgraph sqlite saver);
``sessions.checkpoint_ref`` stores the thread id.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hpca.config import app_dir

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
    checkpoint_ref TEXT
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
    exit_info TEXT
);
"""


def db_path() -> Path:
    return app_dir() / "hpca.db"


def checkpoints_db_path() -> Path:
    """LangGraph checkpoints live in their own file: the checkpointer writes
    through its own aiosqlite connection during graph execution, and sharing
    hpca.db produced writer contention ("database is locked") with tool code
    updating the app tables mid-turn."""
    return app_dir() / "checkpoints.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
