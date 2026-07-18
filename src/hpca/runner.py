"""The single internal subprocess runner (§5.1).

Every local execution goes through here: stdout/stderr captured to files
(registered later by the calling tool), rows in the ``processes`` table
backing the right TUI column, optional timeout, kill support. There is
deliberately no free-form shell tool; tools pass argv lists.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpca.triage import LogFinding


@dataclass
class ProcessRecord:
    pid: int
    name: str
    cmd: str
    state: str  # running | finished | failed | killed
    stdout_path: Path
    stderr_path: Path
    started_at: str
    exit_code: int | None = None
    exit_info: str | None = None


class ProcessRunner:
    def __init__(
        self, conn: sqlite3.Connection, *, session_id: str, log_dir: Path
    ) -> None:
        self._conn = conn
        self._session_id = session_id
        self._log_dir = log_dir
        self._records: dict[int, ProcessRecord] = {}
        self._monitors: dict[int, asyncio.Task] = {}
        self._kill_requested: set[int] = set()

    async def start(
        self,
        argv: list[str],
        *,
        name: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        background: bool = False,
    ) -> ProcessRecord:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns()
        stdout_path = self._log_dir / f"{stamp}-{name}.out"
        stderr_path = self._log_dir / f"{stamp}-{name}.err"
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
        except Exception:
            stdout_file.close()
            stderr_file.close()
            raise
        record = ProcessRecord(
            pid=proc.pid,
            name=name,
            cmd=shlex.join(argv),
            state="running",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self._conn.execute(
                "INSERT INTO processes (pid, session_id, name, cmd, state, "
                "stdout_path, stderr_path, started_at, background) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.pid,
                    self._session_id,
                    record.name,
                    record.cmd,
                    record.state,
                    str(record.stdout_path),
                    str(record.stderr_path),
                    record.started_at,
                    int(background),
                ),
            )
            self._conn.commit()
        except Exception:
            # never leave an untracked process running
            proc.kill()
            await proc.wait()
            stdout_file.close()
            stderr_file.close()
            raise
        self._records[proc.pid] = record
        self._monitors[proc.pid] = asyncio.ensure_future(
            self._monitor(proc, record, stdout_file, stderr_file, timeout_s)
        )
        return record

    async def _monitor(self, proc, record, stdout_file, stderr_file, timeout_s):
        try:
            if timeout_s is not None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout_s)
                except asyncio.TimeoutError:
                    record.exit_info = f"killed after timeout ({timeout_s}s)"
                    self._kill_requested.add(record.pid)
                    proc.kill()
                    await proc.wait()
            else:
                await proc.wait()
        finally:
            stdout_file.close()
            stderr_file.close()
        record.exit_code = proc.returncode
        if record.pid in self._kill_requested:
            record.state = "killed"
        elif proc.returncode == 0:
            record.state = "finished"
        else:
            record.state = "failed"
        self._conn.execute(
            "UPDATE processes SET state = ?, exit_code = ?, exit_info = ? "
            "WHERE pid = ? AND session_id = ?",
            (
                record.state,
                record.exit_code,
                record.exit_info,
                record.pid,
                self._session_id,
            ),
        )
        self._conn.commit()

    async def wait(self, pid: int) -> ProcessRecord:
        monitor = self._monitors.get(pid)
        if monitor is None:
            raise KeyError(f"No tracked process with pid {pid}")
        await monitor
        return self._records[pid]

    async def kill(self, pid: int) -> None:
        record = self._records.get(pid)
        if record is None:
            raise KeyError(f"No tracked process with pid {pid}")
        if record.state == "running":
            self._kill_requested.add(pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def running_named(self, name: str) -> int | None:
        """Live pid of a still-running process with this name, or None.

        Reads the table, not ``_records``: the TUI builds a fresh runner per
        turn, and the duplicate this guards against is precisely a second
        start in a *later* turn. Liveness is confirmed against the OS so a row
        left at 'running' by a crashed app cannot block the name forever.
        """
        rows = self._conn.execute(
            "SELECT pid FROM processes WHERE session_id = ? AND name = ? "
            "AND state = 'running'",
            (self._session_id, name),
        ).fetchall()
        for row in rows:
            try:
                os.kill(row["pid"], 0)  # signal 0: liveness only
            except ProcessLookupError:
                continue
            except PermissionError:
                pass  # exists, just not ours to signal
            return row["pid"]
        return None

    def get(self, pid: int) -> ProcessRecord:
        return self._records[pid]

    def list(self) -> list[ProcessRecord]:
        return list(reversed(self._records.values()))


TERMINAL_STATES = {"finished", "failed", "killed"}
EVENT_TAIL_LINES = 20
EVENT_TAIL_CHARS = 1500


@dataclass
class ProcessChange:
    """One subprocess reaching a terminal state, for delivery to the agent."""

    pid: int
    session_id: str
    name: str
    state: str
    exit_code: int | None
    stdout_path: Path
    stderr_path: Path
    exit_info: str | None = None


def _tail(path: Path) -> str:
    """Last few lines of a log, bounded — this goes straight into a prompt."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    body = "\n".join(lines[-EVENT_TAIL_LINES:]).strip()
    return body[-EVENT_TAIL_CHARS:]


def analyse_process_failure(change: ProcessChange) -> LogFinding:
    """Best deterministic reading of why a process failed (§5.5 tiers 1–2).

    Both streams are examined because tools disagree about where errors go,
    and the more conclusive finding wins.
    """
    from hpca.triage import LogFinding as _LF
    from hpca.triage import analyse_log, load_signatures

    signatures = load_signatures()
    findings = [
        analyse_log(path, signatures)
        for path in (change.stderr_path, change.stdout_path)
    ]
    findings.sort(key=lambda f: f.tier)  # tier 1 beats tier 2 beats nothing
    return findings[0] if findings else _LF(tier=3)


def format_process_event(
    change: ProcessChange, finding: "LogFinding | None" = None
) -> str:
    """The message the agent receives when a background process ends.

    Failures mirror the wording run_bash already uses for a failed script, so
    the model meets a shape the rest of the session has taught it: state the
    failure, then say to fix and retry. The diagnosis is inline so reacting
    does not cost a read_file round-trip first.

    What goes in depends on how much the deterministic tiers established. A
    matched signature names the failure *class* and carries a hint, which is
    worth more to a small model than raw log text. Keyword candidates are
    offered as possibilities, explicitly unconfirmed. Only when both come up
    empty does this fall back to the blunt tail.
    """
    if change.state == "finished":
        head = f"[process finished] {change.name} (pid {change.pid}) exited 0."
    elif change.state == "killed":
        head = (
            f"[process killed] {change.name} (pid {change.pid}) was killed"
            f"{' — ' + change.exit_info if change.exit_info else ''}."
        )
    else:
        head = (
            f"[process failed] {change.name} (pid {change.pid}) exited "
            f"{change.exit_code}. Read the error, fix the script, and start it "
            "again."
        )
    parts = [head]

    if change.state == "finished":
        body = _tail(change.stdout_path) or _tail(change.stderr_path)
        if body:
            parts.append(f"stdout tail:\n{body}")
        return "\n\n".join(parts)

    if finding is None:
        finding = analyse_process_failure(change)
    if finding.tier == 1:
        parts.append(f"cause: {finding.title}")
        # Stated separately because the excerpt is padded for job logs, where
        # what follows an error matters; a CLI tool errors on its last line,
        # so the key line would otherwise sit at the bottom of a usage dump.
        if finding.matched_line:
            parts.append(f"cause line: {finding.matched_line}")
        if finding.hint:
            parts.append(f"hint: {finding.hint}")
        parts.append(f"log:\n{finding.excerpt}")
    elif finding.tier == 2:
        lines = "\n".join(
            f"  line {c.line_no}: {c.line}" for c in finding.candidates
        )
        parts.append(
            "No known error signature matched. Most likely error lines "
            f"(unconfirmed):\n{lines}"
        )
        parts.append(f"around the first of them:\n{finding.excerpt}")
    else:
        body = _tail(change.stderr_path) or _tail(change.stdout_path)
        if body:
            parts.append(f"stderr tail:\n{body}")
    return "\n\n".join(parts)


def poll_processes(conn: sqlite3.Connection) -> list[ProcessChange]:
    """Subprocesses that ended since the last poll, store-wide.

    Reads the table rather than any one ``ProcessRunner``: the TUI builds a
    fresh runner per turn, so no single instance knows about processes started
    by an earlier one. The monitor keeps the DB current regardless of which
    runner owns the process. Rows are flagged as they are returned, so each
    completion is delivered exactly once even across restarts.

    Only backgrounded scripts qualify. run_script and run_bash block until the
    process ends and return its output as the tool result, so an event for
    those would repeat what the agent has already read.
    """
    placeholders = ", ".join("?" for _ in TERMINAL_STATES)
    rows = conn.execute(
        f"SELECT * FROM processes WHERE notified = 0 AND background = 1 "
        f"AND state IN ({placeholders})",
        tuple(TERMINAL_STATES),
    ).fetchall()
    if not rows:
        return []
    conn.executemany(
        "UPDATE processes SET notified = 1 WHERE pid = ? AND session_id = ?",
        [(row["pid"], row["session_id"]) for row in rows],
    )
    conn.commit()
    return [
        ProcessChange(
            pid=row["pid"],
            session_id=row["session_id"],
            name=row["name"],
            state=row["state"],
            exit_code=row["exit_code"],
            stdout_path=Path(row["stdout_path"]),
            stderr_path=Path(row["stderr_path"]),
            exit_info=row["exit_info"],
        )
        for row in rows
    ]


def running_session_ids(conn: sqlite3.Connection) -> set[str]:
    """Sessions with a still-running sub-process, store-wide.

    The monitor keeps the DB ``state`` current even after the UI leaves the
    session, so this reads the table rather than any one live runner. A row
    left at ``running`` by a crashed previous run is verified against the OS,
    so a stale pid never blocks forever.
    """
    rows = conn.execute(
        "SELECT DISTINCT session_id, pid FROM processes WHERE state = 'running'"
    ).fetchall()
    alive: set[str] = set()
    for row in rows:
        pid = row["pid"]
        try:
            os.kill(pid, 0)  # signal 0: liveness check, does not touch the process
        except ProcessLookupError:
            continue
        except PermissionError:
            pass  # exists but ours to not signal; count it as alive
        alive.add(row["session_id"])
    return alive
