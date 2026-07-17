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
                "stdout_path, stderr_path, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.pid,
                    self._session_id,
                    record.name,
                    record.cmd,
                    record.state,
                    str(record.stdout_path),
                    str(record.stderr_path),
                    record.started_at,
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

    def get(self, pid: int) -> ProcessRecord:
        return self._records[pid]

    def list(self) -> list[ProcessRecord]:
        return list(reversed(self._records.values()))


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
