"""Job DB layer and polling cycle (§5.4).

``JobStore`` wraps the ``jobs``/``job_logs`` tables; ``poll_active`` is one
poll iteration — query sacct for every non-terminal job, persist updates, and
return the state changes so the TUI can notify. Jobs sacct does not know yet
(accounting lag right after submission) simply stay in SUBMITTED.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from hpca.slurm import JobStatus, SlurmClient, TERMINAL_STATES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRow:
    job_id: str
    kind: str
    session_id: str
    profile: str
    submit_time: str
    state: str
    script_key: str
    sbatch_stdout_path: str
    sbatch_stderr_path: str
    snakemake_log_path: str | None
    last_checked: str | None
    exit_info: str | None


@dataclass
class JobLog:
    job_id: str
    rule_or_step: str | None
    log_path: str
    tool_name: str | None


@dataclass
class StateChange:
    job_id: str
    old_state: str
    new_state: str
    status: JobStatus


class JobStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        job_id: str,
        kind: str,
        session_id: str,
        profile: str,
        script_key: str,
        stdout_path: str,
        stderr_path: str,
        snakemake_log_path: str | None = None,
    ) -> JobRow:
        self._conn.execute(
            "INSERT INTO jobs (job_id, kind, session_id, profile, submit_time, "
            "state, script_key, sbatch_stdout_path, sbatch_stderr_path, "
            "snakemake_log_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                kind,
                session_id,
                profile,
                _now(),
                "SUBMITTED",
                script_key,
                stdout_path,
                stderr_path,
                snakemake_log_path,
            ),
        )
        self._conn.commit()
        return self.get(job_id)

    def get(self, job_id: str) -> JobRow | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return self._to_row(row) if row else None

    def list(self, *, session_id: str) -> list[JobRow]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE session_id = ? "
            "ORDER BY submit_time DESC, rowid DESC",
            (session_id,),
        ).fetchall()
        return [self._to_row(r) for r in rows]

    def active(self, *, session_id: str | None = None) -> list[JobRow]:
        rows = self._conn.execute("SELECT * FROM jobs").fetchall()
        jobs = [self._to_row(r) for r in rows]
        return [
            j
            for j in jobs
            if j.state not in TERMINAL_STATES
            and (session_id is None or j.session_id == session_id)
        ]

    def update_status(self, status: JobStatus) -> None:
        exit_info = None
        if status.is_terminal:
            parts = [status.raw_state]
            if status.exit_code is not None:
                parts.append(f"exit {status.exit_code}")
            if status.signal:
                parts.append(f"signal {status.signal}")
            exit_info = ", ".join(parts)
        self._conn.execute(
            "UPDATE jobs SET state = ?, last_checked = ?, exit_info = ? "
            "WHERE job_id = ?",
            (status.state, _now(), exit_info, status.job_id),
        )
        self._conn.commit()

    def add_log(
        self,
        job_id: str,
        *,
        log_path: str,
        rule_or_step: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO job_logs (job_id, rule_or_step, log_path, tool_name) "
            "VALUES (?, ?, ?, ?)",
            (job_id, rule_or_step, log_path, tool_name),
        )
        self._conn.commit()

    def logs(self, job_id: str) -> list[JobLog]:
        rows = self._conn.execute(
            "SELECT * FROM job_logs WHERE job_id = ?", (job_id,)
        ).fetchall()
        return [
            JobLog(
                job_id=r["job_id"],
                rule_or_step=r["rule_or_step"],
                log_path=r["log_path"],
                tool_name=r["tool_name"],
            )
            for r in rows
        ]

    @staticmethod
    def _to_row(row: sqlite3.Row) -> JobRow:
        return JobRow(**{key: row[key] for key in JobRow.__dataclass_fields__})


async def poll_active(slurm: SlurmClient, store: JobStore) -> list[StateChange]:
    """One poll iteration; returns state changes for TUI notification."""
    active = store.active()
    if not active:
        return []
    statuses = await slurm.status([j.job_id for j in active])
    changes: list[StateChange] = []
    for job in active:
        status = statuses.get(job.job_id)
        if status is None or status.state == job.state:
            continue
        store.update_status(status)
        changes.append(
            StateChange(
                job_id=job.job_id,
                old_state=job.state,
                new_state=status.state,
                status=status,
            )
        )
    return changes
