"""Slurm interface (§5.4): deterministic parsing, injectable execution.

Verified against the target site: Slurm 25.05.3, cluster "cubi",
``sacct --parsable2`` emitting an allocation row plus ``.batch``/``.extern``
step rows; ``MaxRSS`` lives on the steps, elapsed times carry days as
``D-HH:MM:SS``. Submission works from compute nodes there, but the optional
``submit_host`` SSH hop (§2) is supported for sites that restrict it.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

SACCT_FIELDS = "JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,Timelimit"

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}

Run = Callable[[list[str]], Awaitable[tuple[int, str, str]]]


class SlurmError(Exception):
    """A Slurm command failed; message carries the command's stderr."""


def parse_duration(text: str) -> int | None:
    """``[D-]HH:MM:SS`` or ``MM:SS`` → seconds."""
    text = text.strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_part, text = text.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_mem(text: str) -> int | None:
    """sacct memory strings (``312400K``, ``25G``, ``2048``) → bytes."""
    text = text.strip()
    if not text:
        return None
    match = re.fullmatch(r"([\d.]+)([KMGTP]?)[cn]?", text)
    if not match:
        return None
    value = float(match.group(1))
    factor = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4,
              "P": 1024**5}[match.group(2)]
    return int(value * factor)


@dataclass
class JobStatus:
    job_id: str
    state: str  # normalized first token, e.g. CANCELLED
    raw_state: str
    exit_code: int | None = None
    signal: int | None = None
    elapsed_s: int | None = None
    max_rss_bytes: int | None = None
    reqmem: str = ""
    timelimit: str = ""
    steps: list[dict] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def parse_sacct(text: str) -> dict[str, JobStatus]:
    """Parse ``sacct --parsable2 --noheader`` output into per-job statuses."""
    jobs: dict[str, JobStatus] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 7:
            continue
        job_field, raw_state, exit_code, elapsed, maxrss, reqmem, timelimit = fields[:7]
        base_id, _, step = job_field.partition(".")
        if step:  # .batch / .extern / numbered steps: fold into parent
            parent = jobs.get(base_id)
            if parent is None:
                continue
            parent.steps.append({"step": step, "state": raw_state, "maxrss": maxrss})
            rss = parse_mem(maxrss)
            if rss is not None and rss > (parent.max_rss_bytes or 0):
                parent.max_rss_bytes = rss
            continue
        state = raw_state.split()[0] if raw_state else "UNKNOWN"
        code, _, sig = exit_code.partition(":")
        jobs[base_id] = JobStatus(
            job_id=base_id,
            state=state,
            raw_state=raw_state,
            exit_code=int(code) if code.isdigit() else None,
            signal=int(sig) if sig.isdigit() else None,
            elapsed_s=parse_duration(elapsed),
            max_rss_bytes=parse_mem(maxrss),
            reqmem=reqmem,
            timelimit=timelimit,
        )
    return jobs


async def _default_run(argv: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


class SlurmClient:
    def __init__(self, *, submit_host: str | None = None, run: Run | None = None):
        self._submit_host = submit_host
        self._run = run or _default_run

    def _wrap(self, argv: list[str]) -> list[str]:
        if self._submit_host:
            return ["ssh", self._submit_host, "--"] + argv
        return argv

    async def submit(self, script: str, args: list[str]) -> str:
        rc, stdout, stderr = await self._run(
            self._wrap(["sbatch"] + args + [str(script)])
        )
        if rc != 0:
            raise SlurmError(f"sbatch failed: {stderr.strip() or stdout.strip()}")
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            raise SlurmError(f"Could not parse job id from sbatch output: {stdout!r}")
        return match.group(1)

    async def test_only(self, script: str, args: list[str]) -> tuple[bool, str]:
        """``sbatch --test-only`` dry run (§5.2); returns (ok, message)."""
        rc, stdout, stderr = await self._run(
            self._wrap(["sbatch", "--test-only"] + args + [str(script)])
        )
        message = (stderr + stdout).strip()
        return rc == 0, message

    async def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        if not job_ids:
            return {}
        rc, stdout, stderr = await self._run(
            self._wrap(
                [
                    "sacct",
                    "--parsable2",
                    "--noheader",
                    f"--format={SACCT_FIELDS}",
                    "-j",
                    ",".join(job_ids),
                ]
            )
        )
        if rc != 0:
            raise SlurmError(f"sacct failed: {stderr.strip()}")
        return parse_sacct(stdout)

    async def cancel(self, job_id: str) -> None:
        rc, stdout, stderr = await self._run(self._wrap(["scancel", job_id]))
        if rc != 0:
            raise SlurmError(f"scancel failed: {stderr.strip() or stdout.strip()}")
