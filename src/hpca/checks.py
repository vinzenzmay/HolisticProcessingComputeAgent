"""Deterministic syntax checks (§5.2) — no LLM involved.

Every script is checked before any execution. A missing checker binary is
reported as *skipped*, never as a failure — index/tool coverage is always
partial and must not hard-block the user.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

CHECKER = ["bash", "-n"]
CHECKER_LABEL = "bash -n"


@dataclass
class CheckResult:
    ok: bool
    checker: str
    errors: str = ""
    skipped: bool = False


async def syntax_check(path: Path) -> CheckResult:
    argv = [*CHECKER, str(path)]

    if shutil.which(argv[0]) is None:
        return CheckResult(
            ok=True,
            checker=CHECKER_LABEL,
            errors=f"checker {argv[0]!r} not found on this host — check skipped",
            skipped=True,
        )

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stderr + b"\n" + stdout).decode(errors="replace").strip()
    if proc.returncode == 0:
        return CheckResult(ok=True, checker=CHECKER_LABEL)
    return CheckResult(ok=False, checker=CHECKER_LABEL, errors=output)
