"""Deterministic syntax/dry-run checks (§5.2) — no LLM involved.

Every script is checked with the native mechanism of its language before any
execution. A missing checker binary (no R, no snakemake on this host) is
reported as *skipped*, never as a failure — index/tool coverage is always
partial and must not hard-block the user.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ScriptKind = str  # bash | python | R | snakemake

CHECKERS: dict[str, list[str]] = {
    "bash": ["bash", "-n"],
    "python": [sys.executable, "-m", "py_compile"],
    "R": ["Rscript", "--no-init-file", "-e"],
    "snakemake": ["snakemake", "--dry-run", "--quiet", "-s"],
}


@dataclass
class CheckResult:
    ok: bool
    checker: str
    errors: str = ""
    skipped: bool = False


async def syntax_check(kind: ScriptKind, path: Path) -> CheckResult:
    if kind not in CHECKERS:
        raise ValueError(
            f"Unknown script kind {kind!r}; expected one of {sorted(CHECKERS)}"
        )
    if kind == "bash":
        argv = ["bash", "-n", str(path)]
        label = "bash -n"
    elif kind == "python":
        argv = [sys.executable, "-m", "py_compile", str(path)]
        label = "py_compile"
    elif kind == "R":
        argv = ["Rscript", "--no-init-file", "-e", f'invisible(parse(file="{path}"))']
        label = "Rscript parse()"
    else:  # snakemake
        argv = ["snakemake", "--dry-run", "--quiet", "-s", str(path)]
        label = "snakemake -n"

    if shutil.which(argv[0]) is None:
        return CheckResult(
            ok=True,
            checker=label,
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
        return CheckResult(ok=True, checker=label)
    return CheckResult(ok=False, checker=label, errors=output)
