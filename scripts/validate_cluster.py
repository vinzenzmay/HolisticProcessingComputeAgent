#!/usr/bin/env python3
"""Validate HPCA's Slurm integration against the real cluster.

Run this ON the cluster (compute node is fine — submission verified to work
there). It needs no third-party packages, only Python ≥ 3.11 and this repo
checked out (hpca.slurm is stdlib-only):

    python3 scripts/validate_cluster.py [extra sbatch args ...]

e.g. if your site needs a partition/account:

    python3 scripts/validate_cluster.py --partition=short --account=myacct

It submits two tiny jobs (~30 s of sleep): one runs to completion, one is
cancelled immediately. It prints a PASS/FAIL summary — paste the full output
back into the chat.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hpca.slurm import SlurmClient, SlurmError  # noqa: E402

POLL_S = 5
MAX_WAIT_S = 300

JOB_SCRIPT = """\
#!/bin/bash
echo "hpca-validate: start on $(hostname)"
sleep 30
echo "hpca-validate: done"
"""

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


async def wait_terminal(client: SlurmClient, job_id: str) -> object:
    last_state = None
    for _ in range(MAX_WAIT_S // POLL_S):
        await asyncio.sleep(POLL_S)
        statuses = await client.status([job_id])
        status = statuses.get(job_id)
        if status is None:
            print(f"  … job {job_id}: not in sacct yet (accounting lag)")
            continue
        if status.state != last_state:
            print(f"  … job {job_id}: {status.state}")
            last_state = status.state
        if status.is_terminal:
            return status
    return None


async def main() -> None:
    extra_args = sys.argv[1:]
    client = SlurmClient()
    workdir = Path(tempfile.mkdtemp(prefix="hpca-validate-"))
    script = workdir / "validate.sh"
    script.write_text(JOB_SCRIPT)
    print(f"work dir: {workdir}")

    # 1. sbatch --test-only gate
    out = str(workdir / "job-%j.out")
    err = str(workdir / "job-%j.err")
    sbatch_args = ["-o", out, "-e", err, "--time=00:05:00"] + extra_args
    ok, message = await client.test_only(str(script), sbatch_args)
    record("sbatch --test-only accepts the job", ok, message.splitlines()[0] if message else "")
    if not ok:
        print("Aborting: fix the sbatch args (partition/account?) and rerun.")
        return

    # 2. submit + run to completion
    try:
        job_id = await client.submit(str(script), sbatch_args)
        record("submit returns a job id", True, job_id)
    except SlurmError as e:
        record("submit returns a job id", False, str(e))
        return

    status = await wait_terminal(client, job_id)
    record(
        "job reaches a terminal state",
        status is not None,
        f"{status.state}" if status else f"still not terminal after {MAX_WAIT_S}s",
    )
    if status:
        record("terminal state is COMPLETED", status.state == "COMPLETED",
               status.raw_state)
        record("exit code parsed as 0", status.exit_code == 0, str(status.exit_code))
        record(
            "elapsed parsed", status.elapsed_s is not None and status.elapsed_s >= 25,
            f"{status.elapsed_s}s",
        )
        record(
            "MaxRSS parsed from steps",
            status.max_rss_bytes is not None and status.max_rss_bytes > 0,
            f"{status.max_rss_bytes}",
        )
    stdout_file = Path(out.replace("%j", job_id))
    content = stdout_file.read_text() if stdout_file.exists() else ""
    record(
        "stdout log at the -o path with expected content",
        "hpca-validate: done" in content,
        str(stdout_file),
    )

    # 3. submit + cancel
    try:
        job2 = await client.submit(str(script), sbatch_args)
        print(f"  … submitted job {job2} for cancellation test")
        await client.cancel(job2)
        status2 = await wait_terminal(client, job2)
        record(
            "cancelled job reported as CANCELLED",
            status2 is not None and status2.state == "CANCELLED",
            status2.raw_state if status2 else "no terminal state seen",
        )
    except SlurmError as e:
        record("cancel roundtrip", False, str(e))

    failed = [r for r in results if not r[1]]
    print()
    print(f"=== {len(results) - len(failed)}/{len(results)} checks passed ===")
    if failed:
        print("Please paste this full output back to the assistant.")


if __name__ == "__main__":
    asyncio.run(main())
