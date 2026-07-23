"""Tests for hpca.slurm: sacct parsing and the Slurm client (§5.4, §5.5).

The canned outputs mirror real output from the target site (Slurm 25.05.3,
cluster "cubi", sacct --parsable2), including allocation + .batch/.extern
step rows and day-carrying elapsed times.
"""

import pytest

from hpca.slurm import (
    JobStatus,
    SlurmClient,
    SlurmError,
    parse_duration,
    parse_mem,
    parse_sacct,
)

# Real sample from the cluster (job still running)
SACCT_RUNNING = """\
27744534|RUNNING|0:0|1-04:42:02||25G|5-00:00:00
27744534.batch|RUNNING|0:0|1-04:42:02|||
27744534.extern|RUNNING|0:0|1-04:42:02|||
"""

SACCT_COMPLETED = """\
100|COMPLETED|0:0|00:10:30||4G|02:00:00
100.batch|COMPLETED|0:0|00:10:30|312400K||
100.extern|COMPLETED|0:0|00:10:31|1200K||
"""

SACCT_FAILED = """\
101|FAILED|1:0|00:00:12||4G|02:00:00
101.batch|FAILED|1:0|00:00:12|9800K||
"""

SACCT_OOM = """\
102|OUT_OF_MEMORY|0:125|00:05:00||2G|01:00:00
102.batch|OUT_OF_MEMORY|0:125|00:05:00|2097000K||
"""

SACCT_CANCELLED = """\
103|CANCELLED by 5810|0:0|00:01:00||1G|01:00:00
"""


class TestParseDuration:
    def test_days_format(self):
        assert parse_duration("1-04:42:02") == 1 * 86400 + 4 * 3600 + 42 * 60 + 2

    def test_plain_hms(self):
        assert parse_duration("00:10:30") == 630

    def test_minutes_seconds(self):
        assert parse_duration("42:02") == 42 * 60 + 2

    def test_empty_is_none(self):
        assert parse_duration("") is None


class TestParseMem:
    def test_kilobytes(self):
        assert parse_mem("312400K") == 312400 * 1024

    def test_gigabytes(self):
        assert parse_mem("25G") == 25 * 1024**3

    def test_plain_bytes(self):
        assert parse_mem("2048") == 2048

    def test_empty_is_none(self):
        assert parse_mem("") is None


class TestParseSacct:
    def test_running_job_from_real_sample(self):
        jobs = parse_sacct(SACCT_RUNNING)
        status = jobs["27744534"]
        assert status.state == "RUNNING"
        assert not status.is_terminal
        assert status.exit_code == 0
        assert status.elapsed_s == 1 * 86400 + 4 * 3600 + 42 * 60 + 2
        assert status.reqmem == "25G"
        assert status.timelimit == "5-00:00:00"
        assert status.max_rss_bytes is None  # steps carry no RSS while running

    def test_completed_job_maxrss_from_steps(self):
        status = parse_sacct(SACCT_COMPLETED)["100"]
        assert status.state == "COMPLETED"
        assert status.is_terminal
        # MaxRSS lives on the steps, not the allocation row: take the max
        assert status.max_rss_bytes == 312400 * 1024

    def test_failed_job_exit_code(self):
        status = parse_sacct(SACCT_FAILED)["101"]
        assert status.state == "FAILED"
        assert status.exit_code == 1
        assert status.is_terminal

    def test_oom_signal(self):
        status = parse_sacct(SACCT_OOM)["102"]
        assert status.state == "OUT_OF_MEMORY"
        assert status.signal == 125
        assert status.is_terminal

    def test_cancelled_by_user_normalized(self):
        status = parse_sacct(SACCT_CANCELLED)["103"]
        assert status.state == "CANCELLED"
        assert status.raw_state == "CANCELLED by 5810"
        assert status.is_terminal

    def test_multiple_jobs(self):
        jobs = parse_sacct(SACCT_RUNNING + SACCT_FAILED)
        assert set(jobs) == {"27744534", "101"}

    def test_empty_output(self):
        assert parse_sacct("") == {}


class FakeRun:
    """Records argv calls, returns scripted (rc, stdout, stderr) tuples."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    async def __call__(self, argv):
        self.calls.append(argv)
        return self.responses.pop(0)


class TestSlurmClient:
    async def test_submit_parses_job_id(self):
        run = FakeRun([(0, "Submitted batch job 27744534\n", "")])
        client = SlurmClient(run=run)
        job_id = await client.submit("/tmp/job.sh", ["-o", "/tmp/out.log"])
        assert job_id == "27744534"
        assert run.calls[0][0] == "sbatch"
        assert "/tmp/job.sh" in run.calls[0]

    async def test_submit_on_cluster_suffix(self):
        # federated clusters print "... on cluster cubi"
        run = FakeRun([(0, "Submitted batch job 99 on cluster cubi\n", "")])
        client = SlurmClient(run=run)
        assert await client.submit("/tmp/job.sh", []) == "99"

    async def test_submit_failure_raises_with_stderr(self):
        run = FakeRun([(1, "", "sbatch: error: Invalid partition\n")])
        client = SlurmClient(run=run)
        with pytest.raises(SlurmError, match="Invalid partition"):
            await client.submit("/tmp/job.sh", [])

    async def test_test_only_ok(self):
        run = FakeRun(
            [(0, "", "sbatch: Job 1 to start at 2026-07-16T15:00:00 ...\n")]
        )
        client = SlurmClient(run=run)
        ok, message = await client.test_only("/tmp/job.sh", [])
        assert ok
        assert "--test-only" in run.calls[0]

    async def test_test_only_failure_returns_message(self):
        run = FakeRun([(1, "", "sbatch: error: Invalid qos\n")])
        client = SlurmClient(run=run)
        ok, message = await client.test_only("/tmp/job.sh", [])
        assert not ok
        assert "Invalid qos" in message

    async def test_status_queries_sacct(self):
        run = FakeRun([(0, SACCT_RUNNING, "")])
        client = SlurmClient(run=run)
        statuses = await client.status(["27744534"])
        assert statuses["27744534"].state == "RUNNING"
        argv = run.calls[0]
        assert argv[0] == "sacct"
        assert "--parsable2" in argv and "--noheader" in argv
        assert any("27744534" in a for a in argv)

    async def test_cancel(self):
        run = FakeRun([(0, "", "")])
        client = SlurmClient(run=run)
        await client.cancel("42")
        assert run.calls[0] == ["scancel", "42"]

    async def test_cancel_failure_raises(self):
        run = FakeRun([(1, "", "scancel: error: Invalid job id\n")])
        client = SlurmClient(run=run)
        with pytest.raises(SlurmError, match="Invalid job id"):
            await client.cancel("42")

    async def test_submit_host_wraps_in_ssh(self):
        run = FakeRun([(0, "Submitted batch job 7\n", "")])
        client = SlurmClient(submit_host="login01", run=run)
        await client.submit("/tmp/job.sh", [])
        assert run.calls[0][:3] == ["ssh", "login01", "--"]
        assert "sbatch" in run.calls[0]


class TestJobStates:
    """squeue-based liveness (§4.2 layer 1): running? -> keep, gone -> reap."""

    async def test_empty_list_makes_no_call(self):
        run = FakeRun([])
        client = SlurmClient(run=run)
        assert await client.job_states([]) == {}
        assert run.calls == []

    async def test_running_job_mapped(self):
        run = FakeRun([(0, "27744534|RUNNING\n", "")])
        client = SlurmClient(run=run)
        states = await client.job_states(["27744534"])
        assert states == {"27744534": "RUNNING"}
        argv = run.calls[0]
        assert argv[0] == "squeue"
        assert "-h" in argv
        assert "%i|%T" in argv
        assert any("27744534" in a for a in argv)

    async def test_missing_id_is_absent(self):
        # Queried two jobs, squeue only reports the live one; the other is gone.
        run = FakeRun([(0, "27744534|RUNNING\n", "")])
        client = SlurmClient(run=run)
        states = await client.job_states(["27744534", "999999"])
        assert states == {"27744534": "RUNNING"}
        assert "999999" not in states

    async def test_pending_state_preserved(self):
        run = FakeRun([(0, "50|PENDING\n", "")])
        client = SlurmClient(run=run)
        assert await client.job_states(["50"]) == {"50": "PENDING"}

    async def test_invalid_job_id_is_not_running_not_error(self):
        # A gone job makes squeue exit non-zero with "Invalid job id
        # specified" — that is authoritative "not running", not a failure.
        run = FakeRun(
            [(1, "", "slurm_load_jobs error: Invalid job id specified\n")]
        )
        client = SlurmClient(run=run)
        assert await client.job_states(["999999"]) == {}

    async def test_invalid_job_id_still_parses_partial_stdout(self):
        run = FakeRun(
            [(1, "100|RUNNING\n", "slurm_load_jobs error: Invalid job id specified\n")]
        )
        client = SlurmClient(run=run)
        assert await client.job_states(["100", "999999"]) == {"100": "RUNNING"}

    async def test_controller_unreachable_raises(self):
        # A real squeue failure must NOT read as "all jobs dead" — callers
        # fall back to probe-only rather than reaping every endpoint.
        run = FakeRun(
            [(1, "", "slurm_load_jobs error: Unable to contact slurm controller\n")]
        )
        client = SlurmClient(run=run)
        with pytest.raises(SlurmError, match="squeue"):
            await client.job_states(["27744534"])

    async def test_submit_host_wraps_in_ssh(self):
        run = FakeRun([(0, "7|RUNNING\n", "")])
        client = SlurmClient(submit_host="login01", run=run)
        await client.job_states(["7"])
        assert run.calls[0][:3] == ["ssh", "login01", "--"]
        assert "squeue" in run.calls[0]
