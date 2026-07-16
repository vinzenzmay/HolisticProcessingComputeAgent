"""Tests for hpca.jobs: job DB rows and the polling update cycle (§5.4)."""

import pytest

from hpca.db import connect, init_db
from hpca.jobs import JobStore, poll_active
from hpca.slurm import JobStatus


@pytest.fixture
def conn(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def store(conn):
    return JobStore(conn)


def add_job(store, job_id="1", session_id="s1"):
    return store.add(
        job_id=job_id,
        kind="sbatch",
        session_id=session_id,
        profile="default",
        script_key="my_script",
        stdout_path=f"/logs/{job_id}.out",
        stderr_path=f"/logs/{job_id}.err",
    )


class TestJobStore:
    def test_add_and_get(self, store):
        add_job(store, "27744534")
        job = store.get("27744534")
        assert job.state == "SUBMITTED"
        assert job.script_key == "my_script"
        assert job.sbatch_stdout_path == "/logs/27744534.out"

    def test_list_newest_first(self, store):
        add_job(store, "1")
        add_job(store, "2")
        assert [j.job_id for j in store.list(session_id="s1")] == ["2", "1"]

    def test_list_filters_session(self, store):
        add_job(store, "1", session_id="s1")
        add_job(store, "2", session_id="s2")
        assert [j.job_id for j in store.list(session_id="s2")] == ["2"]

    def test_update_from_status(self, store):
        add_job(store, "1")
        store.update_status(
            JobStatus(job_id="1", state="RUNNING", raw_state="RUNNING")
        )
        assert store.get("1").state == "RUNNING"

    def test_terminal_update_records_exit_info(self, store):
        add_job(store, "1")
        store.update_status(
            JobStatus(job_id="1", state="FAILED", raw_state="FAILED", exit_code=1)
        )
        job = store.get("1")
        assert job.state == "FAILED"
        assert "exit 1" in job.exit_info

    def test_active_excludes_terminal(self, store):
        add_job(store, "1")
        add_job(store, "2")
        store.update_status(
            JobStatus(job_id="1", state="COMPLETED", raw_state="COMPLETED", exit_code=0)
        )
        assert [j.job_id for j in store.active()] == ["2"]

    def test_job_logs(self, store):
        add_job(store, "1")
        store.add_log("1", rule_or_step="align", log_path="/logs/align.log",
                      tool_name="bwa")
        logs = store.logs("1")
        assert logs[0].log_path == "/logs/align.log"


class StubSlurm:
    def __init__(self, statuses):
        self._statuses = statuses
        self.queried: list[list[str]] = []

    async def status(self, job_ids):
        self.queried.append(sorted(job_ids))
        return {k: v for k, v in self._statuses.items() if k in job_ids}


class TestPollActive:
    async def test_reports_state_changes(self, store):
        add_job(store, "1")
        add_job(store, "2")
        slurm = StubSlurm(
            {
                "1": JobStatus(job_id="1", state="RUNNING", raw_state="RUNNING"),
                "2": JobStatus(
                    job_id="2", state="COMPLETED", raw_state="COMPLETED", exit_code=0
                ),
            }
        )
        changes = await poll_active(slurm, store)
        assert {(c.job_id, c.old_state, c.new_state) for c in changes} == {
            ("1", "SUBMITTED", "RUNNING"),
            ("2", "SUBMITTED", "COMPLETED"),
        }
        assert store.get("2").state == "COMPLETED"

    async def test_unchanged_state_not_reported(self, store):
        add_job(store, "1")
        slurm = StubSlurm(
            {"1": JobStatus(job_id="1", state="RUNNING", raw_state="RUNNING")}
        )
        await poll_active(slurm, store)
        changes = await poll_active(slurm, store)
        assert changes == []

    async def test_job_unknown_to_sacct_stays_submitted(self, store):
        # accounting lag right after submission: sacct may not know the job yet
        add_job(store, "1")
        slurm = StubSlurm({})
        changes = await poll_active(slurm, store)
        assert changes == []
        assert store.get("1").state == "SUBMITTED"

    async def test_no_active_jobs_skips_query(self, store):
        slurm = StubSlurm({})
        await poll_active(slurm, store)
        assert slurm.queried == []

    async def test_terminal_jobs_not_polled_again(self, store):
        add_job(store, "1")
        slurm = StubSlurm(
            {"1": JobStatus(job_id="1", state="FAILED", raw_state="FAILED",
                            exit_code=1)}
        )
        await poll_active(slurm, store)
        await poll_active(slurm, store)
        assert len(slurm.queried) == 1
