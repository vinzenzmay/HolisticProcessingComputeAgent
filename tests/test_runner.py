"""Tests for hpca.runner: the single internal subprocess runner (§5.1, §5.4)."""

import asyncio

import pytest

from hpca.db import connect, init_db
from hpca.runner import ProcessRunner


@pytest.fixture
def conn(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def runner(conn, tmp_path):
    return ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "proc_logs")


class TestRun:
    async def test_captures_stdout_to_file(self, runner):
        proc = await runner.start(["echo", "hello world"], name="greet")
        await runner.wait(proc.pid)
        record = runner.get(proc.pid)
        assert record.state == "finished"
        assert record.exit_code == 0
        assert "hello world" in record.stdout_path.read_text()

    async def test_captures_stderr_separately(self, runner):
        proc = await runner.start(
            ["bash", "-c", "echo out; echo err >&2"], name="both"
        )
        await runner.wait(proc.pid)
        record = runner.get(proc.pid)
        assert "out" in record.stdout_path.read_text()
        assert "err" in record.stderr_path.read_text()
        assert "err" not in record.stdout_path.read_text()

    async def test_nonzero_exit_is_failed(self, runner):
        proc = await runner.start(["bash", "-c", "exit 3"], name="fail")
        await runner.wait(proc.pid)
        record = runner.get(proc.pid)
        assert record.state == "failed"
        assert record.exit_code == 3

    async def test_missing_command_raises(self, runner):
        with pytest.raises(FileNotFoundError):
            await runner.start(["definitely-not-a-command-xyz"], name="nope")

    async def test_db_row_written(self, runner, conn):
        proc = await runner.start(["echo", "hi"], name="dbrow")
        await runner.wait(proc.pid)
        row = conn.execute(
            "SELECT * FROM processes WHERE pid = ?", (proc.pid,)
        ).fetchone()
        assert row["session_id"] == "s1"
        assert "echo" in row["cmd"]
        assert row["state"] == "finished"

    async def test_running_state_while_alive(self, runner):
        proc = await runner.start(["sleep", "5"], name="sleeper")
        record = runner.get(proc.pid)
        assert record.state == "running"
        await runner.kill(proc.pid)

    async def test_cwd_respected(self, runner, tmp_path):
        workdir = tmp_path / "work"
        workdir.mkdir()
        proc = await runner.start(["pwd"], name="whereami", cwd=workdir)
        await runner.wait(proc.pid)
        assert str(workdir) in runner.get(proc.pid).stdout_path.read_text()


class TestKill:
    async def test_kill_marks_killed(self, runner):
        proc = await runner.start(["sleep", "60"], name="victim")
        await runner.kill(proc.pid)
        await runner.wait(proc.pid)
        assert runner.get(proc.pid).state == "killed"

    async def test_kill_unknown_pid_raises(self, runner):
        with pytest.raises(KeyError):
            await runner.kill(999999)


class TestTimeout:
    async def test_timeout_kills_and_marks_killed(self, runner):
        proc = await runner.start(["sleep", "60"], name="slow", timeout_s=0.2)
        await runner.wait(proc.pid)
        record = runner.get(proc.pid)
        assert record.state == "killed"
        assert "timeout" in (record.exit_info or "")


class TestList:
    async def test_lists_session_processes_newest_first(self, runner):
        p1 = await runner.start(["echo", "one"], name="one")
        p2 = await runner.start(["echo", "two"], name="two")
        await runner.wait(p1.pid)
        await runner.wait(p2.pid)
        pids = [r.pid for r in runner.list()]
        assert pids == [p2.pid, p1.pid]


class TestStartFailureCleanup:
    async def test_db_insert_failure_kills_process_and_cleans_up(self, runner):
        import sqlite3

        class FailingConn:
            """sqlite3.Connection attrs are read-only; proxy instead."""

            def __init__(self, real):
                self._real = real

            def execute(self, sql, *params):
                if sql.strip().startswith("INSERT INTO processes"):
                    raise sqlite3.OperationalError("database is locked")
                return self._real.execute(sql, *params)

            def commit(self):
                self._real.commit()

        runner._conn = FailingConn(runner._conn)
        with pytest.raises(sqlite3.OperationalError):
            await runner.start(["sleep", "60"], name="doomed")
        assert runner.list() == []  # no half-tracked record


class TestRunningSessionIds:
    def _insert(self, conn, *, pid, session_id, state):
        conn.execute(
            "INSERT INTO processes (pid, session_id, name, cmd, state, "
            "stdout_path, stderr_path, started_at) "
            "VALUES (?, ?, 'p', 'p', ?, '/o', '/e', 't')",
            (pid, session_id, state),
        )
        conn.commit()

    def test_only_live_running_processes_count(self, conn):
        import os

        from hpca.runner import running_session_ids

        self._insert(conn, pid=os.getpid(), session_id="live", state="running")
        self._insert(conn, pid=os.getpid(), session_id="done", state="finished")
        # a pid that is extremely unlikely to exist: stale 'running' row
        self._insert(conn, pid=2_000_000_000, session_id="stale", state="running")

        ids = running_session_ids(conn)
        assert "live" in ids  # our own pid is alive
        assert "done" not in ids  # not running
        assert "stale" not in ids  # running row, but the pid is gone

    def test_empty_when_nothing_runs(self, conn):
        from hpca.runner import running_session_ids

        assert running_session_ids(conn) == set()
