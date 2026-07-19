"""Tests for hpca.runner: the single internal subprocess runner (§5.1, §5.4)."""

import os
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


# ---------------------------------------------- persisted process history

from hpca.runner import (  # noqa: E402
    count_processes,
    kill_unowned,
    list_processes,
    reconcile_orphans,
    script_path_for,
)


def insert_process(conn, *, pid, session_id="s1", name="job", state="finished",
                   started_at="2026-07-19T10:00:00+00:00", cmd="bash /tmp/x.sh"):
    conn.execute(
        "INSERT INTO processes (pid, session_id, name, cmd, state, stdout_path, "
        "stderr_path, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pid, session_id, name, cmd, state, "/tmp/o", "/tmp/e", started_at),
    )
    conn.commit()


class TestListProcesses:
    """The panel reads the table, not a runner: a runner only knows what it
    started itself, and the TUI builds a new one every turn."""

    def test_returns_persisted_rows(self, conn):
        insert_process(conn, pid=101, name="align")
        records = list_processes(conn, session_id="s1")
        assert [r.name for r in records] == ["align"]
        assert records[0].pid == 101

    def test_newest_first(self, conn):
        insert_process(conn, pid=1, name="old", started_at="2026-07-19T09:00:00+00:00")
        insert_process(conn, pid=2, name="new", started_at="2026-07-19T11:00:00+00:00")
        assert [r.name for r in list_processes(conn, session_id="s1")] == ["new", "old"]

    def test_scoped_to_the_session(self, conn):
        insert_process(conn, pid=1, session_id="s1", name="mine")
        insert_process(conn, pid=2, session_id="s2", name="theirs")
        assert [r.name for r in list_processes(conn, session_id="s1")] == ["mine"]

    def test_limit_returns_one_extra_so_a_cut_is_detectable(self, conn):
        for pid in range(1, 6):
            insert_process(conn, pid=pid, started_at=f"2026-07-19T10:0{pid}:00+00:00")
        assert len(list_processes(conn, session_id="s1", limit=3)) == 4
        assert count_processes(conn, session_id="s1") == 5

    def test_unknown_session_is_empty(self, conn):
        assert list_processes(conn, session_id="nope") == []

    def test_null_columns_do_not_crash(self, conn):
        """A row written before a later column existed still has to render."""
        conn.execute(
            "INSERT INTO processes (pid, session_id, name, state) "
            "VALUES (?, ?, ?, ?)", (7, "s1", "sparse", "finished"))
        conn.commit()
        record = list_processes(conn, session_id="s1")[0]
        assert record.cmd == ""
        assert record.exit_code is None


class TestReconcileOrphans:
    """A monitor writes the row when its process exits, so a killed hpca
    leaves rows claiming to run forever."""

    def test_dead_running_row_is_settled(self, conn):
        insert_process(conn, pid=999_999, state="running")  # certainly not alive
        assert reconcile_orphans(conn) == 1
        record = list_processes(conn, session_id="s1")[0]
        assert record.state == "unknown"
        assert "last exited" in record.exit_info

    def test_live_process_is_left_alone(self, conn):
        insert_process(conn, pid=os.getpid(), state="running")
        assert reconcile_orphans(conn) == 0
        assert list_processes(conn, session_id="s1")[0].state == "running"

    def test_finished_rows_untouched(self, conn):
        insert_process(conn, pid=999_998, state="finished")
        assert reconcile_orphans(conn) == 0
        assert list_processes(conn, session_id="s1")[0].state == "finished"

    def test_idempotent(self, conn):
        insert_process(conn, pid=999_997, state="running")
        assert reconcile_orphans(conn) == 1
        assert reconcile_orphans(conn) == 0


class TestKillUnowned:
    def test_settles_the_row_even_when_the_process_is_gone(self, conn):
        insert_process(conn, pid=999_996, state="running")
        kill_unowned(conn, pid=999_996, session_id="s1")
        record = list_processes(conn, session_id="s1")[0]
        assert record.state == "killed"
        assert "panel" in record.exit_info


class TestScriptPathFor:
    """Derived from the command line, so it works for processes recorded
    before this existed and there is no second copy of the truth."""

    def test_finds_the_script(self, tmp_path):
        script = tmp_path / "align.sh"
        script.write_text("echo hi\n")
        assert script_path_for(f"bash {script}") == script

    def test_finds_it_past_arguments(self, tmp_path):
        script = tmp_path / "run.py"
        script.write_text("print(1)\n")
        assert script_path_for(f"python3 {script} --threads 8") == script

    def test_ignores_a_missing_file(self, tmp_path):
        assert script_path_for(f"bash {tmp_path / 'gone.sh'}") is None

    def test_no_script_in_a_plain_command(self):
        assert script_path_for("ls -la /data") is None

    def test_unparseable_command_is_not_an_error(self):
        assert script_path_for('bash "unclosed') is None

    def test_empty(self):
        assert script_path_for("") is None


class TestDescribe:
    """A history of `bash_1784458703578782438` rows identifies nothing."""

    def record(self, name, cmd):
        from hpca.runner import ProcessRecord
        from pathlib import Path as P

        return ProcessRecord(
            pid=1, name=name, cmd=cmd, state="finished",
            stdout_path=P("/tmp/o"), stderr_path=P("/tmp/e"),
            started_at="2026-07-19T10:00:00+00:00",
        )

    def test_named_scripts_keep_their_key(self, tmp_path):
        from hpca.runner import describe

        script = tmp_path / "align-cohort.sh"
        script.write_text("samtools view in.bam\n")
        assert describe(self.record("align-cohort", f"bash {script}")) == "align-cohort"

    def test_throwaway_shows_its_first_command(self, tmp_path):
        from hpca.runner import describe

        script = tmp_path / "bash_123.sh"
        script.write_text("#!/bin/bash\nset -euo pipefail\nnproc --all\n")
        assert describe(self.record("bash_123", f"bash {script}")) == "nproc --all"

    def test_long_command_is_truncated(self, tmp_path):
        from hpca.runner import describe

        script = tmp_path / "bash_1.sh"
        script.write_text("find /data -name '*.bam' -maxdepth 3 -type f | head -20\n")
        described = describe(self.record("bash_1", f"bash {script}"))
        assert len(described) <= 40
        assert described.endswith("…")

    def test_falls_back_to_the_name_when_the_script_is_gone(self, tmp_path):
        from hpca.runner import describe

        assert describe(
            self.record("bash_9", f"bash {tmp_path / 'gone.sh'}")
        ) == "bash_9"

    def test_empty_script_falls_back(self, tmp_path):
        from hpca.runner import describe

        script = tmp_path / "bash_2.sh"
        script.write_text("# only a comment\n")
        assert describe(self.record("bash_2", f"bash {script}")) == "bash_2"


class TestFirstCommand:
    def test_skips_shebang_comments_and_preamble(self, tmp_path):
        from hpca.runner import first_command

        script = tmp_path / "s.sh"
        script.write_text("#!/bin/bash\n# a note\nset -euo pipefail\n\nuname -r\n")
        assert first_command(script) == "uname -r"

    def test_missing_file(self, tmp_path):
        from hpca.runner import first_command

        assert first_command(tmp_path / "nope.sh") == ""
