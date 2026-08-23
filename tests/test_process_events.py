"""Tests for background-process completion reaching the agent (§5.4).

The gap these cover: start_background_script hands a process to the runner and the turn
ends. Nothing used to detect that it exited, so the agent's "I'll check on it"
was a promise the runtime could not keep.
"""

import asyncio

import pytest

from hpca.db import connect, init_db
from hpca.runner import (
    ProcessChange,
    ProcessRunner,
    format_process_event,
    poll_processes,
)


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "hpca.db")
    init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def runner(conn, tmp_path):
    return ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs")


class TestPollProcesses:
    async def test_running_process_is_not_reported(self, conn, runner):
        await runner.start(["sleep", "5"], name="slow", background=True)
        assert poll_processes(conn) == []

    async def test_failure_is_reported_once(self, conn, runner):
        record = await runner.start(
            ["bash", "-c", "echo boom >&2; exit 3"], name="doomed", background=True
        )
        await runner.wait(record.pid)

        changes = poll_processes(conn)
        assert [c.name for c in changes] == ["doomed"]
        assert changes[0].state == "failed"
        assert changes[0].exit_code == 3
        assert changes[0].session_id == "s1"

        # exactly once: a second poll must not replay it
        assert poll_processes(conn) == []

    async def test_success_is_reported(self, conn, runner):
        record = await runner.start(["bash", "-c", "echo done"], name="ok", background=True)
        await runner.wait(record.pid)
        changes = poll_processes(conn)
        assert changes[0].state == "finished"
        assert changes[0].exit_code == 0

    async def test_completion_survives_a_new_runner(self, conn, tmp_path):
        """The TUI builds a fresh ProcessRunner per turn, so the watcher must
        read the table rather than any one runner's in-memory records."""
        first = ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs")
        record = await first.start(["bash", "-c", "exit 1"], name="from-turn-1", background=True)
        await first.wait(record.pid)

        ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs")  # turn 2
        changes = poll_processes(conn)
        assert [c.name for c in changes] == ["from-turn-1"]

    async def test_completion_while_app_was_down_is_delivered(self, conn, runner):
        """notified lives in the DB, so a process that ended between runs is
        still announced the next time the watcher polls."""
        record = await runner.start(["bash", "-c", "exit 1"], name="overnight", background=True)
        await runner.wait(record.pid)
        # simulate a restart: nothing in memory, only the table
        assert [c.name for c in poll_processes(conn)] == ["overnight"]


class TestOnlyBackgroundWork:
    """run_bash blocks and hands its output back as the tool
    result, so announcing it again would tell the agent the same thing
    twice — and a failing run_bash would look like a fresh crash to react to."""

    async def test_foreground_failure_produces_no_event(self, conn, runner):
        record = await runner.start(["bash", "-c", "exit 1"], name="run_bash_probe")
        await runner.wait(record.pid)
        assert poll_processes(conn) == []

    async def test_background_failure_still_produces_an_event(self, conn, runner):
        record = await runner.start(
            ["bash", "-c", "exit 1"], name="started_script", background=True
        )
        await runner.wait(record.pid)
        assert [c.name for c in poll_processes(conn)] == ["started_script"]

    async def test_start_script_marks_its_process_background(self, tmp_path, conn):
        """The wiring that matters: the tool the agent actually calls."""
        from hpca.agent.builtin_tools import default_tool_registry
        from hpca.agent.context import ToolContext
        from hpca.config import Settings

        ctx = ToolContext(
            workdir=tmp_path,
            runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
            settings=Settings(),
            scripts_dir=tmp_path / "scripts",
        )
        tools = default_tool_registry()
        create = tools.get("create_script")
        await create.handler(
            create.params.model_validate(
                {"name": "job", "content_lines": ["exit 4"]}
            ),
            ctx,
        )
        start = tools.get("start_background_script")
        await start.handler(
            start.params.model_validate({"name": "job"}), ctx
        )
        await asyncio.sleep(0.5)
        changes = poll_processes(conn)
        assert [c.name for c in changes] == ["job"]
        assert changes[0].state == "failed"


class TestNoDuplicateStart:
    """A live session started one script twice (pids 38429 and 38443), both
    writing the same VCF. Corrupted output, not just wasted CPU."""

    @pytest.fixture
    def ctx(self, conn, tmp_path):
        from hpca.agent.context import ToolContext
        from hpca.config import Settings

        return ToolContext(
            workdir=tmp_path,
            runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
            settings=Settings(),
            scripts_dir=tmp_path / "scripts",
        )

    async def _make(self, ctx, key, lines):
        from hpca.agent.builtin_tools import default_tool_registry

        tools = default_tool_registry()
        create = tools.get("create_script")
        await create.handler(
            create.params.model_validate(
                {"name": key, "content_lines": lines}
            ),
            ctx,
        )
        return tools.get("start_background_script")

    async def test_second_start_is_refused(self, ctx):
        start = await self._make(ctx, "slow_job", ["sleep 5"])
        first = await start.handler(
            start.params.model_validate({"name": "slow_job"}), ctx
        )
        second = await start.handler(
            start.params.model_validate({"name": "slow_job"}), ctx
        )
        assert "Started" in first
        assert "NOT started" in second
        assert "already running" in second
        assert len(ctx.runner.list()) == 1

    async def test_restart_allowed_once_it_has_finished(self, ctx):
        start = await self._make(ctx, "quick_job", ["true"])
        first = await start.handler(
            start.params.model_validate({"name": "quick_job"}), ctx
        )
        pid = int(first.split("pid ")[1].split(")")[0])
        await ctx.runner.wait(pid)
        again = await start.handler(
            start.params.model_validate({"name": "quick_job"}), ctx
        )
        assert "Started" in again

    async def test_stale_running_row_does_not_block_forever(self, conn, ctx):
        """A row left at 'running' by a crashed app must not wedge the name."""
        start = await self._make(ctx, "ghost", ["true"])
        await start.handler(
            start.params.model_validate({"name": "ghost"}), ctx
        )
        conn.execute(
            "UPDATE processes SET state = 'running', pid = 999999 WHERE name = 'ghost'"
        )
        conn.commit()
        assert ctx.runner.running_named("ghost") is None


class TestFormatProcessEvent:
    async def test_failure_names_the_error_and_asks_for_a_fix(self, conn, runner):
        record = await runner.start(
            ["bash", "-c", "echo 'mamba: command not found' >&2; exit 127"],
            name="sniffles_run",
            background=True,
        )
        await runner.wait(record.pid)
        text = format_process_event(poll_processes(conn)[0])

        assert "[process failed]" in text
        assert "sniffles_run" in text
        assert "127" in text
        assert "mamba: command not found" in text  # the cause, inline
        assert "fix the script" in text  # mirrors run_bash's failure wording

    async def test_success_carries_stdout(self, conn, runner):
        record = await runner.start(
            ["bash", "-c", "echo '187 reads extracted'"], name="extract",
            background=True
        )
        await runner.wait(record.pid)
        text = format_process_event(poll_processes(conn)[0])
        assert "[process finished]" in text
        assert "187 reads extracted" in text

    def test_log_tail_is_bounded(self, tmp_path):
        """This text goes straight into a prompt; a 100k-line log must not."""
        stderr = tmp_path / "big.err"
        stderr.write_text("\n".join(f"line {i}" for i in range(100_000)))
        change = ProcessChange(
            pid=1,
            session_id="s1",
            name="noisy",
            state="failed",
            exit_code=1,
            stdout_path=tmp_path / "missing.out",
            stderr_path=stderr,
        )
        text = format_process_event(change)
        assert len(text) < 2000
        assert "line 99999" in text  # the tail, which is where errors are

    def test_missing_log_file_does_not_raise(self, tmp_path):
        change = ProcessChange(
            pid=1,
            session_id="s1",
            name="gone",
            state="failed",
            exit_code=1,
            stdout_path=tmp_path / "nope.out",
            stderr_path=tmp_path / "nope.err",
        )
        assert "[process failed]" in format_process_event(change)
