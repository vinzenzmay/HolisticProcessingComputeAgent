"""Tests for hpca.agent.job_tools: submit_job / job_status / cancel_job (§5.1)."""

import pytest

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.job_tools import add_job_tools
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.jobs import JobStore
from hpca.runner import ProcessRunner
from hpca.slurm import JobStatus, SlurmClient


class FakeRun:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    async def __call__(self, argv):
        self.calls.append(argv)
        return self.responses.pop(0)


@pytest.fixture
def tools():
    registry = default_tool_registry()
    add_job_tools(registry)
    return registry


def make_ctx(tmp_path, fake_run):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    return ToolContext(
        workdir=tmp_path,
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        session_id="s1",
        profile="default",
        slurm=SlurmClient(run=fake_run),
        jobs=JobStore(conn),
        job_log_dir=tmp_path / "job_logs",
    )


async def call(tools, tool_name, ctx, **kwargs):
    tool = tools.get(tool_name)
    return await tool.handler(tool.params.model_validate(kwargs), ctx)


def register_script(ctx, tmp_path):
    """Put a kept script called 'my_job' where script_path will find it."""
    ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
    script = ctx.scripts_dir / "my_job.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    return script


class TestSubmitJob:
    async def test_test_only_gate_then_submit(self, tools, tmp_path):
        run = FakeRun(
            [
                (0, "", "sbatch: Job 999 to start at ...\n"),  # --test-only
                (0, "Submitted batch job 27744534\n", ""),  # real submission
            ]
        )
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        result = await call(tools, "submit_job", ctx, name="my_job")
        assert "27744534" in result
        assert "--test-only" in run.calls[0]
        assert "--test-only" not in run.calls[1]
        # job recorded with resolved %j log paths, registered in path registry
        job = ctx.jobs.get("27744534")
        assert job is not None
        assert "27744534" in job.sbatch_stdout_path
        assert "%j" not in job.sbatch_stdout_path
        # the result names the log paths outright: nothing to look a key up in
        assert job.sbatch_stdout_path in result

    async def test_test_only_failure_blocks_submission(self, tools, tmp_path):
        run = FakeRun([(1, "", "sbatch: error: Invalid partition specified\n")])
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        result = await call(tools, "submit_job", ctx, name="my_job")
        assert "NOT submitted" in result
        assert "Invalid partition" in result
        assert len(run.calls) == 1  # no real submission attempted
        assert ctx.jobs.active() == []

    async def test_extra_args_passed_through(self, tools, tmp_path):
        run = FakeRun(
            [(0, "", ""), (0, "Submitted batch job 1\n", "")]
        )
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        await call(
            tools, "submit_job", ctx,
            name="my_job", args="--mem=8G --time=01:00:00",
        )
        assert "--mem=8G" in run.calls[1]

    async def test_an_unknown_script_is_refused_not_raised(self, tools, tmp_path):
        """It reaches the model as a sentence it can act on, listing what does
        exist — the shape specs/specs-edit-eval.md §1 asks every refusal to have."""
        ctx = make_ctx(tmp_path, FakeRun([]))
        result = await call(tools, "submit_job", ctx, name="ghost")
        assert "NOT submitted" in result and "ghost" in result


class TestJobStatus:
    async def test_reports_live_state(self, tools, tmp_path):
        run = FakeRun(
            [
                (0, "", ""),
                (0, "Submitted batch job 5\n", ""),
                (0, "5|RUNNING|0:0|00:01:00||4G|01:00:00\n", ""),
            ]
        )
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        await call(tools, "submit_job", ctx, name="my_job")
        result = await call(tools, "job_status", ctx, job_id="5")
        assert "RUNNING" in result

    async def test_accounting_lag_reported_as_submitted(self, tools, tmp_path):
        run = FakeRun(
            [(0, "", ""), (0, "Submitted batch job 6\n", ""), (0, "", "")]
        )
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        await call(tools, "submit_job", ctx, name="my_job")
        result = await call(tools, "job_status", ctx, job_id="6")
        assert "SUBMITTED" in result

    async def test_unknown_job_id(self, tools, tmp_path):
        ctx = make_ctx(tmp_path, FakeRun([(0, "", "")]))
        result = await call(tools, "job_status", ctx, job_id="404")
        assert "unknown" in result.lower() or "not" in result.lower()


class TestCancelJob:
    async def test_cancel_flagged_destructive(self, tools):
        assert tools.get("cancel_job").destructive is True

    async def test_cancels_and_updates_store(self, tools, tmp_path):
        run = FakeRun(
            [
                (0, "", ""),
                (0, "Submitted batch job 7\n", ""),
                (0, "", ""),  # scancel
            ]
        )
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        await call(tools, "submit_job", ctx, name="my_job")
        result = await call(tools, "cancel_job", ctx, job_id="7")
        assert "cancel" in result.lower()
        assert run.calls[2][:2] == ["scancel", "7"]
        assert ctx.jobs.get("7").state == "CANCELLING"


VALID_EXPLANATION = '{"why": "Ran out of memory loading the index.", "current_state": "Job was OOM-killed.", "suggested_fix": "Resubmit with --mem=50G.", "finickiness": "easy", "justification": "Simple resource bump."}'


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        from hpca.llm import ChatResponse

        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


class TestGetJobReport:
    async def test_report_with_explanation(self, tools, tmp_path):
        run = FakeRun(
            [
                (0, "", ""),
                (0, "Submitted batch job 8\n", ""),
                (0, "8|OUT_OF_MEMORY|0:125|00:05:00||25G|01:00:00\n", ""),
            ]
        )
        ctx = make_ctx(tmp_path, run)
        ctx.llm = FakeLLM([VALID_EXPLANATION])
        register_script(ctx, tmp_path)
        await call(tools, "submit_job", ctx, name="my_job")
        result = await call(tools, "get_job_report", ctx, job_id="8")
        assert "OUT_OF_MEMORY" in result
        assert "Out of memory" in result  # state-derived signature title
        assert "finickiness: easy" in result
        assert "--mem=50G" in result

    async def test_unknown_job(self, tools, tmp_path):
        ctx = make_ctx(tmp_path, FakeRun([]))
        result = await call(tools, "get_job_report", ctx, job_id="404")
        assert "not in the job DB" in result

    async def test_without_llm_returns_deterministic_header(self, tools, tmp_path):
        run = FakeRun(
            [
                (0, "", ""),
                (0, "Submitted batch job 9\n", ""),
                (0, "9|FAILED|1:0|00:01:00||4G|01:00:00\n", ""),
            ]
        )
        ctx = make_ctx(tmp_path, run)
        register_script(ctx, tmp_path)
        await call(tools, "submit_job", ctx, name="my_job")
        result = await call(tools, "get_job_report", ctx, job_id="9")
        assert "FAILED" in result
