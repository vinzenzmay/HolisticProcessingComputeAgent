"""Tests for hpca.agent.watch_tools: watch_log / watch_job / list / unwatch."""

import pytest

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.watch_tools import add_watch_tools
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.slurm import SlurmClient
from hpca.watches import KIND_JOB, KIND_LOG, LOG_PRESENT, WatchStore

SQUEUE_RUNNING = "27744534|RUNNING|node042|1-04:42:02|3-19:17:58|snakemake_run|None\n"
SACCT_DONE = "27744534|COMPLETED|0:0|01:00:00||4G|02:00:00\n"


class FakeRun:
    """Answers Slurm calls by which command was asked for, not by call order —
    watch_job only reaches sacct when squeue came up empty."""

    def __init__(self, *, squeue="", sacct=""):
        self.squeue, self.sacct = squeue, sacct
        self.calls: list[list[str]] = []

    async def __call__(self, argv):
        self.calls.append(argv)
        if argv[0] == "squeue":
            return 0, self.squeue, ""
        if argv[0] == "sacct":
            return 0, self.sacct, ""
        return 1, "", f"unexpected {argv[0]}"


@pytest.fixture
def tools():
    return add_watch_tools(default_tool_registry())


def make_ctx(tmp_path, *, run=None):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    return ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        session_id="s1",
        profile="default",
        slurm=SlurmClient(run=run) if run is not None else None,
        watches=WatchStore(conn),
    )


async def call(tools, name, ctx, **kwargs):
    tool = tools.get(name)
    return await tool.handler(tool.params(**kwargs), ctx)


class TestWatchLog:
    async def test_a_log_lands_in_the_panel_with_its_state(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "sniffles.log"
        log.write_text("calling SVs\n")
        result = await call(tools, "watch_log", ctx, path=str(log), label="sniffles")
        assert "Watching" in result
        [watch] = ctx.watches.list(profile="default")
        assert (watch.kind, watch.label) == (KIND_LOG, "sniffles")
        # Polled once on registration. "present", not "writing": whether
        # anything is still writing cannot be read off the file's mtime.
        assert watch.state == LOG_PRESENT

    async def test_a_registry_key_works_as_well_as_a_path(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        ctx.registry.register("run_log", log)
        await call(tools, "watch_log", ctx, path="run_log")
        assert ctx.watches.list(profile="default")[0].target == str(log)

    async def test_the_path_is_registered_so_later_turns_can_name_it(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, path=str(log))
        assert log in ctx.registry.list().values()

    async def test_a_log_that_does_not_exist_yet_is_watched_but_flagged(
        self, tools, tmp_path
    ):
        """A job that has not written its log yet is the normal case; a typo
        is not, so the result has to make the difference visible."""
        ctx = make_ctx(tmp_path)
        result = await call(tools, "watch_log", ctx, path=str(tmp_path / "later.log"))
        assert "does not exist yet" in result
        assert len(ctx.watches.list(profile="default")) == 1

    async def test_a_relative_path_is_refused_with_what_to_do_instead(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        result = await call(tools, "watch_log", ctx, path="sniffles.log")
        assert "Not watched" in result
        assert ctx.watches.list(profile="default") == []

    async def test_watching_the_same_log_twice_does_not_double_the_box(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, path=str(log))
        await call(tools, "watch_log", ctx, path=str(log), label="renamed")
        [watch] = ctx.watches.list(profile="default")
        assert watch.label == "renamed"


class TestWatchJob:
    async def test_a_queued_job_is_watched_and_named_after_itself(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        result = await call(tools, "watch_job", ctx, job_id="27744534")
        assert "RUNNING" in result
        [watch] = ctx.watches.list(profile="default")
        assert (watch.kind, watch.target, watch.label) == (
            KIND_JOB, "27744534", "snakemake_run"
        )
        assert watch.state == "RUNNING"

    async def test_a_job_that_just_finished_is_still_worth_a_box(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue="", sacct=SACCT_DONE))
        result = await call(tools, "watch_job", ctx, job_id="27744534")
        assert "no longer queued" in result
        assert len(ctx.watches.list(profile="default")) == 1

    async def test_an_id_slurm_has_never_heard_of_is_refused(self, tools, tmp_path):
        """A typo'd id would otherwise sit in the panel forever saying nothing."""
        ctx = make_ctx(tmp_path, run=FakeRun(squeue="", sacct=""))
        result = await call(tools, "watch_job", ctx, job_id="99999")
        assert "does not know job" in result
        assert ctx.watches.list(profile="default") == []

    async def test_without_slurm_it_says_so(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        assert "no Slurm connection" in await call(
            tools, "watch_job", ctx, job_id="1"
        )


class TestListAndUnwatch:
    async def test_listing_shows_both_kinds(self, tools, tmp_path):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, path=str(log), label="sniffles")
        await call(tools, "watch_job", ctx, job_id="27744534")
        result = await call(tools, "list_watches", ctx)
        assert "sniffles" in result and "27744534" in result

    async def test_listing_nothing_says_nothing(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        assert await call(tools, "list_watches", ctx) == "Nothing is being watched."

    async def test_unwatch_by_label(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, path=str(log), label="sniffles")
        assert "Stopped watching" in await call(
            tools, "unwatch", ctx, target="sniffles"
        )
        assert ctx.watches.list(profile="default") == []

    async def test_unwatch_by_file_name(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, path=str(log))
        await call(tools, "unwatch", ctx, target="run.log")
        assert ctx.watches.list(profile="default") == []

    async def test_no_match_lists_what_there_is_instead_of_guessing(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, path=str(log), label="sniffles")
        result = await call(tools, "unwatch", ctx, target="minimap")
        assert "No watch matches" in result and "sniffles" in result

    async def test_an_ambiguous_name_is_reported_not_guessed_at(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        for name in ("sample_a.log", "sample_b.log"):
            path = tmp_path / name
            path.write_text("x")
            await call(tools, "watch_log", ctx, path=str(path))
        result = await call(tools, "unwatch", ctx, target="sample")
        assert "matches several" in result
        assert len(ctx.watches.list(profile="default")) == 2
