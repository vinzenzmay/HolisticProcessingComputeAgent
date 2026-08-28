"""Tests for hpca.agent.watch_tools: watch_log / watch_job / list / unwatch."""

from pathlib import Path

import pytest

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.watch_tools import add_watch_tools
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.runner import ProcessRunner
from hpca.slurm import SlurmClient
from hpca.watches import KIND_JOB, KIND_LOG, LOG_PRESENT, WatchStore

SQUEUE_RUNNING = "27744534|RUNNING|node042|1-04:42:02|3-19:17:58|align_run|None\n"
SQUEUE_PENDING = "27744535|PENDING||0:00||call_svs|Priority\n"
SACCT_DONE = "27744534|COMPLETED|0:0|01:00:00||4G|02:00:00\n"
SACCT_OTHER_DONE = "27744536|COMPLETED|0:0|01:00:00||4G|02:00:00\n"


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
        workdir=tmp_path,
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
        result = await call(
            tools, "watch_log", ctx, paths=[str(log)], label="sniffles"
        )
        assert "Watching" in result
        [watch] = ctx.watches.list(profile="default")
        assert (watch.kind, watch.label) == (KIND_LOG, "sniffles")
        # Polled once on registration. "present", not "writing": whether
        # anything is still writing cannot be read off the file's mtime.
        assert watch.state == LOG_PRESENT

    async def test_a_relative_path_is_anchored_at_the_workdir(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, paths=["run.log"])
        assert ctx.watches.list(profile="default")[0].target == str(log)

    async def test_a_log_that_does_not_exist_yet_is_watched_but_flagged(
        self, tools, tmp_path
    ):
        """A job that has not written its log yet is the normal case; a typo
        is not, so the result has to make the difference visible."""
        ctx = make_ctx(tmp_path)
        result = await call(
            tools, "watch_log", ctx, paths=[str(tmp_path / "later.log")]
        )
        assert "does not exist yet" in result
        assert len(ctx.watches.list(profile="default")) == 1

    async def test_an_empty_path_is_refused_with_what_to_do_instead(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        result = await call(tools, "watch_log", ctx, paths=["   "])
        assert "Not watched" in result
        assert ctx.watches.list(profile="default") == []

    async def test_watching_the_same_log_twice_does_not_double_the_box(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(log)])
        await call(tools, "watch_log", ctx, paths=[str(log)], label="renamed")
        [watch] = ctx.watches.list(profile="default")
        assert watch.label == "renamed"

    async def test_several_logs_are_pinned_by_one_call(self, tools, tmp_path):
        """The point of the array: a pipeline's logs are found together, so
        they are pinned together rather than one decision at a time."""
        ctx = make_ctx(tmp_path)
        names = ["align.log", "call.log", "qc.log"]
        for name in names:
            (tmp_path / name).write_text("x")
        result = await call(
            tools, "watch_log", ctx, paths=[str(tmp_path / n) for n in names]
        )
        assert "Watching 3 logs" in result
        assert all(name in result for name in names)
        watched = {Path(w.target).name for w in ctx.watches.list(profile="default")}
        assert watched == set(names)

    async def test_a_bare_string_is_read_as_one_path(self, tools, tmp_path):
        """What the backend actually sends when it means one file; wrapping it
        beats spending a retry on the brackets (see ``_as_string_list``)."""
        ctx = make_ctx(tmp_path)
        (tmp_path / "run.log").write_text("x")
        await call(tools, "watch_log", ctx, paths="run.log")
        assert len(ctx.watches.list(profile="default")) == 1

    async def test_one_file_named_twice_is_one_box(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        result = await call(tools, "watch_log", ctx, paths=[str(log), "run.log"])
        assert len(ctx.watches.list(profile="default")) == 1
        # And it reads as the single watch it is, not as "Watching 2 logs".
        assert "Watching 2" not in result

    async def test_a_blank_entry_does_not_cost_the_others(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        (tmp_path / "run.log").write_text("x")
        result = await call(tools, "watch_log", ctx, paths=["", "run.log"])
        assert len(ctx.watches.list(profile="default")) == 1
        assert "1 blank entry ignored" in result

    async def test_a_label_is_dropped_when_it_would_name_several_boxes(
        self, tools, tmp_path
    ):
        """One label over three files would put the same name on all three,
        which is worse than the file names it replaced."""
        ctx = make_ctx(tmp_path)
        for name in ("a.log", "b.log"):
            (tmp_path / name).write_text("x")
        await call(
            tools,
            "watch_log",
            ctx,
            paths=[str(tmp_path / "a.log"), str(tmp_path / "b.log")],
            label="pipeline",
        )
        labels = {w.label for w in ctx.watches.list(profile="default")}
        assert labels == {""}


class TestWatchJob:
    async def test_a_queued_job_is_watched_and_named_after_itself(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        result = await call(tools, "watch_job", ctx, job_ids=["27744534"])
        assert "RUNNING" in result
        [watch] = ctx.watches.list(profile="default")
        assert (watch.kind, watch.target, watch.label) == (
            KIND_JOB, "27744534", "align_run"
        )
        assert watch.state == "RUNNING"

    async def test_a_job_that_just_finished_is_still_worth_a_box(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue="", sacct=SACCT_DONE))
        result = await call(tools, "watch_job", ctx, job_ids=["27744534"])
        assert "no longer queued" in result
        assert len(ctx.watches.list(profile="default")) == 1

    async def test_an_id_slurm_has_never_heard_of_is_refused(self, tools, tmp_path):
        """A typo'd id would otherwise sit in the panel forever saying nothing."""
        ctx = make_ctx(tmp_path, run=FakeRun(squeue="", sacct=""))
        result = await call(tools, "watch_job", ctx, job_ids=["99999"])
        assert "does not know job" in result
        assert ctx.watches.list(profile="default") == []

    async def test_a_blank_id_is_counted_not_swallowed(self, tools, tmp_path):
        """A blank element is the model losing an argument mid-array; the
        count is what tells it the call was short one job."""
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        result = await call(tools, "watch_job", ctx, job_ids=["27744534", " "])
        assert "1 blank entry ignored" in result
        assert len(ctx.watches.list(profile="default")) == 1

    async def test_without_slurm_it_says_so(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        assert "no Slurm connection" in await call(
            tools, "watch_job", ctx, job_ids=["1"]
        )

    async def test_several_jobs_cost_one_squeue(self, tools, tmp_path):
        """The reason the array exists on this side: one squeue for the batch,
        not one process per id."""
        run = FakeRun(squeue=SQUEUE_RUNNING + SQUEUE_PENDING)
        ctx = make_ctx(tmp_path, run=run)
        result = await call(
            tools, "watch_job", ctx, job_ids=["27744534", "27744535"]
        )
        assert "Watching 2 jobs" in result
        assert "RUNNING" in result and "PENDING" in result
        assert [argv[0] for argv in run.calls] == ["squeue"]
        assert len(ctx.watches.list(profile="default")) == 2

    async def test_ids_may_arrive_as_numbers(self, tools, tmp_path):
        """A job id reads as an integer, so the model sends one; pydantic v2
        will not coerce it, and refusing it would spend a retry on quoting."""
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        await call(tools, "watch_job", ctx, job_ids=[27744534])
        [watch] = ctx.watches.list(profile="default")
        assert watch.target == "27744534"

    async def test_the_unknown_ones_are_named_and_the_rest_still_watched(
        self, tools, tmp_path
    ):
        """A typo among real ids must not cost the real ones their boxes."""
        run = FakeRun(squeue=SQUEUE_RUNNING, sacct=SACCT_OTHER_DONE)
        ctx = make_ctx(tmp_path, run=run)
        result = await call(
            tools, "watch_job", ctx, job_ids=["27744534", "27744536", "99999"]
        )
        assert "Watching 2 jobs" in result
        assert "does not know job 99999" in result
        # One squeue, then one sacct for everything squeue did not know.
        assert [argv[0] for argv in run.calls] == ["squeue", "sacct"]
        assert "27744536,99999" in run.calls[1]
        watched = {w.target for w in ctx.watches.list(profile="default")}
        assert watched == {"27744534", "27744536"}


class TestListAndUnwatch:
    async def test_listing_shows_both_kinds(self, tools, tmp_path):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(log)], label="sniffles")
        await call(tools, "watch_job", ctx, job_ids=["27744534"])
        result = await call(tools, "list_watches", ctx)
        assert "sniffles" in result and "27744534" in result

    async def test_listing_nothing_says_nothing(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        assert await call(tools, "list_watches", ctx) == "Nothing is being watched."

    async def test_unwatch_by_label(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(log)], label="sniffles")
        assert "Stopped watching" in await call(
            tools, "unwatch", ctx, targets=["sniffles"]
        )
        assert ctx.watches.list(profile="default") == []

    async def test_unwatch_by_file_name(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(log)])
        await call(tools, "unwatch", ctx, targets=["run.log"])
        assert ctx.watches.list(profile="default") == []

    async def test_no_match_lists_what_there_is_instead_of_guessing(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        log = tmp_path / "run.log"
        log.write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(log)], label="sniffles")
        result = await call(tools, "unwatch", ctx, targets=["minimap"])
        assert "No watch matches" in result and "sniffles" in result

    async def test_an_ambiguous_name_is_reported_not_guessed_at(
        self, tools, tmp_path
    ):
        ctx = make_ctx(tmp_path)
        for name in ("sample_a.log", "sample_b.log"):
            path = tmp_path / name
            path.write_text("x")
            await call(tools, "watch_log", ctx, paths=[str(path)])
        result = await call(tools, "unwatch", ctx, targets=["sample"])
        assert "matches several" in result
        assert len(ctx.watches.list(profile="default")) == 2

    async def test_several_watches_go_in_one_call(self, tools, tmp_path):
        ctx = make_ctx(tmp_path, run=FakeRun(squeue=SQUEUE_RUNNING))
        for name in ("align.log", "call.log"):
            (tmp_path / name).write_text("x")
        await call(
            tools,
            "watch_log",
            ctx,
            paths=[str(tmp_path / "align.log"), str(tmp_path / "call.log")],
        )
        await call(tools, "watch_job", ctx, job_ids=["27744534"])
        result = await call(
            tools, "unwatch", ctx, targets=["align.log", "27744534"]
        )
        assert "Stopped watching 2:" in result
        left = [w.target for w in ctx.watches.list(profile="default")]
        assert left == [str(tmp_path / "call.log")]

    async def test_one_name_that_misses_does_not_cost_the_others(
        self, tools, tmp_path
    ):
        """Partial application: a name the panel does not have is reported,
        and the names around it are still dropped."""
        ctx = make_ctx(tmp_path)
        (tmp_path / "run.log").write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(tmp_path / "run.log")])
        result = await call(
            tools, "unwatch", ctx, targets=["run.log", "minimap"]
        )
        assert "Stopped watching 1:" in result
        assert "No watch matches 'minimap'" in result
        assert ctx.watches.list(profile="default") == []

    async def test_an_ambiguous_element_is_refused_by_itself(
        self, tools, tmp_path
    ):
        """The refusal is per element, so an ambiguous name keeps its own two
        watches and nothing else's."""
        ctx = make_ctx(tmp_path)
        for name in ("sample_a.log", "sample_b.log", "qc.log"):
            (tmp_path / name).write_text("x")
        await call(
            tools,
            "watch_log",
            ctx,
            paths=[str(tmp_path / n) for n in
                   ("sample_a.log", "sample_b.log", "qc.log")],
        )
        result = await call(tools, "unwatch", ctx, targets=["sample", "qc.log"])
        assert "matches several watches" in result
        left = {Path(w.target).name for w in ctx.watches.list(profile="default")}
        assert left == {"sample_a.log", "sample_b.log"}

    async def test_two_names_for_one_watch_drop_it_once(self, tools, tmp_path):
        """Resolved against one snapshot: the second name is not a phantom
        miss just because the first already removed the row."""
        ctx = make_ctx(tmp_path)
        (tmp_path / "run.log").write_text("x")
        await call(
            tools, "watch_log", ctx, paths=[str(tmp_path / "run.log")],
            label="sniffles",
        )
        result = await call(
            tools, "unwatch", ctx, targets=["sniffles", "run.log"]
        )
        assert "No watch matches" not in result
        assert ctx.watches.list(profile="default") == []

    async def test_a_bare_string_is_read_as_one_target(self, tools, tmp_path):
        ctx = make_ctx(tmp_path)
        (tmp_path / "run.log").write_text("x")
        await call(tools, "watch_log", ctx, paths=[str(tmp_path / "run.log")])
        await call(tools, "unwatch", ctx, targets="run.log")
        assert ctx.watches.list(profile="default") == []


# --------------------------------------------------------- integration tests

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from hpca.agent.graph import build_graph, run_turn  # noqa: E402
from hpca.agent.prompts import orchestrator_system_prompt  # noqa: E402
from hpca.agent.tools import ToolRegistry  # noqa: E402
from hpca.config import LLMSettings  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402

from tests.live_backend import LIVE_KEY, LIVE_MODEL, LIVE_URL, integration  # noqa: E402

SQUEUE_THREE = (
    SQUEUE_RUNNING
    + SQUEUE_PENDING
    + "27744536|RUNNING|node043|00:12:31|3-23:47:29|qc_run|None\n"
)


def live_llm() -> LLMClient:
    return LLMClient(
        LLMSettings(
            base_url=LIVE_URL,
            model=LIVE_MODEL,
            api_key=LIVE_KEY,
            request_timeout_s=120,
            # these test the shape of a call, not reasoning
            enable_thinking=False,
        )
    )


def counting_watch_tools(calls: list[str]) -> ToolRegistry:
    """The watch tools alone, each handler recording that it ran.

    The count is the assertion. "Are all three watched?" passes just as well
    when the model made three calls, which is the behaviour the array exists
    to replace — so what is measured is how many calls it took, not only what
    ended up in the panel.
    """
    registry = add_watch_tools(ToolRegistry())
    for tool in registry:

        async def counting(args, ctx, *, name=tool.name, inner=tool.handler):
            calls.append(name)
            return await inner(args, ctx)

        tool.handler = counting
    return registry


def live_graph(ctx, calls: list[str], llm: LLMClient):
    return build_graph(
        llm=llm,
        tools=counting_watch_tools(calls),
        checkpointer=InMemorySaver(),
        ctx=ctx,
        # The prompt the app actually ships with the watch tools; the batching
        # sentence lives there, not in the schemas.
        system_prompt_fn=lambda: orchestrator_system_prompt(watch_tools=True),
    )


@integration
class TestWatchToolsLive:
    """Does the live backend actually fill the arrays in one call?

    A small model handed a list-typed argument is the measured weak spot (the
    ``middleware`` module docstring: a bare string where a list belonged, 6 of
    6 generations). These tools are only cheaper than the one-target versions
    if the model batches, so that is what is checked here rather than in a
    unit test where the arguments are written by hand.
    """

    async def test_three_job_ids_ride_one_call(self, tmp_path):
        run = FakeRun(squeue=SQUEUE_THREE)
        ctx = make_ctx(tmp_path, run=run)
        calls: list[str] = []
        llm = live_llm()
        try:
            await run_turn(
                live_graph(ctx, calls, llm),
                session_id="live-watch-jobs",
                user_text=(
                    "Jobs 27744534, 27744535 and 27744536 are mine. Pin all "
                    "three to the panel so I can keep an eye on them."
                ),
            )
        finally:
            await llm.close()
        assert calls.count("watch_job") == 1, calls
        assert {w.target for w in ctx.watches.list(profile="default")} == {
            "27744534",
            "27744535",
            "27744536",
        }
        # And the batching is what the cluster sees: one squeue, not three.
        assert [argv[0] for argv in run.calls] == ["squeue"]
        # The live model labels a batch call — "RNA-seq pipeline jobs",
        # "pipeline", "align" — which is one name for three boxes, so the
        # label is dropped and each job keeps its own squeue name. Measured:
        # it sent a label on 4 of 6 batch calls, so this is the common case
        # rather than the corner.
        assert {w.label for w in ctx.watches.list(profile="default")} == {
            "align_run",
            "call_svs",
            "qc_run",
        }

    async def test_three_log_paths_ride_one_call(self, tmp_path):
        ctx = make_ctx(tmp_path)
        names = ("align.log", "call.log", "qc.log")
        for name in names:
            (tmp_path / name).write_text("running\n")
        calls: list[str] = []
        llm = live_llm()
        try:
            await run_turn(
                live_graph(ctx, calls, llm),
                session_id="live-watch-logs",
                user_text=(
                    f"My pipeline writes {tmp_path / 'align.log'}, "
                    f"{tmp_path / 'call.log'} and {tmp_path / 'qc.log'}. "
                    "Pin all three logs to the panel."
                ),
            )
        finally:
            await llm.close()
        assert calls.count("watch_log") == 1, calls
        watched = {Path(w.target).name for w in ctx.watches.list(profile="default")}
        assert watched == set(names)

    async def test_two_names_ride_one_unwatch(self, tmp_path):
        """Seeded directly: this measures the removal call, not the model's
        ability to register three watches first."""
        ctx = make_ctx(tmp_path)
        for name in ("align.log", "call.log", "qc.log"):
            (tmp_path / name).write_text("x")
            ctx.watches.add(
                kind=KIND_LOG,
                target=str(tmp_path / name),
                profile="default",
                session_id="s1",
            )
        calls: list[str] = []
        llm = live_llm()
        try:
            await run_turn(
                live_graph(ctx, calls, llm),
                session_id="live-unwatch",
                user_text=(
                    "align.log and call.log are finished. Stop watching both "
                    "of them; leave qc.log pinned."
                ),
            )
        finally:
            await llm.close()
        assert calls.count("unwatch") == 1, calls
        left = [Path(w.target).name for w in ctx.watches.list(profile="default")]
        assert left == ["qc.log"]
