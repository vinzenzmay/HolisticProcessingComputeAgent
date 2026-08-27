"""Tests for hpca.agent.builtin_tools (§5.1): script tools over the runner/registry."""

import threading
import tracemalloc

import pytest

from hpca.agent import builtin_tools
from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.builtin_tools import script_names, script_path
from hpca.agent.context import ToolContext
from hpca.agent.history import ELISION_SENTINEL, omitted_list
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.paths import PathError
from hpca.runner import ProcessRunner


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield ToolContext(
        workdir=tmp_path,
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
    )
    conn.close()


@pytest.fixture
def tools():
    return default_tool_registry()


async def call(tools, tool_name, ctx, **kwargs):
    tool = tools.get(tool_name)
    args = tool.params.model_validate(kwargs)
    return await tool.handler(args, ctx)


class TestCreateScript:
    async def test_valid_bash_script_written_and_registered(self, tools, ctx):
        result = await call(
            tools,
            "create_script",
            ctx,
            kind="bash",
            name="hello_sh",
            content_lines=["echo hello"],
        )
        path = script_path("hello_sh", ctx)
        # strict mode is injected, then the model's lines
        assert path.read_text() == "set -euo pipefail\necho hello\n"
        assert "hello_sh" in result
        assert "ok" in result.lower()

    async def test_invalid_bash_script_is_not_kept(self, tools, ctx):
        result = await call(
            tools,
            "create_script",
            ctx,
            kind="bash",
            name="bad_sh",
            content_lines=["if [ 1 -eq 1 ]; then", "echo unclosed"],
        )
        assert "syntax" in result.lower()
        assert script_path("bad_sh", ctx) is None

    async def test_invalid_python_reports_error_text(self, tools, ctx):
        result = await call(
            tools,
            "create_script",
            ctx,
            kind="python",
            name="bad_py",
            content_lines=["def broken(:", "    pass"],
        )
        assert "SyntaxError" in result

    async def test_a_name_already_taken_raises(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="x", content_lines=["echo 1"],
        )
        with pytest.raises(PathError, match="already exists"):
            await call(
                tools, "create_script", ctx,
                kind="bash", name="x", content_lines=["echo 2"],
            )

    async def test_a_name_whose_script_is_gone_can_be_reused(self, tools, ctx):
        # A name is free exactly when no file answers to it, so deleting the
        # script frees the name — there is no separate registration to go stale.
        await call(
            tools, "create_script", ctx,
            kind="bash", name="x", content_lines=["echo 1"],
        )
        script_path("x", ctx).unlink()
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="x", content_lines=["echo 2"],
        )
        assert "echo 2" in script_path("x", ctx).read_text()
        assert "syntax check ok" in result

    async def test_content_carrying_the_elision_marker_is_refused(self, tools, ctx):
        """The model handing its own elided record back as script content is
        the observed failure the marker exists to catch (see
        hpca.agent.history): writing it puts the placeholder on disk in place
        of the script, and every rewrite after that shrinks the file further."""
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="x",
            content_lines=["echo start", omitted_list(["echo body"] * 40)],
        )
        assert "NOT created" in result
        assert "line 2" in result  # which line, without reprinting it
        # The refusal must not quote the placeholder back. Measured on the live
        # 27B: a refusal carrying the marker put it into the context, the model
        # composed its next call out of the refusal it had just read, and the
        # same call was refused seventeen times until the decision budget died.
        assert ELISION_SENTINEL not in result
        # Nothing may reach disk, and the name must stay free for the retry.
        assert not (ctx.scripts_dir / "x.sh").exists()
        assert script_path("x", ctx) is None

    async def test_the_legacy_elision_wording_is_refused_too(self, tools, ctx):
        """Pre-0.23.3 sessions and the files already written from one carry the
        old marker, so the guard has to know that wording as well."""
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="x",
            content_lines=["echo start", "... 22 more lines elided ..."],
        )
        assert "NOT created" in result
        assert not (ctx.scripts_dir / "x.sh").exists()

    async def test_the_refusal_names_the_offending_line(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="x",
            content_lines=["echo one", "echo two", "... 7 more lines elided ..."],
        )
        assert "line 3 of content_lines" in result

    async def test_a_script_that_merely_talks_about_elision_is_written(
        self, tools, ctx
    ):
        """The guard looks for the marker itself, not for the words around it —
        an ordinary script saying 'lines' and a number is not a placeholder."""
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="x",
            content_lines=["echo 'skipping 22 more lines'", "echo done"],
        )
        assert "ok" in result.lower()
        assert "skipping 22 more lines" in script_path("x", ctx).read_text()


class TestScriptNameCarryingItsOwnSuffix:
    """The observed loop: the model writes the name the way every script it has
    ever read is written — `validate_hg002.sh` — the kind's suffix is appended
    to that, and the file lands as `validate_hg002.sh.sh`. Self-consistent, so
    nothing refuses it and the doubled name comes straight back out of the
    result the model then quotes into run_bash."""

    async def test_a_bash_name_ending_in_sh_is_not_doubled(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="validate_hg002.sh", content_lines=["echo hi"],
        )
        assert (ctx.scripts_dir / "validate_hg002.sh").is_file()
        assert not (ctx.scripts_dir / "validate_hg002.sh.sh").exists()
        # and the result names the script the way the model must ask for it
        assert "validate_hg002.sh.sh" not in result
        assert "{validate_hg002}" in result

    async def test_a_python_name_ending_in_py_is_not_doubled(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", name="report.py", content_lines=["print(1)"],
        )
        assert (ctx.scripts_dir / "report.py").is_file()
        assert not (ctx.scripts_dir / "report.py.py").exists()

    async def test_a_suffix_of_the_other_kind_is_stripped_too(self, tools, ctx):
        # Same slip about the same field; the kind decides the suffix.
        await call(
            tools, "create_script", ctx,
            kind="bash", name="convert.py", content_lines=["echo hi"],
        )
        assert (ctx.scripts_dir / "convert.sh").is_file()

    async def test_a_dot_that_is_not_a_script_suffix_is_kept(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="align.v2", content_lines=["echo hi"],
        )
        assert (ctx.scripts_dir / "align.v2.sh").is_file()

    async def test_the_suffixed_name_still_resolves_afterwards(self, tools, ctx):
        # The model that wrote `validate_hg002.sh` once will write it again, in
        # `{braces}` and in start_background_script — it has to keep reaching
        # the script it made.
        await call(
            tools, "create_script", ctx,
            kind="bash", name="validate_hg002.sh", content_lines=["echo hi"],
        )
        assert script_path("validate_hg002.sh", ctx) == (
            ctx.scripts_dir / "validate_hg002.sh"
        )
        assert script_path("validate_hg002", ctx) == (
            ctx.scripts_dir / "validate_hg002.sh"
        )

    async def test_the_name_is_taken_under_either_spelling(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="validate_hg002", content_lines=["echo 1"],
        )
        with pytest.raises(PathError, match="already exists"):
            await call(
                tools, "create_script", ctx,
                kind="bash", name="validate_hg002.sh", content_lines=["echo 2"],
            )

    async def test_a_doubled_script_from_an_older_session_still_resolves(
        self, tools, ctx
    ):
        # App dirs already hold `<name>.sh.sh` files written before this, and
        # the only name they ever answered to is the doubled one.
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        legacy = ctx.scripts_dir / "old_job.sh.sh"
        legacy.write_text("echo legacy\n")
        assert script_path("old_job.sh", ctx) == legacy
        assert "old_job.sh" in script_names(ctx)


class TestReadFile:
    async def test_reads_registered_file(self, tools, ctx, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\n")
        result = await call(tools, "read_file", ctx, path=str(f))
        assert "line1" in result and "line2" in result

    async def test_long_file_returns_contiguous_window_with_hint(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1000)))
        result = await call(tools, "read_file", ctx, path=str(f), max_lines=20)
        # Contiguous head window, no middle omission — the model pages instead.
        assert "line0" in result and "line19" in result
        assert "line20" not in result
        assert "line999" not in result
        assert "file continues: lines 21-1000" in result
        assert "start_line=21" in result

    async def test_paging_window_with_start_line(self, tools, ctx, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1000)))
        result = await call(
            tools, "read_file", ctx, path=str(f),
            start_line=501, max_lines=10,
        )
        assert result.splitlines()[0] == "line500"
        assert "line509" in result
        assert "file continues: lines 511-1000" in result
        assert "start_line=511" in result

    async def test_start_line_beyond_eof_is_instructive(self, tools, ctx, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(50)))
        result = await call(
            tools, "read_file", ctx, path=str(f), start_line=100
        )
        assert "50" in result and "start_line" in result

    async def test_start_line_with_small_remainder_no_hint(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(100)))
        result = await call(
            tools, "read_file", ctx, path=str(f),
            start_line=91, max_lines=20,
        )
        assert result.splitlines()[0] == "line90"
        assert "line99" in result
        assert "file continues" not in result

    async def test_a_path_that_is_not_there_says_so(self, tools, ctx, tmp_path):
        result = await call(tools, "read_file", ctx, path="nope")
        assert "Nothing at" in result
        assert str(tmp_path / "nope") in result  # resolved, not echoed back

    async def test_the_body_is_the_whole_result(self, tools, ctx, tmp_path):
        """Nothing is appended to a read. The body is what the model copies
        edit_file's old_lines out of, so a trailing note would be a line of the
        file as far as the next call can tell."""
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\n")
        result = await call(tools, "read_file", ctx, path=str(f))
        assert result == "line1\nline2"

    async def test_a_read_carries_no_note(self, tools, ctx, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\n")
        assert await call(tools, "read_file", ctx, path=str(f)) == "line1"

    async def test_a_nested_path_reads_the_same(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        (d / "a.txt").write_text("inner\n")
        result = await call(
            tools, "read_file", ctx, path=str(d / "a.txt")
        )
        assert result == "inner"

    async def test_literal_path_to_a_directory_is_listed(self, tools, ctx, tmp_path):
        d = tmp_path / "tools"
        d.mkdir()
        (d / "run.sh").write_text("echo hi\n")
        result = await call(tools, "read_file", ctx, path=str(d))
        assert "is a directory" in result
        assert "run.sh" in result

    async def test_a_relative_path_reads_from_the_workdir(
        self, tools, ctx, tmp_path
    ):
        """What the registry answered with UnknownKeyError. The model wrote a
        path it could see; there is no reason for that to be an error."""
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "out.txt").write_text("done\n")
        assert await call(tools, "read_file", ctx, path="results/out.txt") == "done"

    async def test_a_missing_file_comes_back_as_a_sentence(
        self, tools, ctx, tmp_path
    ):
        """Not a bare FileNotFoundError: the read has to come back as something
        the model can act on."""
        result = await call(
            tools, "read_file", ctx, path=str(tmp_path / "results" / "run.log")
        )
        assert "Nothing at" in result
        assert "Check the spelling" in result

    async def test_a_directory_is_listed_not_an_error(self, tools, ctx, tmp_path):
        d = tmp_path / "tools"
        d.mkdir()
        (d / "run.sh").write_text("echo hi\n")
        (d / "sub").mkdir()
        result = await call(tools, "read_file", ctx, path=str(d))
        assert "is a directory" in result
        assert "run.sh" in result
        assert "sub/" in result  # trailing slash marks nested directories
        assert "full path" in result

    async def test_a_deep_path_is_read_for_real(self, tools, ctx, tmp_path):
        d = tmp_path / "pkg"
        (d / "src").mkdir(parents=True)
        (d / "src" / "main.py").write_text("print('hello')\n")
        result = await call(
            tools, "read_file", ctx, path=str(d / "src/main.py")
        )
        assert "hello" in result

    async def test_a_missing_file_inside_a_directory_is_a_useful_error(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "pkg"
        d.mkdir()
        result = await call(tools, "read_file", ctx, path=str(d / "nope.txt"))
        assert "Nothing at" in result

class TestStartScript:
    async def test_runs_and_names_its_logs(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="greeter",
            content_lines=['echo "hi from script"'],
        )
        result = await call(
            tools, "start_background_script", ctx, name="greeter"
        )
        assert "pid" in result.lower()
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        # the log paths are in the result outright, ready for read_file
        assert str(record.stdout_path) in result
        assert "hi from script" in record.stdout_path.read_text()

    async def test_passes_args(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="argsy", content_lines=['echo "arg1=$1"'],
        )
        await call(
            tools, "start_background_script", ctx,
            name="argsy", args="banana",
        )
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        assert "arg1=banana" in record.stdout_path.read_text()

    async def test_python_script_started_with_python(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", name="pyhello",
            content_lines=['print("python says hi")'],
        )
        await call(tools, "start_background_script", ctx, name="pyhello")
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        assert record.state == "finished"
        assert "python says hi" in record.stdout_path.read_text()


class TestToolSuiteShape:
    def test_default_registry_has_expected_tools(self, tools):
        assert set(tools.names()) == {
            "create_script",
            "read_file",
            "run_bash",
            "start_background_script",
            "list_scripts",
        }

    def test_no_tool_is_unconditionally_destructive(self, tools):
        # run_bash gates conditionally (see TestRunBashDestructiveGate); no
        # builtin is destructive on every call.
        assert all(not t.destructive for t in tools)


class TestCreateScriptSaysWhichLanguagesItHas:
    """The two places a model is told, at the two moments it can still act on
    it. Asked for a snakemake workflow, the live model called create_script
    with kind=bash — reasonably, given a one-line description that reads like a
    superset and a `kind` field labelled only "Script language". No kind would
    have worked: a Snakefile is neither bash nor python, so the refusal had to
    come from the description, before the call."""

    def test_the_tool_description_closes_the_pair(self, tools):
        description = next(t for t in tools if t.name == "create_script").description
        assert "ONLY" in description
        assert "Snakefile" in description
        # and says where the file that is not one of the two goes instead
        assert "create_file" in description

    def test_and_so_does_the_kind_field(self):
        from hpca.agent.builtin_tools import CreateScriptParams

        kind = CreateScriptParams.model_fields["kind"].description
        assert "only two" in kind
        assert "Snakefile" in kind
        assert "create_file" in kind

    def test_the_tool_still_takes_the_two_it_does_have(self, tools):
        from hpca.agent.builtin_tools import CreateScriptParams

        assert CreateScriptParams(
            kind="python", name="x", content_lines=["print(1)"]
        ).kind == "python"
        with pytest.raises(Exception):
            CreateScriptParams(kind="snakemake", name="x", content_lines=["rule all:"])


class TestSingleLineScriptGate:
    """The live model mangles newline escapes in JSON strings under guided
    decoding, so content arrives as an array of lines; a "script" that is
    only a shebang line would still pass `bash -n` vacuously — reject it."""

    async def test_shebang_only_script_rejected(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="oneliner",
            content_lines=["#!/bin/bash for i in 1 2 3; do echo $i; done"],
        )
        assert "NOT created" in result
        assert "oneliner" not in script_names(ctx)

    async def test_single_command_line_without_shebang_ok(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="shortie", content_lines=["echo hi"],
        )
        assert "ok" in result.lower()

    async def test_multiline_with_shebang_ok(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="proper",
            content_lines=["#!/bin/bash", "for i in 1 2 3; do echo $i; done"],
        )
        assert "ok" in result.lower()


class TestRunBash:
    """run_bash writes, syntax-checks and runs a one-shot script in a single
    call — the look-around workhorse, one round instead of create+run's two."""

    async def test_writes_and_runs_in_one_call(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["echo hello", "echo world"],
        )
        assert "hello" in result and "world" in result
        assert "exit 0" in result
        # exactly one process ran; no separate create step
        assert len(ctx.runner.list()) == 1

    async def test_a_find_search_comes_back(self, tools, ctx, tmp_path):
        planted = tmp_path / "refs" / "GRCh38.fa"
        planted.parent.mkdir(parents=True)
        planted.write_text(">chr1\n")
        result = await call(
            tools, "run_bash", ctx,
            content_lines=[f"find {tmp_path} -iname '*GRCh38*' 2>/dev/null"],
        )
        assert str(planted) in result

    async def test_syntax_error_is_not_run(self, tools, ctx):
        before = len(ctx.runner.list())
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["if [ 1 -eq 1 ]; then", "echo unclosed"],
        )
        assert "NOT run" in result and "syntax" in result.lower()
        assert len(ctx.runner.list()) == before  # nothing executed

    async def test_failure_returns_stderr_and_says_to_fix(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["echo oops >&2", "exit 2"],
        )
        assert "FAILED, exit 2" in result
        assert "oops" in result
        assert "run_bash again" in result

    async def test_timeout_kills_and_explains(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx, content_lines=["sleep 30"], timeout_s=1
        )
        assert "TIMED OUT" in result

    async def test_long_output_is_bounded(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx, content_lines=["seq 1 500"]
        )
        assert len(result) < 6000
        assert "500" in result  # the tail — what a search prints last
        assert "omitted" in result

    async def test_shebang_only_is_rejected(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx, content_lines=["#!/bin/bash echo hi"]
        )
        assert "NOT run" in result

    async def test_no_output_is_stated_not_silent(self, tools, ctx):
        result = await call(tools, "run_bash", ctx, content_lines=["true"])
        assert "(no output)" in result

    async def test_throwaway_scripts_are_not_registered(self, tools, ctx):
        await call(tools, "run_bash", ctx, content_lines=["echo x"])
        await call(tools, "run_bash", ctx, content_lines=["echo y"])
        # nothing clutters the registry the model reasons over: the script is
        # throwaway, and output that fit needs no log key
        assert script_names(ctx) == []
        assert ctx.runner.list()[0].state == "finished"

    async def test_clipped_output_names_a_log_that_can_be_read(self, tools, ctx):
        result = await call(tools, "run_bash", ctx, content_lines=["seq 1 500"])
        assert "omitted" in result and "read_file" in result
        # the path it points at must actually hold the rest — the whole point
        # of naming it is that the follow-up read works
        record = ctx.runner.list()[0]
        assert str(record.stdout_path) in result
        assert "1\n" in record.stdout_path.read_text()


class TestLargeOutputIsNotReadWhole:
    """Keeping 4000 characters must not cost reading the whole log.

    The UI and the agent share one event loop, so a synchronous read here is a
    freeze the user sits through, and on a cluster node the log is on NFS.
    Measured before the fix, on a 104 MB log: 170ms of blocked loop and 269 MB
    of resident memory, to keep 4000 characters of it.
    """

    LOG_MB = 32

    def _big_log(self, tmp_path):
        path = tmp_path / "big.txt"
        chunk = ("y" * 79 + "\n") * 13_000  # ~1 MB
        with path.open("w") as handle:
            for _ in range(self.LOG_MB):
                handle.write(chunk)
        return path

    async def test_a_large_log_is_not_pulled_into_memory(self, tools, ctx, tmp_path):
        """Python's own allocation peak across the call.

        A whole-file read cannot hide from it — the text has to exist — while
        a window read is invisible next to the threshold. Asserted on memory
        rather than on elapsed time because a warm page cache makes the second
        one say nothing.
        """
        big = self._big_log(tmp_path)
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            result = await call(
                tools, "run_bash", ctx,
                content_lines=[f"cat {big}", "echo THE_VERY_LAST_LINE"],
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert "ran (exit 0)" in result
        assert peak < 4 * 1024 * 1024, (
            f"{peak / 1e6:.1f} MB allocated to summarise a {self.LOG_MB} MB log"
        )

    async def test_the_answer_is_still_the_end_of_the_log_and_says_it_was_cut(
        self, tools, ctx, tmp_path
    ):
        big = self._big_log(tmp_path)
        result = await call(
            tools, "run_bash", ctx,
            content_lines=[f"cat {big}", "echo THE_VERY_LAST_LINE"],
        )
        assert "THE_VERY_LAST_LINE" in result  # the tail, not the head
        assert "omitted" in result
        # and the pointer at the rest still has to be the file that holds it
        record = ctx.runner.list()[0]
        assert str(record.stdout_path) in result
        assert len(result) < 6000

    async def test_a_failure_at_the_top_of_a_huge_stderr_is_still_quoted(
        self, tools, ctx, tmp_path
    ):
        """Why the citation scan reads both ends and not just the tail.

        A script without `set -e` fails on line 1 and keeps going; bash's
        message about line 1 is then buried under everything printed after
        it, and a model told 'line 1' that cannot see line 1 can only rewrite
        the whole script.
        """
        big = self._big_log(tmp_path)
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["not_a_command_xyz", f"cat {big} >&2"],
        )
        assert "> 1 | not_a_command_xyz" in result

    async def test_the_reads_do_not_run_on_the_event_loop(self, tools, ctx, monkeypatch):
        """Bounded is not enough: one NFS round trip on the loop is a stutter.

        The assertion is the thread the read ran on, which is the property
        being claimed, rather than which call was used to get off the loop.
        """
        loop_thread = threading.get_ident()
        threads: list[int] = []
        real = builtin_tools.read_tail

        def spy(path, max_bytes):
            threads.append(threading.get_ident())
            return real(path, max_bytes)

        monkeypatch.setattr(builtin_tools, "read_tail", spy)
        await call(tools, "run_bash", ctx, content_lines=["echo hi"])
        assert threads, "run_bash did not read its logs at all"
        assert loop_thread not in threads


class TestRunBashFailingLines:
    """A bash message names a line number; the script is a throwaway file the
    model never reads back, so the line itself has to come with it."""

    async def test_the_line_bash_points_at_is_quoted_back(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["echo one", "not_a_command_xyz"],
        )
        assert "FAILED" in result
        assert "> 2 | not_a_command_xyz" in result

    async def test_neighbouring_lines_come_with_it(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["echo one", "not_a_command_xyz"],
        )
        assert "1 | echo one" in result

    async def test_a_mid_script_failure_is_quoted_though_the_run_exits_zero(
        self, tools, ctx
    ):
        # run_bash is lenient by design, so a failed command in the middle
        # leaves the exit code to whatever ran last. Without the quote, "exit
        # 0" is all the model would see.
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["not_a_command_xyz", "echo carried on"],
        )
        assert "exit 0" in result
        assert "> 1 | not_a_command_xyz" in result

    async def test_a_successful_run_quotes_nothing(self, tools, ctx):
        result = await call(tools, "run_bash", ctx, content_lines=["echo fine"])
        assert "|" not in result

    async def test_a_failure_with_no_line_reference_is_unchanged(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx, content_lines=["echo oops >&2", "exit 2"]
        )
        assert "FAILED, exit 2" in result
        assert "point at" not in result

    async def test_another_program_numbering_its_own_input_is_not_quoted(
        self, tools, ctx
    ):
        # "line 1" here is python's, about its own -c source, not the script's.
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["python3 -c 'import nosuchmodule_xyz'"],
        )
        assert "FAILED" in result
        assert "|" not in result

    async def test_the_quoted_line_is_the_expanded_one(self, tools, ctx, tmp_path):
        # What ran is what is quoted: `{name}` resolved, as the script has it.
        await call(
            tools, "create_script", ctx,
            kind="bash", name="data", content_lines=["echo hi"],
        )
        script = ctx.scripts_dir / "data.sh"
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["not_a_command_xyz {data}"],
        )
        assert f"not_a_command_xyz {script}" in result


class TestRunBashScriptInterpolation:
    """`{name}` in a run_bash line expands to a kept script's path, which is how
    a kept script is run now that run_script is gone. A name is the only thing
    that expands: a data file is written as the path it is."""

    async def test_a_data_path_is_written_out_not_braced(
        self, tools, ctx, tmp_path
    ):
        target = tmp_path / "reads.bam"
        target.write_text("BAMDATA\n")
        result = await call(
            tools, "run_bash", ctx, content_lines=[f"cat {target}"]
        )
        assert "BAMDATA" in result

    async def test_a_kept_script_is_run_by_name(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", name="qc",
            content_lines=["import sys", "print('qc ran', sys.argv[1])"],
        )
        result = await call(
            tools, "run_bash", ctx, content_lines=["python3 {qc} --strict"]
        )
        assert "qc ran --strict" in result
        assert "exit 0" in result

    async def test_awk_body_is_left_alone(self, tools, ctx):
        # {print $1} is not a key reference; substituting it would break the
        # single most common look-around idiom.
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["echo 'a b' | awk '{print $1}'"],
        )
        assert "exit 0" in result
        assert result.strip().endswith("a")

    async def test_shell_variables_are_left_alone(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["X=hello", "echo ${X}"],
        )
        assert "hello" in result

    async def test_unresolved_reference_is_named_when_the_run_fails(
        self, tools, ctx, tmp_path
    ):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="reads", content_lines=["echo hi"],
        )
        result = await call(tools, "run_bash", ctx, content_lines=["cat {readz}"])
        assert "{readz}" in result
        assert "reads" in result  # the scripts that do exist, for the retry

    async def test_a_successful_run_is_not_nagged_about_braces(self, tools, ctx):
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["echo 'a b' | awk '{print $2}'"],
        )
        assert "did not match" not in result


class TestRunBashDestructiveGate:
    """run_bash trips the §5.3 destructive gate when — and only when — its
    script's leading command would destroy something. This lets plan/auto mode
    run benign look-around unattended while still pausing on `rm -rf`."""

    def gates(self, tools, content_lines):
        tool = tools.get("run_bash")
        args = tool.params.model_validate({"content_lines": content_lines})
        return tool.gates(args, None)

    @pytest.mark.parametrize(
        "lines",
        [
            ["ls -la"],
            ["find / -name '*.rm' 2>/dev/null"],  # rm only in a pattern
            ["# rm this later"],  # rm only in a comment
            ["samtools view x.bam | head"],
            ["echo 'rm is dangerous'"],  # rm inside a string, not a command
            ["grep -r rm ."],  # rm is an argument, not the command
        ],
    )
    def test_benign_look_around_does_not_gate(self, tools, lines):
        assert not self.gates(tools, lines)

    @pytest.mark.parametrize(
        "lines",
        [
            ["rm -rf /data/x"],
            ["dd if=/dev/zero of=/data/x"],
            ["/bin/rm x"],  # absolute path resolves to rm
            ["scancel 123"],
            ["mkfs.ext4 /dev/sdb1"],
            ["sudo shred -u secret"],  # wrapper skipped -> shred
            ["find . -name '*.tmp' | xargs rm"],  # destructive tail of a pipe
            ["echo hi", "truncate -s 0 log"],  # a later line
            ["DEBUG=1 chmod 000 file"],  # leading assignment skipped -> chmod
        ],
    )
    def test_destructive_commands_gate(self, tools, lines):
        assert self.gates(tools, lines)

    def test_describe_names_the_flagged_command(self, tools):
        tool = tools.get("run_bash")
        args = tool.params.model_validate({"content_lines": ["rm -rf x", "dd if=a"]})
        details = tool.describe_call(args, None)
        assert "rm" in details and "dd" in details


class TestBashFailFast:
    """A bash execution script that runs a failing command must report the
    failure, not march on to a success echo and exit 0 (this bit a real
    sniffles run: the process showed 'finished (exit 0)' while it had failed)."""

    async def test_failure_before_a_success_echo_is_not_masked(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="run_thing",
            content_lines=[
                "false",                       # the "tool" fails
                'echo "Done. Output: out.vcf"',  # would otherwise mask it
            ],
        )
        record = await ctx.runner.start(
            ["bash", str(script_path("run_thing", ctx))], name="run_thing"
        )
        record = await ctx.runner.wait(record.pid)
        assert record.state == "failed"
        assert record.exit_code != 0
        # the success line never ran
        assert "Done" not in record.stdout_path.read_text()

    async def test_strict_mode_is_injected_after_the_shebang(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="shebanged",
            content_lines=["#!/bin/bash", "echo hi"],
        )
        text = script_path("shebanged", ctx).read_text()
        lines = text.splitlines()
        assert lines[0] == "#!/bin/bash"
        assert lines[1] == "set -euo pipefail"

    async def test_strict_mode_prepended_when_no_shebang(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="bare", content_lines=["echo hi"],
        )
        assert script_path("bare", ctx).read_text().startswith(
            "set -euo pipefail\n"
        )

    async def test_the_models_own_set_e_is_respected(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", name="own",
            content_lines=["set -e", "echo hi"],
        )
        text = script_path("own", ctx).read_text()
        assert text.count("set -e") == 1  # not doubled

    async def test_the_success_note_warns_against_unconditional_done(
        self, tools, ctx
    ):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", name="noted", content_lines=["echo hi"],
        )
        assert "fail-fast" in result

    async def test_non_bash_scripts_are_not_touched(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", name="py", content_lines=["print('hi')"],
        )
        text = script_path("py", ctx).read_text()
        assert "set -euo pipefail" not in text  # python fails on exception anyway

    async def test_run_bash_stays_lenient_for_exploration(self, tools, ctx):
        # `command -v missing` returns non-zero; exploration must still get the
        # output, not abort — run_bash reports the exit code rather than
        # fail-fasting like an execution script
        result = await call(
            tools, "run_bash", ctx,
            content_lines=[
                "command -v this_tool_does_not_exist_xyz || echo NOTFOUND",
                "echo still-running",
            ],
        )
        assert "still-running" in result  # did not abort at the missing tool
        assert "exit 0" in result


class TestRunBashLengthLimit:
    """run_bash is for looking around, not for writing files.

    Measured live: asked for a design document with only run_bash available,
    the model puts the whole thing into ONE array element of 5-8k characters —
    the long-string case the array-of-lines design exists to avoid — and the
    §5.2 code-vs-docs gate never sees it, because run_bash only runs `bash -n`.
    The limit is what routes that work to create_file/create_script instead.
    """

    def params(self, lines):
        from hpca.agent.builtin_tools import RunBashParams

        return RunBashParams.model_validate({"content_lines": lines})

    def test_a_look_around_script_is_fine(self):
        lines = ["find /data -maxdepth 3 -iname '*.bam' 2>/dev/null | head -20"] * 20
        assert self.params(lines).content_lines == lines

    def test_a_script_over_the_limit_is_refused_before_it_can_run(self):
        import pytest as _pytest
        from pydantic import ValidationError

        from hpca.agent.builtin_tools import RUN_SCRIPT_MAX_CHARS

        with _pytest.raises(ValidationError) as caught:
            self.params(["echo " + "x" * 200] * 40)
        message = str(caught.value)
        assert str(RUN_SCRIPT_MAX_CHARS) in message
        assert "create_file" in message and "create_script" in message

    def test_the_refusal_says_how_to_write_a_file_too_long_for_one_call(self):
        import pytest as _pytest
        from pydantic import ValidationError

        with _pytest.raises(ValidationError) as caught:
            self.params(["x" * 3000])
        assert "edit_file" in str(caught.value)

    def test_one_pathological_element_counts_the_same_as_many(self):
        # The live failure shape: the whole document as a single string.
        import pytest as _pytest
        from pydantic import ValidationError

        with _pytest.raises(ValidationError):
            self.params(["cat << 'EOF' > specs.md\n" + "line\n" * 1000])
