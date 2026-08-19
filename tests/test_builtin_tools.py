"""Tests for hpca.agent.builtin_tools (§5.1): script tools over the runner/registry."""

import pytest

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.history import ELISION_SENTINEL, omitted_list
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.registry import PathRegistry, RegistryError, UnknownKeyError
from hpca.runner import ProcessRunner


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
    )
    conn.close()


@pytest.fixture
def tools():
    return default_tool_registry()


async def call(tools, name, ctx, **kwargs):
    tool = tools.get(name)
    args = tool.params.model_validate(kwargs)
    return await tool.handler(args, ctx)


class TestCreateScript:
    async def test_valid_bash_script_written_and_registered(self, tools, ctx):
        result = await call(
            tools,
            "create_script",
            ctx,
            kind="bash",
            registry_key="hello_sh",
            content_lines=["echo hello"],
        )
        path = ctx.registry.resolve("hello_sh")
        # strict mode is injected, then the model's lines
        assert path.read_text() == "set -euo pipefail\necho hello\n"
        assert "hello_sh" in result
        assert "ok" in result.lower()

    async def test_invalid_bash_script_not_registered(self, tools, ctx):
        result = await call(
            tools,
            "create_script",
            ctx,
            kind="bash",
            registry_key="bad_sh",
            content_lines=["if [ 1 -eq 1 ]; then", "echo unclosed"],
        )
        assert "syntax" in result.lower()
        with pytest.raises(UnknownKeyError):
            ctx.registry.resolve("bad_sh")

    async def test_invalid_python_reports_error_text(self, tools, ctx):
        result = await call(
            tools,
            "create_script",
            ctx,
            kind="python",
            registry_key="bad_py",
            content_lines=["def broken(:", "    pass"],
        )
        assert "SyntaxError" in result

    async def test_duplicate_key_raises(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x", content_lines=["echo 1"],
        )
        with pytest.raises(RegistryError, match="already exists"):
            await call(
                tools, "create_script", ctx,
                kind="bash", registry_key="x", content_lines=["echo 2"],
            )

    async def test_key_naming_another_live_file_raises(self, tools, ctx, tmp_path):
        other = tmp_path / "data.txt"
        other.write_text("payload\n")
        ctx.registry.register("x", other)
        with pytest.raises(RegistryError, match="already names"):
            await call(
                tools, "create_script", ctx,
                kind="bash", registry_key="x", content_lines=["echo 1"],
            )
        assert ctx.registry.resolve("x") == other

    async def test_key_whose_script_is_gone_can_be_recreated(self, tools, ctx):
        # A key naming a file that no longer exists names nothing. Refusing it
        # would burn the key for the session, the trap Registry.register fixed.
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x", content_lines=["echo 1"],
        )
        ctx.registry.resolve("x").unlink()
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x", content_lines=["echo 2"],
        )
        assert "echo 2" in ctx.registry.resolve("x").read_text()
        assert "stale" not in result  # same path back, so nothing was repointed

    async def test_repointing_a_dead_key_is_reported(self, tools, ctx, tmp_path):
        ctx.registry.register("x", tmp_path / "never_written.sh")
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x", content_lines=["echo 1"],
        )
        assert "never_written.sh" in result
        assert ctx.registry.resolve("x") == ctx.scripts_dir / "x.sh"

    async def test_content_carrying_the_elision_marker_is_refused(self, tools, ctx):
        """The model handing its own elided record back as script content is
        the observed failure the marker exists to catch (see
        hpca.agent.history): writing it puts the placeholder on disk in place
        of the script, and every rewrite after that shrinks the file further."""
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x",
            content_lines=["echo start", omitted_list(["echo body"] * 40)],
        )
        assert "NOT created" in result
        assert "line 2" in result  # which line, without reprinting it
        # The refusal must not quote the placeholder back. Measured on the live
        # 27B: a refusal carrying the marker put it into the context, the model
        # composed its next call out of the refusal it had just read, and the
        # same call was refused seventeen times until the decision budget died.
        assert ELISION_SENTINEL not in result
        # Nothing may reach disk, and the key must stay free for the retry.
        assert not (ctx.scripts_dir / "x.sh").exists()
        with pytest.raises(UnknownKeyError):
            ctx.registry.resolve("x")

    async def test_the_legacy_elision_wording_is_refused_too(self, tools, ctx):
        """Pre-0.23.3 sessions and the files already written from one carry the
        old marker, so the guard has to know that wording as well."""
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x",
            content_lines=["echo start", "... 22 more lines elided ..."],
        )
        assert "NOT created" in result
        assert not (ctx.scripts_dir / "x.sh").exists()

    async def test_the_refusal_names_the_offending_line(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="x",
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
            kind="bash", registry_key="x",
            content_lines=["echo 'skipping 22 more lines'", "echo done"],
        )
        assert "ok" in result.lower()
        assert "skipping 22 more lines" in ctx.registry.resolve("x").read_text()


class TestReadFile:
    async def test_reads_registered_file(self, tools, ctx, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\n")
        ctx.registry.register("data", f)
        result = await call(tools, "read_file", ctx, registry_key="data")
        assert "line1" in result and "line2" in result

    async def test_long_file_returns_contiguous_window_with_hint(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1000)))
        ctx.registry.register("big", f)
        result = await call(tools, "read_file", ctx, registry_key="big", max_lines=20)
        # Contiguous head window, no middle omission — the model pages instead.
        assert "line0" in result and "line19" in result
        assert "line20" not in result
        assert "line999" not in result
        assert "file continues: lines 21-1000" in result
        assert "start_line=21" in result

    async def test_paging_window_with_start_line(self, tools, ctx, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1000)))
        ctx.registry.register("big", f)
        result = await call(
            tools, "read_file", ctx, registry_key="big",
            start_line=501, max_lines=10,
        )
        assert result.splitlines()[0] == "line500"
        assert "line509" in result
        assert "file continues: lines 511-1000" in result
        assert "start_line=511" in result

    async def test_start_line_beyond_eof_is_instructive(self, tools, ctx, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(50)))
        ctx.registry.register("big", f)
        result = await call(
            tools, "read_file", ctx, registry_key="big", start_line=100
        )
        assert "50" in result and "start_line" in result

    async def test_start_line_with_small_remainder_no_hint(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(100)))
        ctx.registry.register("big", f)
        result = await call(
            tools, "read_file", ctx, registry_key="big",
            start_line=91, max_lines=20,
        )
        assert result.splitlines()[0] == "line90"
        assert "line99" in result
        assert "file continues" not in result

    async def test_unknown_key_raises_with_available(self, tools, ctx, tmp_path):
        ctx.registry.register("known", tmp_path / "k.txt")
        with pytest.raises(UnknownKeyError, match="known"):
            await call(tools, "read_file", ctx, registry_key="nope")

    async def test_reads_a_literal_path_and_reports_its_new_key(
        self, tools, ctx, tmp_path
    ):
        """A path the model just saw in `ls` output is usable as it stands —
        no register_path round-trip first (§4.3, path-or-key)."""
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\n")
        result = await call(tools, "read_file", ctx, registry_key=str(f))
        assert "line1" in result and "line2" in result
        assert ctx.registry.resolve("data.txt") == f
        # The note is its own last line: the body above it is what the model
        # copies edit_file's old_lines out of, and must stay untouched.
        assert result.splitlines()[-1] == f"({f} is registered as 'data.txt')"

    async def test_a_plain_key_read_carries_no_note(self, tools, ctx, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\n")
        ctx.registry.register("data", f)
        assert await call(tools, "read_file", ctx, registry_key="data") == "line1"

    async def test_key_and_subpath_read_carries_no_note(self, tools, ctx, tmp_path):
        """The auto-registration a subpath already did is not a path argument,
        so it must not start announcing itself."""
        d = tmp_path / "run"
        d.mkdir()
        (d / "a.txt").write_text("inner\n")
        ctx.registry.register("run_dir", d)
        result = await call(
            tools, "read_file", ctx, registry_key="run_dir", subpath="a.txt"
        )
        assert result == "inner"

    async def test_literal_path_to_a_directory_is_listed(self, tools, ctx, tmp_path):
        d = tmp_path / "tools"
        d.mkdir()
        (d / "run.sh").write_text("echo hi\n")
        result = await call(tools, "read_file", ctx, registry_key=str(d))
        assert "is a directory" in result
        assert "run.sh" in result
        assert "is registered as 'tools'" in result

    async def test_a_relative_path_is_still_an_unknown_key(
        self, tools, ctx, tmp_path
    ):
        ctx.registry.register("known", tmp_path / "k.txt")
        with pytest.raises(UnknownKeyError, match="known"):
            await call(tools, "read_file", ctx, registry_key="results/out.txt")

    async def test_a_key_pointing_at_nothing_says_so(self, tools, ctx, tmp_path):
        """register_path takes a path before it exists, and a registered file
        can be deleted from under its key. Either way the read must come back
        as a sentence the model can act on, not a bare FileNotFoundError."""
        ctx.registry.register("planned", tmp_path / "results" / "run.log")
        result = await call(tools, "read_file", ctx, registry_key="planned")
        assert "Nothing at" in result
        assert "planned" in result

    async def test_directory_key_is_listed_not_an_error(self, tools, ctx, tmp_path):
        d = tmp_path / "tools"
        d.mkdir()
        (d / "run.sh").write_text("echo hi\n")
        (d / "sub").mkdir()
        ctx.registry.register("tools_dir", d)
        result = await call(tools, "read_file", ctx, registry_key="tools_dir")
        assert "is a directory" in result
        assert "run.sh" in result
        assert "sub/" in result  # trailing slash marks nested directories
        assert "subpath" in result

    async def test_subpath_reads_file_inside_registered_directory(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "pkg"
        (d / "src").mkdir(parents=True)
        (d / "src" / "main.py").write_text("print('hello')\n")
        ctx.registry.register("pkg", d)
        result = await call(
            tools, "read_file", ctx, registry_key="pkg", subpath="src/main.py"
        )
        assert "hello" in result

    async def test_subpath_autoregisters_the_file_for_reuse(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "pkg"
        d.mkdir()
        (d / "notes.txt").write_text("body\n")
        ctx.registry.register("pkg", d)
        await call(tools, "read_file", ctx, registry_key="pkg", subpath="notes.txt")
        assert ctx.registry.resolve("notes.txt") == d / "notes.txt"

    async def test_missing_subpath_is_a_useful_error(self, tools, ctx, tmp_path):
        d = tmp_path / "pkg"
        d.mkdir()
        ctx.registry.register("pkg", d)
        result = await call(
            tools, "read_file", ctx, registry_key="pkg", subpath="nope.txt"
        )
        assert "No such file" in result

    async def test_subpath_cannot_escape_the_directory(self, tools, ctx, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret\n")
        d = tmp_path / "pkg"
        d.mkdir()
        ctx.registry.register("pkg", d)
        result = await call(
            tools, "read_file", ctx, registry_key="pkg", subpath="../secret.txt"
        )
        assert "escapes" in result
        assert "top secret" not in result


class TestStartScript:
    async def test_runs_and_registers_logs(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="greeter",
            content_lines=['echo "hi from script"'],
        )
        result = await call(
            tools, "start_background_script", ctx, registry_key="greeter"
        )
        assert "pid" in result.lower()
        assert "greeter_stdout" in result
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        stdout = ctx.registry.resolve("greeter_stdout")
        assert "hi from script" in stdout.read_text()

    async def test_passes_args(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="argsy", content_lines=['echo "arg1=$1"'],
        )
        await call(
            tools, "start_background_script", ctx,
            registry_key="argsy", args="banana",
        )
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        assert "arg1=banana" in ctx.registry.resolve("argsy_stdout").read_text()

    async def test_python_script_started_with_python(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", registry_key="pyhello",
            content_lines=['print("python says hi")'],
        )
        await call(tools, "start_background_script", ctx, registry_key="pyhello")
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        assert record.state == "finished"
        assert "python says hi" in ctx.registry.resolve("pyhello_stdout").read_text()


class TestListPaths:
    async def test_lists_registered_keys(self, tools, ctx, tmp_path):
        ctx.registry.register("alpha", tmp_path / "a.txt")
        ctx.registry.register("beta", tmp_path / "b.txt")
        result = await call(tools, "list_paths", ctx)
        assert "alpha" in result and "beta" in result

    async def test_empty_registry(self, tools, ctx):
        result = await call(tools, "list_paths", ctx)
        assert "no paths" in result.lower()


class TestRegistryShape:
    def test_default_registry_has_expected_tools(self, tools):
        assert set(tools.names()) == {
            "create_script",
            "read_file",
            "run_bash",
            "start_background_script",
            "list_paths",
        }

    def test_no_tool_is_unconditionally_destructive(self, tools):
        # run_bash gates conditionally (see TestRunBashDestructiveGate); no
        # builtin is destructive on every call.
        assert all(not t.destructive for t in tools)


class TestSingleLineScriptGate:
    """The live model mangles newline escapes in JSON strings under guided
    decoding, so content arrives as an array of lines; a "script" that is
    only a shebang line would still pass `bash -n` vacuously — reject it."""

    async def test_shebang_only_script_rejected(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="oneliner",
            content_lines=["#!/bin/bash for i in 1 2 3; do echo $i; done"],
        )
        assert "NOT created" in result
        assert "oneliner" not in ctx.registry.list()

    async def test_single_command_line_without_shebang_ok(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="shortie", content_lines=["echo hi"],
        )
        assert "ok" in result.lower()

    async def test_multiline_with_shebang_ok(self, tools, ctx):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="proper",
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
        assert ctx.registry.list() == {}
        assert ctx.runner.list()[0].state == "finished"

    async def test_clipped_output_registers_a_log_that_can_be_read(self, tools, ctx):
        result = await call(tools, "run_bash", ctx, content_lines=["seq 1 500"])
        assert "omitted" in result and "read_file" in result
        # the key it points at must actually exist — the whole point of
        # registering lazily is that the follow-up read works
        key = next(k for k in ctx.registry.list() if k.startswith("bash_stdout"))
        assert key in result
        assert "1\n" in ctx.registry.resolve(key).read_text()


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
        # What ran is what is quoted: `{key}` resolved, as the script has it.
        ctx.registry.register("data", tmp_path / "data")
        result = await call(
            tools, "run_bash", ctx,
            content_lines=["not_a_command_xyz {data}"],
        )
        assert f"not_a_command_xyz {tmp_path / 'data'}" in result


class TestRunBashKeyInterpolation:
    """`{key}` in a run_bash line expands to the registered path, which is how
    a kept script is run now that run_script is gone — and it keeps the
    "tools take keys, never literal paths" rule instead of carving it open."""

    async def test_key_expands_to_its_path(self, tools, ctx, tmp_path):
        target = tmp_path / "reads.bam"
        target.write_text("BAMDATA\n")
        ctx.registry.register("reads", target)
        result = await call(tools, "run_bash", ctx, content_lines=["cat {reads}"])
        assert "BAMDATA" in result

    async def test_a_registered_script_is_run_by_key(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", registry_key="qc",
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
        ctx.registry.register("reads", tmp_path / "reads.bam")
        result = await call(tools, "run_bash", ctx, content_lines=["cat {readz}"])
        assert "{readz}" in result
        assert "reads" in result  # the keys that do exist, for the retry

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
            kind="bash", registry_key="run_thing",
            content_lines=[
                "false",                       # the "tool" fails
                'echo "Done. Output: out.vcf"',  # would otherwise mask it
            ],
        )
        record = await ctx.runner.start(
            ["bash", str(ctx.registry.resolve("run_thing"))], name="run_thing"
        )
        record = await ctx.runner.wait(record.pid)
        assert record.state == "failed"
        assert record.exit_code != 0
        # the success line never ran
        assert "Done" not in record.stdout_path.read_text()

    async def test_strict_mode_is_injected_after_the_shebang(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="shebanged",
            content_lines=["#!/bin/bash", "echo hi"],
        )
        text = ctx.registry.resolve("shebanged").read_text()
        lines = text.splitlines()
        assert lines[0] == "#!/bin/bash"
        assert lines[1] == "set -euo pipefail"

    async def test_strict_mode_prepended_when_no_shebang(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="bare", content_lines=["echo hi"],
        )
        assert ctx.registry.resolve("bare").read_text().startswith(
            "set -euo pipefail\n"
        )

    async def test_the_models_own_set_e_is_respected(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="own",
            content_lines=["set -e", "echo hi"],
        )
        text = ctx.registry.resolve("own").read_text()
        assert text.count("set -e") == 1  # not doubled

    async def test_the_success_note_warns_against_unconditional_done(
        self, tools, ctx
    ):
        result = await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="noted", content_lines=["echo hi"],
        )
        assert "fail-fast" in result

    async def test_non_bash_scripts_are_not_touched(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", registry_key="py", content_lines=["print('hi')"],
        )
        text = ctx.registry.resolve("py").read_text()
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
