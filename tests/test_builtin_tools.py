"""Tests for hpca.agent.builtin_tools (§5.1): script tools over the runner/registry."""

import pytest

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
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
        with pytest.raises(RegistryError, match="already registered"):
            await call(
                tools, "create_script", ctx,
                kind="bash", registry_key="x", content_lines=["echo 2"],
            )


class TestReadFile:
    async def test_reads_registered_file(self, tools, ctx, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\n")
        ctx.registry.register("data", f)
        result = await call(tools, "read_file", ctx, registry_key="data")
        assert "line1" in result and "line2" in result

    async def test_long_file_truncated_head_tail(self, tools, ctx, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1000)))
        ctx.registry.register("big", f)
        result = await call(tools, "read_file", ctx, registry_key="big", max_lines=20)
        assert "line0" in result
        assert "line999" in result
        assert "line500" not in result
        assert "omitted" in result

    async def test_unknown_key_raises_with_available(self, tools, ctx, tmp_path):
        ctx.registry.register("known", tmp_path / "k.txt")
        with pytest.raises(UnknownKeyError, match="known"):
            await call(tools, "read_file", ctx, registry_key="nope")

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
        result = await call(tools, "start_script", ctx, registry_key="greeter")
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
        await call(tools, "start_script", ctx, registry_key="argsy", args="banana")
        record = ctx.runner.list()[0]
        await ctx.runner.wait(record.pid)
        assert "arg1=banana" in ctx.registry.resolve("argsy_stdout").read_text()

    async def test_python_script_started_with_python(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="python", registry_key="pyhello",
            content_lines=['print("python says hi")'],
        )
        await call(tools, "start_script", ctx, registry_key="pyhello")
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
            "run_script",
            "start_script",
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


class TestRunScript:
    """run_script waits and hands the output back — this is how the agent
    looks around the system (find a file, check a program exists)."""

    async def test_returns_stdout(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="hello",
            content_lines=["#!/bin/bash", "echo found-it"],
        )
        result = await call(tools, "run_script", ctx, registry_key="hello")
        assert "found-it" in result
        assert "exit 0" in result

    async def test_a_find_style_search_comes_back(self, tools, ctx, tmp_path):
        planted = tmp_path / "refs" / "GRCh38.primary.fa"
        planted.parent.mkdir(parents=True)
        planted.write_text(">chr1\nACGT\n")
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="findref",
            content_lines=[
                "#!/bin/bash",
                f"find {tmp_path} -maxdepth 3 -iname '*GRCh38*' 2>/dev/null | head",
            ],
        )
        result = await call(tools, "run_script", ctx, registry_key="findref")
        assert str(planted) in result  # the agent can now register this path

    async def test_failure_is_reported_with_stderr(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="boom",
            content_lines=["#!/bin/bash", "echo to-stderr >&2", "exit 3"],
        )
        result = await call(tools, "run_script", ctx, registry_key="boom")
        assert "FAILED with exit code 3" in result
        assert "to-stderr" in result

    async def test_timeout_kills_and_says_how_to_recover(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="slow",
            content_lines=["#!/bin/bash", "sleep 30"],
        )
        result = await call(
            tools, "run_script", ctx, registry_key="slow", timeout_s=1
        )
        assert "TIMED OUT" in result
        assert "-maxdepth" in result or "Narrow" in result

    async def test_long_output_is_bounded_and_points_at_the_log(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="chatty",
            content_lines=["#!/bin/bash", "seq 1 500"],
        )
        result = await call(tools, "run_script", ctx, registry_key="chatty")
        assert len(result) < 6000  # the prompt is not flooded
        assert "500" in result  # the tail, which is what a search prints last
        assert "omitted" in result and "read_file" in result

    async def test_no_output_is_stated_not_silent(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="quiet",
            content_lines=["#!/bin/bash", "true"],
        )
        result = await call(tools, "run_script", ctx, registry_key="quiet")
        assert "(no output)" in result

    async def test_logs_are_registered_for_follow_up(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="logged",
            content_lines=["#!/bin/bash", "echo x"],
        )
        await call(tools, "run_script", ctx, registry_key="logged")
        assert any("logged_stdout" in key for key in ctx.registry.list())

    async def test_the_run_is_tracked_like_any_process(self, tools, ctx):
        await call(
            tools, "create_script", ctx,
            kind="bash", registry_key="tracked",
            content_lines=["#!/bin/bash", "echo x"],
        )
        await call(tools, "run_script", ctx, registry_key="tracked")
        records = ctx.runner.list()
        assert [r.name for r in records] == ["tracked"]
        assert records[0].state == "finished"

    async def test_unknown_key_is_a_useful_error(self, tools, ctx):
        with pytest.raises(Exception):
            await call(tools, "run_script", ctx, registry_key="nope")


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

    async def test_throwaway_scripts_are_not_registered(self, tools, ctx):
        await call(tools, "run_bash", ctx, content_lines=["echo x"])
        await call(tools, "run_bash", ctx, content_lines=["echo y"])
        # no bash_* keys clutter the registry the model reasons over
        assert not any(k.startswith("bash_") for k in ctx.registry.list())
        assert ctx.runner.list()[0].state == "finished"


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
