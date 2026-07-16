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
        assert path.read_text() == "echo hello\n"  # joined + trailing newline
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
            "start_script",
            "list_paths",
        }

    def test_no_tool_is_destructive_yet(self, tools):
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
