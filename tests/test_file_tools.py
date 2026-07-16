"""Tests for hpca.agent.file_tools (§5.1, §5.3): key-based, trash-backed ops."""

import pytest

from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.registry import PathRegistry, UnknownKeyError
from hpca.runner import ProcessRunner
from hpca.trash import TrashManager


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        trash=TrashManager(tmp_path / "trash", backup_limit_bytes=1024 * 1024),
    )
    conn.close()


@pytest.fixture
def tools():
    return add_file_tools(ToolRegistry())


async def call(tools, name, ctx, **kwargs):
    tool = tools.get(name)
    return await tool.handler(tool.params.model_validate(kwargs), ctx)


def gates(tools, name, ctx, **kwargs):
    tool = tools.get(name)
    return tool.gates(tool.params.model_validate(kwargs), ctx)


class TestRegisterPath:
    async def test_registers_existing_file(self, tools, ctx, tmp_path):
        f = tmp_path / "input.bam"
        f.write_text("x")
        result = await call(tools, "register_path", ctx, key="input_bam", path=str(f))
        assert "Registered" in result
        assert ctx.registry.resolve("input_bam") == f

    async def test_missing_path_not_registered(self, tools, ctx, tmp_path):
        result = await call(
            tools, "register_path", ctx, key="ghost", path=str(tmp_path / "nope")
        )
        assert "NOT registered" in result
        assert "ghost" not in ctx.registry.list()


class TestListDir:
    async def test_lists_with_pattern(self, tools, ctx, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.bam").write_text("x")
        (d / "b.bam").write_text("x")
        (d / "notes.txt").write_text("x")
        (d / "sub").mkdir()
        ctx.registry.register("data", d)
        result = await call(tools, "list_dir", ctx, dir_key="data", pattern="*.bam")
        assert "a.bam" in result and "b.bam" in result
        assert "notes.txt" not in result
        full = await call(tools, "list_dir", ctx, dir_key="data")
        assert "sub/" in full

    async def test_bounded_listing(self, tools, ctx, tmp_path):
        d = tmp_path / "many"
        d.mkdir()
        for i in range(150):
            (d / f"f{i:03}.txt").write_text("x")
        ctx.registry.register("many", d)
        result = await call(tools, "list_dir", ctx, dir_key="many")
        assert len(result.splitlines()) <= 101
        assert "omitted" in result


class TestDeleteFile:
    async def test_gated_when_key_resolves(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        ctx.registry.register("x", f)
        assert gates(tools, "delete_file", ctx, registry_key="x") is True

    async def test_unresolvable_key_not_gated(self, tools, ctx):
        # no pointless approval modal; the handler's error feeds the retry loop
        assert gates(tools, "delete_file", ctx, registry_key="ghost") is False

    async def test_deletes_via_trash_and_unregisters(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        ctx.registry.register("x", f)
        result = await call(tools, "delete_file", ctx, registry_key="x")
        assert not f.exists()
        assert "recoverable" in result
        assert "x" not in ctx.registry.list()
        assert ctx.trash.list()[0].original_path == f

    async def test_oversized_states_no_backup(self, tools, ctx, tmp_path):
        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        f = tmp_path / "big.bin"
        f.write_text("more than two bytes")
        ctx.registry.register("big", f)
        result = await call(tools, "delete_file", ctx, registry_key="big")
        assert "WITHOUT backup" in result


class TestMoveFile:
    async def test_fresh_target_not_gated_and_reassigns_key(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "a.txt"
        f.write_text("content")
        d = tmp_path / "dest"
        d.mkdir()
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        assert gates(tools, "move_file", ctx, source_key="a", dest_dir_key="dest") is False
        await call(tools, "move_file", ctx, source_key="a", dest_dir_key="dest")
        assert not f.exists()
        assert (d / "a.txt").read_text() == "content"
        assert ctx.registry.resolve("a") == d / "a.txt"

    async def test_move_over_existing_gated_and_target_trashed(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "a.txt"
        f.write_text("new content")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old content")
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        assert gates(tools, "move_file", ctx, source_key="a", dest_dir_key="dest") is True
        result = await call(tools, "move_file", ctx, source_key="a", dest_dir_key="dest")
        assert (d / "a.txt").read_text() == "new content"
        assert "trash" in result
        trashed = ctx.trash.list()[0]
        assert trashed.trashed_path.read_text() == "old content"

    async def test_rename_via_new_name(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        d = tmp_path / "dest"
        d.mkdir()
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        await call(
            tools, "move_file", ctx,
            source_key="a", dest_dir_key="dest", new_name="b.txt",
        )
        assert (d / "b.txt").exists()

    async def test_unknown_source_raises(self, tools, ctx, tmp_path):
        d = tmp_path / "dest"
        d.mkdir()
        ctx.registry.register("dest", d)
        with pytest.raises(UnknownKeyError):
            await call(tools, "move_file", ctx, source_key="nope", dest_dir_key="dest")


class TestCopyFile:
    async def test_copy_registers_new_key(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        d = tmp_path / "dest"
        d.mkdir()
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        assert gates(tools, "copy_file", ctx, source_key="a", dest_dir_key="dest") is False
        result = await call(tools, "copy_file", ctx, source_key="a", dest_dir_key="dest")
        assert f.exists()
        assert (d / "a.txt").read_text() == "content"
        assert "a_copy" in result

    async def test_copy_over_existing_gated(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("new")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old")
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        assert gates(tools, "copy_file", ctx, source_key="a", dest_dir_key="dest") is True


class TestDescribeCall:
    async def test_delete_description_shows_resolved_path(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello")
        ctx.registry.register("x", f)
        tool = tools.get("delete_file")
        text = tool.describe_call(tool.params.model_validate({"registry_key": "x"}), ctx)
        assert str(f) in text
        assert "recoverable" in text

    async def test_delete_description_warns_no_backup(self, tools, ctx, tmp_path):
        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        f = tmp_path / "big.bin"
        f.write_text("many bytes here")
        ctx.registry.register("big", f)
        tool = tools.get("delete_file")
        text = tool.describe_call(tool.params.model_validate({"registry_key": "big"}), ctx)
        assert "NO BACKUP" in text

    async def test_move_description_shows_both_paths_and_overwrite(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "a.txt"
        f.write_text("new")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old")
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        tool = tools.get("move_file")
        text = tool.describe_call(
            tool.params.model_validate({"source_key": "a", "dest_dir_key": "dest"}), ctx
        )
        assert str(f) in text and str(d / "a.txt") in text
        assert "OVERWRITES" in text
