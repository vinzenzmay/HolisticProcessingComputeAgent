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

    async def test_subpath_deletes_file_inside_directory_keeping_dir_key(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "run"
        d.mkdir()
        victim = d / "old.log"
        victim.write_text("stale")
        ctx.registry.register("run_dir", d)
        result = await call(
            tools, "delete_file", ctx, registry_key="run_dir", subpath="old.log"
        )
        assert not victim.exists()
        assert "recoverable" in result
        # the directory key survives; only the inner file was removed
        assert ctx.registry.resolve("run_dir") == d

    async def test_subpath_delete_is_gated_when_it_resolves(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "run"
        d.mkdir()
        (d / "old.log").write_text("stale")
        ctx.registry.register("run_dir", d)
        assert (
            gates(tools, "delete_file", ctx, registry_key="run_dir", subpath="old.log")
            is True
        )

    async def test_missing_subpath_is_a_useful_error(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        ctx.registry.register("run_dir", d)
        # a bad subpath is not gated, and the handler explains rather than crashes
        assert (
            gates(tools, "delete_file", ctx, registry_key="run_dir", subpath="ghost")
            is False
        )
        result = await call(
            tools, "delete_file", ctx, registry_key="run_dir", subpath="ghost"
        )
        assert "No such path" in result

    async def test_subpath_cannot_escape_the_directory(self, tools, ctx, tmp_path):
        outside = tmp_path / "keepme.txt"
        outside.write_text("precious")
        d = tmp_path / "run"
        d.mkdir()
        ctx.registry.register("run_dir", d)
        result = await call(
            tools, "delete_file", ctx, registry_key="run_dir", subpath="../keepme.txt"
        )
        assert "escapes" in result
        assert outside.exists()


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

    async def test_subpath_moves_file_from_directory_and_registers_it(
        self, tools, ctx, tmp_path
    ):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("content")
        d = tmp_path / "dest"
        d.mkdir()
        ctx.registry.register("src_dir", src)
        ctx.registry.register("dest", d)
        result = await call(
            tools, "move_file", ctx,
            source_key="src_dir", subpath="a.txt", dest_dir_key="dest",
        )
        assert not (src / "a.txt").exists()
        assert (d / "a.txt").read_text() == "content"
        # the source directory key is untouched; the moved file gets its own key
        assert ctx.registry.resolve("src_dir") == src
        assert ctx.registry.resolve("a.txt") == d / "a.txt"
        assert "a.txt" in result


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

    async def test_subpath_copies_file_from_directory(self, tools, ctx, tmp_path):
        src = tmp_path / "src"
        (src / "nested").mkdir(parents=True)
        (src / "nested" / "a.txt").write_text("content")
        d = tmp_path / "dest"
        d.mkdir()
        ctx.registry.register("src_dir", src)
        ctx.registry.register("dest", d)
        result = await call(
            tools, "copy_file", ctx,
            source_key="src_dir", subpath="nested/a.txt", dest_dir_key="dest",
        )
        assert (src / "nested" / "a.txt").exists()  # original untouched
        assert (d / "a.txt").read_text() == "content"
        assert "a.txt_copy" in result


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

    async def test_delete_description_resolves_subpath(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        (d / "old.log").write_text("stale")
        ctx.registry.register("run_dir", d)
        tool = tools.get("delete_file")
        text = tool.describe_call(
            tool.params.model_validate({"registry_key": "run_dir", "subpath": "old.log"}),
            ctx,
        )
        assert str(d / "old.log") in text
