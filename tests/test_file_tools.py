"""Tests for hpca.agent.file_tools (§5.1, §5.3): key-based, trash-backed ops."""

import pytest

from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools, edit_preview
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
        # names the tool that undoes it, not just "the trash": without that the
        # model tells the user to go dig through the app dir themselves
        assert "recoverable" in result and "restore_file" in result
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


class TestRestoreFile:
    async def test_not_gated(self, tools, ctx):
        # restore never overwrites, so it needs no approval modal
        assert gates(tools, "restore_file", ctx, path="/anything") is False

    async def test_restores_a_deleted_file_and_registers_it(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        ctx.registry.register("x", f)
        await call(tools, "delete_file", ctx, registry_key="x")
        result = await call(tools, "restore_file", ctx, path=str(f))
        assert f.read_text() == "precious"
        assert str(f) in result
        # usable again straight away: the delete dropped the key, restore adds one
        assert ctx.registry.resolve("x.txt") == f
        assert ctx.trash.list() == []

    async def test_restores_by_file_name(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        ctx.registry.register("x", f)
        await call(tools, "delete_file", ctx, registry_key="x")
        await call(tools, "restore_file", ctx, path="x.txt")
        assert f.read_text() == "precious"

    async def test_empty_path_lists_the_trash(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        ctx.registry.register("x", f)
        await call(tools, "delete_file", ctx, registry_key="x")
        result = await call(tools, "restore_file", ctx)
        assert str(f) in result
        assert not f.exists()  # listing restores nothing

    async def test_empty_trash_says_so(self, tools, ctx):
        assert "empty" in await call(tools, "restore_file", ctx)

    async def test_unknown_path_lists_what_is_there(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        ctx.registry.register("x", f)
        await call(tools, "delete_file", ctx, registry_key="x")
        result = await call(tools, "restore_file", ctx, path="ghost.txt")
        assert "ghost.txt" in result and "no" in result.lower()
        assert str(f) in result  # ... and what it could have meant instead

    async def test_ambiguous_name_asks_for_the_full_path(self, tools, ctx, tmp_path):
        victims = []
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            victim = d / "run.log"
            victim.write_text(name)
            ctx.registry.register(name, victim)
            await call(tools, "delete_file", ctx, registry_key=name)
            victims.append(victim)
        result = await call(tools, "restore_file", ctx, path="run.log")
        assert all(str(v) in result for v in victims)
        assert not any(v.exists() for v in victims)  # nothing guessed at

    async def test_same_path_deleted_twice_restores_the_newest(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "x.txt"
        for content in ("first", "second"):
            f.write_text(content)
            ctx.registry.register_auto(f)
            await call(tools, "delete_file", ctx, registry_key="x.txt")
        await call(tools, "restore_file", ctx, path=str(f))
        assert f.read_text() == "second"

    async def test_refuses_when_the_path_is_occupied(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        ctx.registry.register("x", f)
        await call(tools, "delete_file", ctx, registry_key="x")
        f.write_text("something new")
        result = await call(tools, "restore_file", ctx, path=str(f))
        assert "already exists" in result
        assert "move_file" in result  # the way out is named, not left to guess
        assert f.read_text() == "something new"
        assert ctx.trash.list()  # the backup is kept, not consumed

    async def test_an_edit_is_undone_by_moving_aside_then_restoring(
        self, tools, ctx, tmp_path
    ):
        # edit_file leaves the previous content in the trash under the *same*
        # path, so a restore only becomes possible once the edited file is out
        # of the way — and it has to be moved, not deleted: deleting it would
        # trash it under that same path and become the newer entry.
        f = tmp_path / "run.sh"
        f.write_text("echo before\n")
        ctx.registry.register("run_sh", f)
        aside = tmp_path / "aside"
        aside.mkdir()
        ctx.registry.register("aside", aside)
        await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo before"],
            new_lines=["echo after"],
        )
        assert "already exists" in await call(tools, "restore_file", ctx, path=str(f))
        await call(
            tools, "move_file", ctx, source_key="run_sh", dest_dir_key="aside"
        )
        result = await call(tools, "restore_file", ctx, path=str(f))
        assert "Restored" in result
        assert f.read_text() == "echo before\n"
        assert (aside / "run.sh").read_text() == "echo after\n"

    async def test_unbacked_deletion_reported_as_unrecoverable(
        self, tools, ctx, tmp_path
    ):
        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        f = tmp_path / "big.bin"
        f.write_text("more than two bytes")
        ctx.registry.register("big", f)
        await call(tools, "delete_file", ctx, registry_key="big")
        result = await call(tools, "restore_file", ctx, path=str(f))
        assert "without a backup" in result
        assert not f.exists()

    async def test_restores_a_file_lost_to_an_overwrite(self, tools, ctx, tmp_path):
        # move_file over an existing target trashes the old file the same way,
        # so the same tool brings it back once the path is free again
        f = tmp_path / "a.txt"
        f.write_text("new content")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old content")
        ctx.registry.register("a", f)
        ctx.registry.register("dest", d)
        await call(tools, "move_file", ctx, source_key="a", dest_dir_key="dest")
        (d / "a.txt").unlink()
        await call(tools, "restore_file", ctx, path=str(d / "a.txt"))
        assert (d / "a.txt").read_text() == "old content"


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


class TestEditFile:
    """Replace an exact run of lines in place, without re-sending the file."""

    @pytest.fixture
    def script(self, ctx, tmp_path):
        path = tmp_path / "run.sh"
        path.write_text("#!/bin/bash\nset -euo pipefail\necho one\necho two\n")
        ctx.registry.register("run_sh", path)
        return path

    async def test_replaces_the_matched_lines_and_leaves_the_rest(
        self, tools, ctx, script
    ):
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo one"],
            new_lines=["echo ONE", "echo one-and-a-half"],
        )
        assert script.read_text() == (
            "#!/bin/bash\nset -euo pipefail\necho ONE\necho one-and-a-half\necho two\n"
        )
        assert "Edited" in result and str(script) in result

    async def test_empty_new_lines_deletes_the_old_ones(self, tools, ctx, script):
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo one"],
            new_lines=[],
        )
        assert script.read_text() == "#!/bin/bash\nset -euo pipefail\necho two\n"
        assert "1 line deleted" in result

    async def test_a_multi_line_block_matches_across_lines(self, tools, ctx, script):
        await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo one", "echo two"],
            new_lines=["echo both"],
        )
        assert script.read_text() == "#!/bin/bash\nset -euo pipefail\necho both\n"

    async def test_the_previous_content_goes_to_the_trash(self, tools, ctx, script):
        await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo two"],
            new_lines=["echo three"],
        )
        entries = ctx.trash.list()
        assert [e.original_path for e in entries] == [script]
        assert "echo two" in entries[0].trashed_path.read_text()
        assert "echo three" in script.read_text()  # the file itself is the edited one

    async def test_no_match_leaves_the_file_alone(self, tools, ctx, script):
        before = script.read_text()
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo nothing like this"],
            new_lines=["echo x"],
        )
        assert "NOT edited" in result
        assert script.read_text() == before
        assert ctx.trash.list() == []  # nothing was overwritten, nothing backed up

    async def test_no_match_points_at_a_line_that_does_occur(self, tools, ctx, script):
        # The usual near-miss: right line, wrong indentation or trailing space.
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["    echo one"],
            new_lines=["echo x"],
        )
        assert "NOT edited" in result
        assert "line 3" in result

    async def test_an_ambiguous_match_is_refused(self, tools, ctx, tmp_path):
        path = tmp_path / "twice.sh"
        path.write_text("echo hi\necho mid\necho hi\n")
        ctx.registry.register("twice", path)
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="twice",
            old_lines=["echo hi"],
            new_lines=["echo bye"],
        )
        assert "NOT edited" in result and "2 times" in result
        assert path.read_text() == "echo hi\necho mid\necho hi\n"

    async def test_subpath_edits_a_file_inside_a_registered_directory(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "run"
        d.mkdir()
        (d / "conf.yaml").write_text("threads: 4\nmem: 8G\n")
        ctx.registry.register("run_dir", d)
        await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_dir",
            subpath="conf.yaml",
            old_lines=["threads: 4"],
            new_lines=["threads: 16"],
        )
        assert (d / "conf.yaml").read_text() == "threads: 16\nmem: 8G\n"

    async def test_a_directory_key_says_to_use_subpath(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        ctx.registry.register("run_dir", d)
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_dir",
            old_lines=["x"],
            new_lines=["y"],
        )
        assert "NOT edited" in result and "subpath" in result

    async def test_a_binary_file_is_refused(self, tools, ctx, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"\x00\x01\x82\xff binary")
        ctx.registry.register("bin", path)
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="bin",
            old_lines=["x"],
            new_lines=["y"],
        )
        assert "NOT edited" in result
        assert path.read_bytes() == b"\x00\x01\x82\xff binary"

    async def test_a_missing_subpath_is_a_useful_error(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        ctx.registry.register("run_dir", d)
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_dir",
            subpath="ghost.txt",
            old_lines=["x"],
            new_lines=["y"],
        )
        assert "No such path" in result

    async def test_unknown_key_raises(self, tools, ctx):
        with pytest.raises(UnknownKeyError):
            await call(
                tools,
                "edit_file",
                ctx,
                registry_key="nope",
                old_lines=["x"],
                new_lines=["y"],
            )


class TestEditSyntaxGate:
    """§5.2's mandatory gate must not be reachable around: an edit rewrites a
    script's content exactly as create_script does."""

    @pytest.fixture
    def script(self, ctx, tmp_path):
        path = tmp_path / "run.sh"
        path.write_text("if true; then\n  echo ok\nfi\n")
        ctx.registry.register("run_sh", path)
        return path

    async def test_an_edit_that_breaks_the_syntax_is_not_applied(
        self, tools, ctx, script
    ):
        before = script.read_text()
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["fi"],
            new_lines=[],  # drops the closing fi
        )
        assert "NOT edited" in result
        assert script.read_text() == before
        assert ctx.trash.list() == []  # never overwritten, so never backed up

    async def test_a_valid_edit_to_a_script_still_applies(self, tools, ctx, script):
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["  echo ok"],
            new_lines=["  echo fine"],
        )
        assert "Edited" in result
        assert "echo fine" in script.read_text()

    async def test_a_non_script_file_is_not_syntax_checked(self, tools, ctx, tmp_path):
        # A dangling `if` is broken bash, but a .txt file is not bash and must
        # not be held to bash's rules.
        path = tmp_path / "notes.txt"
        path.write_text("first\nsecond\n")
        ctx.registry.register("notes", path)
        await call(
            tools,
            "edit_file",
            ctx,
            registry_key="notes",
            old_lines=["second"],
            new_lines=["if true; then"],
        )
        assert path.read_text() == "first\nif true; then\n"


class TestEditGating:
    async def test_an_edit_gates_as_destructive(self, tools, ctx, tmp_path):
        path = tmp_path / "run.sh"
        path.write_text("echo one\n")
        ctx.registry.register("run_sh", path)
        assert gates(
            tools,
            "edit_file",
            ctx,
            registry_key="run_sh",
            old_lines=["echo one"],
            new_lines=["echo two"],
        )

    async def test_an_unresolvable_key_is_not_gated(self, tools, ctx):
        # A dud call goes to the error-feedback loop, not to the user.
        assert not gates(
            tools, "edit_file", ctx, registry_key="nope", old_lines=["x"], new_lines=["y"]
        )

    async def test_the_description_shows_the_path_and_the_backup(
        self, tools, ctx, tmp_path
    ):
        path = tmp_path / "run.sh"
        path.write_text("echo one\n")
        ctx.registry.register("run_sh", path)
        tool = tools.get("edit_file")
        text = tool.describe_call(
            tool.params.model_validate(
                {
                    "registry_key": "run_sh",
                    "old_lines": ["echo one"],
                    "new_lines": ["echo two"],
                }
            ),
            ctx,
        )
        assert str(path) in text
        assert "trash" in text

    async def test_the_description_warns_when_there_is_no_backup(
        self, tools, ctx, tmp_path
    ):
        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        path = tmp_path / "big.txt"
        path.write_text("many bytes here")
        ctx.registry.register("big", path)
        tool = tools.get("edit_file")
        text = tool.describe_call(
            tool.params.model_validate(
                {"registry_key": "big", "old_lines": ["many"], "new_lines": ["few"]}
            ),
            ctx,
        )
        assert "NO BACKUP" in text


class TestEditPreview:
    """What the approval prompt and the chat call box show for one edit."""

    def test_the_lines_are_shown_as_a_diff(self, ctx):
        text = edit_preview(
            {"old_lines": ["echo one", "echo two"], "new_lines": ["echo both"]}, ctx
        )
        assert text.splitlines() == ["- echo one", "- echo two", "+ echo both"]

    def test_a_deletion_shows_only_removed_lines(self, ctx):
        assert edit_preview({"old_lines": ["gone"], "new_lines": []}, ctx) == "- gone"


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
