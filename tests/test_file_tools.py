"""Tests for hpca.agent.file_tools (§5.1, §5.3): key-based, trash-backed ops."""

import pytest

from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools, edit_preview
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.registry import PathRegistry, RegistryError, UnknownKeyError
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


async def call(tools, tool_name, ctx, **kwargs):
    tool = tools.get(tool_name)
    return await tool.handler(tool.params.model_validate(kwargs), ctx)


def gates(tools, tool_name, ctx, **kwargs):
    tool = tools.get(tool_name)
    return tool.gates(tool.params.model_validate(kwargs), ctx)


class TestRegisterPath:
    async def test_registers_existing_file(self, tools, ctx, tmp_path):
        f = tmp_path / "input.bam"
        f.write_text("x")
        result = await call(tools, "register_path", ctx, key="input_bam", path=str(f))
        assert "Registered" in result
        assert ctx.registry.resolve("input_bam") == f

    async def test_a_path_that_does_not_exist_yet_is_still_registered(
        self, tools, ctx, tmp_path
    ):
        """It is how the file the agent is about to create looks. Refusing it
        left no key to name that file with in the call that would create it."""
        target = tmp_path / "results" / "run.log"
        result = await call(tools, "register_path", ctx, key="ghost", path=str(target))
        assert ctx.registry.resolve("ghost") == target
        # …and the model is told, so it does not read it expecting content.
        assert "Registered" in result
        assert "nothing exists at" in result
        assert str(target) in result

    async def test_an_existing_path_is_not_reported_as_missing(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "runs"
        d.mkdir()
        result = await call(tools, "register_path", ctx, key="runs", path=str(d))
        assert "directory" in result
        assert "nothing exists" not in result
        assert "previously pointed" not in result  # nothing was replaced

    async def test_a_typo_can_be_corrected_under_the_same_key(
        self, tools, ctx, tmp_path
    ):
        """The typo and the file that does exist are usually the same key in
        the user's head, and there is no unregister tool: without this the key
        stays pointed at nothing for the rest of the session."""
        real = tmp_path / "reads.bam"
        real.write_text("x")
        typo = tmp_path / "raeds.bam"
        await call(tools, "register_path", ctx, key="reads", path=str(typo))
        result = await call(tools, "register_path", ctx, key="reads", path=str(real))
        assert ctx.registry.resolve("reads") == real
        # the correction is stated, not silent: the model just saw the same
        # "Registered ..." sentence for a path that turned out to be wrong.
        assert "file" in result
        assert "previously pointed" in result
        assert str(typo) in result

    async def test_a_correction_onto_another_missing_path_says_both(
        self, tools, ctx, tmp_path
    ):
        first = tmp_path / "one.log"
        second = tmp_path / "two.log"
        await call(tools, "register_path", ctx, key="out", path=str(first))
        result = await call(tools, "register_path", ctx, key="out", path=str(second))
        assert ctx.registry.resolve("out") == second
        assert "nothing exists at" in result and str(second) in result
        assert "previously pointed" in result and str(first) in result

    async def test_a_key_pointing_at_a_real_file_is_not_repointed(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "a.txt"
        f.write_text("x")
        ctx.registry.register("x", f)
        with pytest.raises(RegistryError, match="pick a different key"):
            await call(
                tools, "register_path", ctx, key="x", path=str(tmp_path / "b.txt")
            )
        assert ctx.registry.resolve("x") == f


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


class TestEditRepairAndFuzz:
    """Absorb model imperfection inside the tool (PI-style): repair the
    arguments and climb a matching ladder instead of bouncing a near-miss back
    for another round-trip plus a re-read of the file."""

    @pytest.fixture
    def notes(self, ctx, tmp_path):
        # .txt on purpose: no §5.2 gate, these tests are about matching only
        path = tmp_path / "notes.txt"
        path.write_text("alpha\nbravo\ncharlie\n")
        ctx.registry.register("notes", path)
        return path

    async def test_embedded_newlines_in_old_and_new_lines_are_split(
        self, tools, ctx, notes
    ):
        # One array element holding two lines still means two lines.
        result = await call(
            tools, "edit_file", ctx,
            registry_key="notes",
            old_lines=["alpha\nbravo"],
            new_lines=["one\ntwo", "three"],
        )
        assert notes.read_text() == "one\ntwo\nthree\ncharlie\n"
        assert "2 lines → 3 lines" in result

    async def test_numbered_listing_prefixes_are_stripped_when_all_carry_one(
        self, tools, ctx, notes
    ):
        # A model copying read_file's numbered listing verbatim.
        result = await call(
            tools, "edit_file", ctx,
            registry_key="notes",
            old_lines=["2: bravo", "3: charlie"],
            new_lines=["BRAVO"],
        )
        assert notes.read_text() == "alpha\nBRAVO\n"
        assert "Edited" in result

    async def test_a_partially_numbered_copy_is_not_stripped(
        self, tools, ctx, notes
    ):
        # One prefixed line among plain ones is genuine content, not a copied
        # listing — refuse rather than mangle.
        before = notes.read_text()
        result = await call(
            tools, "edit_file", ctx,
            registry_key="notes",
            old_lines=["2: bravo", "charlie"],
            new_lines=["x"],
        )
        assert "NOT edited" in result
        assert notes.read_text() == before

    async def test_genuine_numbered_content_matches_before_stripping(
        self, tools, ctx, tmp_path
    ):
        # A YAML-ish "12: value" that really is in the file must match as-is;
        # stripping is a fallback for lines that match nothing.
        path = tmp_path / "map.txt"
        path.write_text("11: ten\n12: twelve\n")
        ctx.registry.register("map", path)
        await call(
            tools, "edit_file", ctx,
            registry_key="map",
            old_lines=["12: twelve"],
            new_lines=["12: TWELVE"],
        )
        assert path.read_text() == "11: ten\n12: TWELVE\n"

    async def test_trailing_whitespace_in_the_file_is_forgiven_and_applied(
        self, tools, ctx, tmp_path
    ):
        # The model cannot even see a trailing space in the file; asking it to
        # copy one is a wasted round-trip. new_lines land verbatim.
        path = tmp_path / "pad.txt"
        path.write_text("keep\nfix me  \nkeep too\n")
        ctx.registry.register("pad", path)
        result = await call(
            tools, "edit_file", ctx,
            registry_key="pad",
            old_lines=["fix me"],
            new_lines=["fixed"],
        )
        assert path.read_text() == "keep\nfixed\nkeep too\n"
        assert "Edited" in result

    async def test_smart_quotes_and_dashes_are_forgiven_and_applied(
        self, tools, ctx, tmp_path
    ):
        # File written with typographic quotes/dashes, model sends ASCII.
        path = tmp_path / "prose.txt"
        path.write_text("start\nsay ‘hi’ — loudly\nend\n")
        ctx.registry.register("prose", path)
        await call(
            tools, "edit_file", ctx,
            registry_key="prose",
            old_lines=["say 'hi' - loudly"],
            new_lines=["say 'bye'"],
        )
        assert path.read_text() == "start\nsay 'bye'\nend\n"

    async def test_leading_whitespace_is_still_meaning_not_fuzz(
        self, tools, ctx, tmp_path
    ):
        # Indentation is meaning (Python); the ladder never strips it, the
        # near-miss advice still points at the line.
        path = tmp_path / "code.txt"
        path.write_text("def f():\n    return 1\n")
        ctx.registry.register("code", path)
        result = await call(
            tools, "edit_file", ctx,
            registry_key="code",
            old_lines=["return 1"],
            new_lines=["return 2"],
        )
        assert "NOT edited" in result and "line 2" in result
        assert path.read_text() == "def f():\n    return 1\n"

    async def test_ambiguity_at_a_fuzzy_level_is_still_refused(
        self, tools, ctx, tmp_path
    ):
        # Two lines that only differ in trailing whitespace both match at the
        # rstrip level — that is 2 occurrences, not a rescue.
        path = tmp_path / "twice.txt"
        path.write_text("echo hi  \nmid\necho hi\t\n")
        ctx.registry.register("twice", path)
        before = path.read_text()
        result = await call(
            tools, "edit_file", ctx,
            registry_key="twice",
            old_lines=["echo hi"],
            new_lines=["echo bye"],
        )
        assert "NOT edited" in result and "2 times" in result
        assert path.read_text() == before

    async def test_a_crlf_file_keeps_its_line_endings(self, tools, ctx, tmp_path):
        path = tmp_path / "win.txt"
        path.write_bytes(b"alpha\r\nbravo\r\ncharlie\r\n")
        ctx.registry.register("win", path)
        await call(
            tools, "edit_file", ctx,
            registry_key="win",
            old_lines=["bravo"],
            new_lines=["BRAVO"],
        )
        assert path.read_bytes() == b"alpha\r\nBRAVO\r\ncharlie\r\n"

    async def test_a_bom_survives_the_edit(self, tools, ctx, tmp_path):
        path = tmp_path / "bom.txt"
        path.write_bytes("﻿alpha\nbravo\n".encode())
        ctx.registry.register("bom", path)
        await call(
            tools, "edit_file", ctx,
            registry_key="bom",
            old_lines=["alpha"],
            new_lines=["ALPHA"],
        )
        assert path.read_bytes() == "﻿ALPHA\nbravo\n".encode()

    async def test_an_identical_replacement_is_an_error_not_a_write(
        self, tools, ctx, notes
    ):
        result = await call(
            tools, "edit_file", ctx,
            registry_key="notes",
            old_lines=["bravo"],
            new_lines=["bravo"],
        )
        assert "NOT edited" in result and "identical" in result
        assert ctx.trash.list() == []  # nothing written, nothing backed up

    async def test_success_is_one_short_line_without_the_undo_lecture(
        self, tools, ctx, notes
    ):
        result = await call(
            tools, "edit_file", ctx,
            registry_key="notes",
            old_lines=["bravo"],
            new_lines=["BRAVO"],
        )
        assert result == f"Edited {notes} at line 2: 1 line → 1 line."
        assert ctx.trash.list()  # the backup itself is still kept

    async def test_success_warns_only_when_no_backup_could_be_kept(
        self, tools, ctx, tmp_path
    ):
        from hpca.trash import TrashManager

        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        path = tmp_path / "big.txt"
        path.write_text("more than two bytes\n")
        ctx.registry.register("big", path)
        result = await call(
            tools, "edit_file", ctx,
            registry_key="big",
            old_lines=["more than two bytes"],
            new_lines=["tiny"],
        )
        assert "NO backup" in result and "cannot be undone" in result

    async def test_create_file_splits_embedded_newlines_too(
        self, tools, ctx, tmp_path
    ):
        ctx.registry.register("project", tmp_path)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="notes.md",
            content_lines=["a\nb", "c"],
        )
        assert (tmp_path / "notes.md").read_text() == "a\nb\nc\n"
        assert "3 lines" in result


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


class TestCreateFile:
    """Writing a document — specs, a README, a config — without a shell.

    Before this, ``create_script`` could only write into the scripts dir under
    a language suffix and ``edit_file`` needs a file to already exist, so the
    only way to author a markdown file was a `cat << 'EOF'` heredoc through
    run_bash: a hundred lines of prose funnelled through bash quoting.
    """

    async def test_writes_the_lines_into_the_registered_directory(
        self, tools, ctx, tmp_path
    ):
        ctx.registry.register("project", tmp_path)
        await call(
            tools, "create_file", ctx,
            dir_key="project",
            name="specs.md",
            content_lines=["# locus-cutter", "", "## Why"],
        )
        assert (tmp_path / "specs.md").read_text() == "# locus-cutter\n\n## Why\n"

    async def test_the_new_file_comes_back_with_a_key(self, tools, ctx, tmp_path):
        ctx.registry.register("project", tmp_path)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="specs.md", content_lines=["hi"],
        )
        keys = [key for key, path in ctx.registry.list().items()
                if path == tmp_path / "specs.md"]
        assert keys and keys[0] in result

    async def test_an_existing_file_is_not_overwritten(self, tools, ctx, tmp_path):
        ctx.registry.register("project", tmp_path)
        (tmp_path / "specs.md").write_text("the real specs\n")
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="specs.md", content_lines=["junk"],
        )
        assert "NOT created" in result and "edit_file" in result
        assert (tmp_path / "specs.md").read_text() == "the real specs\n"

    async def test_a_subdirectory_is_created_on_the_way(self, tools, ctx, tmp_path):
        ctx.registry.register("project", tmp_path)
        await call(
            tools, "create_file", ctx,
            dir_key="project", name="docs/specs.md", content_lines=["hi"],
        )
        assert (tmp_path / "docs" / "specs.md").read_text() == "hi\n"

    async def test_a_name_escaping_the_directory_is_refused(
        self, tools, ctx, tmp_path
    ):
        target = tmp_path / "project"
        target.mkdir()
        ctx.registry.register("project", target)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="../escaped.md", content_lines=["hi"],
        )
        assert "NOT created" in result
        assert not (tmp_path / "escaped.md").exists()

    async def test_an_absolute_name_is_refused(self, tools, ctx, tmp_path):
        ctx.registry.register("project", tmp_path)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="/tmp/escaped.md", content_lines=["hi"],
        )
        assert "NOT created" in result

    async def test_a_file_key_is_not_a_directory(self, tools, ctx, tmp_path):
        f = tmp_path / "reads.bam"
        f.write_text("x")
        ctx.registry.register("reads", f)
        result = await call(
            tools, "create_file", ctx,
            dir_key="reads", name="specs.md", content_lines=["hi"],
        )
        assert "NOT created" in result and "directory" in result

    async def test_a_directory_key_registered_before_it_exists_is_created(
        self, tools, ctx, tmp_path
    ):
        """register_path takes a directory that is not there yet, and the write
        already makes missing parents — so refusing this only sent the model
        into the retry loop for something the tool can just do."""
        ctx.registry.register("results", tmp_path / "results")
        result = await call(
            tools, "create_file", ctx,
            dir_key="results", name="notes.md", content_lines=["hi"],
        )
        assert (tmp_path / "results" / "notes.md").read_text() == "hi\n"
        assert "Created" in result
        # ...and the model is cautioned, not merely informed: this is the one
        # file tool that will happily build a mistyped path instead of failing
        # on it, so the result has to invite a spelling check.
        assert "did not exist and was created" in result
        assert "misspelled" in result
        assert str(tmp_path / "results") in result

    async def test_a_directory_given_as_a_literal_path_is_cautioned_too(
        self, tools, ctx, tmp_path
    ):
        """The case the caution exists for. A literal dir path skips the
        register_path step that used to say "nothing exists there yet", so
        without it the only signal a typo gives is a success message."""
        target = tmp_path / "reslts"  # the typo the user did not make
        result = await call(
            tools, "create_file", ctx,
            dir_key=str(target), name="notes.md", content_lines=["hi"],
        )
        assert (target / "notes.md").is_file()
        assert "did not exist and was created" in result
        assert "misspelled" in result

    async def test_writing_into_an_existing_directory_makes_no_such_claim(
        self, tools, ctx, tmp_path
    ):
        ctx.registry.register("project", tmp_path)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="notes.md", content_lines=["hi"],
        )
        assert "did not exist and was created" not in result

    async def test_a_script_faces_the_same_content_gate(self, tools, ctx, tmp_path):
        # §5.2 is mandatory: a second way to put content into a script file
        # must not be a second way around the syntax/docs gate.
        ctx.registry.register("project", tmp_path)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="qc.py", content_lines=["def broken(:"],
        )
        assert "NOT created" in result and "SyntaxError" in result
        assert not (tmp_path / "qc.py").exists()

    async def test_a_valid_script_is_written(self, tools, ctx, tmp_path):
        ctx.registry.register("project", tmp_path)
        result = await call(
            tools, "create_file", ctx,
            dir_key="project", name="qc.py", content_lines=["print('ok')"],
        )
        assert "NOT created" not in result
        assert (tmp_path / "qc.py").read_text() == "print('ok')\n"

    async def test_creating_a_file_is_not_a_destructive_call(
        self, tools, ctx, tmp_path
    ):
        # It refuses to overwrite, so there is nothing for the §5.3 gate to
        # protect — and a doc write that stops for approval is a doc write the
        # agent stops doing.
        ctx.registry.register("project", tmp_path)
        assert not gates(
            tools, "create_file", ctx,
            dir_key="project", name="specs.md", content_lines=["hi"],
        )


class TestEditTargetPath:
    """edit_target_path is the per-file approval key (§3.5): it must resolve
    exactly what edit_file would touch, and never raise."""

    def test_resolves_a_registered_file(self, ctx, tmp_path):
        from hpca.agent.file_tools import EditFileParams, edit_target_path

        target = tmp_path / "a.txt"
        target.write_text("x\n")
        ctx.registry.register("a", target)
        args = EditFileParams(registry_key="a", old_lines=["x"], new_lines=["y"])
        assert edit_target_path(args, ctx) == target.resolve()

    def test_resolves_a_subpath_to_the_same_key_as_a_direct_key(self, ctx, tmp_path):
        from hpca.agent.file_tools import EditFileParams, edit_target_path

        sub = tmp_path / "dir" / "b.txt"
        sub.parent.mkdir()
        sub.write_text("x\n")
        ctx.registry.register("dir", tmp_path / "dir")
        ctx.registry.register("bfile", sub)
        by_sub = EditFileParams(
            registry_key="dir", subpath="b.txt", old_lines=["x"], new_lines=["y"]
        )
        by_key = EditFileParams(registry_key="bfile", old_lines=["x"], new_lines=["y"])
        assert edit_target_path(by_sub, ctx) == edit_target_path(by_key, ctx)

    def test_unresolvable_returns_none_instead_of_raising(self, ctx):
        from hpca.agent.file_tools import EditFileParams, edit_target_path

        args = EditFileParams(registry_key="nope", old_lines=["x"], new_lines=["y"])
        assert edit_target_path(args, ctx) is None


class TestPlaceholderTracking:
    """Skeleton-then-fill's other half: after every write the result names
    how many `TBD` placeholder lines remain, so the model cannot lose count."""

    async def test_create_with_placeholders_counts_them(self, ctx, tmp_path):

        tools = add_file_tools(ToolRegistry())
        ctx.registry.register("ws", tmp_path)
        result = await call(
            tools,
            "create_file",
            ctx,
            dir_key="ws",
            name="plan.md",
            content_lines=["# t", "## A", "TBD: A", "## B", "TBD: B"],
        )
        assert "2 placeholder line(s) still to fill" in result
        assert "line 3" in result and "TBD: A" in result

    async def test_filling_the_last_placeholder_reports_nothing(self, ctx, tmp_path):
        tools = add_file_tools(ToolRegistry())
        target = tmp_path / "plan.md"
        target.write_text("## A\nTBD: A\n")
        ctx.registry.register("plan", target)
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="plan",
            old_lines=["TBD: A"],
            new_lines=["done content"],
        )
        assert "placeholder" not in result

    async def test_filling_one_of_two_names_the_next(self, ctx, tmp_path):
        tools = add_file_tools(ToolRegistry())
        target = tmp_path / "plan.md"
        target.write_text("## A\nTBD: A\n## B\nTBD: B\n")
        ctx.registry.register("plan", target)
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key="plan",
            old_lines=["TBD: A"],
            new_lines=["content A"],
        )
        assert "1 placeholder line(s) still to fill" in result
        assert "TBD: B" in result


class TestLiteralPaths:
    """A key argument also takes a literal absolute path (§4.3, path-or-key).

    The point is the round-trip a key costs: every agent corpus the model was
    trained on has file tools that take paths, so a path it just saw in `ls`
    output should be usable straight away. It gains a key on the way through,
    which the result reports so the next call can be cheap.
    """

    async def test_edit_file_by_path_and_reports_the_key(self, tools, ctx, tmp_path):
        target = tmp_path / "run.txt"
        target.write_text("alpha\nbeta\n")
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key=str(target),
            old_lines=["beta"],
            new_lines=["gamma"],
        )
        assert target.read_text() == "alpha\ngamma\n"
        assert "is registered as 'run.txt'" in result
        assert ctx.registry.resolve("run.txt") == target

    async def test_create_file_by_directory_path(self, tools, ctx, tmp_path):
        result = await call(
            tools,
            "create_file",
            ctx,
            dir_key=str(tmp_path),
            name="specs.md",
            content_lines=["# specs"],
        )
        assert (tmp_path / "specs.md").read_text() == "# specs\n"
        assert f"{tmp_path} is registered as" in result

    async def test_delete_file_by_path(self, tools, ctx, tmp_path):
        victim = tmp_path / "stale.log"
        victim.write_text("stale")
        result = await call(tools, "delete_file", ctx, registry_key=str(victim))
        assert not victim.exists()
        assert "recoverable" in result
        # The key it got is dropped again by the deletion itself, so the result
        # names none — nothing is left pointing at a file that is gone.
        assert ctx.registry.list() == {}

    async def test_delete_by_path_is_still_gated(self, tools, ctx, tmp_path):
        """The HITL gate keys on resolvability, so it has to see paths too —
        otherwise a literal path would be the way around §5.3."""
        victim = tmp_path / "stale.log"
        victim.write_text("stale")
        assert gates(tools, "delete_file", ctx, registry_key=str(victim)) is True
        # …and asking registers nothing: predicates stay side-effect free
        assert ctx.registry.list() == {}

    async def test_move_file_by_paths(self, tools, ctx, tmp_path):
        source = tmp_path / "a.txt"
        source.write_text("content")
        dest = tmp_path / "dest"
        dest.mkdir()
        result = await call(
            tools, "move_file", ctx,
            source_key=str(source), dest_dir_key=str(dest),
        )
        assert (dest / "a.txt").read_text() == "content"
        # the key follows the file, exactly as it does for a key-named move
        assert ctx.registry.resolve("a.txt") == dest / "a.txt"
        assert "'a.txt'" in result
        assert f"{dest} is registered as" in result

    async def test_copy_file_by_paths(self, tools, ctx, tmp_path):
        source = tmp_path / "a.txt"
        source.write_text("content")
        dest = tmp_path / "dest"
        dest.mkdir()
        result = await call(
            tools, "copy_file", ctx,
            source_key=str(source), dest_dir_key=str(dest),
        )
        assert source.exists()
        assert (dest / "a.txt").read_text() == "content"
        assert "a.txt_copy" in result  # hint off the key, not off the whole path

    async def test_path_inside_a_registered_directory_gets_its_own_key(
        self, tools, ctx, tmp_path
    ):
        """A registered directory does not claim what is under it: the file
        gets a key of its own and the directory key is untouched."""
        ctx.registry.register("ws", tmp_path)
        target = tmp_path / "run.txt"
        target.write_text("alpha\n")
        await call(
            tools, "edit_file", ctx,
            registry_key=str(target), old_lines=["alpha"], new_lines=["beta"],
        )
        assert ctx.registry.list() == {"ws": tmp_path, "run.txt": target}

    async def test_a_key_wins_over_a_path(self, tools, ctx, tmp_path):
        """Precedence is key-first, so a session's own names never change
        meaning — and a key can never be shaped like a path anyway."""
        named = tmp_path / "named.txt"
        named.write_text("alpha\n")
        ctx.registry.register("target", named)
        result = await call(
            tools, "edit_file", ctx,
            registry_key="target", old_lines=["alpha"], new_lines=["beta"],
        )
        assert named.read_text() == "beta\n"
        assert "is registered as" not in result  # nothing new was registered

    async def test_tilde_is_expanded(self, tools, ctx, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "notes.md"
        target.write_text("alpha\n")
        result = await call(
            tools, "edit_file", ctx,
            registry_key="~/notes.md", old_lines=["alpha"], new_lines=["beta"],
        )
        assert target.read_text() == "beta\n"
        assert "is registered as 'notes.md'" in result

    async def test_a_bare_name_still_errors_with_the_key_list(
        self, tools, ctx, tmp_path
    ):
        ctx.registry.register("ws", tmp_path)
        with pytest.raises(UnknownKeyError) as exc:
            await call(
                tools, "edit_file", ctx,
                registry_key="ghost", old_lines=["a"], new_lines=["b"],
            )
        assert "ws" in str(exc.value)

    async def test_a_relative_path_still_errors(self, tools, ctx, tmp_path):
        ctx.registry.register("ws", tmp_path)
        with pytest.raises(UnknownKeyError):
            await call(
                tools, "create_file", ctx,
                dir_key="results/out", name="x.md", content_lines=["x"],
            )
