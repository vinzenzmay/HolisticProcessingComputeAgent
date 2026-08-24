"""Tests for hpca.agent.file_tools (§5.1, §5.3): key-based, trash-backed ops."""

import pytest

from hpca.agent import hints
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools, edit_preview
from hpca.agent.history import ELISION_SENTINEL, omitted_list
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.runner import ProcessRunner
from hpca.trash import TrashManager


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield ToolContext(
        workdir=tmp_path,
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


class TestDeleteFile:
    async def test_gated_when_key_resolves(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        assert gates(tools, "delete_file", ctx, path=str(f)) is True

    async def test_a_path_that_is_not_there_is_not_gated(self, tools, ctx, tmp_path):
        # no pointless approval modal; the handler's error feeds the retry loop
        assert gates(tools, "delete_file", ctx, path=str(tmp_path / "ghost")) is False

    async def test_deletes_via_trash(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        result = await call(tools, "delete_file", ctx, path=str(f))
        assert not f.exists()
        # names the tool that undoes it, not just "the trash": without that the
        # model tells the user to go dig through the app dir themselves
        assert "recoverable" in result and "restore_file" in result
        assert ctx.trash.list()[0].original_path == f

    async def test_oversized_states_no_backup(self, tools, ctx, tmp_path):
        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        f = tmp_path / "big.bin"
        f.write_text("more than two bytes")
        result = await call(tools, "delete_file", ctx, path=str(f))
        assert "WITHOUT backup" in result

    async def test_missing_subpath_is_a_useful_error(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        # a bad subpath is not gated, and the handler explains rather than crashes
        assert (
            gates(tools, "delete_file", ctx, path=str(d / "ghost"))
            is False
        )
        result = await call(
            tools, "delete_file", ctx, path=str(d / "ghost")
        )
        assert "No such path" in result

class TestRestoreFile:
    async def test_not_gated(self, tools, ctx):
        # restore never overwrites, so it needs no approval modal
        assert gates(tools, "restore_file", ctx, path="/anything") is False

    async def test_restores_a_deleted_file_and_registers_it(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        await call(tools, "delete_file", ctx, path=str(f))
        result = await call(tools, "restore_file", ctx, path=str(f))
        assert f.read_text() == "precious"
        assert str(f) in result
        assert ctx.trash.list() == []

    async def test_restores_by_file_name(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        await call(tools, "delete_file", ctx, path=str(f))
        await call(tools, "restore_file", ctx, path="x.txt")
        assert f.read_text() == "precious"

    async def test_empty_path_lists_the_trash(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        await call(tools, "delete_file", ctx, path=str(f))
        result = await call(tools, "restore_file", ctx)
        assert str(f) in result
        assert not f.exists()  # listing restores nothing

    async def test_empty_trash_says_so(self, tools, ctx):
        assert "empty" in await call(tools, "restore_file", ctx)

    async def test_unknown_path_lists_what_is_there(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        await call(tools, "delete_file", ctx, path=str(f))
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
            await call(tools, "delete_file", ctx, path=str(victim))
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
            await call(tools, "delete_file", ctx, path=str(f))
        await call(tools, "restore_file", ctx, path=str(f))
        assert f.read_text() == "second"

    async def test_refuses_when_the_path_is_occupied(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("precious")
        await call(tools, "delete_file", ctx, path=str(f))
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
        aside = tmp_path / "aside"
        aside.mkdir()
        await call(
            tools,
            "edit_file",
            ctx,
            path=str(f),
            old_lines=["echo before"],
            new_lines=["echo after"],
        )
        assert "already exists" in await call(tools, "restore_file", ctx, path=str(f))
        await call(
            tools, "move_file", ctx, source_path=str(f), dest_path=str(aside)
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
        await call(tools, "delete_file", ctx, path=str(f))
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
        await call(tools, "move_file", ctx, source_path=str(f), dest_path=str(d))
        (d / "a.txt").unlink()
        await call(tools, "restore_file", ctx, path=str(d / "a.txt"))
        assert (d / "a.txt").read_text() == "old content"


class TestMoveFile:
    async def test_fresh_target_is_not_gated(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        d = tmp_path / "dest"
        d.mkdir()
        assert gates(
            tools, "move_file", ctx, source_path=str(f), dest_path=str(d)
        ) is False
        await call(tools, "move_file", ctx, source_path=str(f), dest_path=str(d))
        assert not f.exists()
        assert (d / "a.txt").read_text() == "content"

    async def test_move_over_existing_gated_and_target_trashed(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "a.txt"
        f.write_text("new content")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old content")
        assert gates(
            tools, "move_file", ctx, source_path=str(f), dest_path=str(d)
        ) is True
        result = await call(
            tools, "move_file", ctx, source_path=str(f), dest_path=str(d)
        )
        assert (d / "a.txt").read_text() == "new content"
        assert "trash" in result
        trashed = ctx.trash.list()[0]
        assert trashed.trashed_path.read_text() == "old content"

    async def test_rename_via_new_name(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        d = tmp_path / "dest"
        d.mkdir()
        await call(
            tools, "move_file", ctx,
            source_path=str(f),
            dest_path=str(d / "b.txt")
        )
        assert (d / "b.txt").exists()

    async def test_a_source_that_is_not_there_is_refused_not_raised(
        self, tools, ctx, tmp_path
    ):
        d = tmp_path / "dest"
        d.mkdir()
        result = await call(
            tools, "move_file", ctx,
            source_path=str(tmp_path / "nope.txt"), dest_path=str(d),
        )
        assert "NOT moved" in result and "No such path" in result

class TestCopyFile:
    async def test_copy_leaves_the_source_in_place(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        d = tmp_path / "dest"
        d.mkdir()
        assert gates(
            tools, "copy_file", ctx, source_path=str(f), dest_path=str(d)
        ) is False
        result = await call(
            tools, "copy_file", ctx, source_path=str(f), dest_path=str(d)
        )
        assert f.exists()
        assert (d / "a.txt").read_text() == "content"
        assert str(d / "a.txt") in result

    async def test_copy_over_existing_gated(self, tools, ctx, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("new")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old")
        assert gates(
            tools, "copy_file", ctx, source_path=str(f), dest_path=str(d)
        ) is True

class TestEditFile:
    """Replace an exact run of lines in place, without re-sending the file."""

    @pytest.fixture
    def script(self, ctx, tmp_path):
        path = tmp_path / "run.sh"
        path.write_text("#!/bin/bash\nset -euo pipefail\necho one\necho two\n")
        return path

    async def test_replaces_the_matched_lines_and_leaves_the_rest(
        self, tools, ctx, script
    ):
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(script),
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
            path=str(script),
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
            path=str(script),
            old_lines=["echo one", "echo two"],
            new_lines=["echo both"],
        )
        assert script.read_text() == "#!/bin/bash\nset -euo pipefail\necho both\n"

    async def test_the_previous_content_goes_to_the_trash(self, tools, ctx, script):
        await call(
            tools,
            "edit_file",
            ctx,
            path=str(script),
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
            path=str(script),
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
            path=str(script),
            old_lines=["    echo one"],
            new_lines=["echo x"],
        )
        assert "NOT edited" in result
        assert "line 3" in result

    async def test_an_ambiguous_match_is_refused(self, tools, ctx, tmp_path):
        path = tmp_path / "twice.sh"
        path.write_text("echo hi\necho mid\necho hi\n")
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(path),
            old_lines=["echo hi"],
            new_lines=["echo bye"],
        )
        assert "NOT edited" in result and "2 times" in result
        assert path.read_text() == "echo hi\necho mid\necho hi\n"

    async def test_a_directory_key_says_to_use_subpath(self, tools, ctx, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(d),
            old_lines=["x"],
            new_lines=["y"],
        )
        assert "NOT edited" in result and "subpath" in result

    async def test_a_binary_file_is_refused(self, tools, ctx, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"\x00\x01\x82\xff binary")
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(path),
            old_lines=["x"],
            new_lines=["y"],
        )
        assert "NOT edited" in result
        assert path.read_bytes() == b"\x00\x01\x82\xff binary"

class TestEditRepairAndFuzz:
    """Absorb model imperfection inside the tool (PI-style): repair the
    arguments and climb a matching ladder instead of bouncing a near-miss back
    for another round-trip plus a re-read of the file."""

    @pytest.fixture
    def notes(self, ctx, tmp_path):
        # .txt on purpose: no §5.2 gate, these tests are about matching only
        path = tmp_path / "notes.txt"
        path.write_text("alpha\nbravo\ncharlie\n")
        return path

    async def test_embedded_newlines_in_old_and_new_lines_are_split(
        self, tools, ctx, notes
    ):
        # One array element holding two lines still means two lines.
        result = await call(
            tools, "edit_file", ctx,
            path=str(notes),
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
            path=str(notes),
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
            path=str(notes),
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
        await call(
            tools, "edit_file", ctx,
            path=str(path),
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
        result = await call(
            tools, "edit_file", ctx,
            path=str(path),
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
        await call(
            tools, "edit_file", ctx,
            path=str(path),
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
        result = await call(
            tools, "edit_file", ctx,
            path=str(path),
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
        before = path.read_text()
        result = await call(
            tools, "edit_file", ctx,
            path=str(path),
            old_lines=["echo hi"],
            new_lines=["echo bye"],
        )
        assert "NOT edited" in result and "2 times" in result
        assert path.read_text() == before

    async def test_a_crlf_file_keeps_its_line_endings(self, tools, ctx, tmp_path):
        path = tmp_path / "win.txt"
        path.write_bytes(b"alpha\r\nbravo\r\ncharlie\r\n")
        await call(
            tools, "edit_file", ctx,
            path=str(path),
            old_lines=["bravo"],
            new_lines=["BRAVO"],
        )
        assert path.read_bytes() == b"alpha\r\nBRAVO\r\ncharlie\r\n"

    async def test_a_bom_survives_the_edit(self, tools, ctx, tmp_path):
        path = tmp_path / "bom.txt"
        path.write_bytes("﻿alpha\nbravo\n".encode())
        await call(
            tools, "edit_file", ctx,
            path=str(path),
            old_lines=["alpha"],
            new_lines=["ALPHA"],
        )
        assert path.read_bytes() == "﻿ALPHA\nbravo\n".encode()

    async def test_an_identical_replacement_is_an_error_not_a_write(
        self, tools, ctx, notes
    ):
        result = await call(
            tools, "edit_file", ctx,
            path=str(notes),
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
            path=str(notes),
            old_lines=["bravo"],
            new_lines=["BRAVO"],
        )
        assert result == (
            f"Edited {notes} at line 2: 1 line → 1 line. "
            "Do not read it back to check."
        )
        assert ctx.trash.list()  # the backup itself is still kept

    async def test_success_says_not_to_read_the_file_back(
        self, tools, ctx, notes
    ):
        """The one clause the short line pays for.

        Measured on the live 27B: a write that succeeds is followed by a
        read_file on the file just written, purely to confirm it landed. The
        result message is the only place that habit can be talked out of, and
        the round trip it saves is worth the words it costs.
        """
        result = await call(
            tools, "edit_file", ctx,
            path=str(notes),
            old_lines=["bravo"],
            new_lines=["BRAVO"],
        )
        assert "Do not read it back" in result

    async def test_success_warns_only_when_no_backup_could_be_kept(
        self, tools, ctx, tmp_path
    ):
        from hpca.trash import TrashManager

        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        path = tmp_path / "big.txt"
        path.write_text("more than two bytes\n")
        result = await call(
            tools, "edit_file", ctx,
            path=str(path),
            old_lines=["more than two bytes"],
            new_lines=["tiny"],
        )
        assert "NO backup" in result and "cannot be undone" in result

    async def test_create_file_splits_embedded_newlines_too(
        self, tools, ctx, tmp_path
    ):
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "notes.md"),
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
        return path

    async def test_an_edit_that_breaks_the_syntax_is_not_applied(
        self, tools, ctx, script
    ):
        before = script.read_text()
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(script),
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
            path=str(script),
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
        await call(
            tools,
            "edit_file",
            ctx,
            path=str(path),
            old_lines=["second"],
            new_lines=["if true; then"],
        )
        assert path.read_text() == "first\nif true; then\n"


class TestEditGating:
    async def test_an_edit_gates_as_destructive(self, tools, ctx, tmp_path):
        path = tmp_path / "run.sh"
        path.write_text("echo one\n")
        assert gates(
            tools,
            "edit_file",
            ctx,
            path=str(path),
            old_lines=["echo one"],
            new_lines=["echo two"],
        )

    async def test_a_path_that_is_not_there_is_not_gated(self, tools, ctx, tmp_path):
        # A dud call goes to the error-feedback loop, not to the user.
        assert not gates(
            tools, "edit_file", ctx,
            path=str(tmp_path / "nope.txt"), old_lines=["x"], new_lines=["y"],
        )

    async def test_the_description_shows_the_path_and_the_backup(
        self, tools, ctx, tmp_path
    ):
        path = tmp_path / "run.sh"
        path.write_text("echo one\n")
        tool = tools.get("edit_file")
        text = tool.describe_call(
            tool.params.model_validate(
                {
                    "path": str(path),
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
        tool = tools.get("edit_file")
        text = tool.describe_call(
            tool.params.model_validate(
                {"path": str(path), "old_lines": ["many"], "new_lines": ["few"]}
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


class TestWhatTheToolDescriptionsPromise:
    """The description is what the model reads while it is deciding, and the
    result is what it reads afterwards. When the two disagree about whether a
    deletion can be undone, the one that shapes the decision is the one that
    was wrong — so the size limit belongs in both."""

    def test_delete_points_at_edit_file_for_changing_a_file(self, tools):
        # The delete-then-recreate loop this whole line of work started from:
        # a model that wants to change three lines reaches for the tool whose
        # description does not mention the alternative.
        assert "edit_file" in tools.get("delete_file").description

    def test_delete_does_not_promise_an_unconditional_backup(self, tools):
        # It said "(trash-backed)" flat out, which is false for any file above
        # safety.backup_limit_gb — the handler says so, but only once the file
        # is already gone.
        description = tools.get("delete_file").description
        assert "backup size limit" in description

    def test_the_create_exists_hint_routes_to_edit_file(self):
        assert "edit_file" in hints.CREATE_FILE_EXISTS

    def test_the_create_exists_hint_does_not_name_the_delete_tool(self):
        # It used to read "to replace it wholesale, delete_file first (the old
        # version stays recoverable from the trash)" — a named tool plus a
        # reassurance, which is a recipe and was followed as one. The caveat
        # survives; the tool name does not, because the name is the part that
        # gets copied into the next call.
        assert "delete_file" not in hints.CREATE_FILE_EXISTS


class TestDescribeCall:
    async def test_delete_description_shows_resolved_path(self, tools, ctx, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello")
        tool = tools.get("delete_file")
        text = tool.describe_call(
            tool.params.model_validate({"path": str(f)}), ctx
        )
        assert str(f) in text
        assert "recoverable" in text

    async def test_delete_description_warns_no_backup(self, tools, ctx, tmp_path):
        ctx.trash = TrashManager(tmp_path / "trash", backup_limit_bytes=2)
        f = tmp_path / "big.bin"
        f.write_text("many bytes here")
        tool = tools.get("delete_file")
        text = tool.describe_call(
            tool.params.model_validate({"path": str(f)}), ctx
        )
        assert "NO BACKUP" in text

    async def test_move_description_shows_both_paths_and_overwrite(
        self, tools, ctx, tmp_path
    ):
        f = tmp_path / "a.txt"
        f.write_text("new")
        d = tmp_path / "dest"
        d.mkdir()
        (d / "a.txt").write_text("old")
        tool = tools.get("move_file")
        text = tool.describe_call(
            tool.params.model_validate(
                {"source_path": str(f), "dest_path": str(d)}
            ),
            ctx,
        )
        assert str(f) in text and str(d / "a.txt") in text
        assert "OVERWRITES" in text

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
        await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"),
            content_lines=["# locus-cutter", "", "## Why"],
        )
        assert (tmp_path / "specs.md").read_text() == "# locus-cutter\n\n## Why\n"

    async def test_an_existing_file_is_not_overwritten(self, tools, ctx, tmp_path):
        (tmp_path / "specs.md").write_text("the real specs\n")
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"), content_lines=["junk"],
        )
        assert "NOT created" in result and "edit_file" in result
        # The refusal names the two ways forward and stops there. Offering
        # read_file as a third only invited the model to fetch content it is
        # about to replace anyway, and every word here is read on every call.
        assert "read_file" not in result
        assert (tmp_path / "specs.md").read_text() == "the real specs\n"

    async def test_success_says_not_to_read_the_file_back(
        self, tools, ctx, tmp_path
    ):
        """See the same test on edit_file: the 27B confirms its own writes by
        reading them back, which costs a round trip and puts the lines it just
        sent back into the context. Nothing here may point at read_file as the
        way to get the content back — that was the invitation."""
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"), content_lines=["hi"],
        )
        assert "Do not read it back" in result
        assert "read_file" not in result

    async def test_a_subdirectory_is_created_on_the_way(self, tools, ctx, tmp_path):
        await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "docs/specs.md"), content_lines=["hi"],
        )
        assert (tmp_path / "docs" / "specs.md").read_text() == "hi\n"

    async def test_dot_dot_is_folded_rather_than_refused(
        self, tools, ctx, tmp_path
    ):
        """The old interface refused a name that climbed out of its dir_key,
        because the pair (directory, name) made "out of it" a thing that could
        happen. A path has no such inside: `..` is just part of the path, so it
        is normalized and the file lands where it says."""
        target = tmp_path / "project"
        target.mkdir()
        result = await call(
            tools, "create_file", ctx,
            path=str(target / ".." / "beside.md"), content_lines=["hi"],
        )
        assert (tmp_path / "beside.md").read_text() == "hi\n"
        assert str(tmp_path / "beside.md") in result

    async def test_a_file_key_is_not_a_directory(self, tools, ctx, tmp_path):
        f = tmp_path / "reads.bam"
        f.write_text("x")
        result = await call(
            tools, "create_file", ctx,
            path=str(f / "specs.md"), content_lines=["hi"],
        )
        assert "NOT created" in result and "directory" in result

    async def test_a_directory_key_registered_before_it_exists_is_created(
        self, tools, ctx, tmp_path
    ):
        """register_path takes a directory that is not there yet, and the write
        already makes missing parents — so refusing this only sent the model
        into the retry loop for something the tool can just do."""
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "results" / "notes.md"), content_lines=["hi"],
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
        """The case the caution exists for. Nothing else about a mistyped
        directory is visible to the model — the write makes it, so without the
        caution the only signal a typo gives is a success message."""
        target = tmp_path / "reslts"  # the typo the user did not make
        result = await call(
            tools, "create_file", ctx,
            path=str(target / "notes.md"), content_lines=["hi"],
        )
        assert (target / "notes.md").is_file()
        assert "did not exist and was created" in result
        assert "misspelled" in result

    async def test_writing_into_an_existing_directory_makes_no_such_claim(
        self, tools, ctx, tmp_path
    ):
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "notes.md"), content_lines=["hi"],
        )
        assert "did not exist and was created" not in result

    async def test_a_script_faces_the_same_content_gate(self, tools, ctx, tmp_path):
        # §5.2 is mandatory: a second way to put content into a script file
        # must not be a second way around the syntax/docs gate.
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "qc.py"), content_lines=["def broken(:"],
        )
        assert "NOT created" in result and "SyntaxError" in result
        assert not (tmp_path / "qc.py").exists()

    async def test_a_valid_script_is_written(self, tools, ctx, tmp_path):
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "qc.py"), content_lines=["print('ok')"],
        )
        assert "NOT created" not in result
        assert (tmp_path / "qc.py").read_text() == "print('ok')\n"

    async def test_creating_a_file_is_not_a_destructive_call(
        self, tools, ctx, tmp_path
    ):
        # It refuses to overwrite, so there is nothing for the §5.3 gate to
        # protect — and a doc write that stops for approval is a doc write the
        # agent stops doing.
        assert not gates(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"), content_lines=["hi"],
        )


class TestEditTargetPath:
    """edit_target_path is the per-file approval key (§3.5): it must resolve
    exactly what edit_file would touch, and never raise."""

    def test_resolves_a_registered_file(self, ctx, tmp_path):
        from hpca.agent.file_tools import EditFileParams, edit_target_path

        target = tmp_path / "a.txt"
        target.write_text("x\n")
        args = EditFileParams(path=str(target), old_lines=["x"], new_lines=["y"])
        assert edit_target_path(args, ctx) == target.resolve()

    def test_unresolvable_returns_none_instead_of_raising(self, ctx):
        from hpca.agent.file_tools import EditFileParams, edit_target_path

        args = EditFileParams(path="", old_lines=["x"], new_lines=["y"])
        assert edit_target_path(args, ctx) is None


class TestPlaceholderTracking:
    """Skeleton-then-fill's other half: after every write the result names
    how many `TBD` placeholder lines remain, so the model cannot lose count."""

    async def test_create_with_placeholders_counts_them(self, ctx, tmp_path):

        tools = add_file_tools(ToolRegistry())
        result = await call(
            tools,
            "create_file",
            ctx,
            path=str(tmp_path / "plan.md"),
            content_lines=["# t", "## A", "TBD: A", "## B", "TBD: B"],
        )
        assert "2 placeholder line(s) still to fill" in result
        assert "line 3" in result and "TBD: A" in result

    async def test_filling_the_last_placeholder_reports_nothing(self, ctx, tmp_path):
        tools = add_file_tools(ToolRegistry())
        target = tmp_path / "plan.md"
        target.write_text("## A\nTBD: A\n")
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(target),
            old_lines=["TBD: A"],
            new_lines=["done content"],
        )
        assert "placeholder" not in result

    async def test_filling_one_of_two_names_the_next(self, ctx, tmp_path):
        tools = add_file_tools(ToolRegistry())
        target = tmp_path / "plan.md"
        target.write_text("## A\nTBD: A\n## B\nTBD: B\n")
        result = await call(
            tools,
            "edit_file",
            ctx,
            path=str(target),
            old_lines=["TBD: A"],
            new_lines=["content A"],
        )
        assert "1 placeholder line(s) still to fill" in result
        assert "TBD: B" in result


class TestPathArguments:
    """What a path argument means, now that it is the only thing a tool takes.

    The registry's whole job was to spare a small model from re-typing a long
    path; these are the cases that used to need a key and now do not.
    """

    async def test_an_absolute_path_is_used_as_given(self, tools, ctx, tmp_path):
        target = tmp_path / "notes.md"
        target.write_text("alpha\n")
        result = await call(
            tools, "edit_file", ctx,
            path=str(target), old_lines=["alpha"], new_lines=["beta"],
        )
        assert target.read_text() == "beta\n"
        assert str(target) in result

    async def test_a_relative_path_is_taken_from_the_workdir(
        self, tools, ctx, tmp_path
    ):
        """The case the registry answered with UnknownKeyError, listing keys
        the model had never chosen. `.` and `results/x.md` are now ordinary."""
        result = await call(
            tools, "create_file", ctx,
            path="results/out/x.md", content_lines=["x"],
        )
        assert (tmp_path / "results" / "out" / "x.md").read_text() == "x\n"
        assert str(tmp_path / "results" / "out" / "x.md") in result

    async def test_a_bare_dot_is_the_working_directory(self, tools, ctx, tmp_path):
        result = await call(
            tools, "create_file", ctx, path="./plan.md", content_lines=["# plan"],
        )
        assert (tmp_path / "plan.md").read_text() == "# plan\n"
        assert str(tmp_path / "plan.md") in result

    async def test_tilde_is_expanded(self, tools, ctx, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "notes.md"
        target.write_text("alpha\n")
        result = await call(
            tools, "edit_file", ctx,
            path="~/notes.md", old_lines=["alpha"], new_lines=["beta"],
        )
        assert target.read_text() == "beta\n"
        # the result prints the real path, never the ~ the model sent: what it
        # says happened has to be checkable against the filesystem
        assert str(target) in result

    async def test_delete_by_path_is_still_gated(self, tools, ctx, tmp_path):
        """The HITL gate keys on resolvability, so it has to see the path —
        otherwise naming a file differently would be the way around §5.3."""
        victim = tmp_path / "stale.log"
        victim.write_text("stale")
        assert gates(tools, "delete_file", ctx, path=str(victim)) is True
        assert victim.exists()  # asking changed nothing: predicates stay pure

    async def test_move_by_paths(self, tools, ctx, tmp_path):
        source = tmp_path / "a.txt"
        source.write_text("content")
        dest = tmp_path / "dest"
        dest.mkdir()
        result = await call(
            tools, "move_file", ctx,
            source_path=str(source), dest_path=str(dest),
        )
        assert (dest / "a.txt").read_text() == "content"
        assert str(dest / "a.txt") in result

    async def test_copy_by_paths(self, tools, ctx, tmp_path):
        source = tmp_path / "a.txt"
        source.write_text("content")
        dest = tmp_path / "dest"
        dest.mkdir()
        result = await call(
            tools, "copy_file", ctx,
            source_path=str(source), dest_path=str(dest),
        )
        assert source.exists()
        assert (dest / "a.txt").read_text() == "content"
        assert str(dest / "a.txt") in result

    async def test_a_dest_that_is_not_a_directory_is_the_target_name(
        self, tools, ctx, tmp_path
    ):
        """`mv a.txt b.txt` renames. The old interface had no way to say that
        without a separate new_name argument."""
        source = tmp_path / "a.txt"
        source.write_text("content")
        await call(
            tools, "move_file", ctx,
            source_path=str(source), dest_path=str(tmp_path / "b.txt"),
        )
        assert (tmp_path / "b.txt").read_text() == "content"
        assert not source.exists()


class TestElisionMarkerGuard:
    """Content the model copied out of its own history is not content.

    hpca.agent.history replaces a big payload in the assistant copy of a call
    with a descriptor of what it left out. Asked to rewrite a file it had
    written earlier, the model sent that descriptor back as the new
    ``content_lines``: the marker landed on disk, the file shrank to what had
    survived the elision, and the next rewrite elided *that*. Fifteen files in
    one session, and nothing in the tools noticed, because a marker is
    perfectly valid text. So both writing tools refuse it — except in
    ``old_lines``, which is how an already-corrupted file gets repaired.
    """

    @pytest.fixture
    def marker(self):
        return omitted_list([f"line {n}" for n in range(97)])

    async def test_create_file_refuses_a_payload_carrying_the_sentinel(
        self, tools, ctx, tmp_path, marker
    ):
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"),
            content_lines=["# locus-cutter", marker],
        )
        assert "NOT created" in result
        assert not (tmp_path / "specs.md").exists()

    async def test_create_file_refuses_the_pre_0_23_3_wording_too(
        self, tools, ctx, tmp_path
    ):
        """It survives in checkpointed sessions and in the files written from
        one, so the guard has to know the marker it was corrupted with."""
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"),
            content_lines=["# locus-cutter", "... 22 more lines elided ..."],
        )
        assert "NOT created" in result
        assert not (tmp_path / "specs.md").exists()

    async def test_the_refusal_names_the_offending_line(
        self, tools, ctx, tmp_path, marker
    ):
        # The model has to be able to find it: it sent the payload, not a file.
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "specs.md"),
            content_lines=["# locus-cutter", "", marker],
        )
        assert "line 3" in result
        # ...and is told where the real text is, rather than only that it lost.
        assert "read_file" in result
        # But the line is NOT quoted back, which every other refusal here does.
        # Measured on the live 27B (evals/edit_eval.py, second_file_after_first):
        # a refusal carrying the marker returned it to the context, the model
        # built its next call out of the refusal it had just read, and that call
        # was refused in the same words seventeen times until the decision
        # budget ran out. The line number locates the line without reprinting
        # the one string that must not go round again.
        assert ELISION_SENTINEL not in result

    async def test_edit_file_refuses_marker_carrying_new_lines(
        self, tools, ctx, tmp_path, marker
    ):
        path = tmp_path / "specs.md"
        path.write_text("# locus-cutter\n\n## Why\n")
        before = path.read_bytes()
        result = await call(
            tools, "edit_file", ctx,
            path=str(path), old_lines=["## Why"], new_lines=[marker],
        )
        assert "NOT edited" in result
        assert path.read_bytes() == before
        assert ctx.trash.list() == []  # refused before anything was backed up

    async def test_edit_file_accepts_marker_carrying_old_lines(
        self, tools, ctx, tmp_path, marker
    ):
        """The repair path, and the reason old_lines is not guarded: a file
        that already has the marker written into it can only be fixed by
        matching that line and replacing it with the real content."""
        path = tmp_path / "specs.md"
        path.write_text(f"# locus-cutter\n{marker}\n")
        result = await call(
            tools, "edit_file", ctx,
            path=str(path),
            old_lines=[marker],
            new_lines=["## Why", "", "Because the reads are 150bp."],
        )
        assert "Edited" in result
        assert path.read_text() == (
            "# locus-cutter\n## Why\n\nBecause the reads are 150bp.\n"
        )

    async def test_prose_that_merely_talks_about_elision_is_written(
        self, tools, ctx, tmp_path
    ):
        # The guard looks for the marker, not for the subject: a document may
        # say "elided" as often as it likes.
        result = await call(
            tools, "create_file", ctx,
            path=str(tmp_path / "notes.md"),
            content_lines=["3 lines elided from the log", "no marker here"],
        )
        assert "Created" in result
        assert (tmp_path / "notes.md").read_text() == (
            "3 lines elided from the log\nno marker here\n"
        )
