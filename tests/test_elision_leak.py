"""The seam between hpca.agent.history and the tools that write files.

Each side is covered on its own — test_history for when a payload is folded
out of the model's view, test_file_tools and test_builtin_tools for the guard
that refuses one coming back. What is only visible from here is the loop they
close, and the loop is what actually went wrong in the field:

    create_file writes a 40-line TSV
      → the model's record of that call is folded to a description of it
      → the model, still mid-job, writes the second file from that record
      → the description lands on disk and the file is now 4 lines
      → folding *that* leaves 4 lines again, so every retry confirms the loss

and the model, watching its files come back short, concluded its writer was
truncating and spent a dozen rounds bisecting a bug that was never there.

The fix is that the fold is now a property of *age*, not of writing: the last
few calls keep their payloads, so the record a model reaches for while it is
still working is intact and there is nothing wrong to copy. The tool guard
stays as a backstop for the folded records that do exist — old ones, and the
ones already sitting in sessions checkpointed before the change — and these
tests hold both halves: that a fresh record survives, and that a folded one
cannot reach disk.

Deliberately built on the real elision and the real tools rather than on a
marker string either side spells out, so a change to the fold is caught here
instead of quietly passing.
"""

import json

import pytest

from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.history import (
    KEEP_RECENT_CALLS,
    fold_old_payloads,
    tool_call_message,
)
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.trash import TrashManager

# The file from the session this bug was found in, in miniature. Well past the
# fold's character budget, because that is the only kind of payload the fold
# now touches at all — the 24-row annotation that actually got corrupted would
# today be carried whole no matter how old it got, which is itself half the
# fix.
ANNOTATION = [
    "# UMAP cluster annotation - 20260423_results",
    "# Source: 20260423_fig3_streamlined.Rmd, 'umap' block",
    "#",
    "cluster_no\tcell_type\tlong_annotation\tlineage",
] + [
    f"{i}\tCd8_eff_like\tCd8 effector-like cell, cluster {i} of the rpca UMAP\tCd8"
    for i in range(120)
]

# What a session checkpointed before 0.23.3 still carries, and what the files
# already written from one contain. The guard has to know this shape too, or
# resuming such a session walks straight back into the cascade.
LEGACY_ELIDED = ANNOTATION[:3] + [f"... {len(ANNOTATION) - 3} more lines elided ..."]


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


def history_with(write, *, followed_by=0):
    """A call, optionally buried under later ones, as the model would see it.

    Goes through ``fold_old_payloads`` rather than folding by hand: these tests
    are about what the model is actually handed, and hand-rolling that would
    let the two drift apart without anything failing.
    """
    messages = [tool_call_message("create_file", write)]
    for n in range(followed_by):
        messages.append(tool_call_message("read_file", {"registry_key": f"k{n}"}))
        messages.append({"role": "user", "content": f"[tool result] read_file: {n}"})
    return fold_old_payloads(messages)


def payload_of(message):
    """The ``content_lines`` the model is handed — the list itself when the
    record is whole, the descriptor string once it has been folded.

    Decoded rather than substring-matched against the serialized envelope: the
    rows carry tabs, JSON escapes them, and a test that compared raw text would
    fail for a reason that has nothing to do with folding.
    """
    envelope = json.loads(str(message.get("content") or ""))
    return envelope["arguments"]["content_lines"]


class TestAFreshRecordSurvives:
    """The half of the fix that removes the bug rather than catching it."""

    async def test_the_call_just_made_still_carries_its_payload(
        self, tools, ctx, tmp_path
    ):
        """The regression test for the original failure. The model rewrites a
        file while it is still working on it, so the record it reads back is
        minutes old at most — and that record now holds the lines."""
        write = {
            "dir_key": str(tmp_path),
            "name": "annotation.tsv",
            "content_lines": ANNOTATION,
        }
        await call(tools, "create_file", ctx, **write)
        assert payload_of(history_with(write)[0]) == ANNOTATION

    async def test_a_second_file_written_from_that_record_is_correct(
        self, tools, ctx, tmp_path
    ):
        """End to end: the exact move that corrupted fifteen files. The model
        reads its own record and writes the rows it finds there — which is now
        the right thing to do, so the file comes out whole."""
        write = {
            "dir_key": str(tmp_path),
            "name": "annotation.tsv",
            "content_lines": ANNOTATION,
        }
        await call(tools, "create_file", ctx, **write)
        # what the model has in front of it when it composes the next call
        seen = payload_of(history_with(write)[0])
        assert seen == ANNOTATION
        result = await call(
            tools,
            "create_file",
            ctx,
            dir_key=str(tmp_path),
            name="annotation2.tsv",
            content_lines=ANNOTATION,
        )
        assert result.startswith("Created")
        assert (tmp_path / "annotation2.tsv").read_text() == "\n".join(ANNOTATION) + "\n"


class TestAnOldRecordCannotReachDisk:
    """The backstop, for the folded records that do exist: calls old enough to
    have been folded, and sessions checkpointed before the fold moved."""

    def test_a_buried_call_is_folded_and_keeps_no_line(self):
        write = {"dir_key": "d", "name": "annotation.tsv", "content_lines": ANNOTATION}
        view = history_with(write, followed_by=KEEP_RECENT_CALLS)
        folded = payload_of(view[0])
        assert isinstance(folded, str)  # a description, not a shortened list
        assert not any(line in folded for line in ANNOTATION)
        assert str(len(ANNOTATION)) in folded  # it says how much it stands for

    async def test_writing_a_folded_record_back_is_refused(
        self, tools, ctx, tmp_path
    ):
        write = {"dir_key": "d", "name": "annotation.tsv", "content_lines": ANNOTATION}
        folded = payload_of(history_with(write, followed_by=KEEP_RECENT_CALLS)[0])
        result = await call(
            tools,
            "create_file",
            ctx,
            dir_key=str(tmp_path),
            name="annotation2.tsv",
            content_lines=[folded],
        )
        assert result.startswith("NOT created")
        assert not (tmp_path / "annotation2.tsv").exists()

    async def test_the_pre_0_23_3_placeholder_is_refused_too(
        self, tools, ctx, tmp_path
    ):
        """A session checkpointed before the fix still holds the old wording,
        and resuming it must not write that to disk either."""
        result = await call(
            tools,
            "create_file",
            ctx,
            dir_key=str(tmp_path),
            name="annotation3.tsv",
            content_lines=LEGACY_ELIDED,
        )
        assert result.startswith("NOT created")
        assert not (tmp_path / "annotation3.tsv").exists()

    async def test_the_refusal_says_where_the_content_still_is(
        self, tools, ctx, tmp_path
    ):
        """A refusal the model cannot act on is another round of guessing."""
        result = await call(
            tools,
            "create_file",
            ctx,
            dir_key=str(tmp_path),
            name="annotation4.tsv",
            content_lines=LEGACY_ELIDED,
        )
        assert "read_file" in result

    async def test_an_ordinary_table_is_written_unchanged(self, tools, ctx, tmp_path):
        """The guard sits in front of every write, so what it must not do is
        cost a normal one. TSV rows, comment headers and prose about omitted
        lines all pass."""
        content = ANNOTATION + ["# some rows were elided from the source sheet"]
        await call(
            tools,
            "create_file",
            ctx,
            dir_key=str(tmp_path),
            name="ordinary.tsv",
            content_lines=content,
        )
        assert (tmp_path / "ordinary.tsv").read_text() == "\n".join(content) + "\n"


class TestRepairingAFileThatAlreadyCarriesOne:
    """The files written before the fix are still on disk, and getting the
    placeholder out of them is an edit like any other — so the guard must not
    stand in the way of the cleanup it exists to make unnecessary."""

    async def test_the_marker_line_can_be_edited_out(self, tools, ctx, tmp_path):
        broken = tmp_path / "annotation.tsv"
        broken.write_text("\n".join(LEGACY_ELIDED) + "\n")
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key=str(broken),
            old_lines=[LEGACY_ELIDED[-1]],
            new_lines=ANNOTATION[3:],
        )
        assert result.startswith("Edited")
        assert broken.read_text() == "\n".join(ANNOTATION) + "\n"

    async def test_an_edit_that_writes_the_marker_back_is_refused(
        self, tools, ctx, tmp_path
    ):
        intact = tmp_path / "annotation.tsv"
        intact.write_text("\n".join(ANNOTATION) + "\n")
        result = await call(
            tools,
            "edit_file",
            ctx,
            registry_key=str(intact),
            old_lines=ANNOTATION[3:],
            new_lines=[LEGACY_ELIDED[-1]],
        )
        assert result.startswith("NOT edited")
        # The file is what it was: a refused edit that half-applied would be
        # worse than the bug.
        assert intact.read_text() == "\n".join(ANNOTATION) + "\n"
