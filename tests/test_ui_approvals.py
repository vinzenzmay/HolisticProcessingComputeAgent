"""M5b — the inline approval, the refusal that says why, and the yes/no.

The claims are `specs-ui-acceptance.md`'s "Inline approval prompts" and
"Declining with a reason", in its own order, plus §4.3 items 22 and 23 (the
generic confirm, and `confirm.requested` — which the Textual UI never drew at
all, so that one is new behaviour arriving with the port rather than a port of
something).

Everything here is either a frame (a string) or a typed command on a real
`InProcessConnection.pair()`; the transport is never mocked and the core is a
scripted peer. The payload the prompt is asked about is built the way the
graph builds one — the real tool, its real schema blurb, its own
``describe_call`` and the real diff preview — so that what the prompt is
asserted to show is what it shows in the app.
"""

from __future__ import annotations

import pytest

from hpca import protocol
from hpca.agent.context import ToolContext
from hpca.agent.file_tools import add_file_tools
from hpca.agent.modes import script_preview
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.runner import ProcessRunner
from hpca.trash import TrashManager
from hpca.ui.app import CHAT, DECISION, INPUT, SESSIONS, WATCHERS, RowUI
from hpca.ui.approval import approval_details
from hpca.ui.state import Confirm
from tests.ui_harness import connected, plain, widths

ROWS = [
    protocol.SessionRow(session_id="s1", title="the first thing", mode="manual"),
    protocol.SessionRow(session_id="s2", title="the second thing", mode="manual"),
]

# A gate with no `details` of its own: the arguments are all there is to judge
# by, one of them is plumbing, and two of them are the script's own lines.
BASH_GATE = {
    "tool": "run_bash",
    "kind": "execution",
    "description": "Run a registered script as a tracked background process.",
    "arguments": {
        "key": "merge_vcf",
        "content_lines": ["set -euo pipefail", "bcftools merge -o out.vcf"],
        "timeout_s": 600,
    },
    "script": "set -euo pipefail\nbcftools merge -o out.vcf",
}

LONG_SCRIPT = {
    "tool": "run_bash",
    "kind": "execution",
    "arguments": {"key": "big"},
    "script": "\n".join(
        f"srun --partition=medium --time=08:00:00 --mem=64G step_{i}.sh "
        f"--reference /scratch/proj/refs/GRCh38_full_analysis_set.fa"
        for i in range(60)
    ),
}


def entry(seq: int, kind: str = "user", text: str = "", **kw) -> protocol.Entry:
    return protocol.Entry(kind=kind, text=text or f"row {seq}", seq=seq, **kw)


def edit_call_payload(tmp_path):
    """The payload the graph puts up for a gated ``edit_file``.

    Lifted from `tests/test_tui_approvals.py`, which built it the same way and
    for the same reason: a prompt asserted against a hand-written dict proves
    only that the dict was rendered.
    """
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    ctx = ToolContext(
        workdir=tmp_path,
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        trash=TrashManager(tmp_path / "trash", backup_limit_bytes=1024 * 1024),
    )
    path = tmp_path / "run.sh"
    path.write_text("#!/bin/bash\necho one\n")
    tool = add_file_tools(ToolRegistry()).get("edit_file")
    arguments = {
        "path": str(path),
        "old_lines": ["echo one"],
        "new_lines": ["echo ONE"],
    }
    payload = {
        "tool": "edit_file",
        "arguments": arguments,
        "description": tool.description,
        "kind": "destructive",
        "script": script_preview("edit_file", arguments, ctx),
        "details": tool.describe_call(tool.params.model_validate(arguments), ctx),
    }
    conn.close()
    return path, tool.description, payload


@pytest.fixture
async def wire():
    async with connected() as w:
        await w.tell(protocol.Hello(profile="hpc"))
        await w.tell(protocol.SessionRows(rows=list(ROWS)))
        await w.tell(
            protocol.ChatReset(
                session_id="s1", entries=[entry(1, text="edit run.sh", index=0)]
            )
        )
        w.peer.clear()
        yield w


async def parked(wire, payload: dict, session_id: str = "s1"):
    await wire.tell(
        protocol.DecisionRequested(session_id=session_id, payload=dict(payload))
    )
    return wire


# ------------------------------------------------- inline, and not a modal


class TestItIsInlineAndNotAModal:
    """The design decision written down at `tui/approval_screen.py:9-15`: a
    modal would cover another session's chat, and approvals are per session
    while the user may be reading something else."""

    async def test_the_prompt_is_on_screen(self, wire):
        await parked(wire, BASH_GATE)
        assert "── decision ─" in wire.screen()
        assert "Run this — run_bash?" in wire.screen()

    async def test_and_the_conversation_is_still_behind_it(self, wire):
        await parked(wire, BASH_GATE)
        # The whole objection to a modal: what the question is about must not
        # be covered by the question.
        assert "edit run.sh" in wire.screen()
        assert "── chat ─" in wire.screen()
        assert wire.ui.overlay is None, "an overlay is exactly what this is not"

    async def test_and_it_has_the_keys(self, wire):
        await parked(wire, BASH_GATE)
        assert wire.ui.focus == DECISION
        assert "y approve" in plain(wire.frame()[-1])

    @pytest.mark.parametrize("width", [80, 100, 137])
    async def test_every_row_is_exactly_the_terminal_width(self, wire, width):
        await parked(wire, LONG_SCRIPT)
        assert widths(wire.ui.render(width, 40)) == {width}

    @pytest.mark.parametrize("width", [80, 100, 137])
    async def test_and_so_is_the_reason_stage(self, wire, width):
        await parked(wire, LONG_SCRIPT)
        await wire.press("n", *"a script that does not delete the shards")
        assert widths(wire.ui.render(width, 40)) == {width}

    async def test_a_long_script_does_not_eat_the_chat(self, wire):
        # Sixty lines of script, and the prompt still takes at most half the
        # screen — DecisionBar's `max-height: 60%`, in rows.
        await parked(wire, LONG_SCRIPT)
        drawn = [plain(x) for x in wire.ui.render(100, 40)]
        assert any("── decision ─" in x for x in drawn), "the prompt is drawn"
        assert wire.ui._decision_h(100, 40) <= (40 - 2) // 2
        chat = drawn.index("".join(x for x in drawn if "── chat ─" in x))
        decision = next(i for i, x in enumerate(drawn) if "── decision ─" in x)
        assert decision - chat > RowUI.MIN_CHAT, "the conversation is still there"

    @pytest.mark.parametrize(
        "width,height", [(80, 24), (120, 40), (60, 14), (40, 10), (100, 8)]
    )
    async def test_a_terminal_too_small_for_it_still_gets_a_frame(
        self, wire, width, height
    ):
        # The prompt gives up its middle before it gives up the heading and
        # the keys: a question with no visible way to answer it is worse than
        # one whose script is clipped.
        await parked(wire, LONG_SCRIPT)
        await wire.press("n", *"no")
        drawn = wire.ui.render(width, height)
        assert len(drawn) == height
        assert widths(drawn) == {width}

    async def test_an_open_screen_keeps_the_keys_until_it_closes(self, wire):
        # A decision can arrive while the key list (or any overlay) is up. The
        # prompt is drawn under it, not over it, so the screen answers first
        # and the prompt takes over the moment it closes.
        wire.ui.focus = CHAT
        await wire.press("?")
        await parked(wire, BASH_GATE)
        await wire.press("y")  # closes the key list; not an approval
        assert wire.peer.took(protocol.DecisionResolve) == []
        assert wire.ui.overlay is None
        await wire.press("y")
        assert wire.peer.last(protocol.DecisionResolve).approved is True

    async def test_a_decision_does_not_take_the_cursor_out_of_another_column(
        self, wire
    ):
        # A decision surfacing must never yank the user out of the column they
        # are working in — the rule `_set_pending_decision` had.
        wire.ui.focus = WATCHERS
        await parked(wire, BASH_GATE)
        assert wire.ui.focus == WATCHERS


class TestAnsweringIt:
    async def test_y_approves(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("y")
        answer = wire.peer.last(protocol.DecisionResolve)
        assert (answer.session_id, answer.approved, answer.reason) == (
            "s1",
            True,
            "",
        )

    async def test_and_the_prompt_goes(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("y")
        assert "── decision ─" not in wire.screen()
        assert wire.ui.focus == INPUT

    async def test_escape_refuses_with_no_reason(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("esc")
        answer = wire.peer.last(protocol.DecisionResolve)
        assert (answer.approved, answer.reason) == (False, "")

    async def test_approving_never_asks_why(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("y")
        assert "what should be different" not in wire.screen()

    async def test_the_core_clearing_it_also_takes_it_off_the_screen(self, wire):
        # `decision.cleared`: the turn died, or another front-end answered it.
        await parked(wire, BASH_GATE)
        await wire.tell(protocol.DecisionCleared(session_id="s1"))
        assert "── decision ─" not in wire.screen()
        assert wire.ui.focus == INPUT


class TestADecisionInAnotherSession:
    async def test_it_flags_the_sidebar_row(self, wire):
        await parked(wire, BASH_GATE, session_id="s2")
        rows = [x for x in wire.frame() if "the second thing" in x]
        assert "!" in rows[0]

    async def test_and_puts_no_prompt_on_this_one(self, wire):
        await parked(wire, BASH_GATE, session_id="s2")
        assert "── decision ─" not in wire.screen()
        assert wire.ui.focus != DECISION

    async def test_a_new_session_does_not_inherit_it(self, wire):
        await parked(wire, BASH_GATE, session_id="s2")
        await wire.tell(
            protocol.SessionCreated(
                row=protocol.SessionRow(session_id="s3", title="a new one")
            )
        )
        assert "── decision ─" not in wire.screen()
        assert wire.ui.active_id == "s3"

    async def test_opening_it_reveals_the_prompt(self, wire):
        await parked(wire, BASH_GATE, session_id="s2")
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
        await wire.press("enter")
        assert "Run this — run_bash?" in wire.screen()
        assert wire.ui.focus == DECISION

    async def test_and_answering_it_there_names_that_session(self, wire):
        await parked(wire, BASH_GATE, session_id="s2")
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
        await wire.press("enter", "y")
        assert wire.peer.last(protocol.DecisionResolve).session_id == "s2"


# ------------------------------------------- what the prompt says about it


class TestThePromptShowsTheCallAndNothingElse:
    async def test_a_gated_edit_shows_the_path_and_the_diff(self, wire, tmp_path):
        path, _, payload = edit_call_payload(tmp_path)
        await parked(wire, payload)
        shown = wire.ui.render(200, 44)
        text = "\n".join(plain(x) for x in shown)
        assert str(path) in text
        assert "copied to trash" in text
        assert "- echo one" in text and "+ echo ONE" in text

    async def test_the_tool_blurb_and_a_json_blob_stay_off_the_screen(
        self, wire, tmp_path
    ):
        _, description, payload = edit_call_payload(tmp_path)
        await parked(wire, payload)
        text = "\n".join(plain(x) for x in wire.ui.render(200, 44))
        assert description[:40] not in text
        assert "old_lines" not in text and '"path"' not in text

    async def test_the_heading_names_the_tool(self, wire, tmp_path):
        _, _, payload = edit_call_payload(tmp_path)
        await parked(wire, payload)
        assert "Destructive operation — edit_file?" in wire.screen()

    async def test_arguments_reach_the_screen_as_lines(self, wire):
        await parked(wire, BASH_GATE)
        text = "\n".join(plain(x) for x in wire.ui.render(140, 40))
        assert "key: merge_vcf" in text
        assert "timeout_s" not in text, "plumbing"
        assert "content_lines" not in text, "the script block already says it"


class TestTheHelpersTheArgumentsAreBuiltBy:
    """Moved whole from `hpca/tui/approval_screen.py` into `hpca/ui/approval.py`
    — `decision.requested.payload` is an opaque dict, so understanding its
    shape is the UI's job, and these are the part of `tui/` that survives it.
    """

    def test_they_are_lines_not_json(self):
        text = approval_details(
            {"tool": "delete", "arguments": {"target": "results/", "force": True}}
        )
        assert text == "target: results/\nforce: True"

    def test_plumbing_arguments_are_left_out(self):
        text = approval_details(
            {"tool": "run_thing", "arguments": {"key": "run_sh", "timeout_s": 600}}
        )
        assert text == "key: run_sh"

    def test_the_lines_of_the_script_below_are_not_repeated_above_it(self):
        text = approval_details(
            {
                "tool": "run_bash",
                "arguments": {"content_lines": ["squeue -u me"]},
                "script": "squeue -u me",
            }
        )
        assert text == ""

    def test_a_pathological_argument_is_clipped(self):
        text = approval_details({"tool": "x", "arguments": {"blob": "y" * 5000}})
        assert len(text) < 2100 and text.endswith("[clipped]")

    def test_details_replace_the_arguments_entirely(self):
        text = approval_details(
            {
                "tool": "delete_file",
                "arguments": {"registry_key": "notes"},
                "description": "Delete a registered file",
                "details": "rm /home/me/notes.txt\n(12 bytes; recoverable from trash)",
            }
        )
        assert text == "rm /home/me/notes.txt\n(12 bytes; recoverable from trash)"


# ------------------------------------------------------ declining with a reason


class TestDecliningWithAReason:
    async def test_n_opens_the_box_and_answers_nothing_yet(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n")
        assert "what should be different" in wire.screen()
        assert wire.peer.took(protocol.DecisionResolve) == []

    async def test_the_call_stays_on_screen_while_the_reason_is_typed(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n")
        text = "\n".join(plain(x) for x in wire.ui.render(140, 40))
        assert "bcftools merge -o out.vcf" in text, "the script is still there"
        assert "key: merge_vcf" in text

    async def test_the_box_takes_the_keys_the_prompt_was_using(self, wire):
        # "n" in the middle of a sentence is a letter, not a second verdict.
        await parked(wire, BASH_GATE)
        await wire.press("n", *"not yes")
        assert "not yes" in wire.screen()
        assert wire.peer.took(protocol.DecisionResolve) == []

    async def test_enter_sends_the_reason_with_the_refusal(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n", *"use the shard list", "enter")
        answer = wire.peer.last(protocol.DecisionResolve)
        assert (answer.approved, answer.reason) == (False, "use the shard list")

    async def test_an_empty_box_is_the_plain_refusal(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n", "enter")
        answer = wire.peer.last(protocol.DecisionResolve)
        assert (answer.approved, answer.reason) == (False, "")

    async def test_escaping_out_of_the_box_still_refuses(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n", *"half a thought", "esc")
        answer = wire.peer.last(protocol.DecisionResolve)
        assert (answer.approved, answer.reason) == (False, "")

    async def test_a_manually_skipped_script_carries_the_reason_back(self, wire):
        # The execution gate, whose heading asks about the *script* — the same
        # round trip, and the one manual mode is made of.
        await parked(wire, BASH_GATE)
        await wire.press("n")
        assert "different about the script" in wire.screen()
        await wire.press(*"keep the shards", "enter")
        assert wire.peer.last(protocol.DecisionResolve).reason == "keep the shards"

    async def test_shift_enter_is_a_new_line_and_not_a_send(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n", *"one", "shift-enter", *"two")
        assert wire.peer.took(protocol.DecisionResolve) == []
        await wire.press("enter")
        assert wire.peer.last(protocol.DecisionResolve).reason == "one\ntwo"


class TestTheHalfWrittenReasonIsADraft:
    """specs-core-process.md §4.4: the decision moves to the core, the reason
    stays in the UI — "a draft, same class as ``_drafts``"."""

    async def at_the_box(self, wire):
        await parked(wire, BASH_GATE, session_id="s1")
        await parked(wire, BASH_GATE, session_id="s2")
        await wire.press("n", *"the shards are still open")
        return wire

    async def test_it_waits_in_its_own_session_across_a_switch(self, wire):
        await self.at_the_box(wire)
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
        await wire.press("enter")  # to s2, which is parked on its own copy
        assert "the shards are still open" not in wire.screen()
        assert "(y) run script" in wire.screen(), "s2's is still at the question"
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 0
        await wire.press("enter")
        assert "the shards are still open" in wire.screen()

    async def test_and_only_the_finished_string_ever_crosses(self, wire):
        await self.at_the_box(wire)
        assert wire.peer.took(protocol.DecisionResolve) == []

    async def test_the_same_decision_arriving_again_does_not_wipe_it(self, wire):
        # `decision.requested` is re-emitted on subscribe (§4.4); a reconnect
        # must not throw away a refusal someone is in the middle of writing.
        await self.at_the_box(wire)
        await parked(wire, BASH_GATE, session_id="s1")
        assert "the shards are still open" in wire.screen()


# --------------------------------------------------- the generic yes/no


class TestTheGenericConfirm:
    """§4.3 item 22 — the dialog eleven call sites used. The mechanism lands
    here; its callers arrive with their own milestones (session delete is M6,
    quit and the skill screens are M8), and the one live caller today is the
    interrupt (see `test_ui_turn.py`)."""

    async def test_it_draws_over_whatever_is_on_screen(self, wire):
        wire.ui.ask("Really quit?")
        assert "Really quit?" in wire.screen()
        assert "(y) yes · (n) no" in wire.screen()

    @pytest.mark.parametrize("width", [80, 100, 137])
    async def test_and_the_frame_is_still_the_terminal(self, wire, width):
        wire.ui.ask("Really quit?")
        assert widths(wire.ui.render(width, 24)) == {width}

    async def test_y_is_yes_and_n_is_no(self, wire):
        answers = []
        wire.ui.ask("Really quit?", answers.append)
        await wire.press("y")
        wire.ui.ask("Really quit?", answers.append)
        await wire.press("n")
        assert answers == [True, False]

    async def test_escape_is_no_and_not_ask_me_later(self, wire):
        answers = []
        wire.ui.ask("Really quit?", answers.append)
        await wire.press("esc")
        assert answers == [False]
        assert wire.ui.confirm is None

    async def test_it_takes_the_keys_before_the_rows_do(self, wire):
        # A modal that leaked its keys would act on the screen behind it.
        wire.ui.focus = CHAT
        before = wire.ui.chat.cursor
        wire.ui.ask("Really quit?")
        await wire.press("down", "down")
        assert wire.ui.chat.cursor == before

    async def test_and_over_an_overlay_too(self, wire):
        wire.ui.focus = CHAT
        await wire.press("?")  # the key list, an overlay
        wire.ui.ask("Really quit?")
        assert "Really quit?" in wire.screen()
        await wire.press("y")
        assert wire.ui.overlay is not None, "answering it left the screen alone"


class TestConfirmRequested:
    """§4.3 item 23 — the separate channel triage offers a learned log
    signature on. Never wired into the Textual UI at all, so this is new
    behaviour arriving with the port."""

    async def test_the_question_reaches_the_screen(self, wire):
        await wire.tell(
            protocol.ConfirmRequested(
                id="sig-1", question="Remember “OOM killed” as a failure?"
            )
        )
        assert "Remember “OOM killed” as a failure?" in wire.screen()

    async def test_yes_answers_it_by_id(self, wire):
        await wire.tell(protocol.ConfirmRequested(id="sig-1", question="Learn it?"))
        await wire.press("y")
        answer = wire.peer.last(protocol.ConfirmResolve)
        assert (answer.id, answer.confirmed) == ("sig-1", True)

    async def test_and_no_answers_it_too(self, wire):
        await wire.tell(protocol.ConfirmRequested(id="sig-1", question="Learn it?"))
        await wire.press("n")
        answer = wire.peer.last(protocol.ConfirmResolve)
        assert (answer.id, answer.confirmed) == ("sig-1", False)

    async def test_it_names_no_session_because_it_belongs_to_none(self, wire):
        # A triage offer comes from a poll, not from a conversation
        # (`protocol.ConfirmResolve`).
        assert not hasattr(Confirm(id="x", question="?"), "session_id")
        await wire.tell(protocol.ConfirmRequested(id="sig-1", question="Learn it?"))
        await wire.press("y")
        assert not hasattr(wire.peer.last(protocol.ConfirmResolve), "session_id")
