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
from hpca.ui.ansi import (
    BOLD,
    CYAN,
    DIM,
    PULSE_INTERVAL,
    PULSE_PERIOD,
    PULSE_RAMP,
    RED,
    WHITE,
    YELLOW,
)
from hpca.ui.app import CHAT, DECISION, INPUT, OFFER, SESSIONS, WATCHERS, RowUI
from hpca.ui.approval import approval_details
from hpca.ui.keys import PASTE
from hpca.ui.state import Confirm, Display
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


# The other gate: not a script about to run but something about to be lost.
# Hand-written because what is under test here is the *colour* the two kinds
# are drawn in, and a real payload would only add fields nothing reads.
DELETE_GATE = {
    "tool": "delete_file",
    "kind": "destructive",
    "details": "delete /scratch/proj/cohort.bam",
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



class TestThePromptStandsInForTheMessageBox:
    """§4.3 item 21 in the layout: the prompt is inline at the foot of the
    chat column, and what it is inline *in place of* is the message box.

    Both on screen at once was the arrangement before, and it cost twice: the
    ring pointed one cursor at a slot that held two things, and the box offered
    to take a message in a session whose turn cannot move until the question
    above it is answered.
    """

    async def test_the_message_box_is_not_drawn_while_one_is_pending(self, wire):
        await parked(wire, BASH_GATE)
        assert "── decision ─" in wire.screen()
        assert "── message ─" not in wire.screen()

    async def test_and_it_is_back_the_moment_it_is_answered(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("y")
        assert "── message ─" in wire.screen()

    async def test_the_prompt_takes_the_box_s_slot_and_not_a_row_of_its_own(
        self, wire
    ):
        before = wire.ui._heights(40, 100)
        was = wire.ui._footer_h(100, 40)
        await parked(wire, BASH_GATE)
        after = wire.ui._heights(40, 100)
        # The footer is part of the sum: it is as many rows as this row's keys
        # need (`FOOTER_ROWS`), and the prompt offers four where the message
        # box offers eleven — so at 100 columns it hands a row back, and the
        # screen is still the screen once that row is counted.
        assert sum(after) + wire.ui._footer_h(100, 40) == sum(before) + was, (
            "the screen is still the screen"
        )
        assert after[2] == wire.ui._decision_h(100, 40) + wire.ui._status_h()
        assert after[2] > before[2], "and the prompt has the room the box had"

    async def test_the_half_typed_message_waits_behind_it(self, wire):
        # The draft lives on the `SessionState` and never depended on being
        # drawn (§4.4) — which is what makes hiding the box safe.
        wire.ui.focus = INPUT
        await wire.press(*"the shards are still open")
        await parked(wire, BASH_GATE)
        assert "the shards are still open" not in wire.screen()
        await wire.press("y")
        assert "the shards are still open" in wire.screen()
        assert wire.ui.focus == INPUT

    async def test_the_command_menu_goes_with_the_box_it_belongs_to(self, wire):
        # It is the box's own autocomplete: a list of commands offered next to
        # a field that is not on screen is a list nothing can run.
        wire.ui.focus = INPUT
        await wire.press("/")
        assert "── commands" in wire.screen()
        await parked(wire, BASH_GATE)
        assert "── commands" not in wire.screen()
        assert wire.ui.input.text() == "/", "and the draft that named them is intact"

    @pytest.mark.parametrize("width,height", [(80, 24), (120, 40), (60, 14), (100, 8)])
    async def test_the_slot_is_filled_to_exactly_the_rows_it_was_given(
        self, wire, width, height
    ):
        # The frame pads and truncates at the end, so a band that rendered
        # short or long would show up as another row moving rather than as an
        # exception — hence the count, and not just the frame's height.
        await parked(wire, LONG_SCRIPT)
        heights = wire.ui._heights(height, width)
        assert len(wire.ui.render(width, height)) == height
        assert (
            len(wire.ui._render_decision(width, height))
            + len(wire.ui._render_status(width))
            == heights[2]
        )


class TestTheDecisionCannotBeLeftUnanswerable:
    """The lockout, which was real: DECISION was not in the ctrl+↑/ctrl+↓ ring
    while its own keys still moved the cursor out of it, so one ctrl+↑ off an
    unanswered prompt left a turn parked on a question no key sequence could
    reach again. Re-opening the session was the only way back, and nothing on
    screen said so.

    The prompt is in the ring now, in the slot the box would have had, so
    every step of it comes back to the question.
    """

    async def test_the_ring_holds_the_prompt_while_one_is_pending(self, wire):
        assert wire.ui._ring() == [SESSIONS, CHAT, INPUT, WATCHERS]
        await parked(wire, BASH_GATE)
        assert wire.ui._ring() == [SESSIONS, CHAT, DECISION, WATCHERS]

    @pytest.mark.parametrize("key", ["ctrl-up", "ctrl-down", "tab"])
    async def test_the_decision_cannot_be_left_unanswerable(self, wire, key):
        await parked(wire, BASH_GATE)
        seen = []
        for _ in range(4):  # one whole turn of the ring
            await wire.press(key)
            seen.append(wire.ui.focus)
        assert set(seen) == {SESSIONS, CHAT, DECISION, WATCHERS}, "every row"
        assert wire.ui.focus == DECISION, "and back to the one that is waiting"
        assert "Run this — run_bash?" in wire.screen()
        await wire.press("y")
        assert wire.peer.last(protocol.DecisionResolve).approved is True

    @pytest.mark.parametrize("key", ["ctrl-up", "ctrl-down", "tab"])
    async def test_and_not_from_the_reason_box_either(self, wire, key):
        # The second stage is a text field, and walking off it must not lose
        # the half-written refusal or the way back to it.
        await parked(wire, BASH_GATE)
        await wire.press("n", *"keep the shards")
        for _ in range(4):
            await wire.press(key)
        assert wire.ui.focus == DECISION
        assert "keep the shards" in wire.screen()
        await wire.press("enter")
        assert wire.peer.last(protocol.DecisionResolve).reason == "keep the shards"

    async def test_walking_the_ring_never_asks_a_row_for_a_pane_it_has_none_of(
        self, wire
    ):
        # `_handle_row` looks its pane up in a dict with three entries, so a
        # ring that could route a key there with the prompt or the box under
        # the cursor would raise rather than misdraw.
        await parked(wire, BASH_GATE)
        for _ in range(9):
            await wire.press("ctrl-down")
            await wire.press("down")

    async def test_the_cursor_is_never_left_in_a_box_that_is_not_drawn(self, wire):
        # As any unguarded path that aims at the message box would leave it —
        # a paste, a message handed back, ctrl+↓ out of the chat.
        await parked(wire, BASH_GATE)
        wire.ui.focus = INPUT
        assert "── message ─" not in wire.screen()
        assert wire.ui.focus == DECISION
        await wire.press("y")
        assert wire.peer.last(protocol.DecisionResolve).approved is True

    async def test_the_ring_lands_on_the_question_and_not_on_the_box(
        self, wire
    ):
        await parked(wire, BASH_GATE)
        await wire.press("ctrl-up")  # to the chat
        await wire.press("ctrl-down")  # and back down, into the prompt's slot
        assert wire.ui.focus == DECISION
        await wire.press("y")
        assert wire.peer.last(protocol.DecisionResolve).approved is True

    async def test_a_decision_arriving_takes_the_cursor_out_of_the_box(self, wire):
        # It has to: the box is what the prompt is standing in front of.
        wire.ui.focus = INPUT
        await parked(wire, BASH_GATE)
        assert wire.ui.focus == DECISION

    async def test_and_the_prompt_being_cleared_hands_it_back(self, wire):
        await parked(wire, BASH_GATE)
        await wire.tell(protocol.DecisionCleared(session_id="s1"))
        assert wire.ui.focus == INPUT
        assert wire.ui._ring() == [SESSIONS, CHAT, INPUT, WATCHERS]

    async def test_but_not_from_a_row_the_user_walked_off_to(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("ctrl-up")  # the chat, deliberately
        await wire.tell(protocol.DecisionCleared(session_id="s1"))
        assert wire.ui.focus == CHAT

    async def test_a_pasted_block_is_not_dropped_and_does_not_move_the_cursor(
        self, wire
    ):
        # A paste is an unambiguous "I am entering text" and is never dropped,
        # but the box it is aimed at is not on screen: it waits in the draft
        # with whatever was already there.
        wire.ui.focus = INPUT
        await wire.press(*"before ")
        await parked(wire, BASH_GATE)
        await wire.press(PASTE + "pasted while parked")
        assert wire.ui.focus == DECISION
        assert "pasted while parked" not in wire.screen()
        await wire.press("y")
        assert "before pasted while parked" in wire.screen()

    async def test_and_at_the_reason_stage_it_lands_in_the_box_that_is_open(
        self, wire
    ):
        await parked(wire, BASH_GATE)
        await wire.press("n")
        await wire.press(PASTE + "two shards were still open")
        assert "two shards were still open" in wire.screen()
        await wire.press("enter")
        answer = wire.peer.last(protocol.DecisionResolve)
        assert answer.reason == "two shards were still open"


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
        wire.ui.session_pane.cursor = 2
        await wire.press("enter")
        assert "Run this — run_bash?" in wire.screen()
        assert wire.ui.focus == DECISION

    async def test_and_answering_it_there_names_that_session(self, wire):
        await parked(wire, BASH_GATE, session_id="s2")
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 2
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
        wire.ui.session_pane.cursor = 2
        await wire.press("enter")  # to s2, which is parked on its own copy
        assert "the shards are still open" not in wire.screen()
        assert "(y) run script" in wire.screen(), "s2's is still at the question"
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 1
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



class TestTheRuleSaysWhereTheKeysAre:
    """The decision's rule follows the focus convention, and the severity it
    used to carry moves one line down onto the heading.

    Two facts were fighting over one row. Focus is teal-and-bold in every
    other region — sessions, chat, watchers, all drawn by `Pane.render` — and
    the prompt stands in the message box's slot of the same ring, so a rule
    that could not say "the keys are here" left the one region whose keys
    silently do nothing as the one region unable to say so.

    The severity is not dropped for it. An execution gate and a destructive
    one are different warnings and the prompt still says which, in the same
    two colours, on the bold heading line directly under the rule — which is
    larger type than the rule ever was. What is lost is the dashes it was
    painted on.
    """

    @staticmethod
    def rule_row(wire) -> str:
        rows = [x for x in wire.ui.render(120, 40) if "── decision ─" in x]
        assert len(rows) == 1, "the rule is drawn once"
        return rows[0]

    @staticmethod
    def heading(wire, text: str) -> str:
        rows = [x for x in wire.ui.render(120, 40) if text in x]
        assert len(rows) == 1, f"{text!r} is drawn once"
        return rows[0]

    async def test_the_rule_is_teal_when_the_prompt_is_focused(self, wire):
        await parked(wire, BASH_GATE)
        assert wire.ui.focus == DECISION
        assert self.rule_row(wire).startswith(BOLD + CYAN)

    async def test_and_dim_when_it_is_not(self, wire):
        await parked(wire, BASH_GATE)
        wire.ui.focus = CHAT
        assert self.rule_row(wire).startswith(DIM)

    async def test_it_is_the_same_sentence_the_panes_write(self, wire):
        # Not merely "teal": the exact styles `Pane.render` puts on its own
        # title, so the four regions cannot drift into two conventions.
        await parked(wire, BASH_GATE)
        focused = self.rule_row(wire)
        wire.ui.focus = CHAT
        unfocused = self.rule_row(wire)
        chat = [x for x in wire.ui.render(120, 40) if "── chat ─" in x][0]
        wire.ui.focus = CHAT
        assert focused.startswith(BOLD + CYAN) and chat.startswith(BOLD + CYAN)
        assert unfocused.startswith(DIM)

    async def test_a_destructive_gate_still_says_so_in_red(self, wire):
        await parked(wire, DELETE_GATE)
        assert self.heading(wire, "Destructive operation").startswith(RED + BOLD)

    async def test_and_an_execution_gate_in_yellow(self, wire):
        # The distinction the docstring of `ui/approval.py` insists on: a
        # script about to run is not the same warning as something about to be
        # lost, and one colour for both would be a warning that says nothing.
        await parked(wire, BASH_GATE)
        assert self.heading(wire, "Run this —").startswith(YELLOW + BOLD)

    async def test_the_two_kinds_are_still_told_apart_while_focused(self, wire):
        # The thing the naive fix would have broken: the prompt is focused
        # almost all the time, so a severity that only showed when it was not
        # would be a severity nobody ever sees.
        await parked(wire, BASH_GATE)
        assert wire.ui.focus == DECISION
        execution = self.heading(wire, "Run this —")
        await parked(wire, DELETE_GATE)
        assert wire.ui.focus == DECISION
        destructive = self.heading(wire, "Destructive operation")
        assert execution.split("m", 1)[0] != destructive.split("m", 1)[0]

    async def test_the_refusal_box_keeps_the_severity_too(self, wire):
        # The second stage replaces the question but not the warning: what is
        # being refused is still the same class of thing.
        await parked(wire, DELETE_GATE)
        await wire.press("n")
        assert self.heading(wire, "Denied —").startswith(RED + BOLD)

    async def test_the_rule_costs_no_rows_either_way(self, wire):
        # Focus is a colour and never a layout: a prompt that grew a row when
        # the keys arrived would move the conversation behind it.
        await parked(wire, LONG_SCRIPT)
        focused = wire.ui._heights(40, 120)
        wire.ui.focus = CHAT
        assert wire.ui._heights(40, 120) == focused


class TestHowFastTheAnswerLineBreathes:
    """`Display.decision_pulse_seconds` — the period, in seconds, as it
    arrived over the wire. The colour is still a pure function of the clock;
    what the setting changes is how far round the sweep a given instant is."""

    @staticmethod
    def hint(wire, when: float) -> str:
        wire.ui.clock = lambda: when
        rows = [x for x in wire.ui.render(120, 40) if "(y) run script" in x]
        assert len(rows) == 1
        return rows[0]

    async def test_the_period_arrives_with_the_rest_of_the_display_settings(
        self, wire
    ):
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(decision_pulse_seconds=8.0)
            )
        )
        assert wire.ui.display.decision_pulse_seconds == 8.0

    async def test_a_configured_period_is_what_the_sweep_runs_on(self, wire):
        # A quarter of the way round is the far end of the ramp, wherever the
        # user put the quarter mark.
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(decision_pulse_seconds=8.0)
            )
        )
        await parked(wire, BASH_GATE)
        assert self.hint(wire, 2.0).startswith(CYAN)
        assert self.hint(wire, 6.0).startswith(WHITE)

    async def test_and_a_slower_one_is_visibly_slower(self, wire):
        # The same instant, two periods, two colours — which is the whole of
        # what the setting does.
        await parked(wire, BASH_GATE)
        fast = self.hint(wire, PULSE_PERIOD / 4)
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(decision_pulse_seconds=100.0)
            )
        )
        assert self.hint(wire, PULSE_PERIOD / 4) != fast

    async def test_a_period_nothing_can_be_divided_by_does_not_kill_the_frame(
        self, wire
    ):
        # The settings model refuses zero (`config.DisplaySettings`), which is
        # where a user finds out. This is the other half: by the time a number
        # has crossed the wire it is being divided by inside a repaint, and an
        # exception there takes the terminal down with it.
        await wire.tell(
            protocol.DisplayChanged(
                display=protocol.DisplaySettings(decision_pulse_seconds=0.0)
            )
        )
        await parked(wire, BASH_GATE)
        assert widths(wire.ui.render(120, 40)) == {120}
        assert self.hint(wire, PULSE_PERIOD / 4).startswith(CYAN)

    async def test_the_default_is_one_second(self, wire):
        # Changed from 2.4: the line reads as a prompt waiting for an answer,
        # and a two-and-a-half-second cycle is slow enough that a glance
        # catches it standing still.
        assert PULSE_PERIOD == 1.0
        assert wire.ui.display == Display()


class TestTheAnswerLinePulses:
    """The one line on the screen that is drawn in a different colour every
    tenth of a second, and the reason it is: a turn parked on a question is a
    turn nobody is driving, and a dim key hint under a block of script looks
    exactly like the dim key hint under every other row.

    The clock is pinned in every test here, because that is the whole design:
    the colour is a pure function of `RowUI.clock()` (`ansi.pulse`), so a
    frame at a given instant is one answer and not a race.
    """

    @staticmethod
    def hint(wire, when: float) -> str:
        """The styled answer line at that instant — styled, because the style
        is what is under test."""
        wire.ui.clock = lambda: when
        rows = [x for x in wire.ui.render(120, 40) if "(y) run script" in x]
        assert len(rows) == 1, "the answer line is drawn once"
        return rows[0]

    async def test_the_colour_moves_with_the_clock(self, wire):
        await parked(wire, BASH_GATE)
        assert self.hint(wire, 0.0) != self.hint(wire, PULSE_PERIOD / 4)

    async def test_and_the_words_do_not(self, wire):
        await parked(wire, BASH_GATE)
        moment = (self.hint(wire, x) for x in (0.0, PULSE_PERIOD / 4))
        assert len({plain(x) for x in moment}) == 1

    async def test_it_never_leaves_the_two_colours_it_was_given(self, wire):
        await parked(wire, BASH_GATE)
        # A whole cycle at the frame rate the repaint is booked at, which is
        # every colour the line can ever be drawn in.
        steps = int(PULSE_PERIOD / PULSE_INTERVAL) + 1
        drawn = {self.hint(wire, x * PULSE_INTERVAL) for x in range(steps)}
        assert {x.split("m", 1)[0] + "m" for x in drawn} <= set(PULSE_RAMP)
        assert len(drawn) > 2, "a ramp, and not a two-colour blink"

    async def test_and_reaches_both_ends_of_the_sweep(self, wire):
        await parked(wire, BASH_GATE)
        assert self.hint(wire, PULSE_PERIOD / 4).startswith(CYAN)
        assert self.hint(wire, PULSE_PERIOD * 3 / 4).startswith(WHITE)

    async def test_the_frame_books_the_repaint_that_animates_it(self, wire):
        # Without this the colour would be whichever one the keypress that
        # drew the frame landed on, and it would sit there: nothing else wakes
        # an idle UI (`RowUI.next_wake`).
        assert wire.ui.next_wake() is None
        await parked(wire, BASH_GATE)
        assert 0 < wire.ui.next_wake() <= PULSE_INTERVAL

    async def test_and_stops_booking_them_when_it_is_answered(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("y")
        assert wire.ui.next_wake() is None

    async def test_a_prompt_in_another_session_asks_for_no_frames(self, wire):
        # It is not drawn here — it is a "!" in the sidebar — so nothing about
        # this screen changes with the clock (`_render_decision`).
        await parked(wire, BASH_GATE, session_id="s2")
        assert wire.ui.next_wake() is None

    async def test_the_reason_box_does_not_pulse_behind_its_own_cursor(self, wire):
        await parked(wire, BASH_GATE)
        await wire.press("n")
        wire.ui.clock = lambda: 0.0
        first = [x for x in wire.ui.render(120, 40) if "(enter) send" in x]
        wire.ui.clock = lambda: PULSE_PERIOD / 4
        assert [x for x in wire.ui.render(120, 40) if "(enter) send" in x] == first

    async def test_how_tall_the_prompt_is_does_not_depend_on_what_time_it_is(
        self, wire
    ):
        # `decision_height` is asked before the frame is laid out, and a
        # layout that moved with the clock would relay the whole screen out
        # ten times a second.
        await parked(wire, LONG_SCRIPT)
        wire.ui.clock = lambda: 0.0
        first = wire.ui._heights(40, 120)
        wire.ui.clock = lambda: PULSE_PERIOD / 4
        assert wire.ui._heights(40, 120) == first


# --------------------------------------------------- the generic yes/no


class TestTheGenericConfirm:
    """§4.3 item 22 — the dialog eleven call sites used. The mechanism lands
    here; its callers arrive with their own milestones (session delete is M6,
    quit and the skill screens are M8), and the one live caller today is the
    interrupt (see `test_ui_turn.py`)."""

    async def test_it_draws_instead_of_whatever_is_on_screen(self, wire):
        wire.ui.ask("Really quit?")
        assert "Really quit?" in wire.screen()
        assert "(y) yes · (n) no" in wire.screen()

    async def test_and_it_is_the_only_thing_left_to_read(self, wire):
        # Three rows spliced into a full screen read as another band of it.
        # The question is a gate, so the screen it gates is cleared.
        await wire.tell(protocol.SessionRows(rows=list(ROWS)))
        busy = [plain(row) for row in wire.frame() if plain(row).strip()]
        assert len(busy) > 3, "the frame under the question was already empty"

        wire.ui.ask("Really quit?")
        rows = [plain(row) for row in wire.frame() if plain(row).strip()]
        assert [row.strip() for row in rows] == [
            "── confirm " + "─" * (wire.width - 11),
            "Really quit?",
            "(y) yes · (n) no · (esc) no",
        ]

    async def test_and_no_puts_back_the_frame_it_hid(self, wire):
        # Which is what makes the question cheap to answer wrongly: the frame
        # underneath is built the same way while it is up, so the panes keep
        # the heights and the scroll they had.
        await wire.tell(protocol.SessionRows(rows=list(ROWS)))
        before = wire.frame()
        wire.ui.ask("Really quit?")
        await wire.press("n")
        assert wire.frame() == before

    @pytest.mark.parametrize(
        "height,rows",
        [
            (1, ["Really quit?"]),
            (2, ["Really quit?", "(y) yes · (n) no · (esc) no"]),
        ],
    )
    async def test_a_terminal_too_short_keeps_the_question(self, wire, height, rows):
        # The rule is the decoration, and it is decorating nothing now.
        wire.ui.ask("Really quit?")
        drawn = [plain(row) for row in wire.ui.render(wire.width, height)]
        assert [row.strip() for row in drawn] == rows
        assert widths(drawn) == {wire.width}

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
    behaviour arriving with the port.

    It is raised by a poll rather than by a keypress, which is what makes it
    unlike every other question here: whoever started the job that failed is
    as likely as not reading a different conversation by the time it lands. So
    it waits in the session it is about, standing in that session's message
    box (`app.OFFER`), and a session that is not on screen says so with a mark
    and nothing else.
    """

    async def offered(self, w, session_id="s1", question="Learn it?", offer_id="sig-1"):
        await w.tell(
            protocol.ConfirmRequested(
                id=offer_id, session_id=session_id, question=question
            )
        )
        return w

    async def test_the_question_reaches_the_session_it_is_about(self, wire):
        await self.offered(wire, question="Remember “OOM killed” as a failure?")
        assert "Remember “OOM killed” as a failure?" in wire.screen()

    async def test_it_stands_where_the_message_box_was(self, wire):
        # In the box's slot and not over the frame: the conversation the
        # question is about has to stay readable behind it (`_entry_h`).
        assert "── message" in wire.screen()
        await self.offered(wire)
        screen = wire.screen()
        assert "── offer" in screen
        assert "── message" not in screen
        assert "edit run.sh" in screen, "the chat is still there to read"
        assert "the second thing" in screen, "and so is the sidebar"

    async def test_the_layout_budgets_for_the_rows_it_draws(self, wire):
        """Rows drawn that no band asked for come off the bottom of the frame.

        Which is a silent failure and the reason this is asserted rather than
        eyeballed: the frame is always exactly as tall as the terminal, so an
        entry band that draws four rows where two were planned does not
        overflow — it pushes the watchers column down and two of its rows are
        cut, with nothing anywhere saying they are missing.
        """
        await self.offered(wire, question="a question long enough to wrap " * 6)
        ui, w, h = wire.ui, wire.width, wire.height
        assert ui._entry_h(w, h) == len(ui._render_offer(w, h))
        assert sum(ui._heights(h, w)) == ui._avail(w, h)

    async def test_the_cursor_lands_on_it_so_the_keys_work(self, wire):
        await self.offered(wire)
        assert wire.ui.focus == OFFER

    async def test_yes_answers_it_by_id(self, wire):
        await self.offered(wire)
        await wire.press("y")
        answer = wire.peer.last(protocol.ConfirmResolve)
        assert (answer.id, answer.confirmed) == ("sig-1", True)

    async def test_and_no_answers_it_too(self, wire):
        await self.offered(wire)
        await wire.press("n")
        answer = wire.peer.last(protocol.ConfirmResolve)
        assert (answer.id, answer.confirmed) == ("sig-1", False)

    async def test_escape_is_no_here_too(self, wire):
        await self.offered(wire)
        await wire.press("esc")
        assert wire.peer.last(protocol.ConfirmResolve).confirmed is False

    async def test_answering_gives_the_message_box_back(self, wire):
        await self.offered(wire)
        await wire.press("y")
        assert wire.ui.focus == INPUT
        assert "── message" in wire.screen()

    async def test_a_question_about_another_session_does_not_take_this_one(self, wire):
        # §3.2 property 1: a session that is not on screen may change the
        # sidebar and nothing else.
        await self.offered(wire, session_id="s2", question="Learn the other one?")
        assert "Learn the other one?" not in wire.screen()
        assert "── message" in wire.screen()
        assert wire.ui.focus != OFFER

    async def test_but_the_sidebar_says_it_is_waiting(self, wire):
        await self.offered(wire, session_id="s2")
        row = next(
            line for line in wire.frame() if "the second thing" in plain(line)
        )
        assert "?" in plain(row)

    async def test_and_switching_to_it_is_what_asks(self, wire):
        await wire.tell(protocol.ChatReset(session_id="s2", entries=[entry(1)]))
        await self.offered(wire, session_id="s2", question="Learn the other one?")
        wire.ui.focus = SESSIONS
        wire.ui.session_pane.cursor = 2  # the second session's row
        await wire.press("enter")
        assert "Learn the other one?" in wire.screen()

    async def test_a_decision_outranks_it_and_it_waits(self, wire):
        # Both want the same slot, and of the two the decision is the one
        # holding a turn.
        await self.offered(wire)
        await parked(wire, BASH_GATE)
        screen = wire.screen()
        assert "── decision" in screen
        assert "Learn it?" not in screen
        await wire.press("y")  # answer the decision
        assert "Learn it?" in wire.screen(), "the offer was waiting, not lost"

    async def test_two_questions_are_asked_one_at_a_time(self, wire):
        # The core is holding a continuation per id: a second offer landing on
        # the first would strand it with nothing able to answer it.
        await self.offered(wire, offer_id="sig-1", question="Learn the first?")
        await self.offered(wire, offer_id="sig-2", question="Learn the second?")
        assert "Learn the first?" in wire.screen()
        assert "Learn the second?" not in wire.screen()
        await wire.press("y")
        assert wire.peer.last(protocol.ConfirmResolve).id == "sig-1"
        assert "Learn the second?" in wire.screen()
        await wire.press("n")
        assert wire.peer.last(protocol.ConfirmResolve).id == "sig-2"
        assert "── message" in wire.screen()

    async def test_the_same_question_twice_is_one_question(self, wire):
        # A re-emit on subscribe must not ask twice: the second copy would be
        # unanswerable, the core having freed the continuation on the first.
        await self.offered(wire)
        await self.offered(wire)
        await wire.press("y")
        assert wire.ui.session.offers == []
        assert "── message" in wire.screen()

    async def test_walking_away_does_not_answer_it(self, wire):
        await self.offered(wire)
        await wire.press("ctrl-up")
        assert wire.ui.focus == CHAT
        assert wire.peer.took(protocol.ConfirmResolve) == []
        await wire.press("ctrl-down")
        assert wire.ui.focus == OFFER, "the ring comes back to it"
