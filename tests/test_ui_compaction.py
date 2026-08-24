"""The compaction review, where it now lives: inline, in the session column.

The claims are `specs-ui-acceptance.md`'s "Compaction" bullets, with the two
this move adds — the review stands in the message box's slot rather than over
the whole terminal, and a summary waiting in one conversation stays that
conversation's question. That is the rule the inline approval next door has
(`test_ui_approvals.py`, "inline, and not a modal"), and it is here for the
same reason: `/compact` answers a whole model call after the keystroke, so the
user is as likely as not reading something else by the time the summary lands.

Everything is driven the way the approvals are — a real
`InProcessConnection.pair()` with a scripted core at the far end, `tell` for
what the core said and `press` for what the user typed — so a fold is asserted
as the `compact.resolve` that actually went out.
"""

from __future__ import annotations

import pytest

from hpca import protocol
from hpca.ui.app import CHAT, INPUT, REVIEW, SESSIONS, WATCHERS, RowUI
from hpca.ui.compaction import COMMENT_REFUSAL, CUT_WARNING, HINT
from hpca.ui.keys import PASTE
from tests.ui_harness import connected, plain, widths

ROWS = [
    protocol.SessionRow(session_id="s1", title="the first thing", mode="manual"),
    protocol.SessionRow(session_id="s2", title="the second thing", mode="manual"),
]

SUMMARY = (
    "[earlier in this session]\n"
    "The user is aligning a 40-sample cohort with STAR on /scratch/proj. "
    "Job 8813 died with an OOM at 32G; 40G was the fix. The QC report is "
    "still unread."
)

# A summary nobody reads in one screenful, for the scrolling claims.
LONG = "\n".join(f"summary row {n}" for n in range(80))


def entry(seq: int, kind: str = "user", text: str = "") -> protocol.Entry:
    return protocol.Entry(kind=kind, text=text or f"row {seq}", seq=seq, index=seq - 1)


@pytest.fixture
async def wire():
    async with connected() as w:
        await w.tell(protocol.Hello(profile="hpc"))
        await w.tell(protocol.SessionRows(rows=list(ROWS)))
        await w.tell(
            protocol.ChatReset(
                session_id="s1", entries=[entry(1, text="align the cohort")]
            )
        )
        w.peer.clear()
        yield w


async def switch(wire, session_id: str) -> None:
    """Open a conversation from the sessions column, by name and not by row.

    The list re-sorts as sessions are worked in, so a fixed cursor is a row
    that means something different after the first switch.
    """
    wire.ui.focus = SESSIONS
    pane = wire.ui.session_pane
    for cursor in range(len(pane.flat(wire.inner))):
        pane.cursor = cursor
        if pane.here() == session_id:
            break
    else:
        raise AssertionError(f"no row for {session_id}")
    await wire.press("enter")


async def offered(wire, session_id: str = "s1", **kw):
    """The core has written a summary for that conversation."""
    await wire.tell(
        protocol.CompactProposed(
            **{
                "session_id": session_id,
                "summary": SUMMARY,
                "folded": 46,
                **kw,
            }
        )
    )
    return wire


# ------------------------------------------------- inline, and not a modal


class TestItIsInlineAndNotAModal:
    """A summary is offered about *one* conversation, and it arrives a model
    call after the command — so a screen would be a question about a history
    the user may no longer be looking at."""

    async def test_the_prompt_is_on_screen(self, wire):
        await offered(wire)
        assert "── compact ─" in wire.screen()
        assert "46 messages fold into this summary" in wire.screen()
        assert "died with an OOM at 32G" in wire.screen()

    async def test_and_it_is_not_a_screen(self, wire):
        await offered(wire)
        assert wire.ui.overlay is None, "an overlay is exactly what this is not"

    async def test_the_conversation_is_still_behind_it(self, wire):
        await offered(wire)
        # The whole objection to a modal: what the summary is *of* must not be
        # covered by the offer to fold it.
        assert "align the cohort" in wire.screen()
        assert "── chat ─" in wire.screen()

    async def test_it_stands_where_the_message_box_was(self, wire):
        # "Replace the message panel for the moment": one slot, and the prompt
        # has it while it is up, so there is no box offering to take a message
        # under a question about this conversation's own history.
        assert "── message ─" in wire.screen()
        await offered(wire)
        assert "── message ─" not in wire.screen()
        assert wire.ui.focus == REVIEW

    async def test_and_the_box_comes_back_with_the_draft_still_in_it(self, wire):
        wire.ui.focus = INPUT
        await wire.press(*"half a sentence")
        await offered(wire)
        assert "half a sentence" not in wire.screen()
        await wire.press("d")
        assert "── message ─" in wire.screen()
        assert "half a sentence" in wire.screen()

    async def test_it_has_the_keys(self, wire):
        await offered(wire)
        assert wire.ui.focus == REVIEW
        assert HINT in wire.screen()
        assert "enter fold it in" in plain(wire.frame()[-1])

    async def test_it_does_not_take_the_cursor_out_of_another_column(self, wire):
        # The rule a decision follows: a question arriving must never yank the
        # user out of the column they are working in.
        wire.ui.focus = WATCHERS
        await offered(wire)
        assert wire.ui.focus == WATCHERS
        assert "── compact ─" in wire.screen(), "it is still up, just not aimed at"

    @pytest.mark.parametrize("width", [80, 100, 137])
    async def test_every_row_is_exactly_the_terminal_width(self, wire, width):
        await offered(wire, summary=LONG, guidance="the STAR flags", truncated=True)
        assert widths(wire.ui.render(width, 40)) == {width}

    @pytest.mark.parametrize("width", [80, 100, 137])
    async def test_and_so_is_the_comment_stage(self, wire, width):
        await offered(wire, summary=LONG)
        await wire.press("r", *"you cut it off, and keep the sbatch flags")
        assert widths(wire.ui.render(width, 40)) == {width}

    @pytest.mark.parametrize(
        "width,height", [(80, 24), (120, 40), (60, 14), (40, 10), (100, 8)]
    )
    async def test_a_terminal_too_small_for_it_still_gets_a_frame(
        self, wire, width, height
    ):
        await offered(wire, summary=LONG, truncated=True)
        await wire.press("r", *"finish it")
        drawn = wire.ui.render(width, height)
        assert len(drawn) == height
        assert widths(drawn) == {width}

    async def test_a_long_summary_does_not_eat_the_chat(self, wire):
        await offered(wire, summary=LONG)
        drawn = [plain(x) for x in wire.ui.render(100, 40)]
        assert wire.ui._review_h(100, 40) <= (40 - 2) // 2
        chat = next(i for i, x in enumerate(drawn) if "── chat ─" in x)
        review = next(i for i, x in enumerate(drawn) if "── compact ─" in x)
        assert review - chat > RowUI.MIN_CHAT, "the conversation is still there"

    async def test_an_open_screen_keeps_the_keys_until_it_closes(self, wire):
        # It can land while the key list is up. The prompt is drawn under the
        # screen, not over it, so the screen answers first — and nothing is
        # lost, because the core holds the offer.
        wire.ui.focus = CHAT
        await wire.press("?")
        await offered(wire)
        await wire.press("enter")  # closes the key list; not a fold
        assert wire.peer.took(protocol.CompactResolve) == []
        assert wire.ui.overlay is None
        await wire.press("enter")
        assert wire.peer.last(protocol.CompactResolve).action == "accept"


# ------------------------------------------------- what it says it is folding


class TestWhatItSays:
    async def test_how_many_messages_fold(self, wire):
        await offered(wire, folded=12)
        assert "12 messages fold into this summary" in wire.screen()

    async def test_the_instruction_it_was_written_for(self, wire):
        await offered(wire, guidance="the STAR flags")
        assert "asked to keep: the STAR flags" in wire.screen()

    async def test_a_cut_summary_is_flagged_as_cut(self, wire):
        # The one thing the text cannot say about itself, and the reason the
        # core carries `truncated` at all.
        await offered(wire, truncated=True)
        assert CUT_WARNING in wire.screen()

    async def test_an_untruncated_one_is_not(self, wire):
        await offered(wire)
        assert CUT_WARNING not in wire.screen()

    async def test_a_retry_says_which_attempt_this_is(self, wire):
        await offered(wire, attempt=3)
        assert "attempt 3" in wire.screen()

    async def test_a_first_attempt_does_not(self, wire):
        await offered(wire)
        assert "attempt" not in wire.screen()


# ----------------------------------------------------------- the three answers


class TestAnsweringIt:
    async def test_enter_folds_it_in(self, wire):
        await offered(wire)
        await wire.press("enter")
        assert wire.peer.last(protocol.CompactResolve).action == "accept"

    async def test_and_names_the_session_it_was_offered_for(self, wire):
        await offered(wire)
        await wire.press("enter")
        assert wire.peer.last(protocol.CompactResolve).session_id == "s1"

    async def test_d_discards_it(self, wire):
        await offered(wire)
        await wire.press("d")
        assert wire.peer.last(protocol.CompactResolve).action == "discard"

    async def test_the_prompt_goes_when_it_is_answered(self, wire):
        await offered(wire)
        await wire.press("d")
        assert "── compact ─" not in wire.screen()
        assert wire.ui.focus == INPUT

    async def test_and_cannot_be_answered_twice(self, wire):
        await offered(wire)
        await wire.press("enter", "enter")
        assert len(wire.peer.took(protocol.CompactResolve)) == 1

    async def test_r_asks_again_with_what_the_user_typed(self, wire):
        await offered(wire)
        await wire.press("r", *"finish the last sentence", "enter")
        answer = wire.peer.last(protocol.CompactResolve)
        assert answer.action == "retry"
        assert answer.comment == "finish the last sentence"

    async def test_the_summary_stays_on_screen_while_the_comment_is_typed(self, wire):
        await offered(wire)
        await wire.press("r", *"finish it")
        assert "died with an OOM at 32G" in wire.screen()
        assert "finish it" in wire.screen()

    async def test_an_empty_comment_is_refused_rather_than_sent(self, wire):
        # The same generation again, with the model none the wiser.
        await offered(wire)
        await wire.press("r", "enter")
        assert wire.peer.took(protocol.CompactResolve) == []
        assert COMMENT_REFUSAL in wire.screen()

    async def test_escaping_the_comment_box_keeps_the_summary_up(self, wire):
        await offered(wire)
        await wire.press("r", *"never mind", "esc")
        assert "── compact ─" in wire.screen()
        assert wire.peer.took(protocol.CompactResolve) == []
        await wire.press("enter")  # and it can still be accepted
        assert wire.peer.last(protocol.CompactResolve).action == "accept"

    async def test_the_next_offer_arrives_with_an_empty_box(self, wire):
        # The complaint has been sent; leaving it in the box would invite
        # sending it twice.
        await offered(wire)
        await wire.press("r", *"finish it", "enter")
        await offered(wire, attempt=2)
        assert "finish it" not in wire.screen()

    async def test_a_paste_lands_in_the_comment_box(self, wire):
        await offered(wire)
        await wire.press("r")
        await wire.press(PASTE + "two shards were still open")
        assert "two shards were still open" in wire.screen()
        await wire.press("enter")
        assert wire.peer.last(protocol.CompactResolve).comment == (
            "two shards were still open"
        )


class TestEscapeAnswersNothing:
    """The offer cost a generation, so the way out of the prompt must not be a
    verdict — the core goes on holding it, and `/compact` brings it back."""

    async def test_it_sends_nothing(self, wire):
        await offered(wire)
        await wire.press("esc")
        assert wire.peer.took(protocol.CompactResolve) == []

    async def test_and_gives_the_message_box_back(self, wire):
        # The difference between this prompt and the approval's: a summary
        # holds no turn, so the conversation can still be typed into.
        await offered(wire)
        await wire.press("esc")
        assert "── compact ─" not in wire.screen()
        assert "── message ─" in wire.screen()
        assert wire.ui.focus == INPUT

    async def test_the_summary_is_still_the_sessions(self, wire):
        await offered(wire)
        await wire.press("esc")
        assert wire.ui.session.compaction is not None
        rows = [x for x in wire.frame() if "the first thing" in x]
        assert "?" in rows[0], "the sidebar still says it is waiting"

    async def test_and_compact_stands_it_up_again(self, wire):
        # Without waiting for the core's re-offer: a bare `/compact` on a held
        # summary is answered with the same text, so a keystroke that appeared
        # to do nothing until the round trip landed would be the worse half of
        # the exchange.
        await offered(wire)
        await wire.press("esc")
        await wire.press(*"/compact", "enter")
        assert "── compact ─" in wire.screen()
        assert wire.ui.focus == REVIEW
        assert wire.peer.took(protocol.CompactResolve) == []

    async def test_and_a_brief_after_it_does_not_show_the_old_one(self, wire):
        # There is a new summary coming; the held text is not it.
        await offered(wire)
        await wire.press("esc")
        await wire.press(*"/compact keep the sbatch flags", "enter")
        assert "── compact ─" not in wire.screen()

    async def test_nothing_to_review_says_so(self, wire):
        wire.ui.review_compaction()
        assert "nothing to review" in wire.screen()


# ------------------------------------------------ a summary in another session


class TestASummaryInAnotherSession:
    async def test_it_flags_the_sidebar_row(self, wire):
        await offered(wire, session_id="s2")
        # Any of them: the toast naming the session it is waiting in carries
        # the same title, and it is the row under the sessions rule that has
        # to say so once the toast has gone.
        rows = [x for x in wire.frame() if "the second thing" in x]
        assert any("?" in x for x in rows)

    async def test_and_puts_no_prompt_on_this_one(self, wire):
        await offered(wire, session_id="s2")
        assert "── compact ─" not in wire.screen()
        assert wire.ui.focus != REVIEW
        assert "── message ─" in wire.screen(), "this session's box is untouched"

    async def test_and_says_where_it_is(self, wire):
        await offered(wire, session_id="s2")
        assert "the second thing" in wire.screen()
        assert "/compact opens it" in wire.screen()

    async def test_opening_that_session_reveals_it(self, wire):
        await offered(wire, session_id="s2")
        await switch(wire, "s2")
        assert "46 messages fold into this summary" in wire.screen()

    async def test_and_answering_it_there_names_that_session(self, wire):
        await offered(wire, session_id="s2")
        await switch(wire, "s2")
        wire.ui.focus = REVIEW
        await wire.press("enter")
        assert wire.peer.last(protocol.CompactResolve).session_id == "s2"

    async def test_a_half_typed_complaint_survives_the_switch(self, wire):
        # It lives on the session, like the refusal reason next door: the
        # words typed about one summary are still there on the way back.
        await offered(wire)
        await wire.press("r", *"keep the sbatch flags")
        await switch(wire, "s2")
        assert "keep the sbatch flags" not in wire.screen()
        await switch(wire, "s1")
        assert "keep the sbatch flags" in wire.screen()


# ------------------------------------------------------------- reading it


class TestReadingTheSummary:
    async def test_it_scrolls(self, wire):
        await offered(wire, summary=LONG)
        assert "summary row 0" in wire.screen()
        await wire.press("pgdn")
        assert "summary row 0" not in wire.screen()

    async def test_and_says_how_far_down_it_is(self, wire):
        await offered(wire, summary=LONG)
        assert "/80 ──" in wire.screen()

    async def test_a_summary_that_fits_says_nothing_about_position(self, wire):
        await offered(wire)
        assert "/80 ──" not in wire.screen()

    async def test_end_reaches_the_bottom(self, wire):
        await offered(wire, summary=LONG)
        await wire.press("end")
        assert "summary row 79" in wire.screen()

    async def test_the_arrows_do_not_scroll_the_comment_box(self, wire):
        # At the comment stage they are the editor's, which is where a cursor
        # actually is.
        await offered(wire, summary=LONG)
        await wire.press("r")
        top = wire.ui.session.review.offset
        await wire.press("down", "down")
        assert wire.ui.session.review.offset == top


class TestTheRing:
    """A row of the ring like every other prompt in this slot: its own keys
    must never be the only way out of it."""

    async def test_ctrl_up_goes_to_the_chat_and_comes_back(self, wire):
        await offered(wire)
        await wire.press("ctrl-up")
        assert wire.ui.focus == CHAT
        await wire.press("ctrl-down")
        assert wire.ui.focus == REVIEW

    async def test_ctrl_down_goes_to_the_watchers(self, wire):
        await offered(wire)
        await wire.press("ctrl-down")
        assert wire.ui.focus == WATCHERS

    async def test_walking_off_does_not_answer_it(self, wire):
        await offered(wire)
        await wire.press("ctrl-up", "ctrl-down")
        assert wire.peer.took(protocol.CompactResolve) == []
        assert "── compact ─" in wire.screen()

    async def test_the_comment_survives_the_walk(self, wire):
        await offered(wire)
        await wire.press("r", *"keep the flags", "ctrl-up", "ctrl-down")
        assert "keep the flags" in wire.screen()
        await wire.press("enter")
        assert wire.peer.last(protocol.CompactResolve).comment == "keep the flags"

    async def test_a_decision_takes_the_slot_first(self, wire):
        # Of the two, the decision is the one holding a turn; the summary
        # waits in the core for as long as it takes.
        await offered(wire)
        await wire.tell(
            protocol.DecisionRequested(
                session_id="s1",
                payload={"tool": "delete_file", "kind": "destructive"},
            )
        )
        assert "── decision ─" in wire.screen()
        assert "── compact ─" not in wire.screen()
        await wire.press("y")
        assert "── compact ─" in wire.screen(), "and comes back once it is answered"
