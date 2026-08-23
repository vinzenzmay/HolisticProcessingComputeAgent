"""The state layer: plain dataclasses, and the rows they draw as.

No wire and no pydantic here — that is the point of the module. Everything
below is synchronous and compares strings or ints, which is the property
specs-ui-replacement.md §3.1 is protecting when it says `app.py` must not
import the protocol.
"""

from __future__ import annotations

import re

from hpca.ui import theme
from hpca.ui.ansi import RESET, REVERSE, cell_width, pad
from hpca.ui.rain import (
    KATAKANA,
    SPINNER_STEPS,
    SPINNER_WIDTH,
    spinner,
    spinner_trail,
)
from hpca.ui.state import (
    ChatEntry,
    ChatPart,
    Context,
    Display,
    SessionState,
    Turn,
    entry_item,
    when,
)


def entry(seq: int = 1, kind: str = "user", text: str = "said something", **kw):
    return ChatEntry(kind=kind, text=text, seq=seq, **kw)


# ------------------------------------------------------- an entry as a row


class TestEntriesBecomeRows:
    def test_the_row_is_named_by_the_seq_and_nothing_else(self):
        assert entry_item(entry(7)).key == "7"

    def test_your_own_words_come_back_undecorated(self):
        # Enter on the row hands back what was said, not the line it is drawn
        # as: the rewind quotes it and the copy puts it in the message box.
        item = entry_item(entry(1, text="run it again"))
        assert (item.kind, item.text) == ("user", "run it again")

    def test_and_are_marked_as_yours(self):
        # Amber, where the agent is white: the two colours the conversation is
        # actually read in, and the reason neither is one of the muted ones
        # the rest of the UI signals with. The user's own lines are the few,
        # and they are what a scrollback is searched for.
        assert entry_item(entry(1)).accent == theme.user

    def test_and_the_agent_gets_a_colour_of_its_own(self):
        assert entry_item(entry(1, "assistant", "hello")).accent == theme.agent

    def test_a_message_puts_its_words_under_a_label_of_its_own(self):
        # The head says who is speaking and nothing else, so that every line
        # with words on it is drawn flush at column 0 and can be selected out
        # of the terminal without a label or an indent coming with it.
        item = entry_item(entry(1, "assistant", "first\nsecond\nthird"))
        assert item.head == "hpca"
        assert item.body == ["first", "second", "third"]

    def test_and_yours_says_you(self):
        item = entry_item(entry(1, text="run it again"))
        assert (item.head, item.body) == ("you", ["run it again"])

    def test_an_empty_row_has_nothing_to_open(self):
        # An assistant row is appended before its first token arrives, and a
        # body of one blank line would give it a marker pointing at nothing.
        assert entry_item(entry(1, "assistant", "")).body == []

    def test_a_notice_is_still_collapsed_to_one_line(self):
        # The rows that are the UI talking *about* the conversation keep their
        # one-line form — they are short by construction and nobody pastes
        # them — and a head that carried a newline would break the layout
        # under it.
        item = entry_item(entry(1, "event", "first\nsecond\nthird"))
        assert "\n" not in item.head
        assert "first second third" in item.head

    def test_but_the_whole_thing_is_still_there_to_open(self):
        item = entry_item(entry(1, "assistant", "first\nsecond"))
        assert item.body == ["first", "second"]

    def test_an_error_is_red(self):
        assert entry_item(entry(1, "error", "it fell over")).accent == theme.danger

    def test_a_turn_is_grey_and_says_so_itself(self):
        # Not merely left uncoloured: an accentless row falls through to the
        # renderer's default, which dims the body and leaves the head at full
        # weight — so a closed turn shouted as loudly as a reply, and an
        # opened one was grey on its steps and bright on the line above them.
        assert entry_item(ChatEntry(kind="thinking", seq=3, steps=4)).accent == theme.faint

    def test_a_speaker_line_is_marked_as_a_label(self):
        # What gets it drawn bold. Carried as a flag rather than as an escape
        # in the head, because every line is padded to an exact number of
        # cells before any styling is wrapped around it.
        assert entry_item(entry(1)).label
        assert entry_item(entry(1, "assistant", "hello")).label
        assert entry_item(entry(1, "error", "it fell over")).label

    def test_and_a_summary_line_is_not(self):
        # A turn's head describes what is under it rather than naming who
        # said it, and the weight is there to separate one speaker from the
        # next.
        assert not entry_item(ChatEntry(kind="thinking", seq=3, steps=4)).label
        assert not entry_item(entry(1, "event", "compacted")).label

    def test_a_fold_counts_its_steps(self):
        item = entry_item(
            ChatEntry(
                kind="thinking",
                seq=3,
                steps=4,
                parts=[ChatPart(kind="call", tool="read_file", done=True)],
            )
        )
        assert "4 steps" in item.head

    def test_and_names_the_first_few_tools(self):
        item = entry_item(
            ChatEntry(
                kind="thinking",
                seq=3,
                parts=[
                    ChatPart(kind="call", tool=name, done=True)
                    for name in ("read_file", "edit_file", "run_bash", "list_dir")
                ],
            )
        )
        assert "read_file → edit_file → run_bash …" in item.head

    def test_a_step_is_a_fold_of_its_own(self):
        # One box per turn, opening into its steps, each of those opening into
        # what the tool returned — a tool result is routinely a whole file,
        # and the steps either side of it have to stay readable as a list.
        item = entry_item(
            ChatEntry(
                kind="thinking",
                seq=3,
                parts=[
                    ChatPart(
                        kind="call",
                        tool="read_file",
                        target="/scratch/run.log",
                        result="412 lines",
                        done=True,
                    )
                ],
            )
        )
        assert item.folds[0].head.split() == ["read_file", "/scratch/run.log"]
        assert item.folds[0].body == ["412 lines"]

    def test_a_call_still_running_says_so(self):
        # `Part.done` is the whole of how a client tells a finished tool
        # exchange from one that only looks finished.
        item = entry_item(
            ChatEntry(
                kind="thinking",
                seq=3,
                parts=[ChatPart(kind="call", tool="run_bash", done=False)],
            )
        )
        assert [x.head for x in item.folds] == ["run_bash …"]

    def test_a_result_with_no_call_of_its_own_still_gets_a_row(self):
        # The half-exchange the graph reports when a result arrives with
        # nothing to attach it to: it is still something that happened.
        item = entry_item(
            ChatEntry(
                kind="thinking",
                seq=3,
                parts=[ChatPart(kind="step", text="a note", result="ok", done=True)],
            )
        )
        assert len(item.folds) == 1
        assert "a note" in item.folds[0].head

    def test_a_queued_message_is_drawn_as_the_chat_it_will_become(self):
        # Still the user's words and still copyable as such; the label is what
        # says they have not been sent yet.
        item = entry_item(entry(2, "queued", "the next thing"))
        assert item.kind == "queued"
        assert item.body == ["the next thing"]
        assert "queued" in item.head

    def test_a_kind_nobody_taught_it_still_draws(self):
        # The client drops events it cannot draw; a *kind* it has not seen is a
        # different thing — the row exists and the text is what matters.
        item = entry_item(entry(1, "something-new", "still readable"))
        assert item.body == ["still readable"]


# --------------------------------------------------------------- the chat


class TestTheChatIsAppendOnly:
    def loaded(self) -> SessionState:
        session = SessionState("s1")
        session.reset([entry(1, text="one"), entry(2, "assistant", "two")])
        return session

    def test_a_reset_is_the_transcript(self):
        assert [x.text for x in self.loaded().entries] == ["one", "two"]

    def test_and_marks_the_session_as_loaded(self):
        assert self.loaded().loaded

    def test_a_new_session_is_not(self):
        assert not SessionState("s1").loaded

    def test_an_append_adds_exactly_one_row(self):
        session = self.loaded()
        session.append(entry(3, text="three"))
        assert len(session.chat.items) == 3

    def test_an_update_replaces_the_row_it_names(self):
        session = self.loaded()
        assert session.update(entry(2, "assistant", "two, revised"))
        assert [x.text for x in session.entries] == ["one", "two, revised"]

    def test_and_leaves_it_where_it_was(self):
        session = self.loaded()
        session.update(entry(1, text="one, revised"))
        assert session.chat.items[0].text == "one, revised"

    def test_an_update_for_a_row_that_is_not_here_is_refused(self):
        session = self.loaded()
        assert session.update(entry(9, text="from nowhere")) is False

    def test_and_adds_nothing(self):
        session = self.loaded()
        session.update(entry(9))
        assert len(session.entries) == 2

    def test_an_unnumbered_row_cannot_be_updated(self):
        session = SessionState("s1")
        session.append(entry(0, "error", "the UI wrote this one"))
        assert session.update(entry(0, "error", "revised")) is False

    def test_a_reset_forgets_what_was_open(self):
        # The numbering is re-based, so a key still held would name a row the
        # core has since given to something else. Renumbered here on purpose:
        # a reset back onto seq 1 could not tell a key that survived from one
        # the fresh transcript opened for itself.
        session = self.loaded()
        session.chat.expanded = {"1", "2"}
        session.reset([entry(5, text="only this")])
        assert session.chat.expanded == {"5"}

    def test_a_reset_lets_an_estimate_speak_again(self):
        session = self.loaded()
        session.context.measured = True
        session.reset([entry(1)])
        assert not session.context.measured

    def test_the_rows_are_addressable_after_a_reset(self):
        session = self.loaded()
        session.reset([entry(5, text="renumbered")])
        assert session.update(entry(5, text="revised"))

    def test_but_the_old_names_are_not(self):
        session = self.loaded()
        session.reset([entry(5, text="renumbered")])
        assert session.update(entry(1, text="revised")) is False

    def test_an_entry_can_be_found_by_the_row_it_is_under(self):
        assert self.loaded().entry_at(1).text == "two"

    def test_and_off_the_end_is_nothing_rather_than_an_error(self):
        assert self.loaded().entry_at(99) is None
        assert self.loaded().entry_at(-1) is None


class TestTheNewestRowShowsItself:
    """Exactly one row is open by itself, and it is the last one.

    The chat cleans up behind itself as it grows: whatever was said most
    recently is shown whole, and the moment something newer arrives it folds
    back to a label and a line. What the *user* opened is not part of the
    bargain — that is a row somebody is reading, and a reply landing is not a
    reason to take it away.
    """

    def loaded(self) -> SessionState:
        session = SessionState("s1")
        session.reset([entry(1, text="one"), entry(2, "assistant", "two")])
        return session

    def test_the_last_reply_is_open_without_being_asked(self):
        assert self.loaded().chat.expanded == {"2"}

    def test_a_row_arriving_folds_the_one_it_replaced(self):
        session = self.loaded()
        session.append(entry(3, text="three"))
        assert session.chat.expanded == {"3"}

    def test_and_the_lines_on_screen_agree(self):
        # The closing is done against the cached line list rather than by
        # dropping it, so this is the half that would silently go stale.
        session = self.loaded()
        session.chat.flat(60)
        session.append(entry(3, text="three"))
        rebuilt = [x[1] for x in session.chat.flat(60)]
        session.chat.invalidate()
        assert rebuilt == [x[1] for x in session.chat.flat(60)]

    def test_a_row_the_user_opened_is_left_alone(self):
        # The auto-open is undone by name, not by closing whatever is open.
        session = self.loaded()
        session.chat.expanded.add("1")
        session.append(entry(3, text="three"))
        assert "1" in session.chat.expanded

    def test_a_row_with_nothing_behind_it_opens_nothing(self):
        # A notice is one line to begin with, and a marker pointing at an
        # empty body is worse than no marker.
        session = self.loaded()
        session.append(entry(3, "event", "context compacted"))
        assert session.chat.expanded == set()

    def test_a_reply_that_arrives_empty_opens_when_its_words_do(self):
        # An assistant row is appended before its first token, so the row
        # that has to open is the one that had nothing to open at the time.
        session = self.loaded()
        session.append(ChatEntry(kind="assistant", text="", seq=3))
        assert session.chat.expanded == set()
        session.update(ChatEntry(kind="assistant", text="a re", seq=3))
        assert session.chat.expanded == {"3"}

    def test_and_stays_open_as_the_rest_of_it_streams_in(self):
        session = self.loaded()
        session.append(ChatEntry(kind="assistant", text="", seq=3))
        for text in ("a re", "a reply", "a reply, whole"):
            session.update(ChatEntry(kind="assistant", text=text, seq=3))
        assert session.chat.expanded == {"3"}

    def test_but_one_the_user_folded_away_is_not_reopened(self):
        # The token after the fold would otherwise undo it, ten times a
        # second, which is the one way this could be worse than not opening.
        session = self.loaded()
        session.append(ChatEntry(kind="assistant", text="a re", seq=3))
        session.chat.expanded.discard("3")
        session.update(ChatEntry(kind="assistant", text="a reply", seq=3))
        assert session.chat.expanded == set()

    def test_a_live_turn_keeps_its_steps_while_the_reply_lands(self):
        # Two reasons to be open, undone by two different events: the steps
        # of a running turn belong to `end_turn`, not to the next row.
        session = self.loaded()
        session.start_turn()
        session.append(ChatEntry(kind="thinking", seq=3, steps=2))
        session.append(ChatEntry(kind="assistant", text="done", seq=4))
        assert session.chat.expanded == {"3", "4"}

    def test_an_unqueued_row_hands_the_opening_back(self):
        # `turn.unqueued` takes the newest row away again; the one it leaves
        # behind is the newest now.
        session = self.loaded()
        session.append(entry(3, "queued", "not yet"))
        session.remove(3)
        assert session.chat.expanded == {"2"}


# ---------------------------------------------------------------- the turn


class TestTheTurn:
    def test_a_repeated_activity_keeps_its_clock(self):
        turn = Turn()
        turn.activity_is("reading", "10:00:00")
        turn.activity_is("reading", "10:00:09")
        assert turn.started_at == "10:00:00"

    def test_a_new_activity_takes_the_new_one(self):
        turn = Turn()
        turn.activity_is("reading", "10:00:00")
        turn.activity_is("writing", "10:00:09")
        assert turn.started_at == "10:00:09"

    def test_no_activity_at_all_means_the_work_is_over(self):
        turn = Turn()
        turn.activity_is("reading", "10:00:00")
        turn.activity_is("", "")
        assert turn.activity == ""


class TestTheSpinner:
    """Four cells of katakana, bouncing, with the trail behind where it was."""

    def test_it_is_four_cells_wide_on_every_frame(self):
        for frame in range(len(SPINNER_STEPS) * 3):
            glyphs, styles = spinner(frame)
            assert len(glyphs) == len(styles) == SPINNER_WIDTH
            assert cell_width(glyphs) == SPINNER_WIDTH

    def test_the_head_walks_out_and_back(self):
        heads = [spinner(f)[1].index(spinner_trail()[0]) for f in range(6)]
        assert heads == [0, 1, 2, 3, 2, 1]
        assert spinner(6)[1].index(spinner_trail()[0]) == 0, "and round again"

    def test_the_trail_is_on_the_side_it_came_from(self):
        # Heading right at cell 2: the lit cells behind it are 1 and 0, and
        # the cell it is about to move into is dark. The other way round is
        # what a trail drawn ahead of the head would look like, and it reads
        # as the drop being pushed rather than as it moving.
        styles = spinner(2)[1]
        assert styles[3] == ""
        assert [styles[1], styles[0]] == [spinner_trail()[1], spinner_trail()[2]]
        # And going the other way it is the mirror of that: heading left at
        # cell 2, the one lit cell behind it is 3 and cell 1 is dark.
        styles = spinner(4)[1]
        assert styles[1] == ""
        assert styles[3] == spinner_trail()[1]

    def test_it_fades_rather_than_stopping(self):
        # Distinct styles all the way down, or the "decay" is two shades and a
        # step: at the widest reach all four cells are lit and each is dimmer.
        styles = spinner(3)[1]
        assert list(styles) == list(reversed(spinner_trail()))

    def test_the_glyphs_churn(self):
        assert spinner(0)[0] != spinner(6)[0], "same position, new characters"

    def test_and_are_never_digits(self):
        # The rain mixes them in for the flicker; four cells is not a field,
        # and a lone "7" on the working row is a number somebody tries to read.
        seen = {ch for f in range(200) for ch in spinner(f)[0]} - {" "}
        assert seen <= set(KATAKANA)


class TestThePaintedWorkingRow:
    def paint(self, turn: Turn, now: float, width: int = 60, style: str = ""):
        return turn.paint(now)(pad(turn.line(now), width), style)

    def test_the_line_itself_carries_no_escapes(self):
        # Every row is measured before it is styled (`ansi.pad`), so a spinner
        # that arrived coloured would be a working row padded to the wrong
        # width — and one cell of that is a torn repaint.
        turn = Turn(working=True, activity="reading")
        assert "\x1b" not in turn.line(0.0)
        assert cell_width(turn.line(0.0)) == len(turn.line(0.0))

    def test_the_colours_go_on_over_the_padded_row(self):
        turn = Turn(working=True, activity="reading")
        painted = self.paint(turn, 0.0)
        assert spinner_trail()[0] in painted
        assert re.sub(r"\x1b\[[0-9;]*m", "", painted) == pad(turn.line(0.0), 60)

    def test_and_close_back_into_the_style_the_row_is_drawn_in(self):
        # On the cursor's own row that is REVERSE, and a bare reset after the
        # spinner would end the highlight four cells into the line.
        turn = Turn(working=True, activity="reading")
        painted = self.paint(turn, 0.0, style=REVERSE)
        head, _, rest = painted.partition(RESET + REVERSE)
        assert spinner_trail()[0] in head, "the trail was drawn"
        assert rest.startswith(" reading"), "and the highlight taken up again"

    def test_a_row_too_narrow_to_have_kept_it_is_left_alone(self):
        # `pad` truncates; the answer to a cut-off spinner is a plain line, not
        # colour landing on whatever cells are left.
        turn = Turn(working=True, activity="reading")
        painted = turn.paint(0.0)(pad(turn.line(0.0), 2), "")
        assert "\x1b" not in painted


class TestTheContextMeter:
    def test_it_is_a_percentage_of_the_window(self):
        assert Context(used=2_500, window=10_000).percent == 25

    def test_an_estimate_says_that_it_is_one(self):
        assert Context(used=2_500, window=10_000).label() == "~25% ctx"

    def test_a_measurement_does_not(self):
        assert Context(used=2_500, window=10_000, measured=True).label() == "25% ctx"

    def test_nothing_known_says_nothing(self):
        assert Context().label() == ""

    def test_and_never_divides_by_a_window_it_does_not_have(self):
        assert Context(used=900).percent == 0


class TestWhen:
    """`when`: a core ISO stamp as the local wall clock a person reads.

    On this side of the wire because only this side knows which clock that is
    — the core stamps UTC precisely so it need not be on the same machine
    (specs-core-process.md).
    """

    def test_a_stamp_is_written_day_first_with_seconds(self):
        # Seconds are not decoration: a question and its answer routinely land
        # in the same minute and the log is read to tell them apart.
        got = when("2026-08-21T12:34:56+00:00")
        assert re.fullmatch(r"21-08-2026 \d\d:\d\d:56", got), got

    def test_and_it_is_shown_in_local_time(self):
        # Same instant, two offsets, one answer.
        assert when("2026-08-21T12:34:56+00:00") == when("2026-08-21T14:34:56+02:00")

    def test_no_stamp_is_no_text(self):
        assert when("") == ""

    def test_and_nor_is_something_that_is_not_one(self):
        # A row the UI wrote itself, or a field a future core fills in
        # differently: neither is a reason to raise in the middle of a frame.
        assert when("whenever") == ""


class TestRowsSayWhenTheyHappened:
    AT = "2026-08-21T12:34:56+00:00"

    def test_your_own_message_is_labelled_with_its_time(self):
        item = entry_item(entry(1, text="run it", at=self.AT))
        assert item.head.startswith("you 21-08-2026 ")

    def test_and_so_is_the_agents(self):
        item = entry_item(entry(1, "assistant", "done", at=self.AT))
        assert item.head.startswith("hpca 21-08-2026 ")

    def test_a_queued_one_says_both(self):
        item = entry_item(entry(2, "queued", "next", at=self.AT))
        assert item.head.startswith("you 21-08-2026 ")
        assert item.head.endswith(" · queued")

    def test_an_error_too(self):
        assert entry_item(entry(1, "error", "it fell over", at=self.AT)).head.startswith(
            "error 21-08-2026 "
        )

    def test_a_row_with_no_stamp_is_just_the_label(self):
        # Not "you --" or "you 01-01-1970": a row the core did not stamp says
        # nothing about when rather than saying something false.
        assert entry_item(entry(1, text="run it")).head == "you"

    def test_a_turn_is_not_labelled_with_one(self):
        # Its head is a summary of several messages, and it has no instant to
        # name (`transcript.Entry.at`).
        item = entry_item(ChatEntry(kind="thinking", seq=3, steps=4))
        assert item.head == "4 steps"


class TestTheStampCanBeTurnedOff:
    """`Display.chat_stamps` — the setting for a reader who wants the
    conversation and not the clock, and for a terminal narrow enough that
    nineteen characters of date in front of every message costs a column.

    Which setting is on is not something this layer can look up: the settings
    file is out of a front-end's reach (§4.2 rule 2), so it is handed in — by
    `SessionState`, which was handed it by `RowUI.set_display`, which was told
    by the core.
    """

    AT = "2026-08-21T12:34:56+00:00"

    def test_the_stamp_can_be_turned_off(self):
        item = entry_item(entry(1, text="run it", at=self.AT), stamps=False)
        assert item.head == "you"

    def test_and_the_agents_too(self):
        item = entry_item(entry(1, "assistant", "done", at=self.AT), stamps=False)
        assert item.head == "hpca"

    def test_a_queued_row_keeps_the_half_that_is_not_a_time(self):
        # The " · queued" is what says it has not been sent yet, which is a
        # fact about the message and not about the clock.
        item = entry_item(entry(2, "queued", "next", at=self.AT), stamps=False)
        assert item.head == "you · queued"

    def test_an_error_row_too(self):
        item = entry_item(entry(1, "error", "fell over", at=self.AT), stamps=False)
        assert item.head == "error"

    def test_it_is_the_same_label_a_row_with_no_stamp_draws(self):
        # One shape for "no time on this row", however it came about — a
        # second layout would have nothing else in the pane to line up with.
        off = entry_item(entry(1, text="run it", at=self.AT), stamps=False)
        assert off.head == entry_item(entry(1, text="run it")).head

    def test_nothing_else_about_the_row_moves(self):
        with_stamp = entry_item(entry(1, text="run it", at=self.AT))
        without = entry_item(entry(1, text="run it", at=self.AT), stamps=False)
        assert (without.accent, without.body, without.key, without.label) == (
            with_stamp.accent,
            with_stamp.body,
            with_stamp.key,
            with_stamp.label,
        )

    def test_the_stamp_is_on_unless_something_says_otherwise(self):
        # The default is the one `config.DisplaySettings` carries, so a
        # `SessionState` built by a test draws what a fresh install would.
        assert Display().chat_stamps is True
        assert SessionState("s1").display.chat_stamps is True

    def test_a_session_draws_its_rows_with_what_it_was_given(self):
        session = SessionState("s1", display=Display(chat_stamps=False))
        session.reset([entry(1, text="run it", at=self.AT)])
        assert session.chat.items[0].head == "you"

    def test_and_a_row_arriving_later_is_drawn_the_same_way(self):
        session = SessionState("s1", display=Display(chat_stamps=False))
        session.reset([entry(1, text="run it", at=self.AT)])
        session.append(entry(2, "assistant", "done", at=self.AT))
        assert [x.head for x in session.chat.items] == ["you", "hpca"]

    def test_and_so_is_one_revised_in_place(self):
        session = SessionState("s1", display=Display(chat_stamps=False))
        session.reset([entry(1, "assistant", "", at=self.AT)])
        session.update(entry(1, "assistant", "done", at=self.AT))
        assert session.chat.items[0].head == "hpca"


class TestRestylingAChatThatIsAlreadyOnScreen:
    """What `restyle` is for: the setting changed under a conversation that is
    already drawn, and every row has to be rebuilt without the conversation
    itself being disturbed. `reset` would do the rebuilding and throw away the
    rest — it re-bases the numbering and clears what is open, because it is
    the path a *different transcript* arrives by."""

    AT = "2026-08-21T12:34:56+00:00"

    def rows(self) -> list[ChatEntry]:
        return [
            entry(1, text="run it", at=self.AT),
            entry(2, "assistant", "done\nand dusted", at=self.AT),
        ]

    def test_every_row_is_redrawn(self):
        session = SessionState("s1")
        session.reset(self.rows())
        assert session.chat.items[0].head.startswith("you 21-08-2026")
        session.restyle(Display(chat_stamps=False))
        assert [x.head for x in session.chat.items] == ["you", "hpca"]

    def test_and_the_setting_sticks_for_what_arrives_next(self):
        session = SessionState("s1")
        session.reset(self.rows())
        session.restyle(Display(chat_stamps=False))
        session.append(entry(3, text="again", at=self.AT))
        assert session.chat.items[-1].head == "you"

    def test_the_numbering_is_untouched(self):
        # A reset re-bases it, which would make every open row's key name a
        # different row. Nothing about the conversation changed here.
        session = SessionState("s1")
        session.reset(self.rows())
        session.restyle(Display(chat_stamps=False))
        assert [x.key for x in session.chat.items] == ["1", "2"]

    def test_and_what_was_open_stays_open(self):
        session = SessionState("s1")
        session.reset(self.rows())
        session.chat.expanded.add("1")
        session.restyle(Display(chat_stamps=False))
        assert "1" in session.chat.expanded

    def test_and_what_was_said_is_still_there(self):
        session = SessionState("s1")
        session.reset(self.rows())
        session.restyle(Display(chat_stamps=False))
        assert session.chat.items[1].body == ["done", "and dusted"]
