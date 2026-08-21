"""The state layer: plain dataclasses, and the rows they draw as.

No wire and no pydantic here — that is the point of the module. Everything
below is synchronous and compares strings or ints, which is the property
specs-ui-replacement.md §3.1 is protecting when it says `app.py` must not
import the protocol.
"""

from __future__ import annotations

from hpca.ui.ansi import AMBER, RED, WHITE
from hpca.ui.state import (
    ChatEntry,
    ChatPart,
    Context,
    SessionState,
    Turn,
    entry_item,
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
        # White, where the agent is amber: the two colours the conversation is
        # actually read in, and the reason neither is one of the muted ones
        # the rest of the UI signals with.
        assert entry_item(entry(1)).accent == WHITE

    def test_and_the_agent_gets_a_colour_of_its_own(self):
        assert entry_item(entry(1, "assistant", "hello")).accent == AMBER

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
        assert entry_item(entry(1, "error", "it fell over")).accent == RED

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
        # core has since given to something else.
        session = self.loaded()
        session.chat.expanded = {"1", "2"}
        session.reset([entry(1, text="only this")])
        assert session.chat.expanded == set()

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
