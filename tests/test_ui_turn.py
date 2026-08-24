"""M4b — the turn, on screen: the spinner, the live steps, the meter, the bars.

Everything here is either a frame (a string) or a typed command on a real
`InProcessConnection.pair()`; the transport is never mocked and the core is a
scripted peer. The claims are `specs/specs-ui-acceptance.md`'s, in its own order:
"The spinner", "Thinking box, tool rows, logging", "Context meter", "Agent
modes", and the model-line bullet under "Backends".

Two clocks are driven by hand. The escape window is monotonic and belongs to a
*gesture*; how long a turn has been running is wall time measured against a
stamp the core took, because the two processes share a wall clock and not a
monotonic one.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from hpca import protocol
from hpca.ui import theme
from hpca.ui.app import CHAT, INPUT, SESSIONS, WATCHERS, RowUI
from hpca.ui.rain import KATAKANA as SPINNER_GLYPHS
from hpca.ui.state import Context
from tests.ui_harness import Wire, at_wall, connected, plain, widths

STARTED = "2026-08-21T10:00:00+00:00"
EPOCH = datetime.fromisoformat(STARTED).timestamp()
LATER = "2026-08-21T10:00:09+00:00"

ROWS = [
    protocol.SessionRow(
        session_id="s1",
        title="the first thing",
        profile="hpc",
        mode="auto",
        model="qwen3-27b-fp8",
    ),
    protocol.SessionRow(
        session_id="s2",
        title="the second thing",
        profile="hpc",
        mode="manual",
        model="llama-3.3-70b",
    ),
]


def entry(seq: int, kind: str = "user", text: str = "", **kw) -> protocol.Entry:
    return protocol.Entry(kind=kind, text=text or f"row {seq}", seq=seq, **kw)


def thinking(seq: int, *parts: protocol.Part) -> protocol.Entry:
    return protocol.Entry(
        kind="thinking", text="", seq=seq, steps=len(parts), parts=list(parts)
    )


def call(tool: str, target: str = "", result: str = "", done: bool = False):
    return protocol.Part(
        kind="call", text="", tool=tool, target=target, result=result, done=done
    )


@pytest.fixture
async def wire():
    async with connected() as w:
        # Twelve seconds after the core stamped the turn, so the clock on the
        # working row has something to say.
        at_wall(w.ui, EPOCH + 12)
        await w.tell(protocol.Hello(profile="hpc"))
        await w.tell(protocol.SessionRows(rows=list(ROWS)))
        await w.tell(
            protocol.ChatReset(
                session_id="s1", entries=[entry(1, text="hello there", index=0)]
            )
        )
        w.peer.clear()
        yield w


async def working(wire: Wire, activity: str = "running read_file") -> Wire:
    """A turn in flight in the open session, twelve seconds in."""
    await wire.tell(protocol.TurnStarted(session_id="s1"))
    if activity:
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity=activity, started_at=STARTED
            )
        )
    return wire


def spinner_lines(wire: Wire) -> list[str]:
    return [x for x in wire.frame() if any(f in x for f in SPINNER_GLYPHS)]


def spinner(wire: Wire) -> str:
    lines = spinner_lines(wire)
    assert len(lines) == 1, f"expected one working row, got {lines}"
    return lines[0]


# ------------------------------------------------------------- the spinner


class TestTheWorkingIndicator:
    async def test_it_appears_while_waiting(self, wire):
        await working(wire)
        assert spinner_lines(wire)

    async def test_and_goes_when_the_reply_lands(self, wire):
        await working(wire)
        await wire.tell(protocol.TurnFinished(session_id="s1", reply="done"))
        assert not spinner_lines(wire)

    async def test_and_goes_when_the_turn_fails(self, wire):
        await working(wire)
        await wire.tell(protocol.TurnFailed(session_id="s1", error="boom"))
        assert not spinner_lines(wire)

    async def test_it_sits_after_the_last_message(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=entry(2, text="the newest message")
            )
        )
        screen = wire.screen()
        assert screen.index("the newest message") < screen.index("read_file")

    async def test_and_is_the_last_row_of_the_log(self, wire):
        await working(wire)
        lines = wire.ui.chat.flat(wire.inner)
        assert wire.ui.chat.is_tail(lines[-1][0])

    async def test_at_most_one_per_turn(self, wire):
        await working(wire)
        for step in ("running read_file", "LLM processing", "running run_bash"):
            await wire.tell(
                protocol.TurnActivity(
                    session_id="s1", activity=step, started_at=STARTED
                )
            )
        assert len(spinner_lines(wire)) == 1

    async def test_it_says_the_llm_is_processing(self, wire):
        await working(wire, activity="")
        assert "LLM processing" in spinner(wire)

    async def test_and_names_the_tool_in_flight(self, wire):
        await working(wire, activity="running read_file")
        assert "running read_file" in spinner(wire)

    async def test_steps_are_reported_in_order(self, wire):
        await working(wire, activity="running read_file")
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="running run_bash", started_at=LATER
            )
        )
        assert "running run_bash" in spinner(wire)

    async def test_the_step_is_timed(self, wire):
        await working(wire)
        assert " 12s" in spinner(wire)

    async def test_the_frames_advance(self, wire):
        await working(wire)
        first = spinner(wire)
        at_wall(wire.ui, EPOCH + 12.15)
        assert spinner(wire) != first

    async def test_and_it_is_the_same_row_that_changed(self, wire):
        await working(wire)
        at_wall(wire.ui, EPOCH + 12.15)
        assert len(spinner_lines(wire)) == 1

    async def test_a_new_step_does_not_restart_the_clock(self, wire):
        await working(wire, activity="running read_file")
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="running run_bash", started_at=LATER
            )
        )
        # Nine seconds into the turn the second step began, and the number is
        # "how long since I asked", not "how long has this step taken".
        assert " 3s" in spinner(wire)

    async def test_and_the_same_step_again_does_not_either(self, wire):
        await working(wire, activity="running read_file")
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="running read_file", started_at=LATER
            )
        )
        assert " 12s" in spinner(wire)

    async def test_switching_away_and_back_keeps_the_elapsed_time(self, wire):
        await working(wire)
        await wire.press("ctrl-up", "ctrl-up")  # into the sessions row
        wire.ui.open_session("s2")
        wire.ui.open_session("s1")
        assert " 12s" in spinner(wire)

    async def test_a_backend_call_that_is_not_a_turn_still_times_itself(self, wire):
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="forming memories", started_at=STARTED
            )
        )
        assert " 12s" in spinner(wire)
        assert "forming memories" in spinner(wire)

    async def test_and_says_nothing_about_stopping_it(self, wire):
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="forming memories", started_at=STARTED
            )
        )
        assert "interrupt" not in spinner(wire)

    async def test_but_a_turn_offers_both_ways_out(self, wire):
        await working(wire)
        assert "(enter or esc esc to interrupt)" in spinner(wire)

    async def test_it_goes_when_the_call_that_was_not_a_turn_ends(self, wire):
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="forming memories", started_at=STARTED
            )
        )
        await wire.tell(protocol.TurnActivity(session_id="s1", activity=""))
        assert not spinner_lines(wire)

    async def test_typing_ahead_never_buries_it(self, wire):
        await working(wire)
        for seq in range(2, 6):
            await wire.tell(
                protocol.ChatAppend(
                    session_id="s1", entry=entry(seq, "queued", "and another thing")
                )
            )
        lines = wire.ui.chat.flat(wire.inner)
        assert wire.ui.chat.is_tail(lines[-1][0])

    async def test_and_it_is_still_reachable(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end")
        assert wire.ui.chat.is_tail(wire.ui.chat.current(wire.inner))

    async def test_a_spinner_frame_does_not_relayout_the_log(self, wire):
        # The performance requirement carried over from the Textual widget,
        # which cost 44ms of loop lag at 300 messages for getting it wrong.
        await working(wire)
        before = wire.ui.chat.flat(wire.inner)
        at_wall(wire.ui, EPOCH + 12.15)
        wire.frame()
        assert wire.ui.chat.flat(wire.inner) is before

    async def test_a_background_turn_does_not_draw_one_here(self, wire):
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        await wire.tell(
            protocol.TurnActivity(
                session_id="s2", activity="running run_bash", started_at=STARTED
            )
        )
        assert not spinner_lines(wire)

    async def test_but_it_marks_that_sessions_sidebar_row(self, wire):
        await wire.tell(protocol.TurnStarted(session_id="s2"))
        rows = [x for x in wire.frame() if "the second thing" in x]
        assert "⟳" in rows[0]


class TestStoppingItFromTheRow:
    async def test_enter_on_the_working_row_asks_before_stopping_it(self, wire):
        # The aimed half of the gesture confirms; `esc esc` does not, because
        # the doubling is already the confirmation (M5b, "Interrupting").
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end", "enter")
        assert "Interrupt this turn" in wire.screen()
        assert wire.peer.took(protocol.TurnInterrupt) == []

    async def test_and_confirming_interrupts_the_turn(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end", "enter", "y")
        assert wire.peer.last(protocol.TurnInterrupt).session_id == "s1"

    async def test_and_says_so(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end", "enter", "y")
        assert wire.ui.note == "stopped the turn"

    async def test_declining_leaves_the_turn_running(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end", "enter", "n")
        assert wire.peer.took(protocol.TurnInterrupt) == []
        assert spinner_lines(wire), "the spinner is still there"

    async def test_a_call_that_is_not_a_turn_says_why_it_cannot(self, wire):
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="forming memories", started_at=STARTED
            )
        )
        wire.ui.focus = CHAT
        await wire.press("end", "enter")
        assert "nothing to stop" in wire.ui.note

    async def test_and_sends_nothing(self, wire):
        await wire.tell(
            protocol.TurnActivity(
                session_id="s1", activity="forming memories", started_at=STARTED
            )
        )
        wire.ui.focus = CHAT
        await wire.press("end", "enter")
        assert wire.peer.took(protocol.TurnInterrupt) == []

    async def test_the_working_row_is_not_a_message_to_rewind(self, wire):
        await working(wire)
        wire.ui.focus = CHAT
        await wire.press("end", "enter")
        assert wire.ui.overlay is None


class TestWhenTheSpinnerAsksForARepaint:
    async def test_an_idle_ui_asks_for_none(self, wire):
        assert wire.ui.next_wake() is None

    async def test_a_turn_books_the_next_frame(self, wire):
        await working(wire)
        assert 0 < wire.ui.next_wake() <= 0.1

    async def test_and_the_wake_is_gone_when_the_turn_is(self, wire):
        await working(wire)
        await wire.tell(protocol.TurnFinished(session_id="s1"))
        assert wire.ui.next_wake() is None

    async def test_the_sooner_of_the_two_deadlines_wins(self, wire):
        # The armed escape expires in a second; the spinner turns ten times
        # in that second, and the loop has to wake for the nearer one.
        await working(wire)
        wire.ui.clock = lambda: 100.0
        wire.ui._esc_armed_at = 100.0
        assert wire.ui.next_wake() <= 0.1


# ----------------------------------------------------------- the live steps


class TestLiveStepRows:
    async def test_a_call_appears_as_a_row_while_it_runs(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(2, call("read_file", "/scratch/run.log")),
            )
        )
        assert "/scratch/run.log" in wire.screen()

    async def test_a_call_alone_has_no_result_on_it(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=thinking(2, call("run_bash", "lsof /tmp"))
            )
        )
        step = [x for x in wire.frame() if "lsof /tmp" in x][0]
        assert step.rstrip().endswith("…")

    async def test_the_result_fills_that_same_row(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=thinking(2, call("run_bash", "lsof /tmp"))
            )
        )
        rows = len(wire.ui.chat.items)
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1",
                entry=thinking(
                    2, call("run_bash", "lsof /tmp", "two shards open", done=True)
                ),
            )
        )
        # The result lands *on the call's own row* — as its body, one keypress
        # away, so a four-hundred-line tool result cannot come between the
        # steps either side of it.
        assert len(wire.ui.chat.items) == rows
        assert wire.ui.chat.items[-1].folds[0].body == ["two shards open"]

    async def test_and_the_row_stops_saying_it_is_waiting(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=thinking(2, call("run_bash", "lsof /tmp"))
            )
        )
        assert [x for x in wire.frame() if "lsof /tmp" in x][0].rstrip()[-1] == "…"
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1",
                entry=thinking(
                    2, call("run_bash", "lsof /tmp", "two shards open", done=True)
                ),
            )
        )
        assert [x for x in wire.frame() if "lsof /tmp" in x][0].rstrip()[-1] != "…"

    async def test_and_never_adds_a_second_one(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=thinking(2, call("run_bash", "lsof /tmp"))
            )
        )
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1",
                entry=thinking(
                    2, call("run_bash", "lsof /tmp", "two shards open", done=True)
                ),
            )
        )
        assert wire.screen().count("lsof /tmp") == 1

    async def test_a_rebuild_between_the_two_halves_leaves_the_result_standing(
        self, wire
    ):
        # The reset re-bases the numbering and the update lands on the row it
        # named there, which is what `chat.update` addressing by `seq` buys.
        await working(wire)
        await wire.tell(
            protocol.ChatReset(
                session_id="s1",
                entries=[entry(1), thinking(2, call("run_bash", "lsof /tmp"))],
            )
        )
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1",
                entry=thinking(
                    2, call("run_bash", "lsof /tmp", "two shards open", done=True)
                ),
            )
        )
        assert wire.ui.chat.items[-1].folds[0].body == ["two shards open"]

    async def test_a_live_row_opens_without_being_asked(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(2, call("read_file", "/scratch/run.log")),
            )
        )
        assert "2" in wire.ui.chat.expanded

    async def test_and_folds_back_into_one_box_when_the_turn_ends(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(2, call("read_file", "/scratch/run.log", "ok", True)),
            )
        )
        await wire.tell(protocol.TurnFinished(session_id="s1"))
        # The box the UI opened by itself, and only that one — the messages
        # around it were never folded and are not folded now.
        assert "2" not in wire.ui.chat.expanded
        assert "1 steps" in wire.screen() or "1 step" in wire.screen()

    async def test_but_a_box_the_user_opened_stays_open(self, wire):
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(2, call("read_file", "/scratch/run.log", "ok", True)),
            )
        )
        wire.ui.focus = CHAT
        await wire.press("end", "right")
        await working(wire)
        await wire.tell(protocol.TurnFinished(session_id="s1"))
        assert "2" in wire.ui.chat.expanded

    async def test_reasoning_and_tools_share_one_box_per_turn(self, wire):
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(
                    2,
                    protocol.Part(kind="reasoning", text="think think", done=True),
                    call("read_file", "/scratch/run.log", "412 lines", True),
                ),
            )
        )
        assert "2 steps" in wire.screen()
        assert "412 lines" not in wire.screen()  # closed

    async def test_which_opens_into_its_steps(self, wire):
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(
                    2,
                    protocol.Part(kind="reasoning", text="think think", done=True),
                    call("read_file", "/scratch/run.log", "412 lines", True),
                ),
            )
        )
        wire.ui.focus = CHAT
        # Closed first: the newest row shows itself, so the turn that has just
        # landed is already open and → would be aimed at one of its steps.
        await wire.press("end", "left", "right")
        screen = wire.screen()
        assert "think think" in screen and "/scratch/run.log" in screen
        assert "412 lines" not in screen  # the step is a fold of its own

    async def test_and_each_step_opens_into_what_it_returned(self, wire):
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(
                    2, call("read_file", "/scratch/run.log", "412 lines", True)
                ),
            )
        )
        wire.ui.focus = CHAT
        await wire.press("end", "right", "down", "right")
        assert "412 lines" in wire.screen()

    async def test_the_parts_are_individually_navigable(self, wire):
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(
                    2,
                    call("read_file", "/scratch/one.log", "ok", True),
                    call("run_bash", "lsof /tmp", "ok", True),
                ),
            )
        )
        wire.ui.focus = CHAT
        # As above: closed and opened again, so the cursor starts on the
        # turn's own head line rather than wherever the newest row left it.
        await wire.press("end", "left", "right")
        first = wire.ui.chat.row_key(wire.inner)
        await wire.press("down")
        assert wire.ui.chat.row_key(wire.inner) != first

    async def test_and_hold_the_highlight_across_a_revision(self, wire):
        await working(wire)
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1",
                entry=thinking(
                    2,
                    call("read_file", "/scratch/one.log", "ok", True),
                    call("run_bash", "lsof /tmp"),
                ),
            )
        )
        wire.ui.focus = CHAT
        await wire.press("end", "up")
        on = wire.ui.chat.row_key(wire.inner)
        await wire.tell(
            protocol.ChatUpdate(
                session_id="s1",
                entry=thinking(
                    2,
                    call("read_file", "/scratch/one.log", "ok", True),
                    call("run_bash", "lsof /tmp", "two shards open", done=True),
                ),
            )
        )
        assert wire.ui.chat.row_key(wire.inner) == on

    async def test_no_thinking_means_no_box(self, wire):
        await wire.tell(
            protocol.ChatAppend(
                session_id="s1", entry=entry(2, "assistant", "just an answer")
            )
        )
        assert "steps" not in wire.screen()

    async def test_the_box_survives_reopening_the_session(self, wire):
        await wire.tell(
            protocol.ChatReset(
                session_id="s1",
                entries=[
                    entry(1),
                    thinking(2, call("read_file", "/scratch/run.log", "ok", True)),
                ],
            )
        )
        assert "1 steps" in wire.screen()


# --------------------------------------------------------- the context meter


class TestTheContextMeterState:
    """The widget state `render_bar` and `severity` are fed, which is what
    `specs/specs-ui-acceptance.md` asks survive the port."""

    def test_a_session_with_no_reply_yet_says_so(self):
        assert Context(window=32_768).bar() == (
            "context: window 32,768 · no reply yet"
        )

    def test_and_says_it_even_without_a_window(self):
        assert "window unknown" in Context().bar()

    def test_a_measured_value_replaces_the_placeholder(self):
        context = Context(window=32_768)
        context.measure(12_345)
        assert "12,345 / 32,768 (38%)" in context.bar()

    def test_and_is_not_marked_as_a_guess(self):
        context = Context(window=32_768)
        context.measure(12_345)
        assert "~" not in context.bar()

    def test_an_estimate_is(self):
        context = Context()
        context.estimate(12_345, 32_768)
        assert "~12,345 / 32,768" in context.bar()

    def test_a_measured_value_supersedes_an_estimate(self):
        context = Context()
        context.measure(12_345, 32_768)
        context.estimate(30_000, 32_768)
        assert "12,345 / 32,768" in context.bar()

    def test_reset_clears_back_to_no_reply(self):
        context = Context()
        context.measure(12_345, 32_768)
        context.reset()
        assert "no reply yet" in context.bar()

    def test_but_keeps_the_window_it_learned(self):
        context = Context()
        context.measure(12_345, 32_768)
        context.reset()
        assert "32,768" in context.bar()

    def test_a_window_arriving_after_the_usage_is_applied(self):
        context = Context()
        context.measure(12_345)
        context.measure(12_345, 32_768)
        assert "12,345 / 32,768" in context.bar()

    def test_the_bar_fills_with_the_fraction(self):
        context = Context()
        context.measure(16_384, 32_768)
        assert "[" + "█" * 14 + "─" * 14 + "]" in context.bar()

    def test_seventy_percent_is_a_warning(self):
        context = Context()
        context.measure(23_000, 32_768)
        assert context.severity == "warn"

    def test_ninety_is_a_danger(self):
        context = Context()
        context.measure(30_000, 32_768)
        assert context.severity == "danger"

    def test_and_a_quiet_window_is_neither(self):
        context = Context()
        context.measure(1_000, 32_768)
        assert context.severity == "ok"

    def test_a_session_with_nothing_measured_is_never_alarming(self):
        assert Context(window=1_000).severity == "ok"

    def test_speed_is_appended_to_the_measured_line(self):
        context = Context(speed=4.2)
        context.measure(12_345, 32_768)
        assert context.bar().endswith("· 4.2 tok/s")

    def test_a_fast_turn_needs_no_decimal(self):
        # One decimal only where it carries information; `render_bar`'s own
        # threshold, ported with it.
        context = Context(speed=142.4)
        context.measure(12_345, 32_768)
        assert context.bar().endswith("· 142 tok/s")

    def test_speed_waits_for_a_measured_fill(self):
        assert "tok/s" not in Context(window=32_768, speed=4.2).bar()

    def test_none_clears_it(self):
        context = Context(speed=4.2)
        context.measure(12_345, 32_768)
        context.speed = None
        assert "tok/s" not in context.bar()

    def test_and_so_does_a_reset(self):
        context = Context(speed=4.2)
        context.measure(12_345, 32_768)
        context.reset()
        assert context.speed is None

    def test_the_thinking_level_is_shown_before_the_first_reply(self):
        assert Context(window=32_768, effort="medium").bar().endswith("think medium")

    def test_and_after_the_fill_and_the_speed(self):
        context = Context(speed=4.2, effort="medium")
        context.measure(12_345, 32_768)
        assert context.bar().endswith("· 4.2 tok/s · think medium")

    def test_off_is_shown_rather_than_hidden(self):
        assert "think off" in Context(window=32_768, effort="off").bar()

    def test_and_none_clears_it(self):
        assert "think" not in Context(window=32_768, effort=None).bar()

    def test_the_picture_is_the_first_thing_to_go(self):
        context = Context()
        context.measure(12_345, 32_768)
        assert context.bar(cells=0) == "context 12,345 / 32,768 (38%)"


class TestTheContextMeterOnScreen:
    async def test_an_estimate_reaches_the_row(self, wire):
        await wire.tell(
            protocol.ContextEstimate(session_id="s1", used=12_345, window=32_768)
        )
        assert "~12,345 / 32,768 (38%)" in wire.screen()

    async def test_a_measurement_is_not_marked_as_a_guess(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        assert "] 12,345 / 32,768 (38%)" in wire.screen()

    async def test_the_meter_always_reflects_the_session_on_screen(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s2", prompt_tokens=30_000, max_model_len=32_768
            )
        )
        assert "30,000" not in wire.screen()

    async def test_and_shows_that_sessions_own_number_on_the_way_back(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s2", prompt_tokens=30_000, max_model_len=32_768
            )
        )
        wire.ui.open_session("s2")
        assert "30,000 / 32,768" in wire.screen()

    async def test_a_reset_clears_the_measurement(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        await wire.tell(protocol.ChatReset(session_id="s1", entries=[entry(1)]))
        assert "no reply yet" in wire.screen()

    async def test_a_full_window_is_drawn_in_red(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=31_000, max_model_len=32_768
            )
        )
        line = [x for x in wire.ui.render(120, 40) if "31,000" in plain(x)][0]
        assert "\x1b[38;5;167m" in line


# ---------------------------------------------------------- the mode bar


class TestTheModeBar:
    async def test_it_names_the_mode_and_what_it_means(self, wire):
        assert "mode: auto — works until the task is done" in wire.screen()

    async def test_full_auto_reads_as_two_words(self, wire):
        await wire.tell(
            protocol.SessionRows(
                rows=[ROWS[0].model_copy(update={"mode": "full-auto"}), ROWS[1]]
            )
        )
        assert "mode: full auto — asks for nothing" in wire.screen()

    async def test_the_mode_colours_the_line(self, wire):
        line = [x for x in wire.ui.render(120, 40) if "mode: auto" in plain(x)][0]
        assert "\x1b[38;5;71m" in line  # theme.ok, as auto

    async def test_it_follows_the_session_on_screen(self, wire):
        wire.ui.open_session("s2")
        assert "mode: manual" in wire.screen()

    async def test_and_is_hidden_with_no_session_open(self):
        assert "mode:" not in "\n".join(
            plain(x) for x in RowUI().render(120, 40)
        )


class TestShiftTabCycles:
    """The dial is the message box's key and no other row's.

    Deciding the agent may act unasked is a thought you have *while typing*,
    not one you leave the box to act on — which is why the Textual app bound
    this `priority=True` so it fired from the entry. Everywhere else shift+tab
    is the way back up the ring, as tab is the way down.
    """

    async def test_it_moves_to_the_next_mode(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab")
        assert "mode: full auto" in wire.screen()

    async def test_and_persists_the_choice(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab")
        sent = wire.peer.last(protocol.ModeSet)
        assert (sent.session_id, sent.mode) == ("s1", "full-auto")

    async def test_it_is_a_cycle(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab", "shift-tab", "shift-tab")
        assert [x.mode for x in wire.peer.took(protocol.ModeSet)] == [
            "full-auto",
            "manual",
            "auto",
        ]

    async def test_the_core_still_has_the_last_word(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab")
        await wire.tell(protocol.SessionRows(rows=list(ROWS)))
        assert "mode: auto" in wire.screen()

    @pytest.mark.parametrize("row", [SESSIONS, CHAT, WATCHERS])
    async def test_the_other_columns_leave_the_mode_alone(self, wire, row):
        wire.ui.focus = row
        await wire.press("shift-tab")
        assert wire.peer.took(protocol.ModeSet) == []

    @pytest.mark.parametrize("row", [SESSIONS, CHAT, WATCHERS])
    async def test_and_move_between_rows_instead(self, wire, row):
        wire.ui.focus = row
        await wire.press("shift-tab")
        assert wire.ui.focus != row

    async def test_the_chat_row_moves_the_way_the_other_lists_do(self, wire):
        # It used to be the second place the dial could be turned, which is
        # what made shift+tab mean one thing in three rows and another in the
        # fourth. Now it is the ring's step back, everywhere but the box.
        wire.ui.focus = CHAT
        await wire.press("shift-tab")
        assert wire.ui.focus == SESSIONS

    async def test_the_mode_a_session_is_in_belongs_to_that_session(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab")
        wire.ui.open_session("s2")
        assert "mode: manual" in wire.screen()

    async def test_and_does_not_take_the_focus_out_of_the_box(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab")
        assert wire.ui.focus == INPUT

    async def test_ctrl_up_is_how_you_leave_the_box(self, wire):
        wire.ui.focus = INPUT
        await wire.press("ctrl-up")
        assert wire.ui.focus == CHAT
        assert wire.peer.took(protocol.ModeSet) == []

    async def test_typing_shift_tab_does_not_write_into_the_draft(self, wire):
        wire.ui.focus = INPUT
        await wire.press("shift-tab")
        assert wire.ui.input.text() == ""


# ----------------------------------------------------------- the model line


class TestTheModelLine:
    async def test_it_shows_the_sessions_model(self, wire):
        assert "qwen3-27b-fp8" in wire.screen()

    async def test_it_updates_on_a_session_switch(self, wire):
        wire.ui.open_session("s2")
        screen = wire.screen()
        assert "llama-3.3-70b" in screen and "qwen3-27b-fp8" not in screen

    async def test_it_is_hidden_with_no_session_open(self):
        assert "qwen" not in "\n".join(plain(x) for x in RowUI().render(120, 40))

    async def test_a_session_on_the_bootstrap_client_names_no_model(self, wire):
        await wire.tell(
            protocol.SessionRows(
                rows=[ROWS[0].model_copy(update={"model": ""}), ROWS[1]]
            )
        )
        assert "qwen3-27b-fp8" not in wire.screen()

    async def test_the_top_bar_keeps_the_profile_and_drops_the_model(self, wire):
        header = wire.frame()[0]
        assert "hpc" in header and "qwen3-27b-fp8" not in header


# -------------------------------------------------- the layout under pressure


class TestTheLayout:
    @pytest.mark.parametrize("width", [80, 100, 137])
    async def test_every_row_is_exactly_the_width(self, wire, width):
        await working(wire)
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        drawn = wire.ui.render(width, 24)
        assert len(drawn) == 24
        assert widths(drawn) == {width}

    async def test_the_mode_and_the_meter_share_one_row(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        rows = [x for x in wire.frame() if "mode:" in x or "context" in x]
        assert len(rows) == 1

    async def test_at_eighty_columns_the_mode_loses_its_sentence(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        row = [x for x in wire.ui.render(80, 24) if "mode:" in plain(x)][0]
        assert "works until" not in plain(row)

    async def test_but_never_which_mode_it_is_in(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        assert "mode: auto" in "\n".join(plain(x) for x in wire.ui.render(80, 24))

    async def test_nor_how_full_the_window_is(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        assert "(38%)" in "\n".join(plain(x) for x in wire.ui.render(80, 24))

    async def test_a_wide_terminal_gets_the_whole_sentence(self, wire):
        await wire.tell(
            protocol.TurnUsage(
                session_id="s1", prompt_tokens=12_345, max_model_len=32_768
            )
        )
        screen = "\n".join(plain(x) for x in wire.ui.render(137, 24))
        assert "shift+tab to switch" in screen
        assert "[" + "█" * 11 in screen  # the full 28-cell bar

    async def test_the_chat_still_gets_the_slack(self, wire):
        # Four rows plus a status line on a small terminal, and the chat is
        # still the one that keeps what is left over.
        heights = wire.ui._heights(24, 80)
        assert heights[1] >= wire.ui.MIN_CHAT

    async def test_the_status_row_costs_nothing_with_no_session(self):
        assert RowUI()._status_h() == 0

    @pytest.mark.parametrize("height", [8, 10, 14])
    async def test_a_short_terminal_still_draws_exactly(self, wire, height):
        await working(wire)
        drawn = wire.ui.render(80, height)
        assert len(drawn) == height
        assert widths(drawn) == {80}
