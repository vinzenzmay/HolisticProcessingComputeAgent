"""The thinking box in the chat window, and plain-text session logging."""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import ListView, Static

from hpca.agent.tools import Tool, ToolRegistry
from hpca.config import Settings
from hpca.llm import ChatResponse
from hpca.transcript import RESULT_RULE
from hpca.tui.app import (
    ChatInput,
    DecisionBar,
    HpcaApp,
    StepBox,
    ThinkingBox,
    WorkingIndicator,
)


def is_title_request(json_schema):
    """The app names a session by asking the model (§3 sessions column); that
    call is not one of the queued decisions."""
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})

class FakeLLM:
    """Answers with the queued decisions, thinking out loud alongside them."""

    def __init__(self, outputs, reasoning=None):
        self._outputs = list(outputs)
        self._reasoning = list(reasoning or [])

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(
            content=self._outputs.pop(0),
            reasoning=self._reasoning.pop(0) if self._reasoning else None,
        )

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


class AskParams(BaseModel):
    question: str = Field(description="What to ask the docs")


async def ask_handler(args, ctx):
    """A sub-agent: a tool that runs its own model call (§4.2)."""
    response = await ctx.llm.chat([{"role": "user", "content": args.question}])
    return response.content


def subagent_tools():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="ask_docs",
            description="Ask the docs",
            params=AskParams,
            handler=ask_handler,
        )
    )
    return registry


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def logs_dir(hpca_home, tmp_path):
    """Point session logs at a temp dir instead of ./hpca-logs."""
    directory = tmp_path / "logs"
    settings = Settings()
    settings.logging.dir = str(directory)
    settings.save()
    return directory


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def log_text(directory):
    files = list(directory.glob("*.log"))
    assert len(files) == 1, files
    return files[0].read_text()


class TestThinkingBox:
    async def test_reasoning_and_steps_share_one_collapsed_box(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("ask_docs", question="what is a BAM?"),
                    "binary alignment map",  # answered to the tool's own call
                    respond_json("done"),
                ],
                # the sub-agent's call does not think; only decisions do
                reasoning=["I should ask the docs.", "", "Now I can answer."],
            ),
            tools=subagent_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "explain BAMs")
            boxes = list(app.query(ThinkingBox))
            assert len(boxes) == 1, "one box per turn, not one per step"
            box = boxes[0]
            assert box.collapsed
            rendered = str(box.content)
            assert "enter to expand" in rendered
            assert "2 steps" not in rendered  # one tool step this turn
            assert "1 step" in rendered
            # collapsed: the working is summarised, not spelled out
            assert "I should ask the docs" not in rendered

    async def test_enter_expands_into_collapsible_steps(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([respond_json("42")], reasoning=["Simple arithmetic."]),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "how many?")
            chat_list = app.query_one("#chat-list", ListView)
            box = app.query_one(ThinkingBox)
            chat_list.focus()
            chat_list.index = 1  # the thinking box between question and answer
            await pilot.press("enter")
            await pilot.pause()
            # Expanding reveals the parts as their own collapsed rows, not as
            # inline text on the box itself.
            assert not box.collapsed
            assert "enter to collapse" in str(box.content)
            assert "Simple arithmetic." not in str(box.content)
            step = app.query_one(StepBox)
            assert not step.expanded  # revealed collapsed
            assert "reasoning" in str(step.content)
            assert "Simple arithmetic." not in str(step.content)

            # The step is the next row down; enter opens it individually.
            await pilot.press("down")
            assert chat_list.index == 2
            await pilot.press("enter")
            await pilot.pause()
            assert step.expanded
            assert "Simple arithmetic." in str(step.content)

            # Collapsing the box removes its step rows again.
            chat_list.index = 1
            await pilot.press("enter")
            await pilot.pause()
            assert box.collapsed
            assert list(app.query(StepBox)) == []

    async def test_parts_are_individually_navigable_and_hold_the_highlight(
        self, hpca_home
    ):
        # A turn with reasoning, one tool exchange and more reasoning: three
        # parts, because the call and the result it returned are one row.
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("ask_docs", question="what is a BAM?"),
                    "binary alignment map",
                    respond_json("done"),
                ],
                reasoning=["First thought.", "", "Second thought."],
            ),
            tools=subagent_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "explain BAMs")
            chat_list = app.query_one("#chat-list", ListView)
            chat_list.focus()
            chat_list.index = 1  # the thinking box
            await pilot.press("enter")  # expand
            await pilot.pause()

            steps = list(app.query(StepBox))
            assert [s._step.label() for s in steps] == [
                "reasoning", "ask_docs", "reasoning"
            ]
            assert all(not s.expanded for s in steps)  # each revealed collapsed
            assert chat_list.index == 1  # highlight kept on the box

            # Open the tool exchange individually; the others stay shut.
            await pilot.press("down")  # first reasoning
            await pilot.press("down")  # the ask_docs exchange
            assert chat_list.index == 3
            await pilot.press("enter")
            await pilot.pause()
            assert steps[1].expanded
            assert not any(s.expanded for s in (steps[0], steps[2]))
            # both halves in the one row, the question above the answer
            body = str(steps[1].content)
            assert body.index("what is a BAM?") < body.index("binary alignment map")
            assert RESULT_RULE in body
            assert chat_list.index == 3  # still on the same step

            # Collapse the box: every step row goes away, highlight back on it.
            chat_list.index = 1
            await pilot.press("enter")
            await pilot.pause()
            assert list(app.query(StepBox)) == []
            assert chat_list.index == 1

    async def test_no_thinking_no_box(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("hi")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            assert list(app.query(ThinkingBox)) == []

    async def test_box_survives_reopening_the_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("42")], reasoning=["Thought hard."]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "how many?")
            session = app.active_session
            await app.start_new_session()
            await pilot.pause()
            assert list(app.query(ThinkingBox)) == []
            await app.open_session(session)
            await pilot.pause()
            chat_list = app.query_one("#chat-list", ListView)
            chat_list.focus()
            chat_list.index = 1  # the thinking box
            await pilot.press("enter")  # expand it
            await pilot.press("down")  # onto the reasoning step
            await pilot.press("enter")  # open the step
            await pilot.pause()
            assert "Thought hard." in str(app.query_one(StepBox).content)


async def expand_box(app, pilot, index=1):
    """Open the thinking box at ``index`` and return its revealed step rows."""
    chat_list = app.query_one("#chat-list", ListView)
    chat_list.focus()
    chat_list.index = index
    await pilot.press("enter")
    await pilot.pause()
    return list(app.query(StepBox))


class TestToolCallBoxes:
    """The script and the command the agent ran stay readable in the chat
    after the approval prompt that showed them is gone — and in auto mode,
    where no prompt ever showed them."""

    async def test_the_approved_script_stays_readable_in_the_chat(self, hpca_home):
        # Manual mode: the prompt shows the script, the user runs it, and the
        # prompt disappears. The script must not disappear with it.
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo cohort-listing"]),
                    respond_json("four BAMs"),
                ]
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "what is in the cohort?")
            await pilot.press("y")  # run script
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.query_one("#decision-bar", DecisionBar).display
            steps = await expand_box(app, pilot)
            # one row for the exchange, not one for the script and one for
            # what it printed
            assert [s._step.label() for s in steps] == ["run_bash"]
            # revealed collapsed: the script is a box you open, not a wall
            assert all(not s.expanded for s in steps)
            assert "echo cohort-listing" not in str(steps[0].content)

            await pilot.press("down")  # onto the exchange row
            await pilot.press("enter")  # open it
            await pilot.pause()
            assert steps[0].expanded
            assert "echo cohort-listing" in str(steps[0].content)
            assert RESULT_RULE in str(steps[0].content)  # and what it printed

    async def test_a_skipped_script_is_still_there_to_read(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo scratch-cleanup"]),
                    respond_json("understood"),
                ]
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "clean up")
            await pilot.press("n")  # skip script
            await pilot.pause()
            await pilot.press("enter")  # without a reason
            await app.workers.wait_for_complete()
            await pilot.pause()
            steps = await expand_box(app, pilot)
            assert len(steps) == 1  # the refusal is this row's result half
            # and it reads as a failure on the collapsed row, though the tool
            # never ran to raise anything
            assert steps[0]._step.label() == "run_bash (error)"
            assert "echo scratch-cleanup" in steps[0]._step.text
            assert "SKIPPED" in steps[0]._step.result

    async def test_auto_mode_shows_what_ran_unasked(self, hpca_home):
        # Nothing gates here, so this box is the only account of what ran.
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo auto-run"]),
                    respond_json("done looking"),
                ]
            )
        )
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "check something")
            assert not app.query_one("#decision-bar", DecisionBar).display
            steps = await expand_box(app, pilot)
            assert "echo auto-run" in steps[0]._step.text

    async def test_the_script_survives_reopening_the_session(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo still-here"]),
                    respond_json("nothing queued"),
                ]
            )
        )
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "anything running?")
            session = app.active_session
            await app.start_new_session()
            await pilot.pause()
            await app.open_session(session)
            await pilot.pause()
            steps = await expand_box(app, pilot)
            await pilot.press("down")  # onto the call row
            await pilot.press("enter")  # open it
            await pilot.pause()
            assert "echo still-here" in str(steps[0].content)

    async def test_the_call_is_written_to_the_session_log(self, logs_dir):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo job-42"]),
                    respond_json("it failed"),
                ]
            )
        )
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "why did job 42 fail?")
            text = log_text(logs_dir)
            # One heading for the exchange, with the script and what it
            # printed under it — the log writes the same reading of the turn
            # the chat shows, and the tool is named once in both.
            assert text.count("— run_bash —") == 1
            assert "echo job-42" in text
            assert RESULT_RULE in text


class TestLiveSteps:
    """A turn shows its work while it works. The spinner alone says only that
    something is happening, not what — and a tool can run for minutes."""

    async def test_the_call_is_on_screen_while_the_tool_runs(self, hpca_home):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(args, ctx):
            entered.set()
            await release.wait()
            return "eventually"

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="scan_cohort",
                description="Scans",
                params=AskParams,
                handler=slow,
            )
        )
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("scan_cohort", question="how many BAMs?"),
                    respond_json("four"),
                ]
            ),
            tools=registry,
        )
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "count the BAMs"
            await pilot.press("enter")
            await asyncio.wait_for(entered.wait(), timeout=5)
            await pilot.pause()

            # Mid-run: the call is already a row, and it says what is running
            # and that the answer is still out.
            steps = list(app.query(StepBox))
            assert [s._step.label() for s in steps] == ["scan_cohort …"]
            assert "how many BAMs?" in steps[0]._step.text
            assert not steps[0]._step.done
            # and the spinner is still below it, where the turn continues
            assert app.query(WorkingIndicator)

            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # Once the turn is over the rows fold into the turn's own box.
            assert not app.query(WorkingIndicator)
            assert [b._entry.summary() for b in app.query(ThinkingBox)] == ["1 step"]
            assert list(app.query(StepBox)) == []

    async def test_a_live_row_opens_without_a_box_above_it(self, hpca_home):
        # The rows arrive before the thinking box that will own them exists;
        # opening one mid-turn must not go looking for a parent that is not
        # there yet.
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(args, ctx):
            entered.set()
            await release.wait()
            return "eventually"

        registry = ToolRegistry()
        registry.register(
            Tool(name="scan", description="Scans", params=AskParams, handler=slow)
        )
        app = HpcaApp(
            llm=FakeLLM([tool_json("scan", question="what?"), respond_json("this")]),
            tools=registry,
        )
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "look"
            await pilot.press("enter")
            await asyncio.wait_for(entered.wait(), timeout=5)
            await pilot.pause()

            chat_list = app.query_one("#chat-list", ListView)
            chat_list.focus()
            chat_list.index = 1  # the live call row
            await pilot.press("enter")
            await pilot.pause()
            step = app.query_one(StepBox)
            assert step.expanded
            assert "what?" in str(step.content)

            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()


# The two halves as the graph announces them (its ``on_step``), so the tests
# below can drive the chat's half of the merge without a tool that blocks.
LIVE_CALL = {
    "kind": "call",
    "tool": "run_bash",
    "arguments": {"content_lines": ["echo forty-two"]},
    "script": "echo forty-two",
}
LIVE_RESULT = {"kind": "step", "text": "[tool result] run_bash: ran (exit 0).\n42"}


class TestAResultFillsInTheRowItsCallOpened:
    """The call and the result are announced separately and can be minutes
    apart, and still have to end up as one row: the call goes up the moment it
    is made, and the result is written into that same row when it lands. A
    second row would put the tool's name on screen twice and move everything
    under it just as the user started reading."""

    async def test_a_call_alone_is_one_row_with_no_result_on_it(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("done")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.report_step(app.active_session.session_id, LIVE_CALL)
            await pilot.pause()
            rows = list(app.query(StepBox))
            assert len(rows) == 1
            assert rows[0].awaiting_result
            assert rows[0]._step.label() == "run_bash …"  # the answer is still out
            rows[0].toggle()
            # only the call: an empty "result" heading would read as a tool
            # that answered with nothing
            assert "echo forty-two" in str(rows[0].content)
            assert RESULT_RULE not in str(rows[0].content)

    async def test_the_result_fills_that_row_rather_than_adding_one(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("done")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_id = app.active_session.session_id
            app.report_step(session_id, LIVE_CALL)
            await pilot.pause()
            app.report_step(session_id, LIVE_RESULT)
            await pilot.pause()
            rows = list(app.query(StepBox))
            assert len(rows) == 1, "still the one row, now with both halves"
            assert not rows[0].awaiting_result
            assert rows[0]._step.label() == "run_bash"  # no longer running
            rows[0].toggle()
            body = str(rows[0].content)
            assert body.index("echo forty-two") < body.index("ran (exit 0).")
            assert RESULT_RULE in body

    async def test_a_result_with_no_call_of_its_own_still_gets_a_row(self, hpca_home):
        # Nothing to land on — the announcement is news either way, and a
        # dropped result would leave the turn looking like it did nothing.
        app = HpcaApp(llm=FakeLLM([respond_json("done")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.report_step(app.active_session.session_id, LIVE_RESULT)
            await pilot.pause()
            rows = list(app.query(StepBox))
            assert len(rows) == 1
            assert rows[0]._step.label() == "run_bash"
            assert "ran (exit 0)." in rows[0]._step.text

    async def test_a_rebuild_between_the_halves_leaves_the_result_standing(
        self, hpca_home
    ):
        # Switching session mid-turn redraws the chat and takes the live rows
        # with it. The result then belongs to no row on screen, and must not be
        # written into the widget that is no longer there.
        app = HpcaApp(llm=FakeLLM([respond_json("done")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            session_id = app.active_session.session_id
            app.report_step(session_id, LIVE_CALL)
            await pilot.pause()
            await app._set_chat_messages([])
            await pilot.pause()
            assert app._live_call is None
            app.report_step(session_id, LIVE_RESULT)
            await pilot.pause()
            rows = list(app.query(StepBox))
            assert len(rows) == 1  # the result on its own, the call gone
            assert not rows[0].awaiting_result
            assert "ran (exit 0)." in rows[0]._step.text


class TestEntryStyling:
    async def test_user_and_agent_boxes_differ_in_colour(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("hello back")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello agent")
            statics = [
                s
                for s in app.query("#chat-list Static")
                if set(s.classes) & {"chat-user", "chat-assistant"}
            ]
            assert len(statics) == 2
            user, agent = statics
            assert user.border_title == "you"
            assert agent.border_title == "agent"
            # a one-cell box around each entry, in a colour per speaker
            assert user.styles.border_top[0] == "round"
            assert agent.styles.border_top[0] == "round"
            assert user.styles.border_top[1] != agent.styles.border_top[1]


class TestSessionLogging:
    async def test_turn_is_logged_with_thinking_and_answer(self, logs_dir):
        app = HpcaApp(
            llm=FakeLLM([respond_json("Four BAMs match.")], reasoning=["Count them."])
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "which BAMs?")
            text = log_text(logs_dir)
            assert "session opened" in text
            assert "] user\nwhich BAMs?" in text
            assert "thinking" in text
            assert "Count them." in text
            assert "] agent\nFour BAMs match." in text
            # ordered as it happened
            assert text.index("which BAMs?") < text.index("Count them.")
            assert text.index("Count them.") < text.index("Four BAMs match.")

    async def test_subagent_query_and_reply_are_logged_under_the_tool(self, logs_dir):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("ask_docs", question="what is a BAM?"),
                    "binary alignment map",
                    respond_json("ok"),
                ]
            ),
            tools=subagent_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "explain BAMs")
            text = log_text(logs_dir)
            assert "subagent:ask_docs query" in text
            assert "user: what is a BAM?" in text
            assert "subagent:ask_docs reply" in text

    async def test_reopening_a_session_does_not_relog_it(self, logs_dir):
        from hpca.logs import log_path

        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first question")
            session = app.active_session
            await app.start_new_session()  # a different session, a different file
            await pilot.pause()
            await app.open_session(session)
            await pilot.pause()
            await submit_chat(app, pilot, "second question")
            text = log_path(logs_dir, session).read_text()
            # count logged entries, not mentions: sub-agent queries quote the
            # transcript, and quoting it is not re-logging it
            assert text.count("] user\nfirst question") == 1
            assert text.count("] user\nsecond question") == 1

    async def test_each_session_gets_its_own_file(self, logs_dir):
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            await app.start_new_session()
            await pilot.pause()
            await submit_chat(app, pilot, "second")
            assert len(list(logs_dir.glob("*.log"))) == 2

    async def test_errors_are_logged(self, logs_dir):
        class BrokenLLM:
            async def chat(self, *a, **k):
                from hpca.llm import LLMError

                raise LLMError("backend unreachable")

            async def supports_constrained_decoding(self):
                return True

        app = HpcaApp(llm=BrokenLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hi")
            assert "backend unreachable" in log_text(logs_dir)

    async def test_logging_off_writes_nothing(self, hpca_home, tmp_path):
        directory = tmp_path / "logs"
        settings = Settings()
        settings.logging.dir = str(directory)
        settings.logging.enabled = False
        settings.save()
        app = HpcaApp(llm=FakeLLM([respond_json("hi")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            assert app._log is None
            assert not directory.exists()

    async def test_switching_logging_off_takes_effect_at_once(self, logs_dir):
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "logged question")
            app.settings.logging.enabled = False
            app._refresh_session_log()
            await submit_chat(app, pilot, "private question")
            text = log_text(logs_dir)
            assert "logged question" in text
            assert "private question" not in text
