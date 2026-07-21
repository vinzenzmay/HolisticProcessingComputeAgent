"""The thinking box in the chat window, and plain-text session logging."""

import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import ListView, Static

from hpca.agent.tools import Tool, ToolRegistry
from hpca.config import Settings
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp, StepBox, ThinkingBox


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
        # A turn with reasoning, a tool step, then more reasoning: three parts.
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

            # Open the middle step (the tool) individually; the others stay shut.
            await pilot.press("down")  # first reasoning
            await pilot.press("down")  # the ask_docs step
            assert chat_list.index == 3
            await pilot.press("enter")
            await pilot.pause()
            assert steps[1].expanded and not steps[0].expanded and not steps[2].expanded
            assert "binary alignment map" in str(steps[1].content)
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
