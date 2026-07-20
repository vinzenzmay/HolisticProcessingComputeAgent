"""The working spinner: what the chat shows while a reply is on its way."""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp, WorkingIndicator


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class SlowLLM:
    """Answers only once released, so the wait can be inspected."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.released = asyncio.Event()

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=json.dumps({"title": "a test session"}))
        await self.released.wait()
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


class ListParams(BaseModel):
    where: str = Field(description="Which directory")


async def list_handler(args, ctx):
    return f"3 entries in {args.where}"


def tools():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="list_dir",
            description="List a directory",
            params=ListParams,
            handler=list_handler,
        )
    )
    return registry


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def send(app, pilot, text="which BAMs?"):
    await app.start_new_session()
    await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


async def wait_for_spinner(app, pilot):
    for _ in range(50):
        await pilot.pause()
        spinners = list(app.query(WorkingIndicator))
        if spinners:
            return spinners[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no spinner appeared")


class TestSpinnerLifecycle:
    async def test_it_appears_while_waiting_and_goes_when_the_reply_lands(
        self, hpca_home
    ):
        llm = SlowLLM([respond_json("Four BAMs.")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            spinner = await wait_for_spinner(app, pilot)
            assert spinner.is_attached

            llm.released.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not list(app.query(WorkingIndicator))
            assert any("Four BAMs." in t for t in app.chat_log_texts())

    async def test_it_sits_after_the_last_message(self, hpca_home):
        llm = SlowLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot, "which BAMs?")
            await wait_for_spinner(app, pilot)
            from textual.widgets import ListView

            rows = app.query_one("#chat-list", ListView).children
            assert list(rows[-1].query(WorkingIndicator)), "spinner must come last"
            assert "which BAMs?" in str(rows[-2].query_one("Static").content)
            llm.released.set()
            await app.workers.wait_for_complete()

    async def test_it_goes_when_the_turn_fails(self, hpca_home):
        class BrokenLLM:
            async def chat(self, *a, **k):
                from hpca.llm import LLMError

                raise LLMError("backend unreachable")

            async def supports_constrained_decoding(self):
                return True

        app = HpcaApp(llm=BrokenLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not list(app.query(WorkingIndicator))
            assert any("backend unreachable" in t for t in app.chat_log_texts())

    async def test_one_spinner_per_turn_at_most(self, hpca_home):
        llm = SlowLLM([respond_json("a")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            await wait_for_spinner(app, pilot)
            app.show_working()
            app.show_working()
            await pilot.pause()
            assert len(list(app.query(WorkingIndicator))) == 1
            llm.released.set()
            await app.workers.wait_for_complete()


class TestSpinnerText:
    async def test_it_says_the_llm_is_processing(self, hpca_home):
        llm = SlowLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            spinner = await wait_for_spinner(app, pilot)
            for _ in range(20):  # the graph reports as it starts the call
                await pilot.pause()
                if spinner.activity == "LLM processing":
                    break
                await asyncio.sleep(0.01)
            assert spinner.activity == "LLM processing"
            assert "LLM processing…" in str(spinner.content)
            llm.released.set()
            await app.workers.wait_for_complete()

    async def test_it_names_the_tool_in_flight(self, hpca_home):
        release = asyncio.Event()
        started = asyncio.Event()

        async def slow_list(args, ctx):
            started.set()
            await release.wait()
            return f"3 entries in {args.where}"

        registry = tools()
        registry.get("list_dir").handler = slow_list
        llm = SlowLLM([tool_json("list_dir", where="cohort"), respond_json("3.")])
        llm.released.set()  # the wait under test is the tool's, not the model's
        app = HpcaApp(llm=llm, tools=registry)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            for _ in range(100):
                await pilot.pause()
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            spinner = app.query_one(WorkingIndicator)
            assert spinner.activity == "running list_dir"
            assert "running list_dir…" in str(spinner.content)
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not list(app.query(WorkingIndicator))

    async def test_the_steps_are_reported_in_order(self, hpca_home):
        llm = SlowLLM([tool_json("list_dir", where="cohort"), respond_json("3.")])
        llm.released.set()
        app = HpcaApp(llm=llm, tools=tools())
        async with app.run_test(size=(120, 40)) as pilot:
            seen = []
            app.report_activity = seen.append
            await send(app, pilot)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert seen[:3] == ["LLM processing", "running list_dir", "LLM processing"]

    async def test_frames_advance_and_the_step_is_timed(self, hpca_home):
        llm = SlowLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            spinner = await wait_for_spinner(app, pilot)
            first = str(spinner.content)
            for _ in range(30):  # the frame is on a timer, not on our pauses
                await pilot.pause()
                if str(spinner.content) != first:
                    break
                await asyncio.sleep(0.02)
            assert str(spinner.content) != first
            assert str(spinner.content)[0] in WorkingIndicator.FRAMES
            llm.released.set()
            await app.workers.wait_for_complete()

    async def test_a_new_step_restarts_the_clock(self, hpca_home):
        spinner = WorkingIndicator("LLM processing")
        spinner._started -= 30  # as if it had been thinking for half a minute
        assert spinner.elapsed >= 30
        spinner.set_activity("running list_dir")
        assert spinner.elapsed == 0  # the tool step times itself
        assert spinner.activity == "running list_dir"

    async def test_repeating_the_same_step_does_not_restart_the_clock(self, hpca_home):
        spinner = WorkingIndicator("LLM processing")
        spinner._started -= 30
        spinner.set_activity("LLM processing")
        assert spinner.elapsed >= 30
