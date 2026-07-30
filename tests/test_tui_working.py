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
            # report_activity is now (session_id, activity); record the activity
            app.report_activity = lambda session_id, activity: seen.append(activity)
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

    async def test_a_new_step_does_not_restart_the_clock(self, hpca_home):
        # The number answers "how long since I asked?", so it keeps running
        # across the steps of one turn.
        spinner = WorkingIndicator("LLM processing")
        spinner._started -= 30  # as if it had been thinking for half a minute
        assert spinner.elapsed >= 30
        spinner.set_activity("running list_dir")
        assert spinner.elapsed >= 30
        assert spinner.activity == "running list_dir"

    async def test_repeating_the_same_step_does_not_restart_the_clock(self, hpca_home):
        spinner = WorkingIndicator("LLM processing")
        spinner._started -= 30
        spinner.set_activity("LLM processing")
        assert spinner.elapsed >= 30


class TestClockSurvivesLeavingTheSession:
    """The count is "how long since I sent it", so it must not restart when
    the user looks at another session and comes back — the spinner widget is
    rebuilt on the way back, and it used to bring a fresh clock with it."""

    async def test_switching_away_and_back_keeps_the_elapsed_time(
        self, hpca_home
    ):
        llm = SlowLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(app, pilot)
            await wait_for_spinner(app, pilot)
            waiting = app.active_session
            # as if the user had been waiting on this turn for half a minute
            app._turns[waiting.session_id].started -= 30

            await app.start_new_session()  # switches away
            await pilot.pause()
            assert not list(app.query(WorkingIndicator))  # the new one is idle

            await app.open_session(waiting)
            spinner = await wait_for_spinner(app, pilot)
            assert spinner.elapsed >= 30, "the clock restarted on the way back"
            assert f"{spinner.elapsed}s" in str(spinner.content)

            llm.released.set()
            await app.workers.wait_for_complete()

    async def test_a_backend_call_without_a_turn_still_times_itself(
        self, hpca_home
    ):
        # /conclude and friends have no TurnState to read a start from; they
        # pass a label and the spinner mints its own clock.
        app = HpcaApp(llm=SlowLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.show_working("summarising")
            spinner = await wait_for_spinner(app, pilot)
            assert spinner.activity == "summarising"
            assert spinner.elapsed == 0


class TestSpinnerRepaintIsCheap:
    """The spinner repaints 12.5 times a second for the whole time a reply is
    in flight, and `Static.update` lays out by default — a pass that walks the
    chat log. With a few hundred messages that alone made the TUI crawl
    exactly while the user was waiting (measured: 0.7ms → 44ms p95 loop lag).
    Only a change in the line's width can move anything, so only that lays
    out."""

    def layout_calls(self, spinner):
        calls = []
        spinner.update = lambda content, **kw: calls.append(kw.get("layout"))
        return calls

    async def test_a_frame_advance_does_not_lay_out(self, hpca_home):
        spinner = WorkingIndicator("LLM processing")
        spinner._render_frame()  # first paint establishes the width
        calls = self.layout_calls(spinner)
        spinner._advance()
        spinner._advance()
        assert calls == [False, False]

    async def test_the_first_paint_lays_out(self, hpca_home):
        spinner = WorkingIndicator("LLM processing")
        calls = self.layout_calls(spinner)
        spinner._render_frame()
        assert calls == [True]

    async def test_a_wider_line_lays_out(self, hpca_home):
        spinner = WorkingIndicator("LLM processing")
        spinner._render_frame()
        calls = self.layout_calls(spinner)
        spinner.set_activity("running a_considerably_longer_tool_name")
        assert calls == [True]

    async def test_the_seconds_ticking_over_lays_out(self, hpca_home):
        # " 9s" → " 10s" widens the line, so that frame earns its layout.
        spinner = WorkingIndicator("LLM processing")
        spinner._started -= 9
        spinner._render_frame()
        calls = self.layout_calls(spinner)
        spinner._started -= 1  # now reads 10s
        spinner._advance()
        assert calls == [True]
