"""Tests for the right column: process list, inspect, kill (§3.3)."""

import json

import pytest
from textual.widgets import ListView

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen


def is_title_request(json_schema):
    """The app names a session by asking the model (§3 sessions column); that
    call is not one of the queued decisions."""
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})

class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def start_session_with_process(app, pilot, argv, name):
    """Open a session via chat, then start a tracked process in its runner."""
    await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "hello"
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    record = await app._tool_ctx.runner.start(argv, name=name)
    await app.refresh_processes()
    await pilot.pause()
    return record


class TestProcessList:
    async def test_process_appears_in_right_column(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["sleep", "5"], "sleeper"
            )
            processes_list = app.query_one("#processes-list", ListView)
            assert len(processes_list) == 1
            assert getattr(processes_list.children[0], "data_record").pid == record.pid
            await app._tool_ctx.runner.kill(record.pid)

    async def test_finished_process_shows_state(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["echo", "hi"], "quickie"
            )
            await app._tool_ctx.runner.wait(record.pid)
            await app.refresh_processes()
            await pilot.pause()
            items = app.query_one("#processes-list", ListView).children
            assert getattr(items[0], "data_record").state == "finished"


class TestInspect:
    async def test_enter_opens_inspect_with_logs(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["bash", "-c", "echo needle-out; echo needle-err >&2"],
                "loggy",
            )
            await app._tool_ctx.runner.wait(record.pid)
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, InspectScreen)
            body = app.screen.body_text()
            assert "needle-out" in body
            assert "needle-err" in body
            await pilot.press("escape")
            assert not isinstance(app.screen, InspectScreen)


class TestKill:
    async def test_k_confirms_then_kills(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["sleep", "60"], "victim"
            )
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await app._tool_ctx.runner.wait(record.pid)
            await pilot.pause()
            assert app._tool_ctx.runner.get(record.pid).state == "killed"

    async def test_kill_denied_leaves_process_running(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["sleep", "60"], "survivor"
            )
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("k")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert app._tool_ctx.runner.get(record.pid).state == "running"
            await app._tool_ctx.runner.kill(record.pid)
