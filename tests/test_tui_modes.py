"""Tests for the TUI side of agent modes (§3.5): the mode line above the
entry, shift+tab cycling, and the manual run/skip flow."""

import json

import pytest

from pydantic import BaseModel, Field

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, DecisionBar, HpcaApp
from hpca.tui.mode_bar import ModeBar


def is_title_request(json_schema):
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


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def chat_texts(app):
    return app.chat_log_texts()


def decision_bar(app):
    return app.query_one("#decision-bar", DecisionBar)


def pending_payload(app):
    """The payload of the decision the active session is parked on."""
    return app._pending_decision[app.active_session.session_id]["payload"]


def no_modal(app):
    """The inline decision never pushes a screen — the stack stays at one."""
    return len(app.screen_stack) == 1


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


class TestModeBar:
    async def test_hidden_without_a_session_shown_with_one(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            bar = app.query_one("#mode-bar", ModeBar)
            assert not bar.display
            await app.start_new_session()
            await pilot.pause()
            assert bar.display
            assert bar.mode == "manual"

    async def test_shift_tab_cycles_and_persists(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.active_session.mode == "auto"
            stored = app.session_store.get(app.active_session.session_id)
            assert stored.mode == "auto"
            assert app.query_one("#mode-bar", ModeBar).mode == "auto"
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.active_session.mode == "full-auto"
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.active_session.mode == "manual"

    async def test_new_sessions_start_in_the_configured_default(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]))
        app.settings.agent.default_mode = "auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert app.query_one("#mode-bar", ModeBar).mode == "auto"


class TestManualFlow:
    async def test_run_bash_gates_and_skip_reports_back(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo hi"]),
                    respond_json("understood"),
                ]
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "check something")
            bar = decision_bar(app)
            assert bar.display and bar.kind == "approval"
            assert no_modal(app)  # inline, not a full-screen modal
            assert pending_payload(app)["kind"] == "execution"
            assert "echo hi" in pending_payload(app)["script"]
            await pilot.press("n")  # skip script
            await pilot.pause()
            await pilot.press("enter")  # skip without giving a reason
            await app.workers.wait_for_complete()
            await pilot.pause()
            texts = chat_texts(app)
            assert any("SKIPPED" in t for t in texts)
            assert not any("exit 0" in t for t in texts)
            assert not bar.display  # cleared once answered

    async def test_approve_actually_runs_the_script(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [
                    tool_json("run_bash", content_lines=["echo mode-test"]),
                    respond_json("saw it"),
                ]
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "check something")
            assert decision_bar(app).display and no_modal(app)
            await pilot.press("y")  # run script
            await app.workers.wait_for_complete()
            await pilot.pause()
            texts = chat_texts(app)
            assert any("mode-test" in t and "exit 0" in t for t in texts)


class TestAutoFlow:
    async def test_no_gate_in_auto_mode(self, hpca_home):
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
            assert not decision_bar(app).display  # auto mode never gates
            texts = chat_texts(app)
            assert any("auto-run" in t and "exit 0" in t for t in texts)


class TestFullAutoFlow:
    async def test_destructive_tool_runs_without_approval(self, hpca_home):
        class DeleteParams(BaseModel):
            target: str = Field(description="What to delete")

        async def delete_handler(args, ctx):
            return f"deleted {args.target}"

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="delete",
                description="Delete something",
                params=DeleteParams,
                handler=delete_handler,
                destructive=True,
            )
        )
        app = HpcaApp(
            llm=FakeLLM(
                [tool_json("delete", target="scratch/"), respond_json("it is gone")]
            ),
            tools=registry,
        )
        app.settings.agent.default_mode = "full-auto"
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "clean up scratch")
            assert not decision_bar(app).display  # full-auto never gates
            texts = chat_texts(app)
            assert any("deleted scratch/" in t for t in texts)
            assert any("it is gone" in t for t in texts)
