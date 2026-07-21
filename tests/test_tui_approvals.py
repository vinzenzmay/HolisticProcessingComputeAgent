"""Inline, non-modal decisions (deliverable of feature/3-inline-approvals).

A parked turn's approval or plan handoff renders inside the chat column of the
session it belongs to — never as a full-screen modal — and is answered only
when the chat column is focused. A decision waiting in a session the user has
switched away from lights an "!" in the sidebar instead of stealing focus, and
opening that session reveals its inline prompt.
"""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import Label, ListView

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, DecisionBar, HpcaApp


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


class DeleteParams(BaseModel):
    target: str = Field(description="What to delete")


async def delete_handler(args, ctx):
    return f"deleted {args.target}"


def destructive_tools():
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
    return registry


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


class BlockingLLM(FakeLLM):
    """Parks inside the first real chat() call until released, so the caller
    can switch sessions while the turn is still in flight."""

    def __init__(self, outputs):
        super().__init__(outputs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked_once = False

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        if not self._blocked_once:
            self._blocked_once = True
            self.entered.set()
            await self.release.wait()
        return ChatResponse(content=self._outputs.pop(0))


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def decision_bar(app):
    return app.query_one("#decision-bar", DecisionBar)


def session_row(app, session_id):
    for item in app.query_one("#sessions-list", ListView).children:
        row = getattr(item, "data_session", None)
        if row is not None and row.session_id == session_id:
            return item
    return None


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")


class TestInlineApproval:
    async def test_prompt_is_inline_and_focused_not_a_modal(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            await app.workers.wait_for_complete()
            await pilot.pause()
            bar = decision_bar(app)
            # renders inside the chat column, and nothing was pushed on top
            assert bar.display and bar.kind == "approval"
            assert len(app.screen_stack) == 1
            # brought into focus so its y/n keys work at once
            assert app.focused is bar
            # the session is flagged as pending while it waits
            assert app.active_session.session_id in app._pending_decision

    async def test_answering_resumes_and_clears(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [tool_json("delete", target="results/"), respond_json("it is gone")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not decision_bar(app).display
            assert app.active_session.session_id not in app._pending_decision
            texts = app.chat_log_texts()
            assert any("deleted results/" in t for t in texts)
            assert any("it is gone" in t for t in texts)


class TestBackgroundIndicator:
    async def test_pending_in_background_flags_sidebar_without_overlay(
        self, hpca_home
    ):
        app = HpcaApp(
            llm=BlockingLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            await asyncio.wait_for(app._llm.entered.wait(), timeout=5)
            parked = app.active_session
            # leave the parked session for a fresh one while the turn runs
            await app.start_new_session()
            await pilot.pause()
            other = app.active_session

            app._llm.release.set()  # the turn now surfaces its approval
            await app.workers.wait_for_complete()
            await pilot.pause()

            # the decision landed against the parked session, not the open one
            assert parked.session_id in app._pending_decision
            assert other.session_id not in app._pending_decision
            # no modal, and the open session shows no inline prompt
            assert len(app.screen_stack) == 1
            assert not decision_bar(app).display
            # the parked session's row is flagged with the "!"
            row = session_row(app, parked.session_id)
            assert row.has_class("session-pending")
            assert app._session_row_text(parked).plain.startswith("! ")

    async def test_opening_the_flagged_session_reveals_the_prompt(self, hpca_home):
        app = HpcaApp(
            llm=BlockingLLM(
                [tool_json("delete", target="results/"), respond_json("done")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            await asyncio.wait_for(app._llm.entered.wait(), timeout=5)
            parked = app.active_session
            await app.start_new_session()
            await pilot.pause()
            app._llm.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not decision_bar(app).display  # hidden in the other session

            await app.open_session(parked)
            await pilot.pause()
            # switching to it reveals the inline prompt it was left waiting on
            bar = decision_bar(app)
            assert bar.display and bar.kind == "approval"
            # still pending until answered — opening does not resolve it
            assert parked.session_id in app._pending_decision
