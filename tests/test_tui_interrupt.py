"""Interrupt the model by selecting the working indicator and confirming.

While a turn waits on the LLM a "working" line sits at the end of the chat log
(§ interrupt). Selecting it (enter) opens a yes/no dialog; yes aborts the
in-flight request and hands the message back for editing, no leaves it running.
"""

import asyncio
import json

import pytest
from textual.widgets import ListView, Static

from hpca.llm import ChatResponse
from hpca.tui.app import (
    LLM_WAIT_ACTIVITY,
    ChatInput,
    HpcaApp,
    WorkingIndicator,
)
from hpca.tui.confirm_screen import ConfirmScreen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class BlockingLLM:
    """Parks inside chat() until released, so the turn genuinely sits in the
    'LLM processing' phase — what the interrupt is armed against."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self.entered.set()
        await self.release.wait()
        return ChatResponse(content=json.dumps({"action": "respond", "response": "done"}))

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def send_and_park(app, pilot, text):
    """Submit a message and wait until the turn is parked on the LLM."""
    await app.start_new_session()
    await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await asyncio.wait_for(app._llm.entered.wait(), timeout=5)
    await pilot.pause()


async def select_working_indicator(app, pilot):
    """Move into the log, land on the working indicator (the last line), and
    press enter to select it — the real path a user takes."""
    app.browse_chat_messages()
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


class TestArming:
    async def test_working_indicator_present_while_waiting(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "hello")
            assert app._can_interrupt()
            assert app.query(WorkingIndicator)  # sits at the end of the log
            app._llm.release.set()

    def test_indicator_hints_at_interrupt_only_while_on_the_llm(self):
        assert "enter to interrupt" in WorkingIndicator(LLM_WAIT_ACTIVITY)._frame_text()
        # a tool step or plain "working" is not interruptible: no hint
        assert "enter to interrupt" not in WorkingIndicator("run_bash")._frame_text()
        assert "enter to interrupt" not in WorkingIndicator()._frame_text()

    async def test_no_dialog_when_not_waiting_on_the_llm(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert not app._can_interrupt()
            app._maybe_interrupt_llm()  # no-op: nothing to interrupt
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)


class TestInterrupt:
    async def test_selecting_the_indicator_offers_the_dialog(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "hello")
            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")  # dismiss so teardown is clean
            await pilot.pause()
            app._llm.release.set()

    async def test_confirming_interrupts_and_hands_the_message_back(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "draft with a typo")
            assert app._busy_turn is not None

            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")  # confirm the interrupt
            await pilot.pause()

            assert app._interrupt_worker is not None
            await app._interrupt_worker.wait()
            await pilot.pause()

            chat_input = app.query_one("#chat-input", ChatInput)
            assert chat_input.text == "draft with a typo"  # handed back to edit
            assert app._busy_turn is None
            # the aborted message left the thread: the next turn starts clean
            snap = await app.graph.aget_state(
                {"configurable": {"thread_id": app.active_session.session_id}}
            )
            assert (snap.values or {}).get("messages", []) == []

    async def test_declining_leaves_the_turn_running(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "keep going")
            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")  # decline
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert app._busy_turn is not None  # still running
            assert app._interrupt_worker is None  # nothing fired
            app._llm.release.set()

    async def test_reply_landing_while_the_dialog_is_open_is_a_no_op(self, hpca_home):
        # If the model answers while the confirm dialog sits open, confirming
        # must not fire an interrupt against a turn that is already gone.
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "hello")
            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            # the turn completes underneath the dialog
            app._llm.release.set()
            await pilot.pause()
            await asyncio.sleep(0.05)
            await pilot.pause()
            await pilot.press("y")  # confirm — but there is nothing to abort now
            await pilot.pause()
            assert app._interrupt_worker is None
            assert not app.query(WorkingIndicator)  # the reply landed normally
