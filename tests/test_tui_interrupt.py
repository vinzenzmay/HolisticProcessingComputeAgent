"""Hold-esc-3s to interrupt the model while it waits for the LLM (§ interrupt)."""

import asyncio
import json

import pytest
from textual.widgets import Static

from hpca.llm import ChatResponse
from hpca.tui.app import (
    ESC_IDLE_FACTOR,
    ESC_IDLE_INITIAL,
    ESC_IDLE_MIN,
    ESC_INTERRUPT_TICKS,
    ChatInput,
    HpcaApp,
)


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


class TestArming:
    async def test_esc_does_nothing_when_idle(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert app.handle_esc_hold() is False  # not waiting on the LLM
            assert not app.query_one("#esc-progress", Static).display

    async def test_esc_arms_while_waiting_on_the_llm(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "hello")
            assert app._can_interrupt()
            assert app.handle_esc_hold() is True
            assert app.query_one("#esc-progress", Static).display
            app._cancel_esc_hold()
            assert not app.query_one("#esc-progress", Static).display
            app._llm.release.set()


class TestInterrupt:
    async def test_hold_to_threshold_fires_the_interrupt(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "draft with a typo")
            assert app._busy_turn is not None

            app.handle_esc_hold()  # begin the hold
            for _ in range(ESC_INTERRUPT_TICKS):  # drive the 3s worth of ticks
                app._advance_esc_hold()
            # the tick past the threshold scheduled the rollback worker
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

    async def test_short_press_does_not_interrupt(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "keep going")
            app.handle_esc_hold()
            # a few ticks, then release before 3s
            for _ in range(5):
                app._advance_esc_hold()
            app._cancel_esc_hold()
            await pilot.pause()
            assert app._busy_turn is not None  # still running
            assert app._interrupt_worker is None  # nothing fired
            assert not app.query_one("#esc-progress", Static).display
            app._llm.release.set()


class TestReleaseWatchdog:
    """How fast the progress bar clears after esc is let go. Terminals give no
    key-release, so the watchdog adapts to the measured key-repeat interval."""

    def test_waits_the_full_initial_delay_before_a_repeat_is_seen(self):
        app = HpcaApp(llm=BlockingLLM())
        assert app._esc_gap is None
        # Nothing measured yet: must survive the OS's long initial repeat delay.
        assert app._esc_release_idle() == ESC_IDLE_INITIAL

    def test_fast_repeat_clears_quickly(self):
        app = HpcaApp(llm=BlockingLLM())
        app._esc_gap = 0.03  # ~33 repeats/sec, a typical terminal
        idle = app._esc_release_idle()
        assert idle == ESC_IDLE_MIN  # floored, far below the old 0.75s lag
        assert idle < ESC_IDLE_INITIAL

    def test_moderate_repeat_scales_with_a_margin(self):
        app = HpcaApp(llm=BlockingLLM())
        app._esc_gap = 0.08
        assert app._esc_release_idle() == pytest.approx(0.08 * ESC_IDLE_FACTOR)

    def test_slow_repeat_is_capped_so_a_hold_is_never_cut_off(self):
        app = HpcaApp(llm=BlockingLLM())
        app._esc_gap = 0.4  # a slow-repeat config; interval < cap keeps holds alive
        assert app._esc_release_idle() == ESC_IDLE_INITIAL
