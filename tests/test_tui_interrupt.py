"""Interrupt the model by selecting the working indicator and confirming.

While a turn waits on the LLM a "working" line sits at the end of the chat log
(§ interrupt). Selecting it (enter) opens a yes/no dialog; yes aborts the
in-flight request and hands the message back for editing, no leaves it running.
"""

import asyncio
import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import ListView, Static

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import (
    LLM_WAIT_ACTIVITY,
    ChatInput,
    DecisionBar,
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


class SlowTool:
    """A tool that parks mid-run, so the turn genuinely sits in its "running
    slow_tool" phase — the other half of what an interrupt has to cover."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def reset(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def handler(self, args, ctx):
        self.entered.set()
        await self.release.wait()
        return "finished at last"


SLOW_TOOL = SlowTool()


class SlowParams(BaseModel):
    text: str = Field(default="", description="ignored")


def blocking_tools():
    SLOW_TOOL.reset()
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="slow_tool",
            description="Takes its time",
            params=SlowParams,
            handler=SLOW_TOOL.handler,
        )
    )
    return registry


class ToolThenAnswerLLM:
    """Calls the slow tool once, then answers."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._calls = 0

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self._calls += 1
        if self._calls == 1:
            return ChatResponse(
                content=json.dumps(
                    {"action": "tool_call", "tool": "slow_tool", "arguments": {}}
                )
            )
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "done"})
        )

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def send_and_park(app, pilot, text, until=None):
    """Submit a message and wait until the turn reaches the phase under test —
    parked on the LLM by default, or on whatever ``until`` is set by."""
    await app.start_new_session()
    await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await asyncio.wait_for((until or app._llm.entered).wait(), timeout=5)
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

    def test_the_hint_follows_what_can_actually_be_interrupted(self):
        # Not the phase: a turn is abortable while it runs a tool just as much
        # as while it waits on the model. What is not abortable is a spinner
        # with no turn behind it — a silent backend call.
        for activity in (LLM_WAIT_ACTIVITY, "running run_bash", "working"):
            hint = WorkingIndicator(activity, interruptible=True)._frame_text()
            assert "enter to interrupt" in hint, activity
        assert "enter to interrupt" not in WorkingIndicator(LLM_WAIT_ACTIVITY)._frame_text()

    async def test_a_turn_running_a_tool_is_interruptible(self, hpca_home):
        # The phase a long script spends its minutes in — and the one a user
        # most wants to stop.
        app = HpcaApp(llm=ToolThenAnswerLLM(), tools=blocking_tools())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "run the thing", until=SLOW_TOOL.entered)
            assert app._active_turn().activity == "running slow_tool"
            assert app._can_interrupt()
            indicator = app.query_one(WorkingIndicator)
            assert "enter to interrupt" in indicator._frame_text()

            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            SLOW_TOOL.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # rolled back and handed back for editing, as from the LLM phase
            assert app.active_session.session_id not in app._turns
            assert app.query_one(ChatInput).text == "run the thing"

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
            assert app.active_session.session_id in app._turns

            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")  # confirm the interrupt
            await pilot.pause()

            assert app._interrupt_worker is not None
            await app._interrupt_worker.wait()
            await pilot.pause()

            chat_input = app.query_one("#chat-input", ChatInput)
            assert chat_input.text == "draft with a typo"  # handed back to edit
            assert app.active_session.session_id not in app._turns
            # the aborted message left the thread: the next turn starts clean
            snap = await app.graph.aget_state(
                {"configurable": {"thread_id": app.active_session.session_id}}
            )
            assert (snap.values or {}).get("messages", []) == []

    async def test_message_goes_back_to_its_own_session_not_the_open_one(
        self, hpca_home
    ):
        """Switching sessions while the rollback runs must not drop the
        interrupted message into the entry of whatever is now on screen — it
        waits as a draft in the session it was typed in."""
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "typed in the first session")
            interrupted = app.active_session
            ts = app._turns[interrupted.session_id]
            elsewhere = app.session_store.create(profile="default", title="second")
            await app.open_session(elsewhere)
            await pilot.pause()

            await app._interrupt_turn(
                interrupted, ts.interrupt_keep, ts.user_text, ts.worker
            )
            await pilot.pause()

            chat_input = app.query_one("#chat-input", ChatInput)
            assert chat_input.text == ""  # the open session's entry is untouched
            await app.open_session(interrupted)
            await pilot.pause()
            assert chat_input.text == "typed in the first session"

    async def test_declining_leaves_the_turn_running(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "keep going")
            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")  # decline
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert app.active_session.session_id in app._turns  # still running
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


class ApprovalThenBlockLLM:
    """Asks for a destructive tool first; the call that follows the approval
    parks. That second call belongs to a turn of its own — the one the user is
    left watching after answering the prompt."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._calls = 0

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self._calls += 1
        if self._calls == 1:
            return ChatResponse(
                content=json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "delete",
                        "arguments": {"target": "results/"},
                    }
                )
            )
        self.entered.set()
        await self.release.wait()
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "done"})
        )

    async def supports_constrained_decoding(self):
        return True


class TestTheSpinnerStaysReachable:
    """The user stops the agent by landing on the last line of the log and
    pressing enter, so nothing may take that place while a turn runs."""

    async def test_typing_ahead_does_not_bury_the_spinner(self, hpca_home):
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await send_and_park(app, pilot, "first")
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "and then this"
            await pilot.press("enter")  # queued behind the running turn
            await pilot.pause()

            rows = app.query_one("#chat-list", ListView).children
            assert list(rows[-1].query(WorkingIndicator)), "the spinner must stay last"
            assert "and then this" in str(rows[-2].query_one(Static).content)

            # and enter on that last line still means "stop", not "take back"
            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            app._llm.release.set()


class TestStoppableAfterAnApproval:
    async def test_the_resumed_turn_carries_the_same_anchor(self, hpca_home):
        llm = ApprovalThenBlockLLM()
        app = HpcaApp(llm=llm, tools=destructive_tools())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "delete results"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one("#decision-bar", DecisionBar).display

            await pilot.press("y")  # approve; the turn resumes and parks again
            await asyncio.wait_for(llm.entered.wait(), timeout=5)
            await pilot.pause()

            # the resume has no user message of its own: it borrows the one
            # that started the exchange, which is what the rollback needs.
            ts = app._turns[app.active_session.session_id]
            assert ts.user_text == "delete results"
            assert ts.interrupt_keep is not None
            assert app._can_interrupt()

    async def test_the_spinner_after_an_approval_still_stops_the_turn(self, hpca_home):
        llm = ApprovalThenBlockLLM()
        app = HpcaApp(llm=llm, tools=destructive_tools())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "delete results"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("y")
            await asyncio.wait_for(llm.entered.wait(), timeout=5)
            await pilot.pause()

            session_id = app.active_session.session_id
            await select_working_indicator(app, pilot)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            assert app._interrupt_worker is not None
            await app._interrupt_worker.wait()
            await pilot.pause()

            # handed back for editing, and the whole exchange — message, tool
            # call, tool result — is out of the thread again
            assert chat_input.text == "delete results"
            assert session_id not in app._turns
            snap = await app.graph.aget_state(
                {"configurable": {"thread_id": session_id}}
            )
            assert (snap.values or {}).get("messages", []) == []

    async def test_the_anchor_is_dropped_once_the_exchange_ends(self, hpca_home):
        llm = ApprovalThenBlockLLM()
        app = HpcaApp(llm=llm, tools=destructive_tools())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "delete results"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            session_id = app.active_session.session_id
            await pilot.press("y")
            await asyncio.wait_for(llm.entered.wait(), timeout=5)
            # it survives the turn that parked — the exchange is still running
            assert session_id in app._interrupt_anchor
            llm.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert session_id not in app._interrupt_anchor


class TestItSaysWhyItCannot:
    async def test_a_spinner_that_is_not_a_turn_says_so(self, hpca_home):
        """A silent backend call (/conclude, compacting) has nothing to roll
        back. Enter there used to do nothing at all, which reads as a key that
        stopped working — at the one moment that must not happen."""
        app = HpcaApp(llm=BlockingLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            seen = []
            app.notify = lambda message, **kwargs: seen.append(message)
            app.show_working("compacting context")
            await pilot.pause()

            await select_working_indicator(app, pilot)
            assert not isinstance(app.screen, ConfirmScreen)
            assert seen and "Nothing to interrupt" in seen[0]
