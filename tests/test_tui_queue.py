"""Typing while a turn runs (§3). Messages queue instead of being refused.

A session serialises its own turns — two on one thread_id would interleave
checkpoint writes — so a second message for a busy session queues for it. This
is about never blocking the user's input; turns on *different* sessions run
concurrently (see test_tui_concurrent_turns.py).
"""

import asyncio
import json

import pytest

from textual.widgets import ListView

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp, WorkingIndicator
from hpca.tui.rewind_screen import QueuedScreen

TITLE_REPLY = json.dumps({"title": "a test session"})


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class SlowLLM:
    """Holds a turn open until released, so 'busy' is a real state.

    ``hold`` names a message whose turn is never released, so a test can look
    at the app with that turn genuinely mid-flight rather than racing it.
    """

    def __init__(self, hold: str | None = None):
        self.gate = asyncio.Event()
        self.hold = hold
        self.never = asyncio.Event()  # deliberately never set
        self.seen_user_texts: list[str] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        latest = ""
        for message in messages:
            if message["role"] == "user" and not str(
                message["content"]
            ).startswith(("[tool", "[process", "[job")):
                self.seen_user_texts.append(message["content"])
                latest = str(message["content"])
        if self.hold is not None and self.hold in latest:
            await self.never.wait()
        await self.gate.wait()
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "ok"})
        )

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def send(app, pilot, text):
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


async def open_queued_dialog(app, pilot, text):
    """Activate the first queued row showing ``text``, as the user would."""
    chat_list = app.query_one("#chat-list", ListView)
    row = next(
        index
        for index, item in enumerate(chat_list.children)
        if getattr(getattr(item, "data_entry", None), "kind", None) == "queued"
        and item.data_entry.text == text
    )
    chat_list.focus()
    chat_list.index = row
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    assert isinstance(app.screen, QueuedScreen)


async def cancel_queued(app, pilot, text):
    await open_queued_dialog(app, pilot, text)
    await pilot.press("x")
    # Not workers.wait_for_complete(): the turn this message queued behind is
    # still being held open by the fake backend.
    for _ in range(10):
        await pilot.pause()


class TestQueueing:
    async def test_a_message_sent_while_busy_is_accepted(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            # this session's turn is held open
            assert app.active_session.session_id in app._turns

            await send(app, pilot, "second")
            assert app.queued_texts_for(app.active_session.session_id) == ["second"]

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_the_input_is_cleared_so_typing_continues(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            assert app.query_one("#chat-input", ChatInput).text == ""
            llm.gate.set()
            await app.workers.wait_for_complete()

    async def test_queued_message_is_visible_in_the_transcript(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            assert any(
                entry.kind == "queued" and entry.text == "second"
                for entry in app._chat_entries
            )
            llm.gate.set()
            await app.workers.wait_for_complete()

    async def test_only_one_turn_per_session_runs_at_a_time(self, hpca_home):
        """A session serialises its own turns: two on one thread would
        interleave checkpoint writes, so extra messages queue for THAT session
        rather than starting a second thread on it."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            await send(app, pilot, "third")
            # one turn running for this session, two waiting for it — never a
            # second thread in flight on the same session
            assert list(app._turns) == [session.session_id]
            assert app.queued_texts_for(session.session_id) == ["second", "third"]
            assert llm.seen_user_texts == ["first"]
            llm.gate.set()
            await app.workers.wait_for_complete()

    async def test_queue_drains_in_order_after_the_turn_ends(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            llm.gate.set()
            for _ in range(40):
                await pilot.pause()
                if not app._pending_work and not app._turns:
                    break
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._pending_work == []
            assert llm.seen_user_texts[:1] == ["first"]
            assert "second" in llm.seen_user_texts

    async def test_a_queued_message_survives_the_transcript_rebuild(self, hpca_home):
        """The turn's reply rebuilds the chat from graph messages; a message
        typed meanwhile is not in the graph yet and must not vanish."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            await send(app, pilot, "third")
            llm.gate.set()
            await pilot.pause()
            await pilot.pause()
            # while "second" runs, "third" is still queued and still shown
            texts = [e.text for e in app._chat_entries]
            assert "third" in texts
            await app.workers.wait_for_complete()

    async def test_a_queued_message_survives_the_turn_it_waited_for(
        self, hpca_home
    ):
        """The queue can drain into the next turn before the finished turn's
        reply is drawn. That reply rebuilds the chat from its own snapshot,
        taken before the message it drained even existed — the message is no
        longer queued and not yet in the graph, and must survive anyway."""
        llm = SlowLLM(hold="third")  # "third" runs but never finishes
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            await send(app, pilot, "third")
            llm.gate.set()
            for _ in range(20):
                await pilot.pause()
            assert llm.seen_user_texts[-1] == "third"  # its turn is in flight
            entries = [(e.kind, e.text) for e in app._chat_entries]
            assert ("user", "third") in entries  # shown as sent, not queued
            llm.never.set()
            await app.workers.wait_for_complete()

    async def test_slash_commands_are_refused_not_queued(self, hpca_home):
        """They act on the UI and run their own exclusive workers."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            await send(app, pilot, "first")
            await send(app, pilot, "/conclude")
            assert app._pending_work == []
            llm.gate.set()
            await app.workers.wait_for_complete()


class TestCancelling:
    """A queued message has not reached the model, so it can be taken back —
    the same take-back a turn in flight gets from the working indicator, and
    landing the same way, with the text in the entry to edit and send again."""

    async def test_cancelling_drops_it_from_the_queue_and_the_log(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")

            await cancel_queued(app, pilot, "second")
            assert app.queued_texts_for(session.session_id) == []
            assert not any(e.kind == "queued" for e in app._chat_entries)
            # Landed in the entry, as an interrupted turn's message does.
            assert app.query_one("#chat-input", ChatInput).text == "second"

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "second" not in llm.seen_user_texts  # it never ran

    async def test_the_turn_it_waited_for_still_shows_as_running(self, hpca_home):
        """Cancelling redraws the log, which clears the list the spinner is
        appended to. Losing it there would say the agent had stopped, while the
        turn the message queued behind is still waiting on the model."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            before = next(iter(app.query(WorkingIndicator)))

            await cancel_queued(app, pilot, "second")
            after = list(app.query(WorkingIndicator))
            assert len(after) == 1  # still there, and only one
            assert session.session_id in app._turns
            # Same step, and the same clock — not a spinner restarted at 0s.
            assert after[0].activity == before.activity
            assert after[0]._started == before._started
            # Still the last row, below the messages.
            rows = list(app.query_one("#chat-list", ListView).children)
            assert rows[-1].query(WorkingIndicator)

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not list(app.query(WorkingIndicator))  # gone when done

    async def test_escape_leaves_it_queued(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")

            await open_queued_dialog(app, pilot, "second")
            await pilot.press("escape")
            await pilot.pause()
            assert app.queued_texts_for(session.session_id) == ["second"]
            assert app.query_one("#chat-input", ChatInput).text == ""

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_it_cancels_the_one_that_was_picked(self, hpca_home):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            await send(app, pilot, "third")

            await cancel_queued(app, pilot, "second")
            assert app.queued_texts_for(session.session_id) == ["third"]

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_the_same_message_queued_twice_loses_one_copy(self, hpca_home):
        """Rows are matched by position, not by text: cancelling a duplicate
        must take back one of them, not both."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "again")
            await send(app, pilot, "again")

            await cancel_queued(app, pilot, "again")
            assert app.queued_texts_for(session.session_id) == ["again"]
            assert [e.kind for e in app._chat_entries].count("queued") == 1

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_a_message_that_started_meanwhile_is_not_cancelled(self, hpca_home):
        """The dialog can sit open long enough for the turn ahead to finish and
        the queue to drain into it. Stopping the turn it became is the working
        indicator's question, not this one."""
        llm = SlowLLM(hold="second")  # "second" runs but never finishes
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            queued = next(e for e in app._chat_entries if e.kind == "queued")

            llm.gate.set()  # "first" completes, "second" starts
            for _ in range(20):
                await pilot.pause()
            assert llm.seen_user_texts[-1] == "second"

            await app._cancel_queued(session, queued)
            await pilot.pause()
            assert ("user", "second") in [
                (e.kind, e.text) for e in app._chat_entries
            ]
            assert app.query_one("#chat-input", ChatInput).text == ""

            llm.never.set()
            await app.workers.wait_for_complete()

    async def test_a_session_switch_while_the_dialog_is_open_cancels_nothing(
        self, hpca_home
    ):
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            session = app.active_session
            await send(app, pilot, "first")
            await send(app, pilot, "second")
            queued = next(e for e in app._chat_entries if e.kind == "queued")

            elsewhere = app.session_store.create(profile="default", title="second")
            await app.open_session(elsewhere)
            await pilot.pause()
            await app._cancel_queued(session, queued)
            assert app.queued_texts_for(session.session_id) == ["second"]

            llm.gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()


class TestShutdown:
    async def test_a_reply_landing_after_the_screen_is_gone_does_not_raise(
        self, hpca_home
    ):
        """A turn the queue started can finish while the app is tearing down.
        The rebuild then has no chat column to draw into: it keeps the entries
        and stays quiet, rather than failing the worker on the way out."""
        app = HpcaApp(llm=SlowLLM())
        async with app.run_test() as pilot:
            await app.start_new_session()
            await pilot.pause()
            await app.query_one("#chat-list").remove()  # stand in for teardown
            await app._set_chat_messages(
                [{"role": "user", "content": "landed late"}]
            )
            await app._rerender_chat()
            assert app.chat_log_texts() == ["landed late"]


class TestApprovalInteraction:
    async def test_a_blocked_session_does_not_stall_the_queue(self, hpca_home):
        """A session parked on an approval must wait, but a message for a
        different session should still start."""
        llm = SlowLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test() as pilot:
            await app.start_new_session()
            blocked = app.active_session
            app._awaiting_approval.add(blocked.session_id)
            from hpca.tui.app import PendingWork

            app._pending_work = [
                PendingWork(session_id=blocked.session_id, text="waits", kind="user"),
                PendingWork(session_id="other-session", text="runs", kind="event"),
            ]
            await app.drain_work()
            # the blocked session's message is still queued, the other is gone
            assert [w.text for w in app._pending_work] == ["waits"]
