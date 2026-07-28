"""/compact: the user folds the conversation themselves, and says what for.

The automatic fold (tests/test_compact.py, tests/test_graph.py) waits for the
window to fill. This is the deliberate one: the user picks the moment, and the
text after the command tells the summary what to carry — what to preserve, or
what they are about to do next.
"""

import json

import pytest
from textual.widgets import Static

from hpca.agent import compact
from hpca.llm import ChatResponse
from hpca.tui.app import COMMANDS, ChatInput, HpcaApp


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})
PROMPT_TOKENS = 5000


class FakeLLM:
    """Answers turns with ``respond``; a call with no schema is a summarize."""

    def __init__(self, summary="what happened earlier"):
        self._summary = summary
        self.summarize_calls: list[list[dict]] = []
        self.decide_calls: list[list[dict]] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        if json_schema is None:
            self.summarize_calls.append(list(messages))
            return ChatResponse(content=self._summary)
        self.decide_calls.append(list(messages))
        return ChatResponse(
            content=json.dumps({"action": "respond", "response": "ok"}),
            usage={"prompt_tokens": PROMPT_TOKENS},
        )

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def capture_notifications(app):
    messages: list[str] = []
    original = HpcaApp.notify
    app.notify = lambda message, **kw: (
        messages.append(message),
        original(app, message, **kw),
    )[1]
    return messages


async def submit(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
        await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


async def converse(app, pilot, *texts):
    for text in texts:
        await submit(app, pilot, text)
        await app.workers.wait_for_complete()
        await pilot.pause()


async def thread_state(app):
    snapshot = await app.graph.aget_state(
        {"configurable": {"thread_id": app.active_session.session_id}}
    )
    return snapshot.values or {}


async def run_compact(app, pilot, command="/compact"):
    await submit(app, pilot, command)
    await app.workers.wait_for_complete()
    await pilot.pause()


class TestTheCommandItself:
    def test_it_is_offered_in_the_slash_menu(self):
        assert "compact" in {name for name, _ in COMMANDS}
        usage = next(text for name, text in COMMANDS if name == "compact")
        assert "/compact" in usage

    async def test_typing_slash_lists_it(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "/comp"
            await pilot.pause()
            assert "compact" in {name for name, _ in app._command_matches}


class TestCompacting:
    async def test_it_folds_the_conversation(self, hpca_home):
        llm = FakeLLM(summary="the user asked two things")
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await converse(app, pilot, "first question", "second question")
            await run_compact(app, pilot)
            values = await thread_state(app)
            assert values["compacted"]["upto"] == len(values["messages"]) == 4
            assert "the user asked two things" in (
                values["compacted"]["summary"]["content"]
            )
            # the transcript itself is untouched: the user can still scroll back
            assert values["messages"][0]["content"] == "first question"

    async def test_the_next_turn_runs_on_the_summary(self, hpca_home):
        llm = FakeLLM(summary="everything so far, in brief")
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await converse(app, pilot, "first question")
            await run_compact(app, pilot)
            await converse(app, pilot, "and now?")
            sent = llm.decide_calls[-1]
            assert not any("first question" == str(m["content"]) for m in sent)
            assert any(
                "everything so far, in brief" in str(m["content"]) for m in sent
            )

    async def test_the_instruction_after_the_command_steers_the_summary(
        self, hpca_home
    ):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await converse(app, pilot, "how do I run STAR?")
            await run_compact(
                app, pilot, "/compact keep the STAR flags; next I run the full cohort"
            )
            system = llm.summarize_calls[-1][0]["content"]
            assert "keep the STAR flags" in system
            assert "next I run the full cohort" in system
            # and it stays in the model's view afterwards
            summary = (await thread_state(app))["compacted"]["summary"]["content"]
            assert compact.FOCUS_PREFIX in summary
            assert "next I run the full cohort" in summary

    async def test_a_bare_compact_asks_for_nothing_in_particular(self, hpca_home):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await converse(app, pilot, "how do I run STAR?")
            await run_compact(app, pilot)
            summary = (await thread_state(app))["compacted"]["summary"]["content"]
            assert compact.FOCUS_PREFIX not in summary

    async def test_the_chat_shows_what_was_kept(self, hpca_home):
        """The summary is what the agent remembers from here on, so the user
        sees it — not just a count of what disappeared."""
        app = HpcaApp(llm=FakeLLM(summary="STAR needs 40G; two jobs still queued"))
        async with app.run_test(size=(120, 40)) as pilot:
            await converse(app, pilot, "first question")
            await run_compact(app, pilot)
            notices = [e for e in app._chat_entries if e.kind == "notice"]
            assert notices and "compacted" in notices[-1].text.lower()
            assert "STAR needs 40G" in notices[-1].text
            # and it is really on screen, not only in the app's own record
            shown = [s for s in app.query(Static) if "chat-notice" in s.classes]
            assert shown

    async def test_the_context_meter_stops_showing_the_pre_fold_number(
        self, hpca_home
    ):
        """The backend's last count described the unfolded prompt; after the
        fold it is simply wrong, so it gives way to an estimate of the view."""
        app = HpcaApp(llm=FakeLLM(summary="brief"))
        async with app.run_test(size=(120, 40)) as pilot:
            await converse(app, pilot, "x" * 4000)
            session_id = app.active_session.session_id
            assert app._context_used[session_id] == PROMPT_TOKENS
            await run_compact(app, pilot)
            assert session_id not in app._context_used
            bar = app._context_bar()
            assert 0 < bar._used < PROMPT_TOKENS

    async def test_a_backend_failure_leaves_the_session_alone(self, hpca_home):
        class Failing(FakeLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if json_schema is None:
                    raise RuntimeError("backend down")
                return await FakeLLM.chat(
                    self, messages, json_schema=json_schema, **kwargs
                )

        app = HpcaApp(llm=Failing())
        async with app.run_test(size=(120, 40)) as pilot:
            notifications = capture_notifications(app)
            await converse(app, pilot, "first question")
            await run_compact(app, pilot)
            assert not (await thread_state(app)).get("compacted")
            assert any("backend down" in m for m in notifications)


class TestRefusals:
    async def test_without_a_session_there_is_nothing_to_compact(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            notifications = capture_notifications(app)
            app._handle_slash_command("/compact")
            await pilot.pause()
            assert any("no active session" in m.lower() for m in notifications)

    async def test_an_empty_session_is_left_alone(self, hpca_home):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            notifications = capture_notifications(app)
            await app.start_new_session()
            await run_compact(app, pilot)
            assert not llm.summarize_calls  # the backend is never bothered
            assert any("nothing" in m.lower() for m in notifications)

    async def test_compacting_twice_over_says_so(self, hpca_home):
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            notifications = capture_notifications(app)
            await converse(app, pilot, "first question")
            await run_compact(app, pilot)
            await run_compact(app, pilot)
            assert len(llm.summarize_calls) == 1
            assert any("nothing" in m.lower() for m in notifications)

    async def test_a_session_waiting_on_an_approval_is_not_touched(self, hpca_home):
        """The thread is parked mid-turn on an interrupt; rewriting its state
        underneath the pending decision is not a thing to do quietly."""
        llm = FakeLLM()
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            notifications = capture_notifications(app)
            await converse(app, pilot, "first question")
            app._awaiting_approval.add(app.active_session.session_id)
            await run_compact(app, pilot)
            assert not llm.summarize_calls
            assert any("approval" in m.lower() for m in notifications)
            assert not (await thread_state(app)).get("compacted")
