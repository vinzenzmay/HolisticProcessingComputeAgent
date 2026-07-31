"""Tests for the chat wiring: input → agent worker → chat log, sessions, approval."""

import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import ListView

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, DecisionBar, HpcaApp


def decision_bar(app):
    return app.query_one("#decision-bar", DecisionBar)


class DeleteParams(BaseModel):
    target: str = Field(description="What to delete")


async def delete_handler(args, ctx):
    return f"deleted {args.target}"


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


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


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


def chat_texts(app):
    return app.chat_log_texts()


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()  # the chat entry only exists in a session
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


class TestChatFlow:
    async def test_user_and_assistant_messages_appear(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("hello back")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello agent")
            texts = chat_texts(app)
            assert any("hello agent" in t for t in texts)
            assert any("hello back" in t for t in texts)

    async def test_model_names_the_session_after_the_first_exchange(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.session_store.list(profile="default") == []
            await submit_chat(app, pilot, "start my session")
            sessions = app.session_store.list(profile="default")
            assert len(sessions) == 1
            assert sessions[0].title == "a test session"
            assert app.active_session.title == sessions[0].title

    async def test_opening_message_names_it_when_the_model_cannot(self, hpca_home):
        class NoTitleLLM(FakeLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if is_title_request(json_schema):
                    return ChatResponse(content="not a title object")
                return await super().chat(messages, json_schema=json_schema, **kwargs)

        app = HpcaApp(llm=NoTitleLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "start my session")
            # the placeholder stands rather than the session going nameless
            assert app.active_session.title == "start my session"

    async def test_later_messages_do_not_retitle(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json(), respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first thing")
            named = app.active_session.title
            await submit_chat(app, pilot, "second thing")
            titles = [s.title for s in app.session_store.list(profile="default")]
            assert titles == [named]  # named once, not on every turn

    async def test_second_turn_same_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("one"), respond_json("two")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            await submit_chat(app, pilot, "second")
            assert len(app.session_store.list(profile="default")) == 1
            texts = chat_texts(app)
            assert any("one" in t for t in texts)
            assert any("two" in t for t in texts)

    async def test_llm_failure_shown_as_error(self, hpca_home):
        class BrokenLLM:
            async def chat(self, *a, **k):
                from hpca.llm import LLMError

                raise LLMError("backend unreachable")

            async def supports_constrained_decoding(self):
                return True

        app = HpcaApp(llm=BrokenLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hi")
            assert any("backend unreachable" in t for t in chat_texts(app))


class TestApprovalFlow:
    async def test_destructive_tool_prompts_inline(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            bar = decision_bar(app)
            assert bar.display and bar.kind == "approval"
            assert len(app.screen_stack) == 1  # inline, no modal on the stack
            payload = app._pending_decision[app.active_session.session_id]["payload"]
            assert payload["tool"] == "delete"
            assert "results/" in payload["arguments"]["target"]

    async def test_approve_runs_tool(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [tool_json("delete", target="results/"), respond_json("it is gone")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            texts = chat_texts(app)
            assert any("deleted results/" in t for t in texts)
            assert any("it is gone" in t for t in texts)

    async def test_reject_skips_tool(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM(
                [tool_json("delete", target="results/"), respond_json("understood")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            await pilot.press("n")
            await pilot.pause()
            await pilot.press("enter")  # refuse without giving a reason
            await app.workers.wait_for_complete()
            await pilot.pause()
            texts = chat_texts(app)
            assert not any("deleted results/" in t for t in texts)
            assert any("DENIED" in t for t in texts)


class TestSessionSwitching:
    async def test_sessions_listed_in_left_column(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "my topic")
            sessions_list = app.query_one("#sessions-list", ListView)
            assert len(sessions_list) >= 2  # "(new session)" + the created one

    async def test_switching_loads_history(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([respond_json("answer A"), respond_json("answer B")])
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "question A")
            first = app.active_session

            await app.start_new_session()
            await pilot.pause()
            await submit_chat(app, pilot, "question B")
            assert not any("question A" in t for t in chat_texts(app))

            await app.open_session(first)
            await pilot.pause()
            texts = chat_texts(app)
            assert any("question A" in t for t in texts)
            assert any("answer A" in t for t in texts)
            assert not any("question B" in t for t in texts)

    async def test_new_session_via_ui_selection(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "session one")
            first = app.active_session
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 0  # "(new session)"
            await pilot.press("enter")  # opens the profile picker
            await pilot.pause()
            await pilot.press("enter")  # take the current profile
            await pilot.pause()
            await pilot.pause()
            assert app.active_session is not first
            assert chat_texts(app) == []
            assert app.focused.id == "chat-input"

    async def test_selecting_a_session_focuses_its_chat_entry(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "session one")
            first = app.active_session
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1  # the session created above
            await pilot.press("enter")
            await pilot.pause()
            assert app.active_session.session_id == first.session_id
            assert app.focused.id == "chat-input"

    async def test_open_session_is_highlighted_in_the_column(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "session one")
            sessions_list = app.query_one("#sessions-list", ListView)
            assert sessions_list.index == 1  # not "(new session)"


class TestChatLogBrowsing:
    async def test_up_leaves_the_entry_and_down_returns_to_it(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("hello back")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello agent")
            chat_list = app.query_one("#chat-list", ListView)
            assert len(chat_list) == 2
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.text = "half typed"
            app.focus_chat_input()
            await pilot.pause()

            await pilot.press("up")  # into the log, at the newest message
            assert app.focused is chat_list
            assert chat_list.index == 1
            await pilot.press("up")  # older messages
            assert chat_list.index == 0
            await pilot.press("down")
            assert app.focused is chat_list
            assert chat_list.index == 1
            await pilot.press("down")  # past the newest: back to the entry
            assert app.focused is chat_input
            assert chat_input.text == "half typed"
            assert chat_input.cursor_location == (0, len("half typed"))

    async def test_up_in_an_empty_log_stays_in_the_entry(self, hpca_home):
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await pilot.press("up")
            assert app.focused.id == "chat-input"


class RecordingLLM(FakeLLM):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append(list(messages))
        return await super().chat(messages, json_schema=json_schema, **kwargs)


async def test_profile_memories_injected_into_system_prompt(hpca_home):
    from hpca.profiles import MemoryScope, Profile

    profile = Profile.load("default")
    profile.add_memory("The cluster is called cubi.", scope=MemoryScope.SYSTEM_PROMPT)
    profile.add_memory("User prefers verbose logs.", scope=MemoryScope.SYSTEM_PROMPT)
    profile.save()

    llm = RecordingLLM([respond_json("ok")])
    app = HpcaApp(llm=llm)
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "hello")
        system = llm.calls[0][0]
        assert system["role"] == "system"
        assert "cubi" in system["content"]
        assert "verbose logs" in system["content"]


async def test_memories_are_frozen_per_session_and_refresh_at_boundaries(hpca_home):
    """Redesign Phase 1: the memory snapshot is frozen per session so the
    system-prompt prefix stays byte-stable for the backend's prefix cache.
    A note written by another session or instance mid-session does NOT shift
    the prompt; it is picked up at the next session boundary."""
    from hpca.profiles import MemoryScope, Profile

    llm = RecordingLLM(
        [respond_json("ok"), respond_json("ok again"), respond_json("ok third")]
    )
    app = HpcaApp(llm=llm)
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "hello")
        assert "STAR needs 40G" not in llm.calls[0][0]["content"]

        # another instance (or session) memorizes something
        profile = Profile.load("default")
        profile.add_memory(
            "STAR needs 40G on this cluster.", scope=MemoryScope.SYSTEM_PROMPT
        )
        profile.save()

        # mid-session the frozen snapshot keeps the prompt stable
        seen = len(llm.calls)
        await submit_chat(app, pilot, "hello again")
        assert not any(
            call[0]["role"] == "system" and "STAR needs 40G" in call[0]["content"]
            for call in llm.calls[seen:]
        )

        # reopening the session is a boundary: the note is picked up
        await app.open_session(app.active_session)
        seen = len(llm.calls)
        await submit_chat(app, pilot, "and again")
        assert any(
            call[0]["role"] == "system" and "STAR needs 40G" in call[0]["content"]
            for call in llm.calls[seen:]
        )
        # and the refreshed per-session memory snapshot carries it too
        assert "STAR needs 40G" in app.profile_memory.scope_text(
            MemoryScope.SYSTEM_PROMPT
        )


async def test_turns_are_indexed_and_recallable_across_sessions(hpca_home):
    """Redesign Phase 2: user/assistant turns land in the episodic index and
    a later session can recall them with session_search — no model call."""
    llm = RecordingLLM([respond_json("STAR needs 40G on this cluster")])
    app = HpcaApp(llm=llm)
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "how much memory does STAR need")
        first_session = app.active_session

        # a fresh session recalls the earlier one
        hits = app.episodic.search("STAR", profile="default")
        assert len(hits) == 1
        assert hits[0].session_id == first_session.session_id
        assert hits[0].goal == "how much memory does STAR need"
        assert hits[0].resolution == "STAR needs 40G on this cluster"

        # deleting the session forgets its transcript from search too
        await app._delete_session(first_session)
        assert app.episodic.search("STAR", profile="default") == []


async def test_tool_traffic_is_not_indexed(hpca_home):
    """Tool results ride the user role; indexing them would drown BM25 in
    tool vocabulary."""
    llm = RecordingLLM(
        [tool_json("delete", target="scratch"), respond_json("removed it")]
    )
    app = HpcaApp(llm=llm, tools=destructive_tools())
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "please delete scratch")
        await pilot.pause()
        if decision_bar(app).display:
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
        rows = app.episodic.window(app.active_session.session_id)
        contents = [r["content"] for r in rows]
        assert "please delete scratch" in contents
        assert not any("[tool result]" in c for c in contents)


class UsageLLM(FakeLLM):
    """A backend that reports token usage, as vLLM does."""

    def __init__(self, outputs, prompt_tokens=1234, window=32_000):
        super().__init__(outputs)
        self._prompt_tokens = prompt_tokens
        self._window = window

    async def chat(self, messages, *, json_schema=None, **kwargs):
        response = await super().chat(messages, json_schema=json_schema, **kwargs)
        response.usage = {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": 20,
        }
        return response

    async def context_window(self):
        return self._window


async def test_context_bar_reports_measured_usage(hpca_home):
    """The bar shows the backend's own prompt_tokens, not an estimate."""
    from hpca.tui.context_bar import ContextBar

    app = HpcaApp(llm=UsageLLM([respond_json("ok")], prompt_tokens=8000))
    async with app.run_test(size=(120, 40)) as pilot:
        bar = app.query_one("#context-bar", ContextBar)
        assert "no reply yet" in bar.text  # nothing measured before the turn
        await submit_chat(app, pilot, "hello")
        assert "8,000 / 32,000" in bar.text
        assert "(25%)" in bar.text
        assert "~" not in bar.text  # measured, so not marked as an estimate


async def test_context_window_discovered_from_the_backend(hpca_home):
    """No configuration needed: the backend says how it was launched."""
    app = HpcaApp(llm=UsageLLM([respond_json("ok")], window=32_768))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app._active_max_model_len() == 32_768


async def test_a_small_window_turns_the_bar_red(hpca_home):
    """The case this exists for: on 32k a long turn saturates the window."""
    from hpca.tui.context_bar import ContextBar

    app = HpcaApp(llm=UsageLLM([respond_json("ok")], prompt_tokens=30_000))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "hello")
        bar = app.query_one("#context-bar", ContextBar)
        assert bar.has_class("context-danger")
        assert "(94%)" in bar.text


async def test_reopening_a_session_estimates_before_the_next_reply(hpca_home):
    """A long session should show it is nearly full before you send, not
    after the reply that overflows it."""
    from hpca.tui.context_bar import ContextBar

    app = HpcaApp(llm=UsageLLM([respond_json("ok")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "hello")
        session = app.active_session
        # ~30k tokens at 4 chars/token, i.e. 94% of the 32k window
        await app.graph.aupdate_state(
            {"configurable": {"thread_id": session.session_id}},
            {"messages": [{"role": "user", "content": "x" * 120_000}]},
        )
        await app.close_session()
        bar = app.query_one("#context-bar", ContextBar)
        assert "no reply yet" in bar.text  # leaving clears the old number
        await app.open_session(session)
        assert "~" in bar.text  # estimated from the stored history
        assert bar.has_class("context-danger")


async def test_switching_session_does_not_show_the_previous_context(hpca_home):
    from hpca.tui.context_bar import ContextBar

    app = HpcaApp(llm=UsageLLM([respond_json("a"), respond_json("b")]))
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "first")
        bar = app.query_one("#context-bar", ContextBar)
        assert "1,234" in bar.text or "8,000" in bar.text or "/" in bar.text
        await app.start_new_session()
        assert "no reply yet" in bar.text
