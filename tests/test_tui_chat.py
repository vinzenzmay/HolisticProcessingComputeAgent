"""Tests for the chat wiring: input → agent worker → chat log, sessions, approval."""

import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import ListView

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.approval_screen import ApprovalScreen


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
    async def test_destructive_tool_opens_modal(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "delete results")
            assert isinstance(app.screen, ApprovalScreen)
            rendered = app.screen.details_text()
            assert "delete" in rendered
            assert "results/" in rendered

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
            await pilot.press("enter")
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
    from hpca.profiles import Profile

    profile = Profile.load("default")
    profile.add_memory("The cluster is called cubi.", tier=1)
    profile.add_memory("User prefers verbose logs.", tier=2)
    profile.save()

    llm = RecordingLLM([respond_json("ok")])
    app = HpcaApp(llm=llm)
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "hello")
        system = llm.calls[0][0]
        assert system["role"] == "system"
        assert "cubi" in system["content"]
        assert "verbose logs" in system["content"]


async def test_memories_written_after_startup_reach_the_next_turn(hpca_home):
    """Memories are shared through the profile file: a note made in another
    session or another running hpca instance must be in this turn's prompt,
    not only what was on disk when this instance started."""
    from hpca.profiles import Profile

    llm = RecordingLLM([respond_json("ok"), respond_json("ok again")])
    app = HpcaApp(llm=llm)
    async with app.run_test(size=(120, 40)) as pilot:
        await submit_chat(app, pilot, "hello")
        assert "STAR needs 40G" not in llm.calls[0][0]["content"]

        # another instance (or session) memorizes something
        profile = Profile.load("default")
        profile.add_memory("STAR needs 40G on this cluster.", tier=1)
        profile.save()

        await submit_chat(app, pilot, "hello again")
        # the first turn predates the memory, so any prompt carrying it is
        # from the second turn (calls include titler traffic; search them all)
        assert any(
            call[0]["role"] == "system" and "STAR needs 40G" in call[0]["content"]
            for call in llm.calls
        )
        # and the tool context the turn carries has it too
        assert "STAR needs 40G" in app._tool_ctx.tier1_text
