"""Session titles: written by the model, renamed by hand, or re-asked for."""

import asyncio
import json

import pytest
from textual.widgets import Input, Label, ListView

from hpca.llm import ChatResponse
from hpca.tui.app import (
    ChatInput,
    HpcaApp,
    WORKING_MARK,
    WorkingIndicator,
)
from hpca.tui.rename_screen import RenameScreen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    """Queued decisions, plus a title whenever the app asks for one."""

    def __init__(self, outputs, titles=("a test session",)):
        self._outputs = list(outputs)
        self._titles = list(titles)
        self.title_calls = 0

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            self.title_calls += 1
            title = self._titles.pop(0) if len(self._titles) > 1 else self._titles[0]
            return ChatResponse(content=json.dumps({"title": title}))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def session_rows(app):
    """The titles as the sessions column shows them."""
    rows = app.query_one("#sessions-list", ListView).children
    return [str(row.query_one("Label").content) for row in rows]


class TestAutoTitle:
    async def test_column_shows_the_summary_not_the_raw_message(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([respond_json("Four BAMs.")], titles=["Cohort BAM inventory"])
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(
                app, pilot, "which BAM files are sitting in the cohort dir?"
            )
            assert session_rows(app) == ["(new session)", "Cohort BAM inventory"]

    async def test_titled_once_per_session(self, hpca_home):
        llm = FakeLLM([respond_json("a"), respond_json("b")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            await submit_chat(app, pilot, "second")
            assert llm.title_calls == 1

    async def test_each_new_session_is_titled(self, hpca_home):
        llm = FakeLLM([respond_json("a"), respond_json("b")], titles=["one", "two"])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            await app.start_new_session()
            await pilot.pause()
            await submit_chat(app, pilot, "second")
            assert llm.title_calls == 2
            assert sorted(session_rows(app)[1:]) == ["one", "two"]

    async def test_reopened_session_is_not_retitled(self, hpca_home):
        llm = FakeLLM([respond_json("a"), respond_json("b")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            session = app.active_session
            await app.start_new_session()
            await pilot.pause()
            await app.open_session(session)
            await pilot.pause()
            await submit_chat(app, pilot, "another question")
            assert llm.title_calls == 1  # it was named when it was new

    async def test_a_slash_command_alone_does_not_title(self, hpca_home):
        llm = FakeLLM([])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = r"\memorize something"
            await pilot.press("enter")
            await pilot.pause()
            assert llm.title_calls == 0


class TestManualRename:
    async def test_r_opens_the_dialog_prefilled_and_saves(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, RenameScreen)
            name = app.screen.query_one("#rename-input", Input)
            assert name.value == "a test session"  # editable, not retyped
            name.value = "STAR OOM investigation"
            await pilot.press("enter")
            await pilot.pause()
            assert app.active_session.title == "STAR OOM investigation"
            assert session_rows(app)[1] == "STAR OOM investigation"
            saved = app.session_store.list(profile="default")[0]
            assert saved.title == "STAR OOM investigation"

    async def test_escape_keeps_the_old_name(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            before = app.active_session.title
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("r")
            await pilot.press("escape")
            await pilot.pause()
            assert app.active_session.title == before

    async def test_an_empty_name_is_refused(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            before = app.active_session.title
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("r")
            await pilot.pause()
            app.screen.query_one("#rename-input", Input).value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert app.active_session.title == before

    async def test_a_hand_written_name_is_not_overwritten(self, hpca_home):
        llm = FakeLLM([respond_json("a")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()  # named by hand before the first message
            await pilot.pause()
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("r")
            await pilot.pause()
            app.screen.query_one("#rename-input", Input).value = "my own name"
            await pilot.press("enter")
            await pilot.pause()

            await submit_chat(app, pilot, "hello")
            assert app.active_session.title == "my own name"
            assert llm.title_calls == 0  # the model is not asked to second-guess


class TestLLMRetitle:
    async def test_t_asks_the_model_for_a_new_title(self, hpca_home):
        llm = FakeLLM([respond_json("a")], titles=["first name", "second name"])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            assert app.active_session.title == "first name"
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.active_session.title == "second name"
            assert session_rows(app)[1] == "second name"

    async def test_rename_keys_are_inert_on_the_new_session_row(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 0  # "(new session)" is not a session
            assert sessions_list.check_action("rename_session", ()) is False
            assert sessions_list.check_action("retitle_session", ()) is False
            await pilot.press("r")
            await pilot.pause()
            assert not isinstance(app.screen, RenameScreen)
            sessions_list.index = 1
            assert sessions_list.check_action("rename_session", ()) is True

    async def test_a_failed_title_leaves_the_name_alone(self, hpca_home):
        class BrokenTitler(FakeLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if is_title_request(json_schema):
                    raise RuntimeError("backend fell over")
                return await super().chat(messages, json_schema=json_schema, **kwargs)

        app = HpcaApp(llm=BrokenTitler([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            assert app.active_session.title == "hello"  # the placeholder stands
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("t")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.active_session.title == "hello"
            assert app.is_running  # reported, not crashed

    async def test_retitling_the_open_session_shows_the_chat_spinner(self, hpca_home):
        """Retitling the session the user is looking at parks the chat-bottom
        spinner, labelled "writing a title", until the call returns."""
        gate = asyncio.Event()

        class GatedTitler(FakeLLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.armed = False  # the opening auto-title must not block submit

            async def chat(self, messages, *, json_schema=None, **kwargs):
                if is_title_request(json_schema) and self.armed:
                    await gate.wait()
                return await super().chat(messages, json_schema=json_schema, **kwargs)

        llm = GatedTitler([respond_json("a")], titles=["first name", "second name"])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            assert app.active_session.title == "first name"
            llm.armed = True
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("t")
            for _ in range(50):  # let the worker park on the gated title call
                if app.query(WorkingIndicator):
                    break
                await pilot.pause()
            indicators = app.query(WorkingIndicator)
            assert indicators, "the open session should show the chat spinner"
            assert indicators.first(WorkingIndicator).activity == "writing a title"
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.query(WorkingIndicator)
            assert app.active_session.title == "second name"

    async def test_retitling_a_background_session_lights_the_row_not_the_chat(
        self, hpca_home
    ):
        """Retitling a session that is highlighted but NOT open lights its
        sidebar row glyph — never the chat spinner, which belongs to the open
        session the user is looking at."""
        gate = asyncio.Event()

        class GatedTitler(FakeLLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.armed = False

            async def chat(self, messages, *, json_schema=None, **kwargs):
                if is_title_request(json_schema) and self.armed:
                    await gate.wait()
                return await super().chat(messages, json_schema=json_schema, **kwargs)

        llm = GatedTitler(
            [respond_json("a"), respond_json("b")],
            titles=["A first", "B first", "A second"],
        )
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            first = app.active_session
            await app.start_new_session()
            await pilot.pause()
            await submit_chat(app, pilot, "second")
            second = app.active_session
            llm.armed = True

            def row_for(session):
                for item in app.query_one("#sessions-list", ListView).children:
                    rs = getattr(item, "data_session", None)
                    if rs is not None and rs.session_id == session.session_id:
                        return item
                return None

            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = next(
                i
                for i, item in enumerate(sessions_list.children)
                if getattr(item, "data_session", None)
                and item.data_session.session_id == first.session_id
            )
            # A is highlighted, B is still the open/active session.
            assert app.active_session.session_id == second.session_id
            assert app._highlighted_session().session_id == first.session_id

            await pilot.press("t")
            for _ in range(50):  # let the worker park on the gated title call
                if first.session_id in app._busy_sessions:
                    break
                await pilot.pause()
            # No chat spinner: A isn't the open chat.
            assert not app.query(WorkingIndicator)
            row = row_for(first)
            assert row.has_class("session-working")
            assert str(row.query_one("Label").content).startswith(WORKING_MARK)

            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not row_for(first).has_class("session-working")
            assert not str(row_for(first).query_one("Label").content).startswith(
                WORKING_MARK
            )
            assert app.session_store.get(first.session_id).title == "A second"


class TestTitleLogging:
    async def test_titling_and_renaming_are_logged(self, hpca_home, tmp_path):
        from hpca.config import Settings

        directory = tmp_path / "logs"
        settings = Settings()
        settings.logging.dir = str(directory)
        settings.save()
        app = HpcaApp(llm=FakeLLM([respond_json("a")], titles=["Cohort BAMs"]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            text = list(directory.glob("*.log"))[0].read_text()
            assert "subagent:title query" in text  # the titler is a sub-agent
            assert "subagent:title reply" in text
            assert "session renamed\nCohort BAMs (by llm)" in text


class TestDeleteSession:
    async def test_d_asks_then_deletes_and_keeps_the_log(self, hpca_home, tmp_path):
        from hpca.config import Settings
        from hpca.logs import log_path
        from hpca.tui.confirm_screen import ConfirmScreen

        directory = tmp_path / "chatlogs"
        settings = Settings()
        settings.logging.dir = str(directory)
        settings.save()
        app = HpcaApp(llm=FakeLLM([respond_json("a")], titles=["Cohort BAMs"]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "which BAMs?")
            session = app.active_session
            log = log_path(directory, session)
            assert log.exists()

            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            assert "Cohort BAMs" in app.screen._question
            assert "log" in app.screen._question  # the dialog says the log stays

            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.session_store.list(profile="default") == []
            assert session_rows(app) == ["(new session)"]
            # the transcript is the durable record; it outlives the session
            assert log.exists()
            assert "which BAMs?" in log.read_text()

    async def test_declining_keeps_the_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("d")
            await pilot.press("n")
            await pilot.pause()
            assert len(app.session_store.list(profile="default")) == 1

    async def test_deleting_the_open_session_empties_the_chat(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            assert app.query_one("#chat-input", ChatInput).display
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("d")
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.active_session is None
            assert app.chat_log_texts() == []
            # nothing to type into once the session it belonged to is gone
            assert not app.query_one("#chat-input", ChatInput).display
            assert app.focused_column_id == "sessions"

    async def test_deleting_another_session_leaves_the_open_one_alone(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "first")
            first = app.active_session
            await app.start_new_session()
            await pilot.pause()
            await submit_chat(app, pilot, "second")
            second = app.active_session

            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = [
                i
                for i, item in enumerate(sessions_list.children)
                if getattr(item, "data_session", None)
                and item.data_session.session_id == first.session_id
            ][0]
            await pilot.press("d")
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.active_session.session_id == second.session_id
            assert app.query_one("#chat-input", ChatInput).display
            assert [s.session_id for s in app.session_store.list(profile="default")] == [
                second.session_id
            ]

    async def test_the_chat_history_is_dropped_with_the_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("an answer")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "a question")
            session = app.active_session
            snapshot = await app.graph.aget_state(
                {"configurable": {"thread_id": session.session_id}}
            )
            assert (snapshot.values or {}).get("messages")

            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 1
            await pilot.press("d")
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            after = await app.graph.aget_state(
                {"configurable": {"thread_id": session.session_id}}
            )
            assert not (after.values or {}).get("messages")

    async def test_delete_is_inert_on_the_new_session_row(self, hpca_home):
        from hpca.tui.confirm_screen import ConfirmScreen

        app = HpcaApp(llm=FakeLLM([respond_json("a")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "hello")
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 0  # "(new session)" is not a session
            assert sessions_list.check_action("delete_session", ()) is False
            await pilot.press("d")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert len(app.session_store.list(profile="default")) == 1
