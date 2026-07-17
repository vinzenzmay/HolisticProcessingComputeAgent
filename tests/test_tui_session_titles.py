"""Session titles: written by the model, renamed by hand, or re-asked for."""

import json

import pytest
from textual.widgets import Input, ListView

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
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
