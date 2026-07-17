"""Tests for job wiring in the TUI: right column, poller, cancel via (k)."""

import json

import pytest
from textual.widgets import ListView

from hpca.llm import ChatResponse
from hpca.slurm import SlurmClient
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


class FakeRun:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    async def __call__(self, argv):
        self.calls.append(argv)
        return self.responses.pop(0)


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def open_session_and_add_job(app, pilot, job_id="27744534"):
    await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "hello"
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    app.job_store.add(
        job_id=job_id,
        kind="sbatch",
        session_id=app.active_session.session_id,
        profile="default",
        script_key="my_script",
        stdout_path=f"/logs/{job_id}.out",
        stderr_path=f"/logs/{job_id}.err",
    )
    await app.refresh_processes()
    await pilot.pause()


class TestJobsInRightColumn:
    async def test_job_row_listed(self, hpca_home):
        run = FakeRun([])
        app = HpcaApp(llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=run))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            items = app.query_one("#processes-list", ListView).children
            assert len(items) == 1
            assert getattr(items[0], "data_job").job_id == "27744534"

    async def test_job_tools_registered_when_slurm_available(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]), slurm=SlurmClient(run=FakeRun([])))
        assert "submit_job" in app._tools.names()
        assert "cancel_job" in app._tools.names()

    async def test_job_tools_absent_without_slurm(self, hpca_home, monkeypatch):
        monkeypatch.setenv("PATH", str(hpca_home))  # no sbatch anywhere
        app = HpcaApp(llm=FakeLLM([]))
        assert "submit_job" not in app._tools.names()


class TestPollJobs:
    async def test_state_change_updates_column(self, hpca_home):
        run = FakeRun(
            [(0, "27744534|RUNNING|0:0|00:01:00||25G|5-00:00:00\n", "")]
        )
        app = HpcaApp(llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=run))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            await app.poll_jobs()
            await pilot.pause()
            assert app.job_store.get("27744534").state == "RUNNING"
            items = app.query_one("#processes-list", ListView).children
            assert getattr(items[0], "data_job").state == "RUNNING"


class TestCancelJob:
    async def test_k_confirms_then_cancels(self, hpca_home):
        run = FakeRun([(0, "", "")])  # scancel
        app = HpcaApp(llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=run))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert run.calls[0][:2] == ["scancel", "27744534"]
            assert app.job_store.get("27744534").state == "CANCELLING"

    async def test_deny_leaves_job_alone(self, hpca_home):
        run = FakeRun([])
        app = HpcaApp(llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=run))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("k")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert run.calls == []
            assert app.job_store.get("27744534").state == "SUBMITTED"


class TestInspectJob:
    async def test_enter_shows_job_details(self, hpca_home):
        app = HpcaApp(
            llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=FakeRun([]))
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, InspectScreen)
            body = app.screen.body_text()
            assert "27744534" in body
            assert "my_script" in body
