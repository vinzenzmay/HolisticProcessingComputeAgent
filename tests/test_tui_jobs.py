"""Tests for job wiring in the TUI: tool registration and the sacct poll.

The right column used to list the session's submitted jobs, with (k) to cancel
and enter to inspect, and most of this file tested that. The column holds only
watch boxes now — a job worth keeping an eye on is a job worth `watch_job`, and
the box says more than the row ever did. What is left here is the wiring that
outlived the rows: whether the job tools are registered at all, and whether the
poll writes what sacct says back to the store.
"""

import json

import pytest

from hpca.llm import ChatResponse
from hpca.slurm import SlurmClient
from hpca.tui.app import ChatInput, HpcaApp


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


class FakeRun:
    """Scripted sacct/squeue output, with the last answer left standing.

    A running app polls jobs on its own 5s timer, so a test that also calls
    poll_jobs by hand can see one more sacct call than it scripted — which
    happens only when the steps before it take longer than the interval, i.e.
    under load. Repeating the last response keeps that extra call from running
    the script dry (poll_jobs swallows the IndexError as "polling failed" and
    leaves the state untouched). An empty script still raises: a test that
    scripts nothing means slurm is not supposed to be called at all.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []
        self._last = None

    async def __call__(self, argv):
        self.calls.append(argv)
        if self.responses:
            self._last = self.responses.pop(0)
        elif self._last is None:
            raise IndexError(f"FakeRun: no scripted response for {argv}")
        return self._last


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
    await pilot.pause()


class TestJobTools:
    async def test_job_tools_registered_when_slurm_available(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([]), slurm=SlurmClient(run=FakeRun([])))
        assert "submit_job" in app._tools.names()
        assert "cancel_job" in app._tools.names()

    async def test_job_tools_absent_without_slurm(self, hpca_home, monkeypatch):
        monkeypatch.setenv("PATH", str(hpca_home))  # no sbatch anywhere
        app = HpcaApp(llm=FakeLLM([]))
        assert "submit_job" not in app._tools.names()


class TestPollJobs:
    async def test_a_state_change_is_written_back(self, hpca_home):
        run = FakeRun(
            [(0, "27744534|RUNNING|0:0|00:01:00||25G|5-00:00:00\n", "")]
        )
        app = HpcaApp(llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=run))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            await app.poll_jobs()
            await pilot.pause()
            assert app.job_store.get("27744534").state == "RUNNING"


class TestTheColumnIgnoresJobs:
    async def test_a_submitted_job_gets_no_box_of_its_own(self, hpca_home):
        """Submitting used to add a row. It is in the chat log already, and a
        job the user wants on screen is one they can `watch_job`."""
        app = HpcaApp(
            llm=FakeLLM([respond_json()]), slurm=SlurmClient(run=FakeRun([]))
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session_and_add_job(app, pilot)
            await app.refresh_watchers()
            await pilot.pause()
            assert list(app.query_one("#watchers-list").children) == []
