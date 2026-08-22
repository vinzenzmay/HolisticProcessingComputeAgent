"""The lag probe's wiring into the row UI (specs-core-process.md §8).

The probe itself is tested in `test_looplag.py`; these are the same claims
`test_tui_looplag.py` makes of the Textual app, moved onto the front-end that
replaces it — because §8's whole point is a number taken before and after,
against the same yardstick, and an instrument that only exists on the side
being replaced measures nothing.

The claims are `specs-ui-acceptance.md`, "Lag instrumentation": off unless
`$HPCA_LOOPLAG`, disabled means no task is ever started, a spike names the
running step, and a run leaves a report block behind.
"""

from __future__ import annotations

import asyncio
import io
import os

import pytest

from hpca.llm import ChatResponse
from hpca.looplag import LoopLagProbe
from hpca.ui import boot
from hpca.ui.app import RowUI
from hpca.ui.boot import start
from hpca.ui.run import Loop
from hpca.ui.screen import Screen
from hpca.ui.state import SessionState


class FakeLLM:
    """Enough of a backend for a run that never sends a turn."""

    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content="")

    async def supports_constrained_decoding(self):
        return True

    async def close(self):
        pass


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_LOCAL_DIR", str(tmp_path / "node-local"))
    return tmp_path / "node-local"


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    """A dev box may well have `sbatch` on it; a test must not care."""
    monkeypatch.setattr("hpca.ui.boot._detect_slurm", lambda settings: None)


def working(session_id: str, activity: str) -> SessionState:
    session = SessionState(session_id)
    session.start_turn()
    session.turn.activity_is(activity, "2026-08-21T10:00:00")
    return session


class TestGate:
    def test_it_is_off_unless_the_env_var_is_set(self, monkeypatch):
        monkeypatch.delenv("HPCA_LOOPLAG", raising=False)
        assert boot._lag_probe(RowUI()).enabled is False

    def test_the_env_var_turns_it_on(self, monkeypatch):
        monkeypatch.setenv("HPCA_LOOPLAG", "row ui")
        assert boot._lag_probe(RowUI()).enabled is True

    async def test_a_disabled_probe_never_starts_a_task(self, monkeypatch):
        # The reason it is safe to leave wired into the loop permanently.
        monkeypatch.delenv("HPCA_LOOPLAG", raising=False)
        probe = boot._lag_probe(RowUI())
        probe.start()
        assert probe.running is False


class TestLabel:
    def test_an_idle_app_blames_nothing(self):
        assert RowUI().activity_label() == "idle"

    def test_a_spike_names_the_running_step(self):
        ui = RowUI([working("s1", "running read_file")])
        assert ui.activity_label() == "running read_file"

    def test_a_background_turn_counts_too(self):
        # A turn in a session the user has left blocks this loop exactly as
        # hard as the open one, so it must be able to take the blame.
        ui = RowUI([working("s1", "LLM processing"), working("s2", "running rag")])
        assert ui.activity_label() == "LLM processing, running rag"

    def test_one_step_is_named_once(self):
        ui = RowUI([working("s1", "running run_bash"), working("s2", "running run_bash")])
        assert ui.activity_label() == "running run_bash"

    def test_a_session_that_finished_is_not_blamed(self):
        ui = RowUI([working("s1", "running read_file")])
        ui.sessions[0].end_turn()
        assert ui.activity_label() == "idle"


class TestTheLoop:
    """The wiring itself: the probe measures the loop it is given to."""

    async def running_loop(self, probe):
        read, write = os.pipe()
        screen = Screen(fd=read, out=io.StringIO())
        loop = Loop(RowUI(), screen, size=lambda: (80, 24), probe=probe)
        task = asyncio.ensure_future(loop.run())
        await asyncio.sleep(0.05)
        return task, write, read

    async def test_the_loop_starts_it_and_stops_it(self):
        probe = LoopLagProbe(interval_s=0.001)
        task, write, read = await self.running_loop(probe)
        assert probe.running is True, "it samples the loop it was given to"
        os.write(write, b"\x03")  # quit
        await asyncio.wait_for(task, 2.0)
        os.close(read), os.close(write)
        assert probe.running is False, "and leaves no task behind it"
        assert probe.samples, "having measured something in between"

    async def test_and_a_loop_with_no_probe_is_the_same_loop(self):
        task, write, read = await self.running_loop(None)
        os.write(write, b"\x03")
        assert await asyncio.wait_for(task, 2.0) == 0
        os.close(read), os.close(write)


class TestTheReport:
    """A run leaves a block behind, or the measurement cannot be compared."""

    async def run_once(self) -> None:
        read, write = os.pipe()
        screen = Screen(fd=read, out=io.StringIO())
        run = asyncio.ensure_future(
            start(llm=FakeLLM(), screen=screen, say=lambda text: None)
        )
        await asyncio.sleep(0.2)
        os.write(write, b"\x03")
        await asyncio.wait_for(run, 20.0)
        os.close(read), os.close(write)

    async def test_a_run_leaves_a_block_behind(self, home, local, monkeypatch):
        monkeypatch.setenv("HPCA_LOOPLAG", "row ui")
        await self.run_once()
        body = (home / "looplag.log").read_text()
        assert "=== looplag" in body
        assert "row ui" in body, "the env var names the run being compared"
        assert "samples" in body

    async def test_a_disabled_run_leaves_nothing(self, home, local, monkeypatch):
        monkeypatch.delenv("HPCA_LOOPLAG", raising=False)
        await self.run_once()
        assert not (home / "looplag.log").exists()
