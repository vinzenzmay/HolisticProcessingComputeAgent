"""The lag probe's wiring into the TUI (specs-core-process.md §8).

The probe itself is tested in test_looplag.py. What matters here is only that
the app switches it on when asked, leaves it off otherwise, names the running
step in a spike, and leaves a report block behind on the way out — because a
before/after measurement nobody can reproduce is worth nothing.
"""

import pytest

from hpca.tui.app import HpcaApp


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


class TestGate:
    def test_it_is_off_unless_the_env_var_is_set(self, hpca_home, monkeypatch):
        monkeypatch.delenv("HPCA_LOOPLAG", raising=False)
        assert HpcaApp()._looplag.enabled is False

    def test_the_env_var_turns_it_on(self, hpca_home, monkeypatch):
        monkeypatch.setenv("HPCA_LOOPLAG", "baseline main")
        assert HpcaApp()._looplag.enabled is True

    def test_a_disabled_probe_never_starts_a_task(self, hpca_home, monkeypatch):
        # The reason it is safe to leave wired in permanently.
        monkeypatch.delenv("HPCA_LOOPLAG", raising=False)
        app = HpcaApp()
        app._looplag.start()
        assert app._looplag.running is False


class TestLabel:
    def test_an_idle_app_blames_nothing(self, hpca_home):
        assert HpcaApp()._current_activity() == "idle"

    def test_a_spike_names_the_running_step(self, hpca_home):
        app = HpcaApp()
        app._turns = {"s1": _turn("running read_file")}
        assert app._current_activity() == "running read_file"

    def test_a_background_turn_counts_too(self, hpca_home):
        # A turn in a session the user has left blocks this loop exactly as
        # hard as the open one, so it must be able to take the blame.
        app = HpcaApp()
        app._turns = {
            "s1": _turn("LLM processing"),
            "s2": _turn("running index_docs"),
        }
        assert app._current_activity() == "LLM processing, running index_docs"

    def test_one_step_is_named_once(self, hpca_home):
        app = HpcaApp()
        app._turns = {"s1": _turn("running run_bash"), "s2": _turn("running run_bash")}
        assert app._current_activity() == "running run_bash"

    def test_it_survives_being_called_before_the_app_is_built(self, hpca_home):
        # The probe starts first thing in on_mount, so the label hook can be
        # called against a half-built app; _turns may not exist yet.
        app = HpcaApp()
        del app._turns
        assert app._current_activity() == "idle"


class TestReport:
    def test_a_run_leaves_a_block_behind(self, hpca_home, monkeypatch):
        monkeypatch.setenv("HPCA_LOOPLAG", "baseline main")
        app = HpcaApp()
        app._looplag.record(0.31)
        assert app._looplag.write_report(hpca_home / "looplag.log", note="baseline")
        body = (hpca_home / "looplag.log").read_text()
        assert "310.0ms" in body and "baseline" in body

    def test_a_disabled_run_leaves_nothing(self, hpca_home, monkeypatch):
        monkeypatch.delenv("HPCA_LOOPLAG", raising=False)
        app = HpcaApp()
        assert app._looplag.write_report(hpca_home / "looplag.log") is False
        assert not (hpca_home / "looplag.log").exists()


class _turn:
    """Stand-in for a TurnState: the label hook reads only `.activity`."""

    def __init__(self, activity: str) -> None:
        self.activity = activity
