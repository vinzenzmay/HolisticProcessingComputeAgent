"""TUI tests: quit confirmation, manage-LLMs screen, backend switcher."""

import json

import pytest
from textual.widgets import ListView

from hpca.config import LLMBackend, Settings
from hpca.discover import DiscoveredBackend
from hpca.llm import ChatResponse
from hpca.tui import manage_llms as manage_module
from hpca.tui import switch_llm as switch_module
from hpca.tui.app import HpcaApp
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.manage_llms import ManageLLMsScreen
from hpca.tui.switch_llm import SwitchLLMScreen


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


QWEN = DiscoveredBackend(
    base_url="http://localhost:51941/v1",
    model="Qwen/Qwen3.6-27B-FP8",
    max_model_len=192000,
)
MINI = DiscoveredBackend(
    base_url="http://localhost:51943/v1",
    model="sentence-transformers/all-MiniLM-L6-v2",
    max_model_len=256,
)


@pytest.fixture
def fake_discovery(monkeypatch):
    async def fake_scan(*args, **kwargs):
        return [QWEN, MINI]

    async def fake_reachable(base_url, **kwargs):
        return "51941" in base_url  # qwen up, everything else down

    monkeypatch.setattr(manage_module, "scan_local_ports", fake_scan)
    monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
    monkeypatch.setattr(switch_module, "is_reachable", fake_reachable)


class TestQuitConfirm:
    async def test_ctrl_q_asks_and_n_stays(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+q")
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert app.is_running

    async def test_ctrl_q_then_y_quits(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+q")
            await pilot.press("y")
            await pilot.pause()
        assert app.return_code == 0


class TestManageScreen:
    async def test_m_opens_and_scan_populates_left(self, hpca_home, fake_discovery):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManageLLMsScreen)
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            left = app.screen.query_one("#llm-discovered", ListView)
            labels = [str(item.query_one("Label").content) for item in left.children]
            assert any("Qwen/Qwen3.6-27B-FP8" in t for t in labels)
            assert any("ctx 192k" in t for t in labels)
            assert any("localhost:51941" in t for t in labels)

    async def test_enter_on_left_configures_and_persists(
        self, hpca_home, fake_discovery
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            left = app.screen.query_one("#llm-discovered", ListView)
            left.focus()
            left.index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.settings.backends) == 1
            assert app.settings.backends[0].model == QWEN.model
            # persisted to disk
            assert len(Settings.load().backends) == 1
            # no longer offered on the left
            assert len(left.children) == 1

    async def test_configured_shows_connection_state(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [
            LLMBackend(model=QWEN.model, base_url=QWEN.base_url, max_model_len=192000),
            LLMBackend(model=MINI.model, base_url=MINI.base_url, max_model_len=256),
        ]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            right = app.screen.query_one("#llm-configured", ListView)
            labels = [str(item.query_one("Label").content) for item in right.children]
            qwen_line = next(t for t in labels if "Qwen" in t)
            mini_line = next(t for t in labels if "MiniLM" in t)
            assert "● connected" in qwen_line
            assert "○ disconnected" in mini_line

    async def test_enter_on_right_sets_default(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [
            LLMBackend(model=QWEN.model, base_url=QWEN.base_url, max_model_len=192000)
        ]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            right = app.screen.query_one("#llm-configured", ListView)
            right.focus()
            right.index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.settings.llm.model == QWEN.model
            assert app.settings.llm.base_url == QWEN.base_url
            assert Settings.load().llm.model == QWEN.model
            labels = [str(i.query_one("Label").content) for i in right.children]
            assert any("★" in t for t in labels)

    async def test_r_removes_configured(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [
            LLMBackend(model=QWEN.model, base_url=QWEN.base_url)
        ]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            right = app.screen.query_one("#llm-configured", ListView)
            right.focus()
            right.index = 0
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            assert QWEN.model in app.screen._question
            await pilot.press("y")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert app.settings.backends == []
            assert Settings.load().backends == []

    async def test_remove_denied_keeps_backend(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [LLMBackend(model=QWEN.model, base_url=QWEN.base_url)]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            right = app.screen.query_one("#llm-configured", ListView)
            right.focus()
            right.index = 0
            await pilot.press("r")
            await pilot.press("n")
            await pilot.pause()
            assert len(app.settings.backends) == 1

    async def test_remove_inert_from_left_panel(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [LLMBackend(model=QWEN.model, base_url=QWEN.base_url)]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            app.screen.query_one("#llm-discovered", ListView).focus()
            await pilot.press("r")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert len(app.settings.backends) == 1

    async def test_arrow_keys_switch_panels(self, hpca_home, fake_discovery):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert app.focused.id == "llm-discovered"
            await pilot.press("right")
            assert app.focused.id == "llm-configured"
            await pilot.press("left")
            assert app.focused.id == "llm-discovered"

    async def test_m_only_from_sessions_column(self, hpca_home, fake_discovery):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app._focus_column("chat")
            await pilot.pause()
            assert app.check_action("manage_llms", ()) is False
            await pilot.press("m")
            await pilot.pause()
            assert not isinstance(app.screen, ManageLLMsScreen)
            app._focus_column("sessions")
            await pilot.pause()
            assert app.check_action("manage_llms", ()) is True
            await pilot.press("m")
            assert isinstance(app.screen, ManageLLMsScreen)
            # inside the manager, (m) is no longer offered
            assert app.check_action("manage_llms", ()) is False

    async def test_footer_offers_add_only_on_discovered_entry(
        self, hpca_home, fake_discovery
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            left = app.screen.query_one("#llm-discovered", ListView)
            right = app.screen.query_one("#llm-configured", ListView)
            assert left.check_action("select_cursor", ()) is True  # add llm to list
            assert right.check_action("select_cursor", ()) is False  # nothing there
            assert right.check_action("remove_llm", ()) is False

    async def test_escape_closes(self, hpca_home, fake_discovery):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ManageLLMsScreen)


class TestSwitcher:
    async def test_l_without_backends_warns(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app._focus_column("chat")
            await pilot.press("l")
            await pilot.pause()
            assert not isinstance(app.screen, SwitchLLMScreen)

    async def test_l_only_available_in_chat_column(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [LLMBackend(model=QWEN.model, base_url=QWEN.base_url)]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            # sessions column focused at start: l is hidden and inert
            assert app.check_action("switch_llm", ()) is False
            await pilot.press("l")
            await pilot.pause()
            assert not isinstance(app.screen, SwitchLLMScreen)
            app._focus_column("chat")
            await pilot.pause()
            assert app.check_action("switch_llm", ()) is True
            await pilot.press("l")
            await pilot.pause()
            assert isinstance(app.screen, SwitchLLMScreen)

    async def test_switch_updates_llm_and_topbar(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [
            LLMBackend(model=QWEN.model, base_url=QWEN.base_url, max_model_len=192000)
        ]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            old_graph = app.graph
            app._focus_column("chat")
            await pilot.press("l")
            assert isinstance(app.screen, SwitchLLMScreen)
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.settings.llm.model == QWEN.model
            assert app.graph is not old_graph
            from hpca.tui.app import TopBar

            assert QWEN.model in app.query_one(TopBar).render_text()
            assert Settings.load().llm.model == QWEN.model

    async def test_switch_escape_changes_nothing(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [LLMBackend(model=QWEN.model, base_url=QWEN.base_url)]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            before = app.settings.llm.model
            app._focus_column("chat")
            await pilot.press("l")
            await pilot.press("escape")
            await pilot.pause()
            assert app.settings.llm.model == before
