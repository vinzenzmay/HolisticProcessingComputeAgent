"""TUI tests: quit confirmation, manage-LLMs screen, backend switcher."""

import json

import pytest
from textual.widgets import Input, ListView
from textual.widgets._toast import Toast

from hpca.config import LLMBackend, Settings
from hpca.discover import DiscoveredBackend
from hpca.llm import ChatResponse
from hpca.tui import backend_form as backend_form_module
from hpca.tui import manage_llms as manage_module
from hpca.tui import switch_llm as switch_module
from hpca.tui.app import HpcaApp
from hpca.tui.backend_form import BackendFormScreen
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
    base_url="http://localhost:20001/v1",
    model="Qwen/Qwen3.6-27B-FP8",
    max_model_len=192000,
)
MINI = DiscoveredBackend(
    base_url="http://localhost:20000/v1",
    model="sentence-transformers/all-MiniLM-L6-v2",
    max_model_len=256,
)
KEYED = DiscoveredBackend(
    base_url="http://localhost:51944/v1",
    model="(api key required)",
    needs_key=True,
)


@pytest.fixture
def fake_discovery(monkeypatch):
    async def fake_scan(*args, **kwargs):
        return [QWEN, MINI]

    async def fake_reachable(base_url, **kwargs):
        return "20001" in base_url  # qwen up, everything else down

    monkeypatch.setattr(manage_module, "scan_local_ports", fake_scan)
    monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
    monkeypatch.setattr(switch_module, "is_reachable", fake_reachable)


class TestQuitConfirm:
    """(q), and only from the sessions column: ctrl+q belongs to zellij."""

    async def test_q_asks_and_n_stays(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.focused_column_id == "sessions"
            await pilot.press("q")
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert app.is_running

    async def test_q_then_y_quits(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("q")
            await pilot.press("y")
            await pilot.pause()
        assert app.return_code == 0

    async def test_ctrl_q_does_nothing(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+q")  # zellij's key, and Textual's own
            await pilot.pause()
            assert app.is_running
            assert not isinstance(app.screen, ConfirmScreen)

    async def test_q_is_typed_in_the_chat_not_a_hotkey(self, hpca_home):
        from hpca.tui.app import ChatInput

        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            assert app.check_action("confirm_quit", ()) is False
            await pilot.press("q")
            await pilot.pause()
            assert app.query_one("#chat-input", ChatInput).text == "q"
            assert not isinstance(app.screen, ConfirmScreen)

    async def test_q_is_inert_on_the_processes_column(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app._focus_column("processes")
            await pilot.pause()
            assert app.check_action("confirm_quit", ()) is False
            await pilot.press("q")
            await pilot.pause()
            assert app.is_running
            assert not isinstance(app.screen, ConfirmScreen)

    async def test_q_does_not_quit_from_another_screen(self, hpca_home, fake_discovery):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert app.check_action("confirm_quit", ()) is False
            await pilot.press("q")
            await pilot.pause()
            assert app.is_running
            assert isinstance(app.screen, ManageLLMsScreen)


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
            assert any("localhost:20001" in t for t in labels)

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

    async def test_enter_on_right_does_not_set_a_default(
        self, hpca_home, fake_discovery
    ):
        # The LLM is chosen per session now; the configured panel has no
        # "set default" — enter on it does nothing and marks no ★.
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
            # the bootstrap default is untouched, and nothing is starred
            assert app.settings.llm.model == "qwen3-6b"
            labels = [str(i.query_one("Label").content) for i in right.children]
            assert not any("★" in t for t in labels)

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
            # the right panel no longer has a "set default" enter action; its
            # per-entry actions are inert with nothing highlighted
            assert right.check_action("remove_llm", ()) is False
            assert right.check_action("toggle_thinking", ()) is False

    async def test_escape_closes(self, hpca_home, fake_discovery):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ManageLLMsScreen)


class TestPortMemory:
    async def test_discovered_ports_remembered_and_persisted(
        self, hpca_home, fake_discovery
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert app.settings.known_llm_ports == [20001, 20000]
            assert Settings.load().known_llm_ports == [20001, 20000]

    async def test_remembered_ports_scanned_first(self, hpca_home, monkeypatch):
        settings = Settings()
        settings.known_llm_ports = [20001]
        settings.backends = [LLMBackend(model=MINI.model, base_url=MINI.base_url)]
        settings.save()
        scanned: list[list[int]] = []

        async def recording_scan(ports, **kwargs):
            scanned.append(list(ports))
            return []

        async def fake_reachable(base_url, **kwargs):
            return False

        monkeypatch.setattr(manage_module, "scan_local_ports", recording_scan)
        monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
        # remembered port first, then the configured backend's, then the rest
        assert scanned[0][:2] == [20001, 20000]
        assert len(scanned[0]) == 64512  # full range still covered

    async def test_port_remembered_when_backend_configured(
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
            assert 20001 in Settings.load().known_llm_ports


class TestSwitcher:
    async def test_ctrl_l_without_backends_warns(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app._focus_column("chat")
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert not isinstance(app.screen, SwitchLLMScreen)

    async def test_ctrl_l_only_available_in_chat_column(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [LLMBackend(model=QWEN.model, base_url=QWEN.base_url)]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            # sessions column focused at start: ctrl+l is hidden and inert
            assert app.check_action("switch_llm", ()) is False
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert not isinstance(app.screen, SwitchLLMScreen)
            await app.start_new_session()  # switching needs a session to switch
            await pilot.pause()
            assert app.focused_column_id == "chat"
            assert app.check_action("switch_llm", ()) is True
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert isinstance(app.screen, SwitchLLMScreen)

    async def test_switch_sets_the_sessions_backend_and_model_line(
        self, hpca_home, fake_discovery
    ):
        settings = Settings()
        settings.backends = [
            LLMBackend(model=QWEN.model, base_url=QWEN.base_url, max_model_len=192000)
        ]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            await pilot.press("ctrl+l")
            assert isinstance(app.screen, SwitchLLMScreen)
            await pilot.press("enter")
            await pilot.pause()
            # the session (not the global default) now points at QWEN
            assert QWEN.model in app.active_session.backend
            assert app.settings.llm.model == "qwen3-6b"  # bootstrap untouched
            from hpca.tui.app import TopBar
            from hpca.tui.context_bar import ModelLine

            # The model shows in the chat-column model line, not the top bar.
            assert QWEN.model in app.query_one(ModelLine).text
            assert QWEN.model not in app.query_one(TopBar).render_text()

    async def test_switch_escape_changes_nothing(self, hpca_home, fake_discovery):
        settings = Settings()
        settings.backends = [LLMBackend(model=QWEN.model, base_url=QWEN.base_url)]
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            before = app.settings.llm.model
            app._focus_column("chat")
            await pilot.press("ctrl+l")
            await pilot.press("escape")
            await pilot.pause()
            assert app.settings.llm.model == before


class TestScanResponsiveness:
    async def test_ui_responsive_and_closable_during_slow_scan(
        self, hpca_home, monkeypatch
    ):
        import asyncio

        async def slow_scan(*args, progress=None, **kwargs):
            await asyncio.sleep(0.6)
            return [QWEN]

        async def fake_reachable(base_url, **kwargs):
            return True

        monkeypatch.setattr(manage_module, "scan_local_ports", slow_scan)
        monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManageLLMsScreen)
            # while the scan runs, the UI must still process input:
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ManageLLMsScreen)
            # the late-finishing scan must not crash against the gone screen
            await asyncio.sleep(0.8)
            await pilot.pause()
            assert app.is_running


class TestIncrementalScanUI:
    async def test_left_panel_fills_while_scan_still_running(
        self, hpca_home, monkeypatch
    ):
        import asyncio

        async def streaming_scan(*args, progress=None, on_found=None, **kwargs):
            on_found(QWEN)  # found early in the sweep
            await asyncio.sleep(0.5)  # rest of the port range
            on_found(MINI)
            return [QWEN, MINI]

        async def fake_reachable(base_url, **kwargs):
            return True

        monkeypatch.setattr(manage_module, "scan_local_ports", streaming_scan)
        monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            left = app.screen.query_one("#llm-discovered", ListView)

            def scan_running():
                return any(
                    w.group == "llm-scan" and w.is_running
                    for w in app.screen.workers
                )

            for _ in range(50):  # wait for the early hit, max ~0.5s
                await pilot.pause()
                if len(left.children) > 0:
                    break
                await asyncio.sleep(0.01)
            assert scan_running(), "scan should still be sweeping"
            assert len(left.children) == 1  # QWEN visible before scan ends
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert len(left.children) == 2


class TestThinkingToggle:
    """Thinking is per backend, off by default, and announces its cost."""

    def configure(self, **overrides):
        settings = Settings()
        settings.backends = [
            LLMBackend(
                model=QWEN.model,
                base_url=QWEN.base_url,
                max_model_len=192000,
                **overrides,
            )
        ]
        settings.save()
        return settings

    async def focus_configured(self, app, pilot):
        await pilot.press("m")
        await app.screen.workers.wait_for_complete()
        await pilot.pause()
        right = app.screen.query_one("#llm-configured", ListView)
        right.focus()
        right.index = 0
        return right

    def labels(self, listview):
        return [str(item.query_one("Label").content) for item in listview.children]

    async def test_marker_shows_only_while_thinking_is_on(
        self, hpca_home, fake_discovery
    ):
        self.configure()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            right = await self.focus_configured(app, pilot)
            assert "thinking" not in self.labels(right)[0]  # quiet when off
            await pilot.press("t")
            await pilot.pause()
            assert "◆ thinking" in self.labels(right)[0]
            await pilot.press("t")
            await pilot.pause()
            assert "thinking" not in self.labels(right)[0]

    async def test_toggle_persists_on_the_backend(self, hpca_home, fake_discovery):
        self.configure()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await self.focus_configured(app, pilot)
            await pilot.press("t")
            await pilot.pause()
            assert app.settings.backends[0].enable_thinking is True
            assert Settings.load().backends[0].enable_thinking is True

    async def test_enabling_pops_up_that_thinking_is_not_better(
        self, hpca_home, fake_discovery
    ):
        self.configure()
        app = HpcaApp(llm=FakeLLM())
        # notifications are off in run_test by default; this is about the popup
        async with app.run_test(size=(120, 40), notifications=True) as pilot:
            await self.focus_configured(app, pilot)
            await pilot.press("t")
            for _ in range(8):  # notify -> call_later -> mount -> render
                await pilot.pause()
            toasts = [str(toast.render()) for toast in app.screen.query(Toast)]
            assert toasts, "enabling thinking must say so on screen"
            assert any("not generally better" in t for t in toasts)
            assert any("133s" in t for t in toasts)  # the measured cost

    async def test_disabling_does_not_warn(self, hpca_home, fake_discovery):
        self.configure(enable_thinking=True)
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40), notifications=True) as pilot:
            await self.focus_configured(app, pilot)
            await pilot.press("t")
            for _ in range(8):
                await pilot.pause()
            toasts = [str(toast.render()) for toast in app.screen.query(Toast)]
            assert not any("not generally better" in t for t in toasts)

    async def test_toggling_the_active_backend_reloads_the_client(
        self, hpca_home, fake_discovery
    ):
        settings = self.configure()
        settings.activate_backend(settings.backends[0])
        settings.save()
        app = HpcaApp()  # owns its client, so it can be rebuilt
        async with app.run_test(size=(120, 40)) as pilot:
            before = app._llm
            await self.focus_configured(app, pilot)
            await pilot.press("t")
            await pilot.pause()
            assert app.settings.llm.enable_thinking is True
            assert app._llm is not before  # the running client picked it up
            assert app._llm._settings.enable_thinking is True

    async def test_toggling_an_inactive_backend_leaves_the_client_alone(
        self, hpca_home, fake_discovery
    ):
        settings = self.configure()
        settings.llm.model = "something-else"  # a different backend is active
        settings.save()
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            before = app._llm
            await self.focus_configured(app, pilot)
            await pilot.press("t")
            await pilot.pause()
            assert app.settings.backends[0].enable_thinking is True
            assert app.settings.llm.enable_thinking is False
            assert app._llm is before

    async def test_thinking_follows_the_backend_when_switching(
        self, hpca_home, fake_discovery
    ):
        # A session's client carries its backend's thinking setting: switching
        # the session's backend gives it a client with that backend's thinking.
        settings = Settings()
        settings.backends = [
            LLMBackend(model=QWEN.model, base_url=QWEN.base_url, enable_thinking=True),
            LLMBackend(model=MINI.model, base_url=MINI.base_url),
        ]
        settings.save()
        app = HpcaApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.start_new_session()
            await pilot.pause()
            app.switch_backend(app.settings.backends[0])
            client = app._client_for(app.active_session)
            assert client._settings.enable_thinking is True
            app.switch_backend(app.settings.backends[1])
            client = app._client_for(app.active_session)
            assert client._settings.enable_thinking is False
            await pilot.pause()

    async def test_toggle_is_inert_without_a_configured_entry(
        self, hpca_home, fake_discovery
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            right = app.screen.query_one("#llm-configured", ListView)
            assert right.check_action("toggle_thinking", ()) is False
            left = app.screen.query_one("#llm-discovered", ListView)
            left.focus()
            await pilot.press("t")  # left panel: not a configured backend
            await pilot.pause()
            assert app.is_running


async def _wait_for_backend(app, pilot, count=1, tries=30):
    for _ in range(tries):
        await pilot.pause()
        if len(app.settings.backends) >= count:
            return
    raise AssertionError("backend was never saved")


@pytest.fixture
def fake_keyed_discovery(monkeypatch):
    """The scan surfaces one key-locked endpoint (as the 401 does)."""

    async def fake_scan(*args, **kwargs):
        return [KEYED]

    async def fake_reachable(base_url, **kwargs):
        return True

    monkeypatch.setattr(manage_module, "scan_local_ports", fake_scan)
    monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
    monkeypatch.setattr(switch_module, "is_reachable", fake_reachable)


@pytest.fixture
def fake_probe(monkeypatch):
    """probe_endpoint used by the form: 'goodkey' (or no key) sees the model."""

    async def probe(base_url, *, api_key=None, **kwargs):
        if api_key in (None, "goodkey"):
            return [
                DiscoveredBackend(
                    base_url=base_url, model="qwen3.6:35B", max_model_len=32768
                )
            ]
        return [
            DiscoveredBackend(
                base_url=base_url, model="(api key required)", needs_key=True
            )
        ]

    monkeypatch.setattr(backend_form_module, "probe_endpoint", probe)


class TestBackendForm:
    async def test_enter_on_keyed_endpoint_opens_form_with_url_locked(
        self, hpca_home, fake_keyed_discovery
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
            assert isinstance(app.screen, BackendFormScreen)
            url = app.screen.query_one("#backend-url", Input)
            assert url.value == KEYED.base_url
            assert url.disabled is True  # base_url came from the scan
            # nothing saved yet — the key hasn't been entered
            assert app.settings.backends == []

    async def test_key_and_autofill_saves_backend(
        self, hpca_home, fake_keyed_discovery, fake_probe
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
            # supply only the key; model + context auto-fill from the probe
            app.screen.query_one("#backend-key", Input).value = "goodkey"
            app.screen.query_one("#backend-model", Input).focus()
            await pilot.press("enter")
            await _wait_for_backend(app, pilot)
            saved = app.settings.backends[0]
            assert saved.api_key == "goodkey"
            assert saved.model == "qwen3.6:35B"  # auto-detected
            assert saved.max_model_len == 32768
            assert saved.base_url == KEYED.base_url
            assert Settings.load().backends[0].api_key == "goodkey"

    async def test_rejected_key_warns_and_keeps_input(
        self, hpca_home, fake_keyed_discovery, fake_probe
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
            app.screen.query_one("#backend-key", Input).value = "wrong"
            app.screen.query_one("#backend-model", Input).focus()
            await pilot.press("enter")
            for _ in range(10):
                await pilot.pause()
            assert isinstance(app.screen, BackendFormScreen)  # still open
            status = str(app.screen.query_one("#backend-status").render())
            assert "rejected" in status.lower()
            assert app.screen.query_one("#backend-key", Input).value == "wrong"
            assert app.settings.backends == []

    async def test_a_opens_blank_manual_form_and_saves(
        self, hpca_home, fake_keyed_discovery, fake_probe
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            app.screen.query_one("#llm-discovered", ListView).focus()
            await pilot.press("a")
            await pilot.pause()
            assert isinstance(app.screen, BackendFormScreen)
            url = app.screen.query_one("#backend-url", Input)
            assert url.disabled is False  # editable for a manual add
            url.value = "http://localhost:52000/v1"
            app.screen.query_one("#backend-key", Input).value = "goodkey"
            app.screen.query_one("#backend-url", Input).focus()
            await pilot.press("enter")
            await _wait_for_backend(app, pilot)
            saved = app.settings.backends[0]
            assert saved.base_url == "http://localhost:52000/v1"
            assert saved.model == "qwen3.6:35B"
            assert saved.api_key == "goodkey"


UNLOCKED = DiscoveredBackend(
    base_url="http://localhost:51945/v1",
    model="pool-unlocked-model",
    max_model_len=4096,
    needs_key=True,
    api_key="poolkey",  # a pooled key already validated against this endpoint
)


@pytest.fixture
def fake_pool_probe(monkeypatch):
    """probe_endpoint on the manage screen: 'goodkey' anywhere in the pool
    unlocks the real model; otherwise the sentinel comes back."""

    async def probe(base_url, *, api_key=None, api_keys=(), **kwargs):
        if api_key == "goodkey" or "goodkey" in tuple(api_keys):
            return [
                DiscoveredBackend(
                    base_url=base_url,
                    model="qwen3.6:35B",
                    max_model_len=32768,
                    needs_key=True,
                    api_key="goodkey",
                )
            ]
        return [
            DiscoveredBackend(
                base_url=base_url, model="(api key required)", needs_key=True
            )
        ]

    monkeypatch.setattr(manage_module, "probe_endpoint", probe)


def _left_labels(app):
    left = app.screen.query_one("#llm-discovered", ListView)
    return [str(item.query_one("Label").content) for item in left.children]


async def _wait_for_left_label(app, pilot, needle, tries=40):
    labels = []
    for _ in range(tries):
        await app.screen.workers.wait_for_complete()
        await pilot.pause()
        labels = _left_labels(app)
        if any(needle in t for t in labels):
            return labels
    return labels


class TestKeyRegistry:
    """Slice C: the key pool re-probes locked endpoints in place, adds
    pool-unlocked entries directly, and no longer leaves stale sentinels."""

    async def test_reprobe_upgrades_sentinel_with_pooled_key(
        self, hpca_home, fake_keyed_discovery, fake_pool_probe
    ):
        settings = Settings()
        settings.llm_api_keys = ["goodkey"]  # a key the pool already knows
        settings.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            # the scan surfaced the bare sentinel (fake_scan ignores the pool)
            assert any("api key required" in t for t in _left_labels(app))
            # an in-place re-probe with the pool key upgrades it
            app.screen._reprobe_discovered()
            labels = await _wait_for_left_label(app, pilot, "qwen3.6:35B")
            assert any("qwen3.6:35B" in t for t in labels)
            assert not any("api key required" in t for t in labels)

    async def test_added_keyed_endpoint_leaves_no_duplicate(
        self, hpca_home, fake_keyed_discovery, fake_probe, fake_pool_probe
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            left = app.screen.query_one("#llm-discovered", ListView)
            left.focus()
            left.index = 0
            await pilot.press("enter")  # opens the form for the bare sentinel
            await pilot.pause()
            app.screen.query_one("#backend-key", Input).value = "goodkey"
            app.screen.query_one("#backend-model", Input).focus()
            await pilot.press("enter")
            await _wait_for_backend(app, pilot)
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            # the just-added endpoint is gone from Discovered — no re-add
            assert not any("51944" in t for t in _left_labels(app))

    async def test_save_time_guard_drops_sentinel_without_a_working_key(
        self, hpca_home, fake_keyed_discovery
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert any("51944" in t for t in _left_labels(app))
            # force-save a backend at that base_url with a key nothing unlocks
            await app.screen._save_backend(
                LLMBackend(model="forced", base_url=KEYED.base_url, api_key="badkey")
            )
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            assert not any("51944" in t for t in _left_labels(app))

    async def test_pool_unlocked_entry_adds_without_form(
        self, hpca_home, monkeypatch
    ):
        async def fake_scan(*args, **kwargs):
            return [UNLOCKED]

        async def fake_reachable(base_url, **kwargs):
            return True

        monkeypatch.setattr(manage_module, "scan_local_ports", fake_scan)
        monkeypatch.setattr(manage_module, "is_reachable", fake_reachable)
        monkeypatch.setattr(switch_module, "is_reachable", fake_reachable)
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            left = app.screen.query_one("#llm-discovered", ListView)
            left.focus()
            left.index = 0
            await pilot.press("enter")
            await _wait_for_backend(app, pilot)
            # saved directly, no form pushed
            assert not isinstance(app.screen, BackendFormScreen)
            assert isinstance(app.screen, ManageLLMsScreen)
            saved = app.settings.backends[0]
            assert saved.api_key == "poolkey"
            assert saved.model == "pool-unlocked-model"
            assert saved.base_url == UNLOCKED.base_url

    async def test_manual_key_feeds_pool_and_reprobes(
        self, hpca_home, fake_keyed_discovery, fake_probe, fake_pool_probe
    ):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            app.screen.query_one("#llm-discovered", ListView).focus()
            await pilot.press("a")  # manual add form
            await pilot.pause()
            assert isinstance(app.screen, BackendFormScreen)
            app.screen.query_one("#backend-url", Input).value = (
                "http://localhost:52000/v1"
            )
            app.screen.query_one("#backend-key", Input).value = "goodkey"
            app.screen.query_one("#backend-url", Input).focus()
            await pilot.press("enter")
            await _wait_for_backend(app, pilot)
            # the manually-typed key joined the pool…
            assert "goodkey" in app.settings.llm_api_keys
            # …and its arrival re-probed the leftover 51944 sentinel in place
            labels = await _wait_for_left_label(app, pilot, "qwen3.6:35B")
            assert any("qwen3.6:35B" in t for t in labels)
