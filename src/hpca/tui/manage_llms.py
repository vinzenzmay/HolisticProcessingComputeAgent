"""Manage-LLMs screen (m): discover, configure, and pick backend LLMs.

Two columns: the left lists endpoints discovered by a localhost port scan
(run when the screen opens — on this HPC setup every backend is an
SSH-tunneled local port); the right lists the configured catalog from
settings, each labeled ● connected / ○ disconnected, ★ marking the active
default. ←/→ switch panels. Key bindings live on the panel widgets, so the
footer only offers "add llm to list (enter)" while the cursor is on a
discovered endpoint, and "set default"/"remove llm" only on a configured
one; removal asks for confirmation.
"""

from __future__ import annotations

import asyncio

from textual import on, work
from textual.app import ComposeResult
from textual.worker import get_current_worker
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Footer, Label, ListItem, ListView, Static

from hpca.config import LLMBackend
from hpca.discover import DiscoveredBackend, is_reachable, scan_local_ports
from hpca.tui.confirm_screen import ConfirmScreen


def backend_line(backend: LLMBackend) -> str:
    return DiscoveredBackend(
        base_url=backend.base_url,
        model=backend.model,
        max_model_len=backend.max_model_len,
        needs_key=backend.api_key is not None,
    ).describe()


class DiscoveredList(ListView):
    """Left panel; its bindings appear in the footer only when focused."""

    BINDINGS = [
        Binding("enter", "select_cursor", "add llm to list", show=True),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "select_cursor":
            return self.highlighted_child is not None
        return True


class ConfiguredList(ListView):
    """Right panel; set-default and remove only offered on an entry."""

    BINDINGS = [
        Binding("enter", "select_cursor", "set default", show=True),
        Binding("r", "remove_llm", "remove llm", show=True),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        if action in ("select_cursor", "remove_llm"):
            return self.highlighted_child is not None
        return True

    def action_remove_llm(self) -> None:
        self.screen.confirm_remove_selected()


class ManageLLMsScreen(Screen):
    BINDINGS = [
        Binding("escape", "close", "back", priority=True),
        Binding("left", "focus_panel('discovered')", "◀ panel", show=False),
        Binding("right", "focus_panel('configured')", "panel ▶", show=False),
        Binding("f5", "rescan", "rescan"),
    ]

    DEFAULT_CSS = """
    #llm-columns { height: 1fr; }
    .llm-panel { width: 1fr; border: solid $panel; }
    .llm-panel:focus-within { border: heavy $accent; }
    .llm-panel-title {
        height: 1;
        text-style: bold;
        text-align: center;
        background: $boost;
    }
    #llm-status { height: 1; color: $text-muted; }
    .llm-connected { color: $success; }
    .llm-disconnected { color: $text-muted; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._discovered: list[DiscoveredBackend] = []
        self._reachable: dict[str, bool] = {}

    def compose(self) -> ComposeResult:
        yield Static("Manage LLM backends", id="llm-title", classes="llm-panel-title")
        with Horizontal(id="llm-columns"):
            with Vertical(classes="llm-panel"):
                yield Static("Discovered", classes="llm-panel-title")
                yield DiscoveredList(id="llm-discovered")
            with Vertical(classes="llm-panel"):
                yield Static("Configured", classes="llm-panel-title")
                yield ConfiguredList(id="llm-configured")
        yield Static("scanning localhost for LLM endpoints…", id="llm-status")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_configured()
        self.query_one("#llm-discovered", ListView).focus()
        self.scan_worker()
        self.reachability_worker()

    # ------------------------------------------------------------ populate

    @work(thread=True, exclusive=True, group="llm-scan")
    def scan_worker(self) -> None:
        """Port scan in its own thread + event loop.

        Probing ~64k localhost ports floods an event loop with connection
        attempts; run on the UI loop it made the whole app unresponsive.
        Only tiny UI updates are marshalled back via call_from_thread.
        """
        worker = get_current_worker()

        def report(done: int, total: int) -> None:
            if not worker.is_cancelled:
                self.app.call_from_thread(
                    self._set_status,
                    f"scanning localhost… {done:,}/{total:,} ports",
                )

        discovered = asyncio.run(scan_local_ports(progress=report))
        if not worker.is_cancelled:
            self.app.call_from_thread(self._apply_scan_results, discovered)

    def _set_status(self, text: str) -> None:
        if self.is_attached:
            self.query_one("#llm-status", Static).update(text)

    async def _apply_scan_results(self, discovered: list[DiscoveredBackend]) -> None:
        if not self.is_attached:
            return  # screen was closed while the scan finished
        self._discovered = discovered
        await self.refresh_discovered()
        self._set_status(
            f"scan finished: {len(discovered)} endpoint(s) found · F5 to rescan"
        )

    @work(group="llm-reach")
    async def reachability_worker(self) -> None:
        for backend in list(self.app.settings.backends):
            self._reachable[backend.base_url] = await is_reachable(backend.base_url)
        await self.refresh_configured()

    def _is_configured(self, discovered: DiscoveredBackend) -> bool:
        return any(
            b.base_url == discovered.base_url and b.model == discovered.model
            for b in self.app.settings.backends
        )

    async def refresh_discovered(self) -> None:
        discovered_list = self.query_one("#llm-discovered", ListView)
        await discovered_list.clear()
        items = []
        for backend in self._discovered:
            if self._is_configured(backend):
                continue
            item = ListItem(Label(Content(backend.describe())))
            item.data_discovered = backend
            items.append(item)
        discovered_list.extend(items)
        if items and discovered_list.index is None:
            discovered_list.index = 0

    async def refresh_configured(self) -> None:
        configured_list = self.query_one("#llm-configured", ListView)
        await configured_list.clear()
        items = []
        for backend in self.app.settings.backends:
            reachable = self._reachable.get(backend.base_url)
            if reachable is None:
                marker, css = "…", "llm-disconnected"
            elif reachable:
                marker, css = "● connected", "llm-connected"
            else:
                marker, css = "○ disconnected", "llm-disconnected"
            star = " ★" if self.app.settings.is_active(backend) else ""
            item = ListItem(
                Label(Content(f"{backend_line(backend)} │ {marker}{star}"),
                      classes=css)
            )
            item.data_configured = backend
            items.append(item)
        configured_list.extend(items)
        if items and configured_list.index is None:
            configured_list.index = 0

    # -------------------------------------------------------------- actions

    def action_close(self) -> None:
        self.dismiss(None)

    def action_focus_panel(self, which: str) -> None:
        self.query_one(f"#llm-{which}", ListView).focus()

    def action_rescan(self) -> None:
        self._set_status("rescanning…")
        self.scan_worker()

    @on(ListView.Selected)
    async def _on_selected(self, event: ListView.Selected) -> None:
        discovered = getattr(event.item, "data_discovered", None)
        configured = getattr(event.item, "data_configured", None)
        if discovered is not None:
            await self._add_backend(discovered)
        elif configured is not None:
            self.app.switch_backend(configured)
            await self.refresh_configured()

    async def _add_backend(self, discovered: DiscoveredBackend) -> None:
        self.app.settings.backends.append(
            LLMBackend(
                model=discovered.model,
                base_url=discovered.base_url,
                max_model_len=discovered.max_model_len,
            )
        )
        self.app.settings.save()
        self._reachable[discovered.base_url] = True  # just probed by the scan
        await self.refresh_discovered()
        await self.refresh_configured()
        self.notify(f"Configured {discovered.model}")

    def confirm_remove_selected(self) -> None:
        configured_list = self.query_one("#llm-configured", ListView)
        backend = getattr(configured_list.highlighted_child, "data_configured", None)
        if backend is None:
            return

        def verdict(confirmed: bool | None) -> None:
            if confirmed:
                self.run_worker(self._remove_backend(backend), group="llm-remove")

        self.app.push_screen(
            ConfirmScreen(f"Really remove {backend.model}?"), verdict
        )

    async def _remove_backend(self, backend: LLMBackend) -> None:
        self.app.settings.backends = [
            b
            for b in self.app.settings.backends
            if not (b.base_url == backend.base_url and b.model == backend.model)
        ]
        self.app.settings.save()
        await self.refresh_discovered()
        await self.refresh_configured()
        self.notify(f"Removed {backend.model}")
