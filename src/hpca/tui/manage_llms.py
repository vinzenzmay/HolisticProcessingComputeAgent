"""Manage-LLMs screen (m): discover, configure, and pick backend LLMs.

Two columns: the left lists endpoints discovered by a localhost port scan
(run when the screen opens — on this HPC setup every backend is an
SSH-tunneled local port); the right lists the configured catalog from
settings, each labeled ● connected / ○ disconnected, ★ marking the active
default. Enter on the left configures a backend; on the right it becomes the
default; (r) removes it.
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Footer, Label, ListItem, ListView, Static

from hpca.config import LLMBackend
from hpca.discover import DiscoveredBackend, is_reachable, scan_local_ports


def backend_line(backend: LLMBackend) -> str:
    return DiscoveredBackend(
        base_url=backend.base_url,
        model=backend.model,
        max_model_len=backend.max_model_len,
        needs_key=backend.api_key is not None,
    ).describe()


class ManageLLMsScreen(Screen):
    BINDINGS = [
        Binding("escape", "close", "back", priority=True),
        Binding("enter", "choose", "add / set default", show=True),
        Binding("r", "remove", "remove"),
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
                yield Static("Discovered (enter: configure)", classes="llm-panel-title")
                yield ListView(id="llm-discovered")
            with Vertical(classes="llm-panel"):
                yield Static(
                    "Configured (enter: set default · r: remove)",
                    classes="llm-panel-title",
                )
                yield ListView(id="llm-configured")
        yield Static("scanning localhost for LLM endpoints…", id="llm-status")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_configured()
        self.query_one("#llm-discovered", ListView).focus()
        self.scan_worker()
        self.reachability_worker()

    # ------------------------------------------------------------ populate

    @work(exclusive=True, group="llm-scan")
    async def scan_worker(self) -> None:
        self._discovered = await scan_local_ports()
        await self.refresh_discovered()
        status = self.query_one("#llm-status", Static)
        status.update(
            f"scan finished: {len(self._discovered)} endpoint(s) found "
            "· F5 to rescan"
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

    # -------------------------------------------------------------- actions

    def action_close(self) -> None:
        self.dismiss(None)

    def action_rescan(self) -> None:
        self.query_one("#llm-status", Static).update("rescanning…")
        self.scan_worker()

    async def action_choose(self) -> None:
        focused = self.focused
        if not isinstance(focused, ListView):
            return
        highlighted = focused.highlighted_child
        discovered = getattr(highlighted, "data_discovered", None)
        configured = getattr(highlighted, "data_configured", None)
        if discovered is not None:
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
        elif configured is not None:
            self.app.switch_backend(configured)
            await self.refresh_configured()

    async def action_remove(self) -> None:
        focused = self.focused
        if not isinstance(focused, ListView):
            return
        configured = getattr(focused.highlighted_child, "data_configured", None)
        if configured is None:
            return
        self.app.settings.backends = [
            b for b in self.app.settings.backends
            if not (b.base_url == configured.base_url and b.model == configured.model)
        ]
        self.app.settings.save()
        await self.refresh_discovered()
        await self.refresh_configured()
        self.notify(f"Removed {configured.model}")

    @on(ListView.Selected)
    async def _on_selected(self, event: ListView.Selected) -> None:
        await self.action_choose()
