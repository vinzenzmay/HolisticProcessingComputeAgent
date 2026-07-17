"""Quick backend switcher (l): pick a configured LLM for the chat."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from hpca.config import LLMBackend
from hpca.discover import is_reachable
from hpca.tui.manage_llms import backend_line


class SwitchLLMScreen(ModalScreen[LLMBackend | None]):
    BINDINGS = [Binding("escape", "cancel", "cancel", priority=True)]

    DEFAULT_CSS = """
    SwitchLLMScreen { align: center middle; }
    #switch-dialog {
        width: 90;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1;
    }
    #switch-title { height: 1; text-style: bold; }
    #switch-hint { color: $text-muted; }
    """

    def __init__(self, backends: list[LLMBackend], active_marker) -> None:
        super().__init__()
        self._backends = backends
        self._is_active = active_marker

    def compose(self) -> ComposeResult:
        with Vertical(id="switch-dialog"):
            yield Static("Switch LLM backend", id="switch-title")
            yield ListView(id="switch-list")
            yield Static("(enter) switch · (esc) cancel", id="switch-hint")

    async def on_mount(self) -> None:
        switch_list = self.query_one("#switch-list", ListView)
        for backend in self._backends:
            star = " ★" if self._is_active(backend) else ""
            item = ListItem(Label(Content(f"{backend_line(backend)} │ …{star}")))
            item.data_backend = backend
            switch_list.append(item)
        switch_list.index = 0
        switch_list.focus()
        self.reachability_worker()

    @work(group="switch-reach")
    async def reachability_worker(self) -> None:
        switch_list = self.query_one("#switch-list", ListView)
        for item in list(switch_list.children):
            backend = getattr(item, "data_backend", None)
            if backend is None:
                continue
            reachable = await is_reachable(backend.base_url)
            marker = "● connected" if reachable else "○ disconnected"
            star = " ★" if self._is_active(backend) else ""
            label = item.query_one(Label)
            label.update(Content(f"{backend_line(backend)} │ {marker}{star}"))
            label.set_classes("llm-connected" if reachable else "llm-disconnected")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(getattr(event.item, "data_backend", None))

    def action_cancel(self) -> None:
        self.dismiss(None)
