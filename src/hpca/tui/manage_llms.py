"""Manage-LLMs screen (m): discover, configure, and pick backend LLMs.

Two columns: the left lists discovered endpoints, the right lists the
configured catalog from settings, each labeled ● connected / ○ disconnected,
with ★ marking the active default.

Discovery runs two ways at once, because a backend can be reached two ways.
On the cluster it is at its node's own ``ip:port`` and is declared by a
manifest in the shared endpoints dir (§4.4) — no port scan can find that, so
those are read and probed directly. Off the cluster the same server is an
SSH-tunneled local port, which only a localhost scan finds. Both feed the same
left panel; whichever finds nothing simply contributes nothing.
←/→ switch panels. Key bindings live on the panel widgets, so the footer only
offers "add llm to list (enter)" while the cursor is on a discovered endpoint,
and "remove llm" only on a configured one; removal asks for confirmation.

Thinking used to be toggled here, per catalog entry. It is a per-session level
now (``/thinking``, hpca.thinking): what a user adjusts is how hard *this*
conversation thinks, and a catalog flag answered that by changing every session
on the backend at once.

A scan that finds nothing reports itself differently depending on the right
panel: a toast when a configured backend still answers (nothing *new* turned
up), and the tunnel recipe in a window that waits for escape when none does —
that text has to be retyped into a shell, so it must hold a selection.
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

from urllib.parse import urlparse

from hpca.autoconnect import offcluster_help
from hpca.cluster_endpoints import discover_cluster_endpoints
from hpca.config import LLMBackend
from hpca.discover import (
    KEY_REQUIRED,
    DiscoveredBackend,
    is_reachable,
    ordered_ports,
    probe_endpoint,
    scan_local_ports,
)
from hpca.tui.backend_form import BackendFormScreen
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen

def _as_discovered(backend: LLMBackend) -> DiscoveredBackend:
    return DiscoveredBackend(
        base_url=backend.base_url,
        model=backend.model,
        max_model_len=backend.max_model_len,
        needs_key=backend.api_key is not None,
    )


def backend_details(backend: LLMBackend) -> str:
    """Context size, key requirement and endpoint — no name."""
    return _as_discovered(backend).details()


def backend_line(backend: LLMBackend) -> str:
    """Name and details on one line, for screens wide enough to hold it."""
    return _as_discovered(backend).describe()


class DiscoveredList(ListView):
    """Left panel; its bindings appear in the footer only when focused."""

    BINDINGS = [
        # esc first so "back" shows leftmost (§ esc/quit ordering); the screen's
        # escape binding handles it, this just orders the footer display.
        Binding("escape", "close", "back", show=True),
        Binding("enter", "select_cursor", "add llm to list", show=True),
        Binding("a", "add_manually", "add llm manually", show=True),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "select_cursor":
            return self.highlighted_child is not None
        return True

    def action_add_manually(self) -> None:
        self.screen.add_manually()


class ConfiguredList(ListView):
    """Right panel; removal is only offered on an entry."""

    # No "set default" — the LLM is chosen per session (at creation, or ctrl+l);
    # this panel only adds and removes catalog entries.
    BINDINGS = [
        Binding("escape", "close", "back", show=True),
        Binding("r", "remove_llm", "remove llm", show=True),
    ]

    def check_action(self, action: str, parameters) -> bool | None:
        if action == "remove_llm":
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
        # Manifest-declared cluster endpoints, kept apart from the port-scan
        # results: the two workers finish independently and the scan replaces
        # its own list wholesale when it does.
        self._cluster: list[DiscoveredBackend] = []
        self._reachable: dict[str, bool] = {}
        # An empty scan means one of two very different things depending on
        # whether anything in the catalog answers, so the verdict waits for
        # the reachability probes even when the scan finishes first.
        self._reachability_done = asyncio.Event()
        # ...and the same for the cluster pass: "the scan found nothing" only
        # means "off the cluster" once the manifest pass has had its say.
        self._cluster_done = asyncio.Event()
        self._scan_finished = False
        self._help_pending = False

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
        self.cluster_worker()
        self.scan_worker()
        self.reachability_worker()

    # ------------------------------------------------------------ populate

    @work(group="llm-cluster")
    async def cluster_worker(self) -> None:
        """Read the endpoints dir and probe what it declares (§4.4).

        This is the only way a cluster endpoint reaches this screen: it lives
        on a compute node's own IP, which the localhost scan cannot see. Only
        the LLM role is listed — the embeddings server is auto-wired to RAG and
        is not a backend anyone picks. Best-effort like startup's auto-connect:
        a missing dir, a dead squeue or an unreachable node just means no rows,
        never an error on top of the screen — but never in silence either. An
        empty left panel on a cluster node with servers running is exactly the
        outcome that needs explaining, and autoconnect.log is where discovery
        already writes the reason it dropped each manifest.
        """
        # Imported here, not at module scope: app.py imports this module.
        from hpca.tui.app import _autoconnect_logger

        logger = _autoconnect_logger()
        try:
            slurm = getattr(self.app, "slurm", None)
            if slurm is None:
                logger.info("manage-llms: no Slurm here — manifest pass skipped")
            else:
                endpoints = await discover_cluster_endpoints(
                    self.app.settings.endpoints.dir_path(),
                    slurm,
                    api_keys=self.app.settings.llm_api_keys,
                )
                self._cluster = list(endpoints.llms)
                logger.info(
                    "manage-llms: %d cluster LLM(s) for the left panel: %s",
                    len(self._cluster),
                    [b.model for b in self._cluster] or "none",
                )
        except Exception:  # discovery must never break the screen
            logger.exception("manage-llms: cluster discovery failed")
            self._cluster = []
        finally:
            # Even a failed pass has to release the verdict: an unset event
            # would leave _report_empty_scan waiting forever.
            self._cluster_done.set()
        if not self.is_attached:
            return
        await self.refresh_discovered()
        if self._scan_finished:  # the scan already printed its count
            self._set_status(self._scan_summary())

    @work(thread=True, exclusive=True, group="llm-scan")
    def scan_worker(self) -> None:
        """Port scan in its own thread + event loop.

        Probing ~64k localhost ports floods an event loop with connection
        attempts; run on the UI loop it made the whole app unresponsive.
        Only tiny UI updates are marshalled back via call_from_thread.
        """
        worker = get_current_worker()
        # Gather the pool up front (like _priority_ports()): the scan tries
        # these keys against 401/403 endpoints so key-locked ports resolve inline.
        keys = self._effective_keys()

        def report(done: int, total: int) -> None:
            if not worker.is_cancelled:
                self.app.call_from_thread(
                    self._set_status,
                    f"scanning localhost… {done:,}/{total:,} ports",
                )

        def found(backend: DiscoveredBackend) -> None:
            # surface each endpoint the moment it is discovered
            if not worker.is_cancelled:
                self.app.call_from_thread(self._append_discovered, backend)

        discovered = asyncio.run(
            scan_local_ports(
                ordered_ports(self._priority_ports()),
                progress=report,
                on_found=found,
                api_keys=keys,
            )
        )
        if not worker.is_cancelled:
            self.app.call_from_thread(self._apply_scan_results, discovered)

    def _effective_keys(self) -> list[str]:
        """The key pool unioned with keys already on configured backends,
        deduped and order-stable — every key we could try against a locked
        endpoint."""
        settings = self.app.settings
        keys = list(settings.llm_api_keys)
        for backend in settings.backends:
            if backend.api_key and backend.api_key not in keys:
                keys.append(backend.api_key)
        return keys

    def _priority_ports(self) -> list[int]:
        """Ports worth trying first: ones that served an LLM before, then
        those of the configured backends."""
        settings = self.app.settings
        configured = (urlparse(b.base_url).port for b in settings.backends)
        return [
            *settings.known_llm_ports,
            *(port for port in configured if port is not None),
        ]

    async def _append_discovered(self, backend: DiscoveredBackend) -> None:
        if not self.is_attached:
            return
        self._discovered.append(backend)
        await self.refresh_discovered()

    def _set_status(self, text: str) -> None:
        if self.is_attached:
            self.query_one("#llm-status", Static).update(text)

    async def _apply_scan_results(self, discovered: list[DiscoveredBackend]) -> None:
        if not self.is_attached:
            return  # screen was closed while the scan finished
        self._discovered = discovered
        self._scan_finished = True
        await self.refresh_discovered()
        if self.app.settings.remember_llm_ports(b.base_url for b in discovered):
            self.app.settings.save()  # next scan starts with these ports
        self._set_status(self._scan_summary())
        if not discovered:
            await self._report_empty_scan()

    def _scan_summary(self) -> str:
        return (
            f"scan finished: {len(self._rows())} endpoint(s) found · F5 to rescan"
        )

    async def _report_empty_scan(self) -> None:
        """Nothing on localhost — what that means depends on the rest.

        The manifest pass may still have found the cluster's own endpoints, in
        which case nothing is wrong and there is nothing to say. Otherwise:
        with a configured backend still answering, the scan simply turned up
        nothing new and a toast says so. With none, this is the off-cluster
        case (§4.6) and the user needs the tunnel recipe: that goes in a
        window they can read at their own pace and select and copy from, not
        a toast that dismisses itself and holds no selection.
        """
        await self._cluster_done.wait()  # a cluster hit means nothing is wrong
        await self._reachability_done.wait()  # the verdict reads the catalog
        if not self.is_attached:
            return  # screen was closed while the probes finished
        if self._cluster:
            return
        if any(self._reachable.get(b.base_url) for b in self.app.settings.backends):
            self.app.notify(
                "Nothing new on localhost — the backends you already have are "
                "on the right.",
                title="No further LLM endpoints found",
            )
            return
        self._show_offcluster_help()

    def _show_offcluster_help(self) -> None:
        """Open the tunnel window — or park it until this screen is back on
        top: a scan can finish while the add-backend form is open, and a modal
        must not land on top of someone mid-typing."""
        if self.app.screen is not self:
            self._help_pending = True
            return
        self._help_pending = False
        self.app.push_screen(
            InspectScreen(
                "No LLM endpoints found - [esc] closes",
                offcluster_help(
                    self.app.settings.endpoints.login_target(),
                    self.app.settings.endpoints.endpoints_dir,
                ),
            )
        )

    def on_screen_resume(self) -> None:
        if self._help_pending:
            self._show_offcluster_help()

    @work(group="llm-reach")
    async def reachability_worker(self) -> None:
        try:
            for backend in list(self.app.settings.backends):
                self._reachable[backend.base_url] = await is_reachable(
                    backend.base_url, api_key=backend.api_key
                )
        finally:
            # Even a cancelled or failed probe run has to release the scan:
            # an unset event would leave _report_empty_scan waiting forever.
            self._reachability_done.set()
        await self.refresh_configured()

    @work(group="llm-reprobe")
    async def _reprobe_discovered(self) -> None:
        """Re-probe only the sentinel entries already in the Discovered list,
        trying the current key pool — no TCP sweep. Sentinels a pooled key
        unlocks are replaced in place with their real model rows."""
        keys = self._effective_keys()
        if not keys:
            return
        resolved: dict[str, list[DiscoveredBackend]] = {}
        for backend in list(self._discovered):
            if backend.model != KEY_REQUIRED:
                continue
            rows = await probe_endpoint(backend.base_url, api_keys=keys)
            if rows and all(row.model != KEY_REQUIRED for row in rows):
                resolved[backend.base_url] = rows
        if not resolved:
            return
        new_discovered: list[DiscoveredBackend] = []
        for backend in self._discovered:
            if backend.model == KEY_REQUIRED and backend.base_url in resolved:
                new_discovered.extend(resolved.pop(backend.base_url, []))
            else:
                new_discovered.append(backend)
        self._discovered = new_discovered
        await self.refresh_discovered()

    def _is_configured(self, discovered: DiscoveredBackend) -> bool:
        return any(
            b.base_url == discovered.base_url and b.model == discovered.model
            for b in self.app.settings.backends
        )

    def _rows(self) -> list[DiscoveredBackend]:
        """Everything discovered, cluster first — a manifest-declared endpoint
        is the one the user is meant to connect to, and it is named, where a
        locked port-scan hit is an anonymous sentinel. Deduped on
        (base_url, model): on a login node with a tunnel to the very server the
        manifest names, both passes report the same row."""
        rows: list[DiscoveredBackend] = []
        seen: set[tuple[str, str]] = set()
        for backend in [*self._cluster, *self._discovered]:
            key = (backend.base_url, backend.model)
            if key in seen:
                continue
            seen.add(key)
            rows.append(backend)
        return rows

    async def refresh_discovered(self) -> None:
        discovered_list = self.query_one("#llm-discovered", ListView)
        previous_index = discovered_list.index
        await discovered_list.clear()
        items = []
        for backend in self._rows():
            if self._is_configured(backend):
                continue
            item = ListItem(Label(Content(backend.describe())))
            item.data_discovered = backend
            items.append(item)
        discovered_list.extend(items)
        if items:  # keep the cursor where it was as entries stream in
            discovered_list.index = min(previous_index or 0, len(items) - 1)

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
            # Two lines: the name, then its state. On one line the state fell
            # off the right edge of the panel, which is where the markers live.
            item = ListItem(
                Label(
                    Content(
                        f"{backend.model}\n"
                        f"{marker} │ {backend_details(backend)}"
                    ),
                    classes=css,
                )
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

    async def action_rescan(self) -> None:
        self._discovered = []
        self._cluster = []
        self._cluster_done = asyncio.Event()
        self._scan_finished = False
        await self.refresh_discovered()
        self._set_status("rescanning…")
        self.cluster_worker()
        self.scan_worker()

    @on(ListView.Selected)
    async def _on_selected(self, event: ListView.Selected) -> None:
        # Only the left panel acts on enter (add to catalog). The right panel's
        # entries are managed by (t)/(r); there is no "set default" any more.
        discovered = getattr(event.item, "data_discovered", None)
        if discovered is not None:
            await self._add_backend(discovered)

    async def _add_backend(self, discovered: DiscoveredBackend) -> None:
        # Pool-unlocked: a pooled key already validated against this exact
        # endpoint during the probe, so save directly with that key — no form.
        if discovered.api_key is not None:
            await self._save_backend(
                LLMBackend(
                    model=discovered.model,
                    base_url=discovered.base_url,
                    max_model_len=discovered.max_model_len,
                    api_key=discovered.api_key,
                )
            )
            return
        # A bare key-locked endpoint came through the scan as "(api key
        # required)" with no real model name — open the form to collect the key
        # (and let it auto-fill the model), rather than saving the placeholder.
        # A cluster manifest, though, names the model behind the 401: prefill
        # it, so the user only has to supply the key.
        if discovered.needs_key:
            self.app.push_screen(
                BackendFormScreen(
                    base_url=discovered.base_url,
                    model="" if discovered.model == KEY_REQUIRED else discovered.model,
                    editable_url=False,
                ),
                self._on_backend_form,
            )
            return
        await self._save_backend(
            LLMBackend(
                model=discovered.model,
                base_url=discovered.base_url,
                max_model_len=discovered.max_model_len,
            )
        )

    def add_manually(self) -> None:
        """Add a backend the scan never surfaced — full form, editable URL."""
        self.app.push_screen(
            BackendFormScreen(editable_url=True), self._on_backend_form
        )

    def _on_backend_form(self, backend: LLMBackend | None) -> None:
        if backend is not None:
            self.run_worker(self._save_backend(backend), group="llm-add")

    async def _save_backend(self, backend: LLMBackend) -> None:
        settings = self.app.settings
        settings.backends.append(backend)
        settings.remember_llm_ports([backend.base_url])
        learned_key = settings.remember_llm_key(backend.api_key)
        settings.save()
        self._reachable[backend.base_url] = True  # just probed, or user-asserted
        # Save-time guard: drop any locked row still sitting at this base_url so
        # the just-configured endpoint can't be added again. The (base_url,
        # model) dedup misses it because the model it displayed — the KEY_REQUIRED
        # sentinel from the scan, or the manifest's name for a cluster endpoint —
        # can differ from what the form finally saved; this closes the
        # force-save / bad-key path too.
        def _superseded(b: DiscoveredBackend) -> bool:
            return b.base_url == backend.base_url and (
                b.model == KEY_REQUIRED or b.model == backend.model or b.needs_key
            )

        self._discovered = [b for b in self._discovered if not _superseded(b)]
        self._cluster = [b for b in self._cluster if not _superseded(b)]
        await self.refresh_discovered()
        await self.refresh_configured()
        self.notify(f"Configured {backend.model}")
        if learned_key:
            # A new key landed in the pool — re-probe the remaining sentinels
            # in place; any it unlocks turn informative without a rescan.
            self._reprobe_discovered()

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
