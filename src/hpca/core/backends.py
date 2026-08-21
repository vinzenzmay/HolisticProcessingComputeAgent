"""Which LLM every session talks to, and the client it talks through.

Lifted out of `HpcaApp`, where the catalog, the per-session clients, the
context meter and cluster auto-connect were fourteen methods interleaved with
the widgets that displayed them. Two things made that code impossible to run
headless, and both are gone here:

**It asked what was on screen.** `_active_model`, `_active_backend`,
`_active_max_model_len` and `_show_context_estimate` all went through
`_reads_session()` — "the session on screen" — to decide whose backend they
meant. Every one of them now takes a `session_id`. The question "is this the
session the user is looking at?" is not one the core is allowed to ask
(`hpca.core.__init__`), and it never had to: the answer only ever decided who
to *tell*, which is now the renderer's problem because every event names its
session (§4.2).

**It reached for widgets.** `self.notify(...)` and `bar.set_used(...)` become
`deps.emit(Notify(...))` and `deps.emit(ContextEstimate(...))`.

What deliberately did **not** move is choosing. `_pick_llm_for_new_session` was
a modal, and picking stays a front-end job; the catalog it picks from, and the
blob the choice is stored as, are here.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Any, Callable

import httpx

from hpca.agent import compact
from hpca.autoconnect import AutoConnectPlan, offcluster_help, plan_auto_connect
from hpca.cluster_endpoints import discover_cluster_endpoints
from hpca.config import LLMBackend, LLMSettings, llm_settings_for
from hpca.core.deps import CoreDeps
from hpca.discover import (
    KEY_REQUIRED,
    DiscoveredBackend,
    is_reachable,
    ordered_ports,
    probe_endpoint,
    scan_local_ports,
)
from hpca.embeddings import EmbeddingClient
from hpca.llm import LLMClient
from hpca.logs import LoggedLLM
from hpca.protocol import ContextEstimate, LLMEntry, Notify, TurnUsage
from hpca.sessions import SessionStore

if TYPE_CHECKING:
    from hpca.logs import SessionLog

# How a client is built from the settings that describe it. A seam rather than
# a bare `LLMClient(...)` call so a test can count constructions and closes
# without a socket ever being opened.
ClientFactory = Callable[[LLMSettings], Any]

# The localhost sweep, as a seam. Injected rather than called directly so a
# test can drive `scan` without opening 64k sockets — and so the thread hop
# around it (see `BackendRegistry._scan_local`) is exercised either way.
PortScanner = Callable[..., Any]


@dataclass
class ScanResult:
    """What one `backend.scan` turned up, and the state it has to be read
    against.

    Three fields because "the scan found nothing" is not one outcome. A
    localhost sweep that finds nothing while the cluster's manifests declare a
    live endpoint is a normal day on a compute node; the same empty sweep with
    every configured backend also unreachable is the off-cluster case, and the
    user needs the tunnel recipe. Only something holding all three can tell
    those apart, which is why the verdict is minted here (`verdict`) rather
    than left to whoever draws the panel.
    """

    local: list[DiscoveredBackend] = field(default_factory=list)
    cluster: list[DiscoveredBackend] = field(default_factory=list)
    reachable: dict[str, bool] = field(default_factory=dict)

    @property
    def found(self) -> int:
        return len(self.local) + len(self.cluster)


def autoconnect_logger(app_dir: Path) -> logging.Logger:
    """The ``hpca.autoconnect`` logger, writing to ``<app_dir>/autoconnect.log``.

    Auto-connect is best-effort and must never interrupt startup, so its
    failures reach nobody's screen and this file is the whole debugging trail —
    including the outcomes that are not failures, since "connected to nothing"
    and "found nothing" look identical from outside.

    Not propagating, and only ever a file handler: the core may share a
    terminal with a TUI, and the root logger's last-resort stderr handler would
    shred the display. A near-copy of `tui/app.py:_file_logger` because nothing
    under `hpca.core` may import that module; parameterised by ``app_dir``
    because the core is handed one rather than asking `config.app_dir()`.
    """
    logger = logging.getLogger("hpca.autoconnect")
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        app_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(app_dir / "autoconnect.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


async def _close_quietly(client: Any) -> None:
    """Close a client without letting it strand the ones queued behind it.

    Only used on teardown and on replacement, which is exactly where a partial
    cleanup is the worst outcome: the caller has already stopped using the
    client, so a raised error buys nothing and costs every socket after it.
    """
    try:
        await client.close()
    except Exception:
        pass


class BackendRegistry:
    """The LLM catalog, the clients built from it, and the context accounting.

    One instance per core. It owns every connection the agent makes to a model
    server — the bootstrap client, the per-session ones, and the embeddings
    client — which is why it is also the thing that closes them.
    """

    def __init__(
        self,
        deps: CoreDeps,
        *,
        sessions: SessionStore | None = None,
        llm: Any = None,
        embedder: Any = None,
        client_factory: ClientFactory = LLMClient,
        on_reload: Callable[[], None] | None = None,
        probe_transport: httpx.AsyncBaseTransport | None = None,
        port_scanner: PortScanner = scan_local_ports,
        logger: logging.Logger | None = None,
    ) -> None:
        self._deps = deps
        # Resolving a session's backend has to be synchronous: `client_for` is
        # what the graph's per-turn resolver calls, and that resolver is a
        # plain callable invoked mid-step. So this holds a real connection
        # instead of going through `deps.db` — one indexed read, and the same
        # one `_backend_of` did on the UI loop.
        self.sessions = sessions or (
            SessionStore(deps.conn) if deps.conn is not None else None
        )
        self._client_factory = client_factory
        # An injected client belongs to whoever passed it in (tests, embedding
        # hosts); only a client we built is ours to replace or to close.
        self._llm = llm if llm is not None else client_factory(deps.settings.llm)
        self._owns_llm = llm is None
        # Per-session clients, built lazily from each session's stored backend
        # and keyed by base_url||model so two sessions on one backend share a
        # client — and so its connection pool. Closed en masse by `aclose`.
        self._session_clients: dict[str, Any] = {}
        # RAG's client. Owned unconditionally, unlike the bootstrap one: it has
        # no injected-by-the-host case, and auto-connect replaces it outright
        # when the cluster turns out to be serving embeddings somewhere else.
        self.embedder = embedder or EmbeddingClient(
            base_url=deps.settings.rag.embedding_base_url,
            model=deps.settings.rag.embedding,
        )
        # Called after the bootstrap client has been replaced, so whoever built
        # something from the old settings (the graph, a session's tool context)
        # can rebuild it. A named hook rather than an implicit reach back into
        # the app, which is what this used to be.
        self._on_reload = on_reload
        self._probe_transport = probe_transport
        self._port_scanner = port_scanner
        self._logger = logger
        # Endpoints a scan turned up that nobody has configured. Kept here
        # rather than in the front-end because they are catalog rows with a
        # flag on them (`protocol.LLMEntry.discovered`) — the same list, so a
        # screen showing two panels is splitting one answer rather than
        # keeping two in step. They outlive the scan so a second `llm.list`
        # still draws them; only a rescan replaces them.
        self._discovered: list[DiscoveredBackend] = []
        # The window the bootstrap backend reports when asked, and the last
        # measured prompt size per session — a different thread is a different
        # context, so this can never be one number.
        self._discovered_window: int | None = None
        self._context_used: dict[str, int] = {}
        # The last generation each session paid for: (completion tokens, wall
        # clock). Kept beside the prompt size rather than divided into a rate
        # here, because the protocol carries both (`protocol.TurnUsage`) and
        # this is the only place that has them.
        self._last_generation: dict[str, tuple[int, float]] = {}
        self._closed = False

    # ------------------------------------------------------------- clients

    @property
    def bootstrap(self) -> Any:
        """The client a session that pinned no backend of its own talks to."""
        return self._llm

    def backend_for(self, session_id: str | None) -> LLMBackend | None:
        """The catalog entry a session talks to, or None for the bootstrap.

        Takes the id because the original took whatever `_reads_session()`
        returned — the session on screen. That reach is the coupling this
        module exists to remove, so the caller now has to say which
        conversation it means.
        """
        blob = self._backend_blob(session_id)
        if not blob:
            return None
        try:
            return LLMBackend.model_validate_json(blob)
        except Exception:
            # A blob from a settings model that has since changed shape must
            # not strand the session: the bootstrap client is at least live.
            return None

    def _backend_blob(self, session_id: str | None) -> str:
        if not session_id or self.sessions is None:
            return ""
        session = self.sessions.get(session_id)
        return session.backend if session is not None else ""

    def client_for(self, session_id: str | None) -> Any:
        """The client a session talks through: its own backend's, built and
        cached on first use, or the bootstrap client when it pinned none."""
        return self.client_for_backend(self.backend_for(session_id))

    def client_for_backend(self, backend: LLMBackend | None) -> Any:
        """The shared client for one backend. Keyed by endpoint and model, not
        by session, which is what makes two sessions on one backend share."""
        if backend is None:
            return self._llm
        key = f"{backend.base_url}||{backend.model}"
        client = self._session_clients.get(key)
        if client is None:
            client = self._client_factory(
                llm_settings_for(backend, self._deps.settings.llm)
            )
            self._session_clients[key] = client
        return client

    def labelled_client(
        self,
        label: str,
        *,
        session_id: str | None,
        log: "SessionLog | None" = None,
    ) -> Any:
        """The client for one of the core's own sub-agent calls (titling,
        /conclude, struggle notes), logged under that name.

        Routed through the session's *own* backend — the same live client chat
        uses — not the bootstrap client, which may point at a backend that is
        no longer running once the session has been switched to another
        (§ per-session LLM). The original fell back to the open session's log
        when handed none; there is no open session here, so a caller that wants
        the traffic recorded passes the log of the session it means.
        """
        client = self.client_for(session_id)
        if log is None:
            return client
        return LoggedLLM(client, log, label=lambda: f"subagent:{label}")

    async def aclose(self) -> None:
        """Close every client this registry built, exactly once.

        Idempotent because shutdown has more than one path into it (a
        `shutdown` command, the idle timer, a dying UI) and closing an
        httpx pool twice is not something to rely on.
        """
        if self._closed:
            return
        self._closed = True
        clients = list(self._session_clients.values())
        self._session_clients.clear()
        if self._owns_llm and self._llm is not None:
            clients.append(self._llm)
        if self.embedder is not None:
            clients.append(self.embedder)
            self.embedder = None
        for client in clients:
            await _close_quietly(client)

    # ------------------------------------------------------------- catalog

    def choices(self) -> list[LLMBackend]:
        """The configured catalog, entries and all, for callers inside the core.

        Choosing is a front-end job — it was a modal — but the list being
        chosen from belongs to the core, and so does what the choice is
        stored as (`backend_for_new_session`).

        Not what crosses the socket: these hold api keys. `catalog` is the
        drawable form, and it is the only one a front-end ever sees.
        """
        return list(self._deps.settings.backends)

    def labels(self) -> dict[str, LLMBackend]:
        """The catalog by the name a command may pin an entry with.

        One place mints these, because two would eventually disagree and the
        disagreement would look like "the backend I picked was ignored": the
        same function fills `LLMEntry.label` for the front-end and resolves
        what comes back in `session.new`.

        The model name where it is unique, ``model @ host:port`` where it is
        not — two vLLMs serving one model on two nodes are two entries a user
        has to be able to tell apart, and a bare index would not survive the
        catalog being reordered, which is the property the stored blob has and
        a label must not lose. A catalog holding the very same model at the
        very same endpoint twice is a duplicated settings entry rather than a
        choice; the second copy is numbered so that every entry still has a
        name, and neither name is silently unreachable.
        """
        counts: dict[str, int] = {}
        for entry in self._deps.settings.backends:
            counts[entry.model] = counts.get(entry.model, 0) + 1
        out: dict[str, LLMBackend] = {}
        for entry in self._deps.settings.backends:
            label = entry.model
            if counts.get(entry.model, 0) > 1:
                label = f"{entry.model} @ {urlparse(entry.base_url).netloc}"
            if label in out:
                seq = 2
                while f"{label} #{seq}" in out:
                    seq += 1
                label = f"{label} #{seq}"
            out[label] = entry
        return out

    def catalog(self, *, reachable: dict[str, bool] | None = None) -> list[LLMEntry]:
        """The catalog as a front-end draws it (`protocol.LLMCatalog`).

        Never the `LLMBackend` objects themselves: they hold api keys, and a
        picker only needs the name, the endpoint, the window and whether a key
        is required — which is exactly what the manage-LLMs line has always
        shown (`discover.DiscoveredBackend.details`).

        ``reachable`` is the probe results by label when there are any. Absent
        means "not asked", which is a third state and not a synonym for
        disconnected: see `probe_catalog`.
        """
        settings = self._deps.settings
        results = reachable or {}
        return [
            LLMEntry(
                label=label,
                model=entry.model,
                base_url=entry.base_url,
                max_model_len=entry.max_model_len,
                # Whether there is a key, never the key. The line says "key:
                # yes"; nothing on a screen ever needed more than that.
                needs_key=entry.api_key is not None,
                active=settings.is_active(entry),
                reachable=results.get(label),
            )
            for label, entry in self.labels().items()
        ]

    async def probe_catalog(self) -> dict[str, bool]:
        """Ask every configured endpoint whether it answers, all at once.

        Concurrent because the endpoints are independent and a probe of a node
        that has gone away costs the full timeout; serialised, a catalog of
        five dead entries would take five timeouts before the first ● could be
        drawn.

        Authenticated where the entry has a key, because `is_reachable` counts
        a 401 as reachable only when it was not given one — a stored key that
        no longer works is a backend the user cannot use, and it should read
        as disconnected rather than as connected.

        Never raises: a probe is a nicety, and an endpoint that fails in a way
        `is_reachable` does not catch must not cost the client its catalog.
        """
        entries = list(self.labels().items())
        if not entries:
            return {}

        async def ask(entry: LLMBackend) -> bool:
            try:
                return await is_reachable(
                    entry.base_url,
                    api_key=entry.api_key,
                    transport=self._probe_transport,
                )
            except Exception:
                return False

        answers = await asyncio.gather(*(ask(entry) for _, entry in entries))
        return {label: bool(ok) for (label, _), ok in zip(entries, answers)}

    def backend_for_new_session(
        self, requested: LLMBackend | str | None = None
    ) -> str | None:
        """The backend blob to store on a session about to be created, or None
        when the label names nothing.

        Stored as JSON rather than as the label itself, so the choice survives
        that entry being dropped from the catalog later
        (`sessions.Session.backend`) — the label is how a *command* names a
        backend, not how a session remembers one.

        What crosses the wire is a label (`protocol.SessionNew`). It used to be
        the blob, while the protocol documented a label, and the mismatch was
        silent in the worst way: an unrecognised value returned "" and the
        session was created against the bootstrap client as though nothing had
        been asked for. So an empty request still means the bootstrap — that is
        a real answer, and what a run with no configured backends gets — but a
        label nothing in the catalog answers to comes back as None, for the
        caller to report rather than absorb.
        """
        if requested is None or requested == "":
            return ""
        if isinstance(requested, LLMBackend):
            return requested.model_dump_json()
        entry = self.labels().get(requested)
        return entry.model_dump_json() if entry is not None else None

    def marks_session_backend(
        self, backend: LLMBackend, *, session_id: str | None
    ) -> bool:
        """Whether ``backend`` is what ``session_id`` currently talks to (the ★).

        session_id is a parameter because the original asked the app which
        session was on screen. Compared by endpoint and model rather than by
        identity: the session stores a copy of the entry, not a reference to
        one, and the catalog list is rebuilt from settings on every load.
        """
        current = self.backend_for(session_id)
        if current is None:
            return self._deps.settings.is_active(backend)
        return (
            current.base_url == backend.base_url
            and current.model == backend.model
        )

    def switch_backend(
        self, session_id: str, backend: LLMBackend, *, busy: bool = False
    ) -> bool:
        """Point one session at a different configured backend.

        ``busy`` is whether THAT session has a turn in flight. The turn table
        belongs to the scheduler, so the answer is passed in rather than read —
        and it is deliberately per session: a background turn in another
        session does not block this switch, because the clients are keyed per
        backend and the checkpoints per thread, so nothing that turn holds is
        disturbed.

        Returns whether the switch happened, which is the caller's cue to
        rebind anything built from the old client (the session's tool context).
        """
        if busy:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text="The agent is mid-reply — switch backends once "
                    "it finishes.",
                )
            )
            return False
        blob = backend.model_dump_json()
        if self.sessions is not None:
            self.sessions.set_backend(session_id, blob)
        self._deps.emit(Notify(text=f"This session now uses {backend.model}"))
        # A rate measured on the old model says nothing about the new one, and
        # leaving it beside a restated fill would attribute one backend's speed
        # to another. The prompt size survives: it counts the same thread.
        self._last_generation.pop(session_id, None)
        # A different backend is a different window, so the fill this session
        # was showing describes the wrong denominator until it is restated.
        self._emit_estimate(session_id)
        return True

    async def set_default(
        self, backend: LLMBackend, *, busy: bool = False
    ) -> bool:
        """Make one backend the default — what every un-pinned session uses.

        The other half of `backend.set` (`switch_backend` is the per-session
        one), and a different operation rather than the same one with a wider
        blast radius: this writes the *settings*, so it outlives the run and
        decides what the next session is created against, while a session
        switch only rewrites one row.

        Added to the catalog if it is not in it, for the same reason
        `ensure_catalog` does it on the auto-connect path: a backend that is
        active but unlisted is one the picker cannot show the user as chosen.
        Its port and key are remembered too, so a later scan finds it first.

        ``busy`` is whether ANY turn is in flight, because the client this
        replaces is the one every un-pinned session is talking through.
        Announced regardless of whether the rebuild happened: what was said is
        that this endpoint is now the default, and a rebuild deferred by a
        running turn (or skipped for an injected client) leaves that true — it
        only delays which client speaks it, exactly as `auto_activate` argues.
        """
        settings = self._deps.settings
        if not any(
            entry.base_url == backend.base_url and entry.model == backend.model
            for entry in settings.backends
        ):
            settings.backends.append(backend)
        settings.activate_backend(backend)
        settings.remember_llm_ports([backend.base_url])
        settings.remember_llm_key(backend.api_key)
        settings.save()
        await self.reload(busy=busy)
        self._deps.emit(
            Notify(text=f"Sessions without a backend of their own now use {backend.model}")
        )
        return True

    def model_for(self, session_id: str | None) -> str:
        """The model name a session talks to, else the bootstrap's."""
        backend = self.backend_for(session_id)
        if backend is not None:
            return backend.model
        return self._deps.settings.llm.model

    def max_model_len_for(self, session_id: str | None) -> int | None:
        """A session's context window: its own backend's, else the probe's
        answer, else the active catalog entry's.

        None means unknown, and with an unknown window compaction stays off —
        better than guessing one and folding history needlessly.
        """
        backend = self.backend_for(session_id)
        if backend is not None:
            return backend.max_model_len
        if self._discovered_window is not None:
            return self._discovered_window
        settings = self._deps.settings
        for candidate in settings.backends:
            if settings.is_active(candidate):
                return candidate.max_model_len
        return None

    async def reload(self, *, busy: bool = False) -> bool:
        """Rebuild the bootstrap client so edited LLM settings apply next turn.

        ``busy`` is whether ANY turn is in flight anywhere: unlike
        `switch_backend` this cannot be scoped to one conversation, because it
        swaps the client every session without a backend of its own is using.

        An injected client belongs to whoever passed it in (tests, embedding
        hosts); only a client we built is ours to replace, so an injected one
        makes this a no-op.

        The window probe is awaited rather than fired off, so a caller that
        wants the new window has it on return; a caller that does not care can
        put the whole call on a task.
        """
        if busy:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text="The agent is mid-reply — the new LLM settings "
                    "apply after this turn.",
                )
            )
            return False
        if not self._owns_llm:
            return False
        old = self._llm
        self._llm = self._client_factory(self._deps.settings.llm)
        # A different model means a different window, and the token counts
        # measured against the old one no longer describe it — drop every
        # session's measurement so each re-measures on its next turn.
        self._discovered_window = None
        self._context_used.clear()
        self._last_generation.clear()
        if self._on_reload is not None:
            self._on_reload()
        if old is not None:
            await _close_quietly(old)
        await self.discover_context_window()
        return True

    # --------------------------------------------------- context accounting

    @property
    def discovered_window(self) -> int | None:
        """What the bootstrap backend last said its window was, if asked."""
        return self._discovered_window

    def measured_for(self, session_id: str | None) -> int | None:
        """The last prompt size the backend reported for a session, if any.

        Public because `ContextEstimate` has no measured/estimated flag, and an
        in-process renderer that draws the two differently has to ask.
        """
        if not session_id:
            return None
        return self._context_used.get(session_id)

    def note_usage(self, session_id: str, usage: dict) -> None:
        """Record what the backend said this session's last prompt cost.

        prompt_tokens is what occupies the window; the completion is spent the
        moment it is generated. Reported per round, so a tool-heavy turn
        visibly fills the meter as it works. Stored for EVERY session rather
        than only the one being looked at — a background turn silently updates
        its own number, so switching to it later shows its current fill instead
        of a stale zero or another session's count. Whether the update reaches
        a screen is the renderer's call now: the event names its session.

        The same report carries the turn's speed — the completion count over
        the wall clock our own client measured around the request (`llm.py`
        puts ``request_seconds`` in this dict, since an OpenAI-style body has
        no timing in it). Recorded before the early return, because a decision
        that generated tokens took time whether or not the backend bothered to
        say what the prompt cost.
        """
        completion = usage.get("completion_tokens")
        seconds = usage.get("request_seconds")
        if completion and seconds:
            self._last_generation[session_id] = (int(completion), float(seconds))
        prompt_tokens = usage.get("prompt_tokens")
        if not prompt_tokens:
            return
        self._context_used[session_id] = int(prompt_tokens)
        self._emit_measured(session_id)

    def _emit_measured(self, session_id: str) -> None:
        """The backend's own count for one session (`turn.usage`).

        A separate event from `context.estimate` because the two are different
        claims about the same number: this one was reported by the model
        server, the other is arithmetic over the stored history. The protocol
        carries no measured/estimated flag, so the distinction has to be the
        event type — and a client that draws them the same way is free to,
        while one that marks an estimate with a "~" can.

        Every path that restates a *known* fill goes through here rather than
        through `_emit_estimate`, including a backend switch: a client that has
        already seen a measured count ignores a later estimate for the same
        session (it is a guess about a number it knows), so restating a new
        window as an estimate would silently fail to move the meter.
        """
        completion, seconds = self._last_generation.get(session_id, (0, None))
        self._deps.emit(
            TurnUsage(
                session_id=session_id,
                prompt_tokens=self._context_used.get(session_id, 0),
                max_model_len=self.max_model_len_for(session_id),
                # Restated on every measurement rather than sent once, so the
                # pair always travels with the fill it was measured beside and
                # a client never has to remember which earlier frame the speed
                # arrived in. Zero and None where there is nothing to report:
                # see `switch_backend`, which drops a rate the new model did
                # not earn.
                completion_tokens=completion,
                request_seconds=seconds,
            )
        )

    def estimate_context(self, session_id: str, values: dict) -> None:
        """How full a reopened session's context already is.

        Estimated from the checkpointed history, since nothing has been sent
        yet this run. Compaction is accounted for: what the model will receive
        is the folded view, not the whole transcript. A measured number for
        this session — its own turn, foreground or background, already reported
        one — always supersedes the estimate.
        """
        if self._context_used.get(session_id):
            self._emit_measured(session_id)
            return
        messages = list(values.get("messages", []))
        compacted = values.get("compacted")
        if compacted:
            messages = [compacted["summary"]] + messages[compacted["upto"] :]
        used = compact.estimate_tokens(messages) if messages else 0
        self._emit_estimate(session_id, used=used)

    def forget_session(self, session_id: str) -> None:
        """Drop a session's measured number — it was closed or deleted.

        Reopening then re-derives the fill from the stored history, which
        reflects compaction and any growth since; a running turn re-stores its
        count the next time it reports.

        The speed goes with it, and for a sharper reason: it describes one
        request made by one backend, so carrying it across a close would put a
        rate from the old model beside a fill measured against the new one.
        """
        self._context_used.pop(session_id, None)
        self._last_generation.pop(session_id, None)

    def _emit_estimate(self, session_id: str, *, used: int | None = None) -> None:
        """Say how full one session's window is, as far as anyone can tell.

        The precedence rule — a measured count always beats an estimate — is
        applied here rather than left to whoever draws it: asked to restate a
        session that has already reported real usage, this says so on the
        measured channel instead (see `_emit_measured`).

        A window of 0 means "not known yet": the protocol field is not
        optional, and an unknown window is exactly what the probe below exists
        to fix.
        """
        if used is None:
            measured = self._context_used.get(session_id)
            if measured:
                self._emit_measured(session_id)
                return
            used = 0
        self._deps.emit(
            ContextEstimate(
                session_id=session_id,
                used=int(used),
                window=self.max_model_len_for(session_id) or 0,
            )
        )

    async def discover_context_window(self) -> int | None:
        """Ask the bootstrap backend how big its window is, and remember it.

        This is why the meter needs no configuration: vLLM reports
        max_model_len per model, so switching from a 192k model to a 32k one
        moves the bar without anyone editing settings. The answer is written
        back to the catalog entry so compaction also benefits, and so the
        number survives a backend that is offline next time.

        Emits nothing. The probe asks about the *bootstrap* backend, which is
        not a question about any one session, and there is no window-only event
        to announce the answer with; the caller — which knows who is listening
        — restates the sessions it cares about. Every failure degrades to
        "window still unknown": a probe is a nicety, and an unreachable backend
        must not stop the turn that is about to discover that for itself.
        """
        try:
            window = await self._llm.context_window()
        except Exception:
            window = None
        if not window:
            return None
        measured = int(window)
        self._discovered_window = measured
        settings = self._deps.settings
        for backend in settings.backends:
            if settings.is_active(backend) and backend.max_model_len != measured:
                backend.max_model_len = measured
                settings.save()
                break
        return measured

    # -------------------------------------------------------- auto-connect

    async def auto_connect(self, *, busy: bool = False) -> AutoConnectPlan | None:
        """Discover live cluster vLLM endpoints and connect with no setup (§4).

        On the cluster (Slurm present) this reads the manifest dir, reconciles
        it against Slurm liveness + a live probe, then applies rule (c): one
        LLM auto-connects, several are announced for the picker, and the
        embeddings server (if up) is always wired to RAG. Off the cluster (no
        Slurm) it is a no-op — the manage-LLMs scan and tunnel template cover
        that path. Every step is best-effort: discovery must never take the
        core down or block the first turn — but every outcome, including a
        swallowed exception, leaves a trail in <app_dir>/autoconnect.log.
        """
        if self._deps.slurm is None:
            return None
        settings = self._deps.settings
        logger = self.logger
        try:
            endpoints = await discover_cluster_endpoints(
                settings.endpoints.dir_path(),
                self._deps.slurm,
                api_keys=settings.llm_api_keys,
                transport=self._probe_transport,
            )
        except Exception:
            logger.exception("auto-connect: discovery failed")
            return None
        plan = plan_auto_connect(
            endpoints, preferred_models=settings.endpoints.preferred_models
        )
        logger.info(
            "auto-connect: %d LLM choice(s); connecting to %s; embeddings %s",
            len(plan.choices),
            plan.connect.model if plan.connect else "none",
            plan.embedding_base_url or "none",
        )
        if plan.embedding_base_url:
            await self.wire_embedding(plan.embedding_base_url)
        if plan.connect is not None:
            await self.auto_activate(plan.connect, busy=busy)
        elif plan.notice:
            self._deps.emit(Notify(text=plan.notice))
        return plan

    def ensure_catalog(self, discovered: DiscoveredBackend) -> LLMBackend:
        """The catalog entry for a discovered endpoint, adding it if new."""
        settings = self._deps.settings
        for backend in settings.backends:
            if (
                backend.base_url == discovered.base_url
                and backend.model == discovered.model
            ):
                return backend
        entry = LLMBackend(
            model=discovered.model,
            base_url=discovered.base_url,
            api_key=discovered.api_key,
            max_model_len=discovered.max_model_len,
        )
        settings.backends.append(entry)
        settings.remember_llm_ports([entry.base_url])
        settings.remember_llm_key(entry.api_key)
        settings.save()
        return entry

    async def auto_activate(
        self, discovered: DiscoveredBackend, *, busy: bool = False
    ) -> bool:
        """Make a discovered LLM the active backend, adding it to the catalog
        first. A no-op when it is already active, so a restart against an
        unchanged endpoint causes no client churn.

        The announcement is unconditional once the catalog has changed, and
        deliberately does not wait on `reload` succeeding: what was announced
        is that this endpoint is now the default. A rebuild refused because a
        turn is in flight, or skipped because the client was injected, still
        leaves that true — it only delays which client speaks it.
        """
        entry = self.ensure_catalog(discovered)
        settings = self._deps.settings
        if settings.is_active(entry):
            return False
        settings.activate_backend(entry)
        settings.save()
        await self.reload(busy=busy)
        self._deps.emit(Notify(text=f"Auto-connected to {discovered.model}"))
        return True

    async def wire_embedding(self, base_url: str) -> bool:
        """Point RAG's embedder at a discovered embeddings server.

        Rebuilding the client is enough: tool contexts read `embedder` when
        they are built, which on the auto-connect path is before any turn runs.
        """
        rag = self._deps.settings.rag
        if rag.embedding_base_url == base_url:
            return False
        rag.embedding_base_url = base_url
        self._deps.settings.save()
        old = self.embedder
        self.embedder = EmbeddingClient(base_url=base_url, model=rag.embedding)
        if old is not None:
            await _close_quietly(old)
        return True

    @property
    def logger(self) -> logging.Logger:
        """The auto-connect trail. Built on first use, as in the original: the
        handler opens a file, and constructing a registry must not."""
        if self._logger is None:
            self._logger = autoconnect_logger(self._deps.app_dir)
        return self._logger
