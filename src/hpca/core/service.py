"""The runtime, assembled: one object, one command in, events out.

This is what a front-end talks to, and — once wave 3 lands — what
``hpca --serve`` runs. It owns the databases, the checkpointer, the graph and
the four domain services, and it exposes exactly two things: :meth:`handle`,
which takes a protocol command, and :meth:`subscribe`, which yields the events
that result. Nothing else is public, because anything else would be a seam a
front-end could reach through and the split would start leaking again.

Three assembly decisions are worth stating, because they are what make the
pieces fit rather than merely coexist:

**Events fan out, they never block.** ``deps.emit`` is synchronous and puts the
event on every subscriber's queue. A service that had to await delivery could
be stalled by a slow client, and a poll timer that can be stalled by a client
is a poll timer that stops noticing that a job died.

**Per-turn dependencies resolve off the invoked ``thread_id``.** The graph is
built once, and every session-varying thing — client, tool context, window,
mode — is a callable the graph invokes with the session id it is running. That
is what lets two turns on two sessions run at once without reading each other's
state, and it is why none of these resolvers may consult "the current session".

**The services do not know about each other.** Where two of them genuinely need
to meet — the scheduler needs a prepared turn, the memory service needs a
client, the pollers need somewhere to put a completion — they are wired here
through small adapters. Each service keeps a constructor that a test can
satisfy with a lambda.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.file_tools import add_file_tools
from hpca.agent.graph import build_graph
from hpca.agent.job_tools import add_job_tools
from hpca.agent.memory_context import build_memory_context, compose_api_content
from hpca.agent.memory_tools import add_memory_tools
from hpca.agent.middleware import uses_native_tools
from hpca.agent.modes import add_plan_tool
from hpca.agent.prompts import (
    build_skill_directive,
    environment_facts,
    orchestrator_system_prompt,
)
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.watch_tools import add_watch_tools
from hpca.core.backends import BackendRegistry
from hpca.core.deps import CoreDeps
from hpca.core.memory_service import MemoryService
from hpca.core.pollers import Pollers
from hpca.core.scheduler import TurnPlan, TurnScheduler
from hpca.protocol import (
    ConfirmRequested,
    ConfirmResolve,
    DecisionResolve,
    Message,
    Notify,
    SessionFocus,
    Shutdown,
    TurnInterrupt,
    TurnSubmit,
    TurnUnqueue,
    TurnUnqueued,
)

logger = logging.getLogger("hpca.core.service")


class AgentService:
    """The whole runtime behind one command/event surface."""

    def __init__(
        self,
        deps: CoreDeps,
        *,
        graph: Any,
        backends: BackendRegistry,
        memory: MemoryService,
        scheduler: TurnScheduler,
        pollers: Pollers,
        sessions,
    ) -> None:
        self._deps = deps
        self._sessions = sessions
        self._graph = graph
        self._backends = backends
        self._memory = memory
        self._scheduler = scheduler
        self._pollers = pollers
        self._subscribers: list[asyncio.Queue] = []
        # Questions raised by a poll and not yet answered, keyed by the id that
        # crossed the wire. The continuation stays here: only the yes/no comes
        # back, so a front-end cannot answer with a different action than the
        # one it was offered.
        self._confirmations: dict[str, Any] = {}
        self._confirm_seq = 0
        self._timers: list[asyncio.Task] = []
        self._stopped = False

    # ------------------------------------------------------------- event fan

    def subscribe(self) -> asyncio.Queue:
        """A queue that receives every event from now on.

        Unbounded on purpose: dropping an event to protect the core would
        desynchronise a renderer that has no way to notice it happened. A
        client too slow to keep up is a client that should be disconnected,
        which is the transport's call, not this one's.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with suppress(ValueError):
            self._subscribers.remove(queue)

    def _fan_out(self, event: Any) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    # ------------------------------------------------------------- dispatch

    async def handle(self, command: Message) -> None:
        """Act on one command. Never raises at the caller.

        A command that fails is reported as a `notify` and the connection
        carries on: the far side of this is a socket, and one bad frame must
        not be able to end a session.
        """
        try:
            await self._dispatch(command)
        except Exception as e:
            logger.exception("command failed: %s", type(command).__name__)
            self._deps.emit(Notify(severity="error", text=str(e)))

    async def _dispatch(self, command: Message) -> None:
        if isinstance(command, SessionFocus):
            # The one sanctioned answer to "what is the user looking at".
            self._deps.focused_session_id = command.session_id
            await self._pollers.refresh_panel(force=True)
            return
        if isinstance(command, TurnSubmit):
            session = self._session(command.session_id)
            if session is None:
                self._deps.emit(
                    Notify(severity="warning", text="That session is gone.")
                )
                return
            # Whether it runs now or waits is the scheduler's answer, and so
            # is saying so: a message that has to queue is drawn by the
            # scheduler as a `queued` chat row, because only the scheduler
            # knows when that row stops waiting (see TurnScheduler.drain).
            self._scheduler.submit_user(
                command.session_id,
                command.text,
                forced_skill=self._skill_named(command.forced_skill, session),
            )
            await self._scheduler.drain()
            return
        if isinstance(command, TurnInterrupt):
            await self._scheduler.interrupt(command.session_id)
            return
        if isinstance(command, TurnUnqueue):
            text = self._scheduler.unqueue(command.session_id, command.seq)
            if text is None:
                # It started while the user was deciding. Stopping the turn it
                # became is turn.interrupt — a different question, and one the
                # UI must ask deliberately rather than have answered for it.
                self._deps.emit(
                    Notify(
                        severity="warning",
                        text="Too late — that message is already running.",
                    )
                )
                return
            self._deps.emit(
                TurnUnqueued(
                    session_id=command.session_id,
                    seq=command.seq,
                    text=text,
                )
            )
            return
        if isinstance(command, DecisionResolve):
            self._scheduler.resolve_decision(
                command.session_id,
                approved=command.approved,
                reason=command.reason,
            )
            return
        if isinstance(command, ConfirmResolve):
            await self._resolve_confirmation(command.id, command.confirmed)
            return
        if isinstance(command, Shutdown):
            await self.stop()
            return
        self._deps.emit(
            Notify(
                severity="warning",
                text=f"Unhandled command: {getattr(command, 'TYPE', command)}",
            )
        )

    # ---------------------------------------------------------- confirmations

    def ask(self, question: str, on_yes) -> None:
        """Put a yes/no to whoever is listening; run ``on_yes`` if they accept.

        Returns immediately — this is called from a poll, and a poll that
        waited for a person would stop being a poll. The continuation is held
        here rather than sent, see :attr:`_confirmations`.
        """
        self._confirm_seq += 1
        key = f"q{self._confirm_seq}"
        self._confirmations[key] = on_yes
        self._deps.emit(ConfirmRequested(id=key, question=question))

    async def _resolve_confirmation(self, key: str, confirmed: bool) -> None:
        on_yes = self._confirmations.pop(key, None)
        if on_yes is None or not confirmed:
            return  # a stale answer, or a no: nothing to run either way
        try:
            await on_yes()
        except Exception as e:
            logger.exception("confirmed action failed")
            self._deps.emit(Notify(severity="error", text=str(e)))

    # ---------------------------------------------------------------- helpers

    def _session(self, session_id: str | None):
        if session_id is None:
            return None
        return self._sessions.get(session_id)

    def _skill_named(self, name: str | None, session) -> Any:
        """The named skill from the *session's* profile, not the core's.

        A turn in a session the user has left must still find its own
        profile's skill; reading the working profile's list is exactly the
        kind of "whatever is current" lookup this split exists to remove.
        """
        if not name:
            return None
        from hpca.skills import load_skills

        profile = getattr(session, "profile", self._deps.profile)
        return next((s for s in load_skills(profile) if s.name == name), None)

    # ------------------------------------------------------------- lifecycle

    def start_timers(self) -> None:
        """Run the pollers on their own cadences.

        Started separately from construction so a test can drive
        ``poll_jobs()`` by hand and never race a timer.
        """
        for interval, poll in self._pollers.timers():
            self._timers.append(asyncio.ensure_future(self._tick(interval, poll)))

    async def _tick(self, interval: float, poll) -> None:
        while not self._stopped:
            await asyncio.sleep(interval)
            if self._stopped:
                return
            try:
                await poll()
            except asyncio.CancelledError:
                raise
            except Exception:  # a poll that dies must not take its timer down
                logger.exception("poll failed")

    async def stop(self) -> None:
        """Quiesce: timers, then turns, then clients. Order matters — a poll
        firing after the databases close would raise into a dead loop."""
        if self._stopped:
            return
        self._stopped = True
        for task in self._timers:
            task.cancel()
        for task in self._timers:
            with suppress(Exception, asyncio.CancelledError):
                await task
        self._timers.clear()
        await self._scheduler.shutdown()
        await self._backends.aclose()


def build_service(
    *,
    settings,
    app_dir,
    db,
    conn,
    checkpointer,
    profile: str = "default",
    slurm=None,
    tools=None,
    llm=None,
    session_store=None,
) -> AgentService:
    """Assemble the runtime. The only place the four services meet.

    Kept a function rather than more constructor arguments on
    :class:`AgentService` so the wiring — which adapter bridges which pair of
    services — is readable in one screen, and so a test can build a service
    with three real components and one fake.
    """
    from hpca.sessions import SessionStore
    from hpca.skills import load_skills

    events: list = []
    service_ref: dict[str, AgentService] = {}

    def emit(event) -> None:
        holder = service_ref.get("service")
        if holder is None:
            events.append(event)  # emitted during assembly; replayed below
            return
        holder._fan_out(event)

    deps = CoreDeps(
        settings=settings,
        app_dir=app_dir,
        db=db,
        emit=emit,
        conn=conn,
        slurm=slurm,
        profile=profile,
    )
    sessions = session_store or SessionStore(conn)

    if tools is None:
        tools = add_ask_docs(add_doc_tools(add_file_tools(default_tool_registry())))
        if slurm is not None:
            add_job_tools(tools)
        # Watching is useful with or without Slurm: watch_job needs a cluster,
        # but watch_log is just a file.
        add_watch_tools(tools)
        add_skill_tools(tools)
        add_memory_tools(tools)
        add_plan_tool(tools)

    backends = BackendRegistry(deps, sessions=sessions, llm=llm)
    memory = MemoryService(
        deps,
        # The two services disagree about whether a session is an object or an
        # id — memory grew up holding Sessions, the registry keys by thread_id
        # because that is what the graph invokes with. Bridged here rather than
        # by widening either signature.
        llm_for=lambda label, s: backends.labelled_client(
            label, session_id=s.session_id if s is not None else None
        ),
        backend_name=lambda s: backends.model_for(
            s.session_id if s is not None else None
        ),
        tools=tools,
    )

    scheduler_ref: dict[str, TurnScheduler] = {}

    def turn_session(session_id: str):
        """The session a turn belongs to, resolvable after the user has moved
        on: the live turn's own copy, else the stored row."""
        sched = scheduler_ref.get("scheduler")
        if sched is not None:
            ts = sched._turns.get(session_id)
            if ts is not None:
                return ts.session
        return sessions.get(session_id)

    def ctx_for_turn(session_id: str) -> ToolContext | None:
        sched = scheduler_ref.get("scheduler")
        if sched is None:
            return None
        ts = sched._turns.get(session_id)
        return ts.plan.ctx if ts is not None else None

    def skills_for(profile: str):
        """This profile's skills. Loaded per turn rather than read off the
        service, because the turn may belong to a different profile than the
        one the core is working under."""
        return load_skills(profile)

    def system_prompt_for(session_id: str) -> str:
        """The prompt for the session running ``session_id``.

        Rendered for that session's own profile, so a background turn uses its
        own memories and its own model tag rather than whichever profile the
        core happens to be working under.
        """
        session = turn_session(session_id)
        profile = getattr(session, "profile", deps.profile)
        sched = scheduler_ref.get("scheduler")
        ts = sched._turns.get(session_id) if sched is not None else None
        turn_memory = ts.plan.memory if ts is not None else memory.snapshot(profile)
        skills = ts.plan.skills if ts is not None else skills_for(profile)
        return orchestrator_system_prompt(
            system_prompt_memories=turn_memory.system_prompt_text(
                active_backend=backends.model_for(session_id)
            ),
            memory_meter=turn_memory.usage_meter(
                settings.memory.system_prompt_token_cap
            ),
            # The skill list is deliberately kept out of the prompt: the model
            # is told skills exist and reaches one via read_skill.
            has_skills=bool(skills),
            session_search="session_search" in tools.names(),
            memory_tool="memory" in tools.names(),
            watch_tools="watch_log" in tools.names(),
            # Asked of the client that will actually carry the turn, not of
            # the global setting: a session pinned to a backend whose entry
            # overrides tool_protocol must be told about the protocol it is
            # really using, or the prompt describes a format the channel has
            # no room for (specs-edit-eval.md §7).
            native_tools=uses_native_tools(backends.client_for(session_id)),
        )

    graph = build_graph(
        llm=lambda sid: backends.client_for(sid),
        tools=tools,
        checkpointer=checkpointer,
        ctx=ctx_for_turn,
        system_prompt_fn=system_prompt_for,
        max_retries=settings.llm.max_retries,
        max_tool_rounds=settings.llm.max_tool_rounds,
        on_activity=lambda sid, act: scheduler_ref["scheduler"].report_activity(
            sid, act
        ),
        max_model_len=lambda sid: backends.max_model_len_for(sid),
        on_usage=lambda sid, usage: backends.note_usage(sid, usage),
        mode_fn=lambda sid: _mode_for(sessions, settings, sid),
        effort_fn=lambda sid: _effort_for(sessions, settings, sid),
    )

    def prepare(session, *, user_text=None, forced_skill=None) -> TurnPlan:
        """Everything a turn needs, frozen before it starts.

        Frozen rather than resolved per round for the same reason the memory
        snapshot is: a profile edit or a backend switch mid-turn must not
        change what the running turn is working from.
        """
        from hpca.logs import open_log

        turn_memory = memory.snapshot(session.profile)
        skills = skills_for(session.profile)
        api_content = None
        if user_text is not None:
            # A "/<skill>" invocation: the model works from the request alone
            # (the command word stripped) plus the skill's procedure on the
            # sidecar. Recall runs on the request, not on the command word.
            if forced_skill is not None:
                request = user_text[1:].partition(" ")[2].strip()
                directive = build_skill_directive(
                    forced_skill.name, forced_skill.description, forced_skill.body
                )
            else:
                request, directive = user_text, ""
            lines = memory.recall_lines(
                request,
                turn_memory,
                session.profile,
                backend=backends.model_for(session.session_id),
            )
            block = build_memory_context(
                lines,
                max_notes=settings.memory.rag_prefetch_count,
                max_chars=settings.memory.rag_prefetch_chars,
            )
            api_content = compose_api_content(
                request, block, environment_facts(), skill_directive=directive
            )
        log = open_log(settings, session)
        return TurnPlan(
            ctx=_make_tool_ctx(
                deps, session, log, skills=skills, backends=backends, tools=tools
            ),
            memory=turn_memory,
            skills=skills,
            api_content=api_content,
            log=log,
        )

    scheduler = TurnScheduler(
        deps,
        graph=graph,
        prepare=prepare,
        session_for=sessions.get,
    )
    scheduler_ref["scheduler"] = scheduler

    pollers = Pollers(
        deps,
        submit_event=scheduler.submit_event,
        llm=lambda: backends.bootstrap,
    )

    service = AgentService(
        deps,
        graph=graph,
        backends=backends,
        memory=memory,
        scheduler=scheduler,
        pollers=pollers,
        sessions=sessions,
    )
    service_ref["service"] = service
    # Triage's "shall I learn this signature?" offer needs somewhere to ask.
    # Wired after construction because the service is what holds the pending
    # question, and the poller is built before it.
    pollers._confirm = lambda question, on_yes: service.ask(question, on_yes)
    for event in events:
        service._fan_out(event)
    return service


def _mode_for(sessions, settings, session_id: str) -> str:
    session = sessions.get(session_id)
    stored = getattr(session, "mode", "") if session is not None else ""
    return stored or settings.agent.default_mode


def _effort_for(sessions, settings, session_id: str) -> str:
    """The session's thinking level (hpca.thinking). Read from the store per
    round, not from a Session copy, so ``/thinking`` reaches a turn already in
    flight — the same rule as the mode above."""
    session = sessions.get(session_id)
    stored = getattr(session, "thinking", "") if session is not None else ""
    return stored or settings.agent.default_thinking


def _make_tool_ctx(deps, session, log, *, skills, backends, tools) -> ToolContext:
    """A tool context bound to one session and its transcript.

    Built per turn, as before, so a turn keeps its own runner and log however
    the rest of the runtime moves on.
    """
    from hpca.config import app_dir as _app_dir
    from hpca.episodic import EpisodicStore
    from hpca.jobs import JobStore
    from hpca.logs import LoggedLLM
    from hpca.rag import RagStore
    from hpca.runner import ProcessRunner
    from hpca.symbols import SymbolIndex
    from hpca.trash import TrashManager
    from hpca.watches import WatchStore

    root = deps.app_dir or _app_dir()
    ctx = ToolContext(
        runner=ProcessRunner(
            deps.conn,
            session_id=session.session_id,
            log_dir=root / "proc_logs",
        ),
        settings=deps.settings,
        scripts_dir=root / "scripts",
        session_id=session.session_id,
        profile=session.profile,
        slurm=deps.slurm,
        jobs=JobStore(deps.conn),
        job_log_dir=root / "job_logs",
        watches=WatchStore(deps.conn),
        llm=backends.client_for(session.session_id),
        trash=TrashManager(
            root / "trash",
            backup_limit_bytes=int(
                deps.settings.safety.backup_limit_gb * 1024**3
            ),
        ),
        symbols=SymbolIndex(deps.conn),
        rag=deps.extras.get("rag") or RagStore(root / "rag.db"),
        embedder=backends.embedder,
        episodic=EpisodicStore(deps.conn),
        skills=skills,
    )
    if log is not None:
        ctx.llm = LoggedLLM(
            ctx.llm, log, label=lambda: f"subagent:{ctx.current_tool or '?'}"
        )
    return ctx
