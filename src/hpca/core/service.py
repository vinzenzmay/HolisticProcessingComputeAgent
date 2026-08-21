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
import hashlib
import logging
from contextlib import suppress
from typing import Any

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.file_tools import add_file_tools
from hpca.agent.graph import (
    build_graph,
    fork_thread,
    rollback_thread,
    thread_message_count,
)
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
from hpca.agent.titler import propose_title
from hpca.agent.watch_tools import add_watch_tools
from hpca.core.backends import BackendRegistry
from hpca.core.deps import CoreDeps
from hpca.core.memory_service import MemoryService
from hpca.core.pollers import Pollers
from hpca.core.scheduler import TurnPlan, TurnScheduler, wire_entry
from hpca.episodic import EpisodicStore
from hpca.protocol import (
    PROTOCOL_VERSION,
    ChatReset,
    ConfirmRequested,
    ConfirmResolve,
    DecisionRequested,
    DecisionResolve,
    Hello,
    Message,
    Notify,
    SessionClose,
    SessionCreated,
    SessionDelete,
    SessionFocus,
    SessionFork,
    SessionList,
    SessionNew,
    SessionOpen,
    SessionRename,
    SessionRetitle,
    SessionRollback,
    SessionRow,
    SessionRows,
    Shutdown,
    TurnInterrupt,
    TurnSubmit,
    TurnUnqueue,
    TurnUnqueued,
)
from hpca.transcript import build_entries
from hpca.watches import WatchStore

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
        checkpointer: Any = None,
    ) -> None:
        self._deps = deps
        self._sessions = sessions
        self._graph = graph
        self._backends = backends
        self._memory = memory
        self._scheduler = scheduler
        self._pollers = pollers
        # Held only so a deleted session's history can go with it. The graph
        # keeps its own reference and nothing else here reaches for it: a
        # service that could read checkpoints would have a second way to know
        # what a thread contains, next to `aget_state`.
        self._checkpointer = checkpointer
        self._subscribers: list[asyncio.Queue] = []
        # Questions raised by a poll and not yet answered, keyed by the id that
        # crossed the wire. The continuation stays here: only the yes/no comes
        # back, so a front-end cannot answer with a different action than the
        # one it was offered.
        self._confirmations: dict[str, Any] = {}
        self._confirm_seq = 0
        self._timers: list[asyncio.Task] = []
        # Command work that outlives its dispatch — titling, today. See _spawn.
        self._tasks: set[asyncio.Task] = set()
        self._stopped = False

    # ------------------------------------------------------------- event fan

    def subscribe(self) -> asyncio.Queue:
        """A queue that receives every event from now on, greeting first.

        Unbounded on purpose: dropping an event to protect the core would
        desynchronise a renderer that has no way to notice it happened. A
        client too slow to keep up is a client that should be disconnected,
        which is the transport's call, not this one's.

        Two frames are put on it before it is handed back, and both go to this
        client only rather than through `_fan_out` — they are a handshake, not
        news, and a second front-end attaching must not make the first one
        re-run its version check or redraw a prompt it is already showing.

        `hello` is the first (§4.2), because everything after it is read
        against a protocol version. A decision the core is parked on follows,
        because it is the one piece of session state that *asks a question*: a
        turn cannot move until it is answered, and a client that never hears
        about it shows a session that appears to have simply stopped. That is
        the latent bug §4.4 names — the parked decision used to live in the
        UI's own memory, so restarting the front-end lost the only copy of a
        question a graph thread was still blocked on.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(
            Hello(
                version=PROTOCOL_VERSION,
                profile=self._deps.profile,
                settings_digest=_settings_digest(self._deps.settings),
            )
        )
        for session_id, payload in self._scheduler.pending_decisions().items():
            queue.put_nowait(
                DecisionRequested(session_id=session_id, payload=dict(payload))
            )
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
        if isinstance(command, SessionList):
            self._emit_rows()
            return
        if isinstance(command, SessionNew):
            self._new_session(command)
            return
        if isinstance(command, SessionOpen):
            if self._known(command.session_id) is not None:
                await self._reset_chat(command.session_id)
            return
        if isinstance(command, SessionClose):
            await self._close_session()
            return
        if isinstance(command, SessionRename):
            self._rename_session(command.session_id, command.title)
            return
        if isinstance(command, SessionRetitle):
            if self._known(command.session_id) is not None:
                # The model writes it, so this takes as long as a small
                # generation does; awaiting it here would stall every command
                # queued behind it on the same socket.
                self._spawn(self._retitle(command.session_id))
            return
        if isinstance(command, SessionDelete):
            await self._delete_session(command.session_id)
            return
        if isinstance(command, SessionFork):
            await self._fork_session(command.session_id, command.index)
            return
        if isinstance(command, SessionRollback):
            await self._rollback_session(command.session_id, command.index)
            return
        if isinstance(command, SessionFocus):
            # The one sanctioned answer to "what is the user looking at".
            self._deps.focused_session_id = command.session_id
            await self._pollers.refresh_panel(force=True)
            return
        if isinstance(command, TurnSubmit):
            session = self._known(command.session_id)
            if session is None:
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
            if await self._scheduler.interrupt(command.session_id) is None:
                return
            # The interrupt rolls the abandoned attempt out of the thread, so
            # the rows drawn for it now describe messages that are gone. A
            # delta cannot take a row off the screen; a reset can, and this is
            # the same case `session.rollback` is — an open of what is left.
            await self._reset_chat(command.session_id)
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

    # ------------------------------------------------------ session lifecycle

    def _emit_rows(self) -> None:
        """The sidebar, whole — the answer to `session.list` and the tail of
        every command that changes what is in it.

        Whole rather than incremental for the same reason `panel.update` is: a
        row is a title, a profile, a mode, a model and a marker or two, and a
        diff would need the core to model what the front-end drew. The chat is the thing that may not be resent
        (§4.2 property 1), and it is resent by nothing here.
        """
        self._deps.emit(
            SessionRows(rows=[self._row(s) for s in self._sessions.list_all()])
        )

    def _row(self, session) -> SessionRow:
        """One session as the sidebar sees it.

        The model is resolved here rather than stored on the row: what the
        session holds is a backend blob, and the front-end may not read the
        database to unpack it (§4.2 rule 2).
        """
        backend = self._backends.backend_for(session.session_id)
        return SessionRow(
            session_id=session.session_id,
            title=session.title,
            profile=session.profile,
            mode=session.mode,
            # Empty for the bootstrap client: the row says what this
            # conversation is *pinned* to, and pinned to nothing is news.
            model=backend.model if backend is not None else "",
            flags=self._flags(session.session_id),
        )

    def _flags(self, session_id: str) -> list[str]:
        """The render markers for one row — the two states a user working in
        another session still has to see (`tui/app.py:_session_row_text`).

        Both are emitted rather than the winner of the two: which mark takes
        precedence is a drawing decision, and the core has no business making
        it for a front-end it cannot see.
        """
        flags: list[str] = []
        if session_id in self._scheduler.pending_decisions():
            flags.append("decision")
        if self._scheduler.is_busy(session_id):
            flags.append("working")
        return flags

    def _known(self, session_id: str):
        """That session, or None — and if None, said out loud.

        A command naming a deleted session is not an error; it is what a
        keypress against a stale sidebar looks like, and every session command
        can be handed one. So it is refused the way every un-carry-out-able
        command is: one warning, no state change, in one place.
        """
        session = self._session(session_id)
        if session is None:
            self._deps.emit(
                Notify(severity="warning", text="That session is gone.")
            )
        return session

    def _new_session(self, command: SessionNew) -> None:
        """Make one, and say which it is.

        `session.created` before `session.rows` on purpose: the UI has to open
        it, and a sidebar cannot say which of its lines is new (see
        `protocol.SessionCreated`).
        """
        settings = self._deps.settings
        session = self._sessions.create(
            profile=command.profile or self._deps.profile,
            mode=settings.agent.default_mode,
            # A blob, not a catalog index, so the choice survives the entry
            # being dropped from the catalog later.
            backend=self._backends.backend_for_new_session(command.backend),
            thinking=settings.agent.default_thinking,
        )
        self._deps.emit(SessionCreated(row=self._row(session)))
        self._emit_rows()

    async def _reset_chat(self, session_id: str) -> None:
        """The whole transcript for one session, renumbered from 1.

        The only frame that may carry a chat wholesale, and so the only place
        this is allowed to be called from: an open, and a rollback — which is
        an open of what is left (§4.2 property 1). Anything else that resends a
        chat is the per-turn rebuild this protocol exists to delete.

        The context estimate goes with it because the two describe the same
        thing: how much of the window this conversation already occupies. A
        rollback in particular leaves the measured number describing a thread
        that no longer exists.
        """
        values = await self._thread_values(session_id)
        entries = [
            wire_entry(entry, seq)
            for seq, entry in enumerate(
                build_entries(
                    list(values.get("messages", [])),
                    values.get("thinking", []),
                    values.get("calls", []),
                ),
                start=1,
            )
        ]
        # What the checkpoint cannot know about: the message of a turn that is
        # running right now, and everything typed ahead behind it. Numbered by
        # the scheduler, which owns the counter this reset just re-based — and
        # which also re-binds a running turn's rows to the names above, so the
        # deltas that follow revise the rows this reset actually drew.
        entries += self._scheduler.rebase_rows(session_id, entries=entries)
        self._deps.emit(ChatReset(session_id=session_id, entries=entries))
        self._backends.estimate_context(session_id, values)

    async def _thread_values(self, session_id: str) -> dict:
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        return snapshot.values or {}

    async def _close_session(self) -> None:
        """Nothing is open any more.

        Carries no session id (§4.1) because there is only ever one thing to
        close: whatever the last `session.focus` named. The panel goes with it
        — watches are session-scoped, so with no session focused there is
        nothing of anyone's to show — and the measured context number is
        dropped, so re-opening re-derives the fill from the stored history the
        way it does after a restart.
        """
        closing = self._deps.focused_session_id
        self._deps.focused_session_id = None
        if closing is not None:
            self._backends.forget_session(closing)
        await self._pollers.refresh_panel(force=True)
        self._emit_rows()

    def _rename_session(self, session_id: str, title: str) -> None:
        if self._known(session_id) is None:
            return
        if not title.strip():
            # A nameless row is a row the user cannot find again. Refused
            # here rather than stored, because the store would take it.
            self._deps.emit(
                Notify(severity="warning", text="A session needs a name.")
            )
            return
        self._sessions.rename(session_id, title)
        self._emit_rows()

    async def _retitle(self, session_id: str) -> None:
        """Ask the model to name this conversation (`session.retitle`).

        Routed through the session's *own* client, not the bootstrap one: a
        session pinned to a backend must not have its title written by
        whichever model the core happens to be holding.
        """
        session = self._session(session_id)
        if session is None:
            return  # deleted while the request was in flight
        messages = list((await self._thread_values(session_id)).get("messages", []))
        if not messages:
            self._deps.emit(
                Notify(severity="warning", text="Nothing to summarize yet.")
            )
            return
        try:
            title = await propose_title(
                self._backends.labelled_client("title", session_id=session_id),
                messages,
            )
        except Exception:
            # Naming is a nicety and the model refusing to do it is ordinary;
            # it is reported, not raised (`tui/app.py:_propose_title`).
            logger.exception("titling failed")
            self._deps.emit(
                Notify(severity="error", text="The model could not write a title.")
            )
            return
        self._sessions.rename(session_id, title)
        self._emit_rows()
        self._deps.emit(Notify(text=f"Renamed to “{title}”"))

    async def _delete_session(self, session_id: str) -> None:
        """Drop a conversation: its row, its history, and what hung off it.

        The plain-text log on disk is deliberately kept — it is the record
        that the session existed at all — and so are the job and process rows,
        which describe real work that outlives the conversation about it (see
        `sessions.SessionStore.delete`). Everything else goes, and the
        episodic index goes for a reason of its own: these are patient-data
        environments, and a deleted conversation must not resurface through a
        search.
        """
        session = self._known(session_id)
        if session is None:
            return
        if self._deps.focused_session_id == session_id:
            self._deps.focused_session_id = None
        # Its queue, its parked decision and its row names go first, so
        # nothing queued for it can start against a thread that is going away.
        self._scheduler.forget_session(session_id)
        self._backends.forget_session(session_id)
        self._sessions.delete(session_id)
        await self._deps.db(
            lambda conn: EpisodicStore(conn).forget_session(session_id)
        )
        # Session-scoped too: left behind, they would be boxes no session can
        # ever show while the pollers went on stat-ing their files forever.
        await self._deps.db(
            lambda conn: WatchStore(conn).forget_session(session_id)
        )
        if self._checkpointer is not None:
            try:
                await self._checkpointer.adelete_thread(session_id)
            except Exception as e:  # the row is already gone; say so, move on
                self._deps.emit(
                    Notify(
                        severity="warning",
                        text=f"Chat history left behind: {e}",
                    )
                )
        self._emit_rows()
        await self._pollers.refresh_panel(force=True)
        self._deps.emit(Notify(text=f"Deleted “{session.title}”"))

    async def _fork_session(self, session_id: str, index: int) -> None:
        """Branch a conversation into a new session, cut before ``index``.

        Ungated on purpose (`protocol.SessionFork`): it only ever reads a
        checkpoint snapshot of the source and only ever writes to a thread
        nothing has touched, and branching off while the agent works is the
        case forking exists for.

        The new session's profile, backend and mode are copied here rather
        than carried on the wire, so a front-end cannot fork a conversation
        into a profile the user never chose.
        """
        source = self._known(session_id)
        if source is None:
            return
        keep = await self._cut_point(session_id, index)
        if keep is None:
            return
        fork = self._sessions.create(
            profile=source.profile,
            title=f"{source.title} (fork)",
            mode=source.mode,
            backend=source.backend,
        )
        try:
            await fork_thread(
                self._graph,
                source_session_id=session_id,
                target_session_id=fork.session_id,
                keep=keep,
            )
        except Exception as e:
            # An empty session nobody asked for is worse than no session: it
            # would sit in the sidebar looking like the fork succeeded.
            self._sessions.delete(fork.session_id)
            self._deps.emit(Notify(severity="error", text=f"Fork failed: {e}"))
            return
        self._deps.emit(SessionCreated(row=self._row(fork)))
        self._emit_rows()
        self._deps.emit(
            Notify(
                text=f"Forked “{source.title}” — "
                "this copy stops before that message."
            )
        )

    async def _rollback_session(self, session_id: str, index: int) -> None:
        """Trim a conversation back to before ``index``, in place.

        Destructive and irreversible, so this half *is* gated: everything
        `rewind_blocker` names either writes to the thread about to be
        shortened or is parked inside it. The answer comes back as the reason
        it gives, which is phrased to be shown.
        """
        if self._known(session_id) is None:
            return
        blocker = self._scheduler.rewind_blocker(session_id)
        if blocker is not None:
            self._deps.emit(
                Notify(severity="warning", text=f"Cannot roll back: {blocker}")
            )
            return
        keep = await self._cut_point(session_id, index)
        if keep is None:
            return
        try:
            await rollback_thread(self._graph, session_id=session_id, keep=keep)
        except Exception as e:
            self._deps.emit(
                Notify(severity="error", text=f"Rollback failed: {e}")
            )
            return
        # The measured fill described the untrimmed thread; re-derive it from
        # what is left, exactly as re-opening the session would.
        self._backends.forget_session(session_id)
        await self._reset_chat(session_id)
        self._deps.emit(
            Notify(text="Rolled back — edit your message and send again.")
        )

    async def _cut_point(self, session_id: str, index: int) -> int | None:
        """``index`` as the ``keep`` length the graph takes, or None.

        The conversion `protocol._Rewind` deliberately keeps off the wire: the
        UI names a message it was shown, `fork_thread` and `rollback_thread`
        want a length, and the two are the same number only while that message
        is still where the user saw it. An index the thread no longer has is
        refused rather than truncating somewhere nobody pointed at — which is
        precisely what would happen if the raw number were passed through.
        """
        count = await thread_message_count(self._graph, session_id=session_id)
        if 0 <= index < count:
            return index
        self._deps.emit(
            Notify(
                severity="warning",
                text="That message is no longer in this conversation.",
            )
        )
        return None

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

    def _spawn(self, coro) -> asyncio.Task:
        """Run a command's slow half without holding up the dispatch loop.

        `handle` is called from a reader loop with one socket behind it, so a
        handler that awaits a generation stalls every command queued after it.
        Kept in a set so `stop` can cancel what is still in flight: a task that
        outlives the databases raises into a loop nobody is reading.

        Failures land where a failed command lands — a `notify` — rather than
        in an unretrieved exception nobody sees. Reported from a done callback
        rather than from a wrapper coroutine, because a wrapper that is
        cancelled before its first step never awaits what it was given, and
        the shutdown path cancels exactly there.
        """
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error("background command failed", exc_info=error)
                self._deps.emit(Notify(severity="error", text=str(error)))

        task.add_done_callback(finished)
        return task

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
        for task in [*self._timers, *self._tasks]:
            task.cancel()
        for task in [*self._timers, *self._tasks]:
            with suppress(Exception, asyncio.CancelledError):
                await task
        self._timers.clear()
        self._tasks.clear()
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
        # Each tool call and its result, the moment it happens. The scheduler
        # takes it because it is the thing that names chat rows: a call has to
        # appear as a row and its result has to land in that same row.
        on_step=lambda sid, step: scheduler_ref["scheduler"].report_step(sid, step),
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
        checkpointer=checkpointer,
    )
    service_ref["service"] = service
    # Triage's "shall I learn this signature?" offer needs somewhere to ask.
    # Wired after construction because the service is what holds the pending
    # question, and the poller is built before it.
    pollers._confirm = lambda question, on_yes: service.ask(question, on_yes)
    for event in events:
        service._fan_out(event)
    return service


def _settings_digest(settings) -> str:
    """A short fingerprint of the settings, for `hello`.

    A digest and not the settings: they hold api keys, and a front-end that
    only needs to notice "these changed under me" has no business being handed
    them (§4.2 rule 2 in spirit — the UI does not read the core's state, it is
    told what it has to draw). Short because it is compared, never inspected.

    A settings object the core cannot serialise still has to produce something
    — the tests build services around fakes — so a failure here is a digest of
    nothing rather than a core that cannot greet a client.
    """
    try:
        blob = settings.model_dump_json()
    except Exception:
        blob = repr(settings)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


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
    from hpca.jobs import JobStore
    from hpca.logs import LoggedLLM
    from hpca.rag import RagStore
    from hpca.runner import ProcessRunner
    from hpca.symbols import SymbolIndex
    from hpca.trash import TrashManager

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
