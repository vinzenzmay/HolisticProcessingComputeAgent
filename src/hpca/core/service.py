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
from datetime import datetime, timezone
from typing import Any

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.context import ToolContext
from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
from hpca.agent.file_tools import add_file_tools
from hpca.agent.graph import (
    build_graph,
    compact_now,
    fork_thread,
    rollback_thread,
    thread_message_count,
)
from hpca.agent.job_tools import add_job_tools
from hpca.agent.memory_context import build_memory_context, compose_api_content
from hpca.agent.memory_tools import add_memory_tools
from hpca.agent.middleware import uses_native_tools
from hpca.agent.modes import MODES, add_plan_tool
from hpca.agent.prompts import (
    build_skill_directive,
    environment_facts,
    orchestrator_system_prompt,
)
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.titler import propose_title
from hpca.agent.watch_tools import add_watch_tools
from hpca.config import LLMBackend
from hpca.core.backends import BackendRegistry
from hpca.core.deps import CoreDeps
from hpca.core.memory_service import KIND_REFLECTION, MemoryService
from hpca.core.pollers import Pollers
from hpca.core.scheduler import TurnPlan, TurnScheduler, wire_entry
from hpca.db import record_command_use
from hpca.episodic import EpisodicStore
from hpca.jobs import JobStore
from hpca.logs import open_log
from hpca.profiles import DEFAULT_PROFILE, Profile
from hpca.protocol import (
    PROTOCOL_VERSION,
    BackendSet,
    ChatReset,
    CommandRun,
    ConfirmRequested,
    ConfirmResolve,
    DecisionRequested,
    DecisionResolve,
    Hello,
    JobCancel,
    LLMCatalog,
    LLMList,
    MemoryResolve,
    Message,
    ModeSet,
    Notify,
    ProcessKill,
    ProfileCreate,
    ProfileDelete,
    ProfileDuplicate,
    ProfileList,
    ProfileRow,
    ProfileRows,
    ProfileSave,
    ProfileSet,
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
    SkillDelete,
    SkillSave,
    ThinkingSet,
    TurnActivity,
    TurnInterrupt,
    TurnInterrupted,
    TurnSubmit,
    TurnUnqueue,
    TurnUnqueued,
    WatchDrop,
    WatchPeek,
    WatchPeeked,
)
from hpca.runner import kill_unowned
from hpca.skills import load_own_skills, load_project_skills, load_skills
from hpca.thinking import EFFORT_HINTS, EFFORTS, XHIGH_WARNING
from hpca.transcript import build_entries
from hpca.watches import KIND_LOG, WatchStore, peek, watch_lines

logger = logging.getLogger("hpca.core.service")

# The slash commands this core answers — the seven built-ins of §4.1. Kept as
# a set rather than inferred from the handler chain because it is also what
# decides whether a command is worth counting for the front-end's frequency
# sort: an unknown one must not teach the menu a name nothing can run.
SLASH_COMMANDS = frozenset(
    {
        "compact",
        "memorize",
        "conclude",
        "thinking",
        "skills-list",
        "skill-remove",
        "skill-creator",
    }
)

# How `/skills-list` marks where a skill resolved from. The profile's own carry
# no tag — they are the removable, unsurprising case; everything else says
# where it came from, because that is what decides whether removing it would
# change another profile.
SKILL_LEVEL_TAGS = {
    "project": "  (project)",
    "profile": "",
    "global": "  (global)",
    "builtin": "  (built-in)",
}

# The name a conversation has before anything has named it — the same string
# `sessions.SessionStore.create` defaults to, passed explicitly below so that
# the two places that care are visibly one decision.
#
# It carries a second meaning the core relies on twice, and it is the only
# durable answer to "has anyone named this?": a title still equal to this was
# written by nobody. That distinguishes a never-named session from one the
# user renamed, which is what stops the model overwriting a hand-written name
# (§ automatic titling) and what makes an untouched session safe to reuse
# (`_reusable_session`). A flag on the row would say the same thing and would
# need a migration to say it; the placeholder is already stored, already
# survives a restart, and is already what the sidebar draws.
UNTITLED_SESSION = "untitled"

# How much of the first message becomes the provisional name. Long enough to
# recognise the conversation in a narrow column, short enough that the row
# does not become the message (`tui/app.py:SESSION_TITLE_MAX`).
SESSION_TITLE_MAX = 40


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
        # Sessions carrying a provisional name — the truncated first message —
        # and waiting for the model to write a real one after the exchange.
        #
        # In memory on purpose, which is what makes "a reopened session is not
        # retitled" fall out rather than need arranging: the set is emptied by
        # a restart, so a conversation whose first exchange happened in an
        # earlier run is never named again behind the user's back. Whether a
        # session was ever named at all is the durable question, and
        # UNTITLED_SESSION answers that one.
        self._untitled: set[str] = set()
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
            await self._new_session(command)
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
            # A conversation nobody has named takes the opening message as a
            # provisional name, and is queued for the model to name properly
            # once this exchange is over (`_name_provisionally`). Here rather
            # than in the scheduler because it is a *typed* message that does
            # it: an event delivered into a session names nothing.
            self._name_provisionally(session, command.text)
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
            text = await self._scheduler.interrupt(command.session_id)
            if text is None:
                return
            # The interrupt rolls the abandoned attempt out of the thread, so
            # the rows drawn for it now describe messages that are gone. A
            # delta cannot take a row off the screen; a reset can, and this is
            # the same case `session.rollback` is — an open of what is left.
            await self._reset_chat(command.session_id)
            # And the message itself comes back to be edited and re-sent —
            # after the reset, because the chat it was in has just been
            # re-stated, and addressed, because it belongs to the session it
            # was typed in and not to whichever one is on screen when this
            # lands (`protocol.TurnInterrupted`).
            self._deps.emit(
                TurnInterrupted(session_id=command.session_id, text=text)
            )
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
        if isinstance(command, MemoryResolve):
            self._resolve_memory(command)
            return
        if isinstance(command, ModeSet):
            self._set_mode(command.session_id, command.mode)
            return
        if isinstance(command, ThinkingSet):
            self._set_thinking(command.session_id, command.effort)
            return
        if isinstance(command, BackendSet):
            await self._set_backend(command)
            return
        if isinstance(command, LLMList):
            self._emit_catalog()
            if command.probe:
                # The probes are round trips to cluster nodes, so they never
                # hold up the frame that lets the screen draw: the catalog has
                # gone out already and the marks arrive in a second one.
                self._spawn(self._probe_catalog())
            return
        if isinstance(command, ProfileList):
            self._emit_profiles()
            return
        if isinstance(command, ProfileSet):
            await self._set_profile(command.name)
            return
        if isinstance(command, ProfileSave):
            self._save_profile(command)
            return
        if isinstance(command, ProfileCreate):
            self._create_profile(command.name)
            return
        if isinstance(command, ProfileDuplicate):
            self._duplicate_profile(command)
            return
        if isinstance(command, ProfileDelete):
            await self._delete_profile(command.name)
            return
        if isinstance(command, SkillSave):
            self._save_skill(command)
            return
        if isinstance(command, SkillDelete):
            self._memory.delete_profile_skill(command.profile, command.name)
            return
        if isinstance(command, WatchPeek):
            await self._peek_watch(command.watch_id)
            return
        if isinstance(command, WatchDrop):
            await self._drop_watch(command.watch_id)
            return
        if isinstance(command, ProcessKill):
            await self._kill_process(command.pid)
            return
        if isinstance(command, JobCancel):
            await self._cancel_job(command.job_id)
            return
        if isinstance(command, CommandRun):
            await self._run_slash(command)
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
            # The stored level, not the resolved one, for the same reason as
            # `mode`: empty means "follows the setting", and a row that
            # answered with the default could not say which of the two it was.
            thinking=session.thinking,
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

    async def _new_session(self, command: SessionNew) -> None:
        """Make one — or hand back the empty one already on screen — and say
        which it is.

        `session.created` before `session.rows` on purpose: the UI has to open
        it, and a sidebar cannot say which of its lines is new (see
        `protocol.SessionCreated`). Emitted for a reused session too: what the
        front-end asked for is a conversation to type in, and it must be told
        which one that is whether or not a row was added.
        """
        settings = self._deps.settings
        profile = command.profile or self._deps.profile
        # A blob, not a label, so the choice survives the entry being dropped
        # from the catalog later; None means the label named nothing, which is
        # said out loud rather than absorbed as "the default" — that silence is
        # what made a front-end sending what `protocol.SessionNew` documented
        # pin nothing at all.
        backend = self._backends.backend_for_new_session(command.backend)
        if backend is None:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"No LLM called “{command.backend}” — this "
                    "conversation talks to the default one.",
                )
            )
            backend = ""
        session = await self._reusable_session()
        if session is not None:
            session = self._retag(session, profile=profile, backend=backend)
        else:
            session = self._sessions.create(
                profile=profile,
                title=UNTITLED_SESSION,
                mode=settings.agent.default_mode,
                backend=backend,
                thinking=settings.agent.default_thinking,
            )
        self._deps.emit(SessionCreated(row=self._row(session)))
        self._emit_rows()

    async def _reusable_session(self):
        """The conversation on screen, when starting a new one would only
        duplicate it — else None.

        "(new session)" pressed twice used to leave two empty rows, and the
        Textual app avoided that by reusing an untouched session and retagging
        it to whatever profile and backend had just been picked. That belongs
        here and could not live anywhere else: the retag is a change to an
        existing session's profile, which no command in §4.1 can express, and
        "untouched" is a question about the checkpointed thread, which §4.2
        rule 2 puts out of a front-end's reach.

        Untouched means four things, and all four have to hold. Nobody has
        named it (the title is still the placeholder — a renamed empty session
        is one the user meant to keep, and retagging it under them would move
        a conversation they had named to another profile). Its thread is
        empty. No turn is running on it. And nothing is queued or parked
        against it — both would land in a thread that is about to change
        profile underneath them.
        """
        session = self._session(self._deps.focused_session_id)
        if session is None or session.title != UNTITLED_SESSION:
            return None
        session_id = session.session_id
        if (
            self._scheduler.is_busy(session_id)
            or self._scheduler.queued_texts_for(session_id)
            or session_id in self._scheduler.pending_decisions()
        ):
            return None
        values = await self._thread_values(session_id)
        return None if values.get("messages") else session

    def _retag(self, session, *, profile: str, backend: str):
        """Point an empty conversation at the profile and LLM just chosen.

        The backend is only rewritten when one was actually asked for: a
        `session.new` that names no backend means "whatever the core would
        pick", and reading that as "unpin this session" would silently undo a
        choice the user made a moment ago in the same empty session.

        The measured window goes with a backend change for the reason
        `switch_backend` gives — a different backend is a different window, and
        the fill this session was showing describes the wrong denominator.
        """
        if session.profile != profile:
            self._sessions.set_profile(session.session_id, profile)
        if backend and backend != session.backend:
            self._sessions.set_backend(session.session_id, backend)
            self._backends.forget_session(session.session_id)
        return self._sessions.get(session.session_id) or session

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
        session = self._known(session_id)
        if session is None:
            return
        if not title.strip():
            # A nameless row is a row the user cannot find again. Refused
            # here rather than stored, because the store would take it.
            self._deps.emit(
                Notify(severity="warning", text="A session needs a name.")
            )
            return
        self._store_title(session, title, by="user")
        self._emit_rows()

    def _store_title(self, session, title: str, *, by: str, log=None) -> None:
        """Write a session's name, record who wrote it, and stop the model
        from writing over it.

        The one place a title is stored, because two things have to happen with
        it and both were missed by having three. It goes in the transcript —
        the log is the durable record of the conversation, and a session that
        changes name halfway through it is otherwise two records that cannot be
        told apart. And it takes the session out of the queue for automatic
        titling: a name a person typed is never overwritten by the model, which
        is the whole reason that queue is a set of ids rather than a rule about
        the title's shape.

        ``log`` is the turn's own log when a turn is what caused this, so the
        line lands in the transcript of the session being renamed rather than
        wherever the user has since navigated (`tui/app.py:_rename_session`
        made the same distinction).
        """
        self._sessions.rename(session.session_id, title)
        self._untitled.discard(session.session_id)
        sink = log if log is not None else open_log(self._deps.settings, session)
        if sink is not None:
            sink.write("session renamed", f"{title} (by {by})")

    def _name_provisionally(self, session, text: str) -> None:
        """Name a never-named conversation after its opening message, and put
        it in the queue for a real title.

        The truncated message is a placeholder and is meant to be replaced:
        it is cut mid-word, and it describes where the conversation started
        rather than what it became. It exists because the alternative is a
        sidebar row that says "untitled" for as long as the first exchange
        takes, and because the model's title has to have something to replace
        if that call fails.

        Not stored through `_store_title`: this is not somebody naming the
        session, and passing it through the one place that records an author
        would also cancel the titling this exists to schedule.
        """
        if session.title != UNTITLED_SESSION:
            return
        provisional = text.strip()[:SESSION_TITLE_MAX] or UNTITLED_SESSION
        self._sessions.rename(session.session_id, provisional)
        self._untitled.add(session.session_id)
        self._emit_rows()

    async def after_turn(self, session, result, plan) -> None:
        """What follows a turn that is not the turn's own business.

        Wired to `TurnScheduler`'s `on_turn_result`, which existed for exactly
        this and had nothing passed to it — so no session was ever titled after
        its first exchange and `session.retitle` was the whole of naming.

        Nothing here may fail a turn, and the scheduler already guarantees that
        by catching; what it cannot do is decide *when*. A turn parked on an
        approval has not finished — the exchange continues in the resume — so
        naming it now would summarise half a conversation and, worse, spend the
        one attempt doing it (`tui/app.py` returns at the same point).
        """
        if getattr(result, "interrupt", None) is not None:
            return
        await self._title_after_turn(
            session, list(getattr(result, "messages", []) or []), plan
        )

    async def _title_after_turn(self, session, messages: list, plan) -> None:
        """Give a freshly-started conversation the model's name for it, once.

        Once, and quietly. The membership test is the whole of the policy:
        a session is put in `_untitled` by its first typed message and taken
        out by the first thing that names it, so a second exchange finds
        nothing to do, a reopened session was never in the set, a slash command
        never put one there, and a hand-written name has already removed it.

        A failure is silent — no toast, no retry. The provisional name is still
        a name, `t` is always there, and a turn's reply is what the user is
        waiting to read; an error toast about the *title* on the back of it
        would be the loudest thing on screen for the smallest reason. The
        explicit `session.retitle` reports, because there somebody asked.
        """
        session_id = session.session_id
        if session_id not in self._untitled:
            return
        self._untitled.discard(session_id)  # one attempt, whatever comes of it
        if not messages:
            return
        try:
            title = await propose_title(
                self._backends.labelled_client(
                    "title", session_id=session_id, log=getattr(plan, "log", None)
                ),
                messages,
            )
        except Exception:
            logger.exception("automatic titling failed")
            return
        if self._session(session_id) is None:
            return  # deleted while the title was being written
        self._store_title(session, title, by="llm", log=getattr(plan, "log", None))
        self._emit_rows()

    async def _retitle(self, session_id: str) -> None:
        """Ask the model to name this conversation (`session.retitle`).

        Routed through the session's *own* client, not the bootstrap one: a
        session pinned to a backend must not have its title written by
        whichever model the core happens to be holding.

        Reported as activity, because this is a silent backend call: from
        outside, a title being written is indistinguishable from a core that
        has stopped answering. The same event a turn uses, for the reason
        `_working` gives — what the user needs to know is that the session is
        busy and for how long, not which part of the core is busy.
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
        # Only when nothing else is holding the session up: a turn running in
        # it is already reporting its own activity, and the empty label below
        # would end that report rather than this one.
        quiet = not self._scheduler.is_busy(session_id)
        if quiet:
            self._working(session_id, "writing a title")
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
        finally:
            if quiet:
                self._working(session_id, "")
        self._store_title(session, title, by="llm")
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
        self._untitled.discard(session_id)
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

    # ------------------------------------------------------------ the dials

    def _set_mode(self, session_id: str, mode: str) -> None:
        """`mode.set`: how much this conversation asks before it acts (§3.5).

        Stored and then restated, rather than acknowledged: the mode is a
        column of the sidebar row, so the frame that says it worked is the
        same frame that redraws it — and it redraws it for every client, which
        a reply addressed to the sender could not.

        An unknown mode is refused here because the set of them is the agent's
        business, which is exactly why the wire carries a plain string
        (`protocol.ModeSet`): validating it at the edge would put a second
        copy of that list in a module that must not have an opinion about it.
        """
        if self._known(session_id) is None:
            return
        if mode not in MODES:
            self._deps.emit(
                Notify(severity="warning", text=f"There is no “{mode}” mode.")
            )
            return
        self._sessions.set_mode(session_id, mode)
        self._emit_rows()

    def _set_thinking(self, session_id: str, effort: str) -> None:
        """`thinking.set`: how hard this conversation reasons (hpca.thinking).

        The same shape as the mode above, including the sidebar restatement —
        the level rides on `SessionRow`, so this is also the event the context
        meter's `· think medium` reads (see `protocol.SessionRow.thinking`).

        xhigh gets a warning rather than a confirmation because it does not
        work: the level is offered since the model advertises it, and a user
        who picks it needs to be told before the first lost turn rather than
        after it. The headline is in the title for the reason `tui/app.py`
        gave — a toast is read in the order it is laid out, and this one has to
        land even if the paragraph under it is skimmed.
        """
        if self._known(session_id) is None:
            return
        if effort not in EFFORTS:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"There is no “{effort}” thinking level.",
                )
            )
            return
        self._sessions.set_thinking(session_id, effort)
        self._emit_rows()
        if effort == "xhigh":
            self._deps.emit(
                Notify(
                    severity="warning",
                    title="Thinking: xhigh — NOT USABLE",
                    text=XHIGH_WARNING,
                    timeout=25,
                )
            )
            return
        self._deps.emit(
            Notify(text=f"Thinking effort for this session: {effort}")
        )

    async def _set_backend(self, command: BackendSet) -> None:
        """`backend.set`: which model answers — one session's, or everyone's.

        Two operations behind one command, told apart by whether a session is
        named (`protocol.BackendSet`), and they are genuinely different: the
        per-session half rewrites one row and is refused only while THAT
        session is mid-reply, while the global half writes the settings, so it
        outlives the run, decides what the next session is created against, and
        is refused while any turn anywhere is in flight (`BackendRegistry`).

        The blob is validated against the real settings model rather than
        modelled a second time in the protocol — which is why it crosses as an
        opaque dict, and why an unusable one is refused here.
        """
        try:
            backend = LLMBackend.model_validate(command.backend)
        except Exception as e:
            # A front-end sending a shape the settings model does not accept is
            # a bug on its side; pinning a session to it would strand the
            # conversation on a backend nothing can build a client from.
            self._deps.emit(
                Notify(severity="error", text=f"Not a usable backend: {e}")
            )
            return
        if command.session_id is None:
            await self._backends.set_default(
                backend, busy=bool(self._scheduler.busy_sessions())
            )
            # Every session that pinned nothing now names a different model.
            self._emit_rows()
            # And the ★ has moved — `set_default` adds an unlisted backend to
            # the catalog on its way past, so this can be a new row as well as
            # a moved mark.
            self._emit_catalog()
            return
        if self._known(command.session_id) is None:
            return
        switched = self._backends.switch_backend(
            command.session_id,
            backend,
            busy=self._scheduler.is_busy(command.session_id),
        )
        if switched:
            self._emit_rows()  # the row carries the model name

    def _emit_catalog(self, reachable: dict[str, bool] | None = None) -> None:
        """The LLM catalog, whole (`protocol.LLMCatalog`).

        The event that was missing: with the settings out of a front-end's
        reach (§4.2 rule 2), nothing carried the configured backends across, so
        the new-session picker and the manage-LLMs screen could only ever be
        drawn from a list a demo filled in.

        Whole rather than incremental for the same reason `session.rows` is,
        and restated wherever it changes — a default switched, an entry added,
        the probes landing.
        """
        self._deps.emit(
            LLMCatalog(
                entries=self._backends.catalog(reachable=reachable),
                probed=reachable is not None,
            )
        )

    async def _probe_catalog(self) -> None:
        """Ask each configured endpoint whether it is up, then say so.

        A second frame rather than a delayed first one. A probe is a round trip
        to a node that may have gone away, so waiting for the slowest of them
        before answering `llm.list` would leave a picker blank for the full
        timeout; the screen draws immediately with nothing marked and the
        ● / ○ arrive when they arrive (`tui/switch_llm.py` did the same with a
        worker that filled in "…" per row).
        """
        self._emit_catalog(await self._backends.probe_catalog())

    # --------------------------------------------------------------- profiles

    def _emit_profiles(self) -> None:
        """The profiles, whole — the answer to `profile.list`.

        Read here rather than derived by a front-end, which is what was
        happening: a picker was assembled out of `hello`'s profile plus
        whatever profiles the sidebar rows named, which misses every profile
        that has no session, and can carry neither the memory count nor the
        provenance because those live in files only the core reads.

        A broken profile file counts nothing rather than taking the listing
        down with it: the screen exists partly so that such a profile can be
        opened and fixed, and it cannot be opened from a screen that failed to
        draw.
        """
        rows = []
        for name in Profile.list_profiles():
            memories, copied_from = 0, ""
            try:
                profile = Profile.load(name)
            except Exception:
                logger.exception("could not read profile %s", name)
            else:
                memories = len(profile.memories)
                copied_from = profile.copied_from
            rows.append(
                ProfileRow(
                    name=name,
                    memories=memories,
                    copied_from=copied_from,
                    # Two different questions: which profile a deleted one's
                    # sessions fall back to, and which one the core is running
                    # under right now (`protocol.ProfileRow`).
                    is_default=name == DEFAULT_PROFILE,
                    working=name == self._deps.profile,
                )
            )
        self._deps.emit(ProfileRows(rows=rows))

    async def _set_profile(self, name: str) -> None:
        """`profile.set`: the profile the core works under when nothing else
        narrows it — a new session's default, and whose watches an unfocused
        panel shows (`CoreDeps.profile`).

        Deliberately not "the open session's profile": a session's profile is
        the session's, and changing what the core is working under must not
        silently move a conversation to another set of memories.

        The panel is repainted because `panel.update` is the one event that
        carries the working profile's name — `hello` also does, but only to a
        client that is connecting, and this can happen at any time.
        """
        if name not in Profile.list_profiles():
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"There is no profile called “{name}”.",
                )
            )
            return
        self._memory.set_working_profile(name)
        self._deps.emit(Notify(text=f"Working profile: {name}"))
        # The listing carries which profile is the working one, so it is now
        # out of date on every client that is showing it.
        self._emit_profiles()
        await self._pollers.refresh_panel(force=True)

    def _save_profile(self, command: ProfileSave) -> None:
        """`profile.save`: write back what the user edited in $EDITOR (§4.4).

        The editor is the front-end's (it needs a terminal to suspend), the
        file is the core's, so the flow ends here with the finished text. Both
        halves report their own outcome, including a memory file that no longer
        parses — the edits are the user's and are never refused, only flagged.
        """
        if command.kind == "memories":
            self._memory.save_profile_memories(command.name, command.text)
            return
        self._memory.save_profile_archive(command.name, command.text)

    def _create_profile(self, name: str) -> None:
        error = self._memory.create_profile(name)
        if error is not None:
            # The name is a filename: what it may contain, and that it may not
            # collide, is the profile store's rule and its wording.
            self._deps.emit(Notify(severity="warning", text=error))
            return
        self._deps.emit(Notify(text=f"Created profile “{name.strip()}”."))
        self._emit_profiles()

    def _duplicate_profile(self, command: ProfileDuplicate) -> None:
        """`profile.duplicate`: same learnings, its own future.

        ``source`` defaults to the working profile, which is what "duplicate
        this one" means from a screen that is already showing it.
        """
        error = self._memory.duplicate_profile(
            command.source or self._deps.profile, command.name
        )
        if error is not None:
            self._deps.emit(Notify(severity="warning", text=error))
            return
        # The copy announces itself (it counts what came along); nothing in the
        # sidebar changed, since sessions belong to conversations rather than
        # to the knowledge that came out of them. The profile listing did: a
        # copy is a new row, and it is the one row that carries a provenance.
        self._emit_profiles()

    async def _delete_profile(self, name: str) -> None:
        """`profile.delete`: drop a profile and everything it learned.

        Three refusals, in the order that costs least to find out: the default
        cannot go at all (it is where deleted profiles' sessions land, so
        removing it would leave them pointing at nothing), a name that does not
        exist is a stale screen, and a profile in use — a reply in flight, or a
        live subprocess under one of its sessions — would strand running work.

        Its sessions are reassigned to the default rather than deleted, which
        is why the sidebar is restated afterwards: every row that named this
        profile now names another.
        """
        if name == DEFAULT_PROFILE:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text="The default profile cannot be deleted — it is where "
                    "other profiles' sessions go.",
                )
            )
            return
        if name not in Profile.list_profiles():
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"There is no profile called “{name}”.",
                )
            )
            return
        blocker = await self._memory.profile_delete_blocker(name)
        if blocker is not None:
            self._deps.emit(Notify(severity="warning", text=blocker))
            return
        await self._memory.delete_profile(name)
        self._emit_rows()
        self._emit_profiles()

    def _save_skill(self, command: SkillSave) -> None:
        """`skill.save`: persist a skill file verbatim, front matter and all.

        Written into the profile's own directory and nowhere else. The global
        (`_shared/`) level is deliberately unreachable from here: one profile
        editing a procedure every other profile can see is a change nobody
        asked for, and the same rule already governs deletion.
        """
        if command.text is None:
            # The shape is shared with `skill.delete`, where the body is
            # meaningless; saving without one would truncate the file.
            self._deps.emit(
                Notify(severity="warning", text="A skill needs a body to save.")
            )
            return
        self._memory.save_skill_file(command.profile, command.name, command.text)

    # ---------------------------------------------------------------- watches

    async def _peek_watch(self, watch_id: int) -> None:
        """`watch.peek`: what is this box saying right now?

        The one read-only command in §4.1, and still a command: the tail lives
        in a file on a node the front-end may not share, and rule 2 of §4.2
        puts the database out of its reach either way.

        A job answers with its state line plus its output, but only when hpca
        submitted it — a job the user sbatch'ed by hand has no known stdout
        path, and the state is then the whole answer. A read that failed comes
        back as the text (`watches.peek` says so in words), because what
        happened to the log is exactly what was asked.
        """
        watch = await self._deps.db(lambda conn: WatchStore(conn).get(watch_id))
        if watch is None:
            self._deps.emit(
                Notify(severity="warning", text="That box is gone.")
            )
            return
        if watch.kind == KIND_LOG:
            text = await self._tail(watch.target)
        else:
            text = " · ".join(part for part in watch_lines(watch) if part)
            row = await self._deps.db(
                lambda conn: JobStore(conn).get(watch.target)
            )
            if row is not None and row.sbatch_stdout_path:
                text += "\n" + await self._tail(row.sbatch_stdout_path)
        self._deps.emit(
            WatchPeeked(watch_id=watch_id, title=watch.title, text=text)
        )

    @staticmethod
    async def _tail(path: str) -> str:
        """`watches.peek`, off the dispatch loop.

        The read is small by construction but the file is a job log on a
        cluster filesystem, where a stat can cost a network round trip — and
        this loop has one socket and every other session's commands behind it.
        The same reason `deps.db` exists, for a file rather than a database.
        """
        return await asyncio.to_thread(peek, path)

    async def _drop_watch(self, watch_id: int) -> None:
        """`watch.drop`: stop watching, and repaint the column.

        Nothing is deleted but the box — the log and the job are untouched —
        which is why there is no confirmation step: the usual reason for
        pressing it is that the run is over and the box has stopped saying
        anything.
        """
        watch = await self._deps.db(lambda conn: WatchStore(conn).get(watch_id))
        removed = await self._deps.db(
            lambda conn: WatchStore(conn).remove(watch_id)
        )
        if not removed:
            self._deps.emit(
                Notify(severity="warning", text="That box is gone.")
            )
            return
        self._deps.emit(Notify(text=f"Stopped watching {watch.title}"))
        await self._pollers.refresh_panel(force=True)

    # ------------------------------------------------------- running work

    async def _kill_process(self, pid: int) -> None:
        """`process.kill`: stop a background subprocess by pid.

        Preferring the runner that started it, when there still is one: its
        monitor is what records how the process ended, so a kill it can see
        settles the row properly. A process from an earlier turn has no live
        monitor — the runner is built per turn — and `kill_unowned` both
        signals it and settles the row itself, which is what stops the history
        claiming it is still running forever.

        The row is looked up by pid alone because that is all the command
        carries, and no store call takes a bare one: the panel that used to
        offer this listed processes per session, and it does not any more.
        """
        row = await self._deps.db(
            lambda conn: conn.execute(
                "SELECT session_id, name, state FROM processes WHERE pid = ?",
                (pid,),
            ).fetchone()
        )
        if row is None:
            self._deps.emit(
                Notify(severity="warning", text=f"No process with pid {pid}.")
            )
            return
        if row["state"] != "running":
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"{row['name']} is not running ({row['state']}).",
                )
            )
            return
        session_id = row["session_id"]
        ctx = self._scheduler.tool_context(session_id)
        runner = getattr(ctx, "runner", None)
        if runner is not None and runner.owns(pid):
            await runner.kill(pid)
            await runner.wait(pid)
        else:
            await self._deps.db(
                lambda conn: kill_unowned(conn, pid=pid, session_id=session_id)
            )
        self._deps.emit(Notify(text=f"Killed {row['name']} (pid {pid})."))

    async def _cancel_job(self, job_id: str) -> None:
        """`job.cancel`: scancel a job hpca submitted.

        The provisional state is written straight away rather than waited for:
        scancel returns before the scheduler has acted, and sacct is what
        confirms it on the next poll (`JobStore.mark`). Until then a box that
        says CANCELLING is the honest answer.
        """
        slurm = self._deps.slurm
        if slurm is None:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text="No cluster is configured — there is nothing to cancel.",
                )
            )
            return
        try:
            await slurm.cancel(job_id)
        except Exception as e:
            self._deps.emit(
                Notify(severity="error", text=f"Cancel failed: {e}")
            )
            return
        await self._deps.db(
            lambda conn: JobStore(conn).mark(job_id, "CANCELLING")
        )
        self._deps.emit(Notify(text=f"Cancelling job {job_id}"))
        await self._pollers.refresh_panel(force=True)

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

    # ------------------------------------------------------ memory proposals

    def _resolve_memory(self, command: MemoryResolve) -> None:
        """`memory.resolve`: write exactly the proposals that were approved.

        Positional against the set the core still holds, and that asymmetry is
        the design point (`protocol.MemoryResolve`): the authoritative objects
        never leave this process, so an approval cannot carry an edited memory
        back in. A short list rejects the rest — an answer that never arrived
        is not an approval.

        An answer to nothing is a stale screen, not an error: a review that has
        already been applied, or one that belonged to a session since deleted.

        The `/conclude` chain continues from here. The self-review and the
        facts the agent flagged mid-session are two rounds of one pass, and
        they cannot be offered together — one session holds one unanswered set,
        so a second offer would overwrite the first. So the flagged batch is
        put up once the reflections have been answered, which is also the order
        the user reads them in.
        """
        pending = self._memory.pending_proposals(command.session_id)
        if pending is None:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text="There is nothing waiting to be reviewed.",
                )
            )
            return
        kind = pending.kind
        kept = self._memory.apply_answered(command.session_id, command.approved)
        followed = False
        if kind == KIND_REFLECTION:
            session = self._session(command.session_id)
            if session is not None:
                followed = self._memory.propose_flagged_edits(session) is not None
        if kept:
            self._deps.emit(
                Notify(
                    text=f"Kept {kept} memor{'y' if kept == 1 else 'ies'}."
                )
            )
        elif not followed:
            # Silent only while a second round is on its way: "nothing kept"
            # ahead of the batch still to be reviewed would read as the end.
            self._deps.emit(Notify(text="Nothing kept."))

    # ----------------------------------------------------------- slash commands

    async def _run_slash(self, command: CommandRun) -> None:
        """`command.run`: the seven built-in slash commands.

        The front-end parses `/name rest` and sends both halves; what ``rest``
        means is each handler's business (a note, a level, a skill name,
        nothing). Which conversation a session-scoped one acts on is named
        explicitly rather than taken from the last `session.focus`, so a
        destructive fold cannot be aimed at whatever the user switched to
        between the keystroke and the frame arriving.

        Three of them are slow — they call a model — and none may hold up the
        dispatch loop, which has one socket behind it. So the gates that can be
        checked cheaply are checked here, synchronously, and only the work goes
        on a task (`_spawn`).
        """
        name = command.name.lstrip("/").strip()
        args = command.args.strip()
        if name in SLASH_COMMANDS:
            # The frequency sort behind the front-end's "/" menu. Counted here
            # because the table is the core's; nothing serves the counts back
            # yet, and recording them anyway is what keeps that possible.
            await self._deps.db(lambda conn: record_command_use(conn, name))
        if name == "compact":
            session = self._session_for_command(command)
            if session is not None:
                self._spawn(self._compact(session, args))
            return
        if name == "memorize":
            if not args:
                self._deps.emit(
                    Notify(severity="warning", text="Usage: /memorize <note>")
                )
                return
            session = self._session_for_command(command)
            if session is not None:
                self._spawn(self._memorize(session, args))
            return
        if name == "conclude":
            session = self._session_for_command(command)
            if session is not None:
                self._spawn(self._conclude(session))
            return
        if name == "thinking":
            self._thinking_command(command, args)
            return
        if name == "skills-list":
            self._list_skills(command.session_id)
            return
        if name == "skill-remove":
            self._remove_skill(command.session_id, args)
            return
        if name == "skill-creator":
            # The one built-in with no core-side shape. It is a form — name,
            # description, body, and which level to write at — and the model's
            # draft is a head start inside that form, not an answer. There is
            # no event that could carry a draft to a front-end and no way for
            # the core to know what came back out of the editing, so the whole
            # command belongs on the other side of the socket, ending in the
            # `skill.save` this core already answers.
            self._deps.emit(
                Notify(
                    severity="warning",
                    text="/skill-creator is a form the front-end owns; the "
                    "finished skill arrives as skill.save.",
                )
            )
            return
        self._deps.emit(
            Notify(severity="warning", text=f"Unknown command: /{name}")
        )

    def _session_for_command(self, command: CommandRun):
        """The conversation a session-scoped slash command acts on, or None.

        None is already reported: either the command arrived without a session
        (a front-end sending `/compact` with nothing open) or it names one that
        has since been deleted, and both are refused the way every
        un-carry-out-able command is — one warning, no state change.
        """
        if command.session_id is None:
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"Open a session first — /{command.name} works on "
                    "one conversation.",
                )
            )
            return None
        return self._known(command.session_id)

    def _thinking_command(self, command: CommandRun, args: str) -> None:
        """`/thinking [level]`: set the session's level, or say what there is.

        With a level it is `thinking.set` typed instead of picked, and goes
        through the same handler so the two cannot drift. Without one the
        front-end is expected to put its own chooser up; the list is answered
        anyway, because the levels and what they cost are the core's to know
        and a client that has no chooser must still be able to find out.
        """
        if args:
            session_id = command.session_id
            if session_id is None:
                self._deps.emit(
                    Notify(
                        severity="warning",
                        text="Open a session first — thinking is per session.",
                    )
                )
                return
            self._set_thinking(session_id, args)
            return
        self._deps.emit(
            Notify(
                title="Thinking effort",
                text="\n".join(
                    f"• {level} — {EFFORT_HINTS.get(level, '')}"
                    for level in EFFORTS
                ),
            )
        )

    def _list_skills(self, session_id: str | None) -> None:
        """`/skills-list`: every skill the profile can see, and from where.

        Profile-scoped, so it answers for the *session's* profile when one is
        named and for the working profile otherwise — a user reading a session
        that belongs to another profile is asking about that one's skills.

        A `notify` with a heading rather than an event of its own: this is a
        block of text with a title, which is exactly what `Notify.title`
        exists for, and inventing an `inspect` event for one read-only listing
        would put a screen shape in the protocol.
        """
        profile = self._profile_for(session_id)
        visible = load_skills(profile, project_root=self._project_root)
        if not visible:
            self._deps.emit(
                Notify(text=f"No skills for profile “{profile}”.")
            )
            return
        lines = []
        for skill in visible:
            lines.append(f"• {skill.name}{SKILL_LEVEL_TAGS.get(skill.level, '')}")
            if skill.description:
                lines.append(f"    {skill.description}")
        self._deps.emit(
            Notify(title=f"Skills · profile “{profile}”", text="\n".join(lines))
        )

    def _remove_skill(self, session_id: str | None, args: str) -> None:
        """`/skill-remove [name]`: delete one of this profile's own skills.

        Named rather than picked, because a picker is a screen: with a name
        this is `skill.delete` typed instead of chosen, and without one it
        answers with what may be removed so that a front-end can offer the
        choice — or a user can simply retype the command with a name.

        Global (`_shared/`) and shipped skills are not offered and are refused
        underneath as well: removing one would change every other profile that
        sees it.
        """
        profile = self._profile_for(session_id)
        if args:
            self._memory.delete_profile_skill(profile, args)
            return
        removable = sorted(
            {s.name for s in load_own_skills(profile)}
            | {s.name for s in load_project_skills(project_root=self._project_root)}
        )
        if not removable:
            self._deps.emit(
                Notify(
                    text=f"Profile “{profile}” has no skills of its own to "
                    "remove.",
                )
            )
            return
        self._deps.emit(
            Notify(
                title="Removable skills",
                text="\n".join(f"• {name}" for name in removable)
                + "\n\nRemove one with /skill-remove <name>.",
            )
        )

    async def _compact(self, session, guidance: str) -> None:
        """`/compact [instruction]`: fold this conversation's history now.

        The automatic fold waits for the window to fill and keeps the recent
        turns verbatim, because it fires unasked. This one is asked for, so it
        folds everything and takes the rest of the typed line as its brief —
        material to preserve, or the step the user is about to take, which is
        the same instruction from the summarizer's point of view.

        **No `chat.reset`, and this is the one to be careful about.** A fold
        looks like the rollback next to it and is not: `rollback_thread`
        removes messages, so the rows drawn for them describe messages that no
        longer exist and only a reset can un-draw them, whereas `compact_now`
        writes a *view* — the stored history is untouched and every row on
        screen still names a message the thread still has. Re-stating the chat
        here would be the per-turn rebuild §4.2 exists to delete, in exchange
        for nothing.

        So what crosses is what changed: the summary the model will work from
        (worth reading once — the user may have named what it had to keep, and
        a small model does not always keep it), and the fill, because the last
        measured count described the unfolded prompt.
        """
        session_id = session.session_id
        if session_id in self._scheduler.pending_decisions():
            # The thread is parked on an interrupt; rewriting the state under
            # an unanswered decision is not something to do quietly.
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"“{session.title}” is waiting on an approval — "
                    "answer that first.",
                )
            )
            return
        self._working(session_id, "compacting context")
        try:
            folded = await compact_now(
                self._graph,
                session_id=session_id,
                llm=self._backends.labelled_client(
                    "compact", session_id=session_id
                ),
                guidance=guidance,
            )
        except Exception as e:
            # Nothing was written: the thread is exactly as it was.
            self._deps.emit(
                Notify(severity="error", text=f"/compact failed: {e}")
            )
            return
        finally:
            self._working(session_id, "")
        if folded is None:
            self._deps.emit(
                Notify(text="Nothing new to compact in this conversation.")
            )
            return
        note = (
            f"Context compacted: {folded['folded']} messages folded into a "
            "summary."
        )
        if guidance:
            note = f"{note} Asked to keep: {guidance}"
        log = open_log(self._deps.settings, session)
        if log is not None:
            log.write("context compacted", f"{note}\n{folded['summary']['content']}")
        # The measured count described the unfolded prompt, so it no longer
        # describes what the next turn will send: drop it and re-derive the
        # fill from the folded view, which `estimate_context` already accounts
        # for.
        self._backends.forget_session(session_id)
        self._backends.estimate_context(
            session_id, await self._thread_values(session_id)
        )
        self._deps.emit(
            Notify(
                title="Context compacted",
                text=f"{note}\n\n{folded['summary']['content']}",
                timeout=20,
            )
        )

    async def _memorize(self, session, note: str) -> None:
        """`/memorize <note>`: turn the note plus the conversation into
        proposals, each of which still needs the user's approval (§5.3).

        Session-scoped even though the note is the substance of it, because
        the answer has to come back addressed: the proposals are held per
        session and `memory.resolve` names one. A memory formed against no
        conversation would have no id to be approved under.
        """
        messages = list((await self._thread_values(session.session_id)).get(
            "messages", []
        ))
        await self._memory.propose_from_note(session, messages, note)

    async def _conclude(self, session) -> None:
        """`/conclude`: what is worth keeping from this conversation.

        Two rounds, offered one after the other because a session holds one
        unanswered set at a time: the model's self-review first, then the facts
        the agent flagged mid-session with the `memory` tool (see
        `_resolve_memory`, which starts the second). With nothing said and
        nothing flagged there is nothing to review, and saying so is cheaper
        than a generation that will propose nothing.
        """
        messages = list((await self._thread_values(session.session_id)).get(
            "messages", []
        ))
        pending = self._memory.pending_edits(session.session_id)
        if not messages and not pending:
            self._deps.emit(
                Notify(severity="warning", text="Nothing to conclude yet.")
            )
            return
        proposed = None
        if messages:
            proposed = await self._memory.review_conversation(
                session, messages, span="whole"
            )
        if proposed is None:
            # No self-review to answer, so nothing will arrive later to start
            # the flagged round: start it here instead.
            proposed = self._memory.propose_flagged_edits(session)
        if proposed is None:
            self._deps.emit(
                Notify(text="Nothing durable to keep from this conversation.")
            )

    def _profile_for(self, session_id: str | None) -> str:
        """The profile a profile-scoped command means: the named session's,
        else the one the core is working under.

        A user reading a session that belongs to another profile is asking
        about that profile's skills, not about the core's — the same reason
        `_skill_named` resolves a forced skill against the session.
        """
        session = self._session(session_id)
        return getattr(session, "profile", None) or self._deps.profile

    def _working(self, session_id: str, activity: str) -> None:
        """Say that a *command* is holding this session up, and since when.

        The same event a turn reports with, deliberately: what the user has to
        know is that the session is busy and for how long, not which part of
        the core is busy on their behalf — `MemoryService._activity` says the
        same for its own sub-agent calls. An empty label ends it.
        """
        self._deps.emit(
            TurnActivity(
                session_id=session_id,
                activity=activity,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    # ---------------------------------------------------------------- helpers

    @property
    def _project_root(self) -> Any:
        """Where project-level skills are read from. The memory service was
        handed one so tests can pin it; the same one is used here rather than
        asking the working directory a second time."""
        return self._memory.project_root

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
    # Declared before the memory service, which needs to ask it a question
    # (below) long before it exists.
    scheduler_ref: dict[str, TurnScheduler] = {}

    def busy_profiles() -> set[str]:
        """Which profiles have a turn in flight — the memory service's half of
        "may this profile be deleted". Read through a callable rather than
        copied, because a stale copy would refuse a deletion that is now fine,
        or allow one that is not."""
        sched = scheduler_ref.get("scheduler")
        return sched.busy_profiles() if sched is not None else set()

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
        busy_profiles=busy_profiles,
        tools=tools,
    )

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
        return sched.tool_context(session_id) if sched is not None else None

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

    async def after_turn(session, result, plan) -> None:
        """Post-turn work, routed to the service the scheduler cannot see.

        The scheduler is built before the service and must stay that way — it
        is handed to the constructor — so the hook goes through the same
        forward reference `emit` uses. Passing *something* here is the fix for
        a real regression: `on_turn_result` exists for exactly this and had
        nothing wired to it, so a conversation was never named after its first
        exchange and `session.retitle` was the whole of titling.
        """
        holder = service_ref.get("service")
        if holder is not None:
            await holder.after_turn(session, result, plan)

    scheduler = TurnScheduler(
        deps,
        graph=graph,
        prepare=prepare,
        session_for=sessions.get,
        on_turn_result=after_turn,
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
