"""Who runs next, and what happens while they do (specs-core-process.md §7).

This is `HpcaApp`'s turn machinery — `_turns`, `_pending_work`, `drain_work`,
`_deliver_event`, the interrupt path and the approval park — with every reach
into the UI replaced by an emitted event. The invariants are unchanged and are
worth restating, because they are the whole reason this is a scheduler rather
than a function call:

* **One turn per session at a time.** Two turns on one ``thread_id`` would
  interleave checkpoint writes. Anything arriving for a busy session waits.
* **Different sessions run concurrently.** A turn in one session must never
  block another's; they have separate threads, separate clients, separate
  contexts.
* **A session parked on an approval is not free.** Its thread cannot move
  until the answer arrives, so its queue waits too — but only its own.

Two things moved deliberately while extracting.

**Pending decisions live here now.** They were UI-process memory
(`app.py:_pending_decision`), surfaced only from that dict, so restarting with
a session parked on an interrupt left it parked with nothing on screen to
answer it. Owned by the scheduler they survive the UI, and
:meth:`TurnScheduler.pending_decisions` re-emits them on subscribe (§4.4).

**Turn preparation is injected.** Deciding what a turn is *for* — its memory
snapshot, its skills, its recalled context, its tool context — belongs to the
memory and backend services. The scheduler takes a :class:`TurnPreparer` and
stays ignorant of all of it, which is what lets it be tested with a two-line
fake instead of a profile tree and a fake LLM.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Awaitable, Callable, Protocol

from langgraph.types import Command

from hpca.agent.graph import (
    deliver_event,
    rollback_thread,
    run_turn,
    thread_message_count,
)
from hpca.core.deps import CoreDeps
from hpca.protocol import (
    DecisionCleared,
    DecisionRequested,
    Notify,
    TurnActivity,
    TurnFailed,
    TurnFinished,
    TurnStarted,
)

logger = logging.getLogger("hpca.core.scheduler")

# The activity string a turn wears while it is parked on the model. Interrupt
# is offered only in this phase: it is the one point where telling the backend
# to stop means anything, and the only one with a prompt to hand back.
LLM_WAIT_ACTIVITY = "LLM processing"


@dataclass
class TurnPlan:
    """Everything one turn needs, resolved before it starts.

    Produced by the :class:`TurnPreparer`, held on the :class:`TurnState` for
    as long as the turn runs. Frozen at the start of the turn rather than read
    per round, so a profile edit or a session switch mid-turn cannot change
    what the running turn is working from.
    """

    ctx: Any = None
    memory: Any = None
    skills: list[Any] = field(default_factory=list)
    # The user message as the *model* should see it: recalled memory and the
    # volatile date/time appended. The transcript keeps the clean text.
    api_content: str | None = None
    log: Any = None


class TurnPreparer(Protocol):
    def __call__(
        self, session: Any, *, user_text: str | None, forced_skill: Any = None
    ) -> TurnPlan: ...


class SessionLookup(Protocol):
    def __call__(self, session_id: str) -> Any: ...


@dataclass
class PendingWork:
    """One turn's worth of input waiting for a session's orchestrator."""

    session_id: str
    text: str
    kind: str  # "user" — typed and waiting | "event" — background completion
    forced_skill: Any = None


@dataclass
class TurnState:
    """Everything one in-flight turn owns, kept per session so turns on
    different sessions never read each other's client, context, memory,
    interrupt bookkeeping or activity.

    A session has a turn in flight ⇔ its id is a key in ``TurnScheduler._turns``.
    """

    session: Any
    plan: TurnPlan
    task: asyncio.Task | None = None
    interrupt_keep: int | None = None
    # The user message this turn is running (None for a resume or an event).
    # The interrupt hands it back to the entry for editing.
    user_text: str | None = None
    activity: str = "working"
    # When this turn began. Held here rather than on any renderer, because a
    # widget is rebuilt every time a session is re-opened and reading the clock
    # off it restarted a two-minute wait at zero.
    started: float = field(default_factory=monotonic)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TurnScheduler:
    """Runs turns, one per session at a time, many sessions at once."""

    def __init__(
        self,
        deps: CoreDeps,
        *,
        graph: Any,
        prepare: TurnPreparer,
        session_for: SessionLookup,
        on_turn_result: Callable[[Any, Any, TurnPlan], Awaitable[None]] | None = None,
    ) -> None:
        self._deps = deps
        self._graph = graph
        self._prepare = prepare
        self._session_for = session_for
        # Called with (session, TurnResult, plan) after a turn produces one.
        # The transcript log, the episodic index and session titling all hang
        # off this; none of them is the scheduler's business, and none of them
        # may be allowed to fail a turn.
        self._on_turn_result = on_turn_result
        self._turns: dict[str, TurnState] = {}
        self._pending: list[PendingWork] = []
        self._awaiting_approval: set[str] = set()
        # session_id -> the graph interrupt payload it is parked on. See the
        # module docstring: this used to live in the UI and die with it.
        self._decisions: dict[str, dict] = {}
        self._shutting_down = False
        # The coalesced drain scheduled by submit_event; see _schedule_drain.
        self._drain_task: asyncio.Task | None = None

    # ------------------------------------------------------------- inspection

    def is_busy(self, session_id: str) -> bool:
        return session_id in self._turns

    def activity_of(self, session_id: str) -> str | None:
        ts = self._turns.get(session_id)
        return ts.activity if ts is not None else None

    def queued_texts_for(self, session_id: str) -> list[str]:
        return [
            work.text
            for work in self._pending
            if work.kind == "user" and work.session_id == session_id
        ]

    def pending_decisions(self) -> dict[str, dict]:
        """Every parked decision, for re-emitting when a client subscribes.

        A copy: a caller iterating this while a turn resolves one would
        otherwise mutate under itself.
        """
        return dict(self._decisions)

    def busy_sessions(self) -> set[str]:
        return set(self._turns)

    # --------------------------------------------------------------- intake

    def submit_user(
        self, session_id: str, text: str, *, forced_skill: Any = None
    ) -> bool:
        """Accept a typed message. Returns whether it has to wait its turn.

        Accepted either way — only its turn may have to queue, which is what
        makes typing ahead look like it worked.
        """
        queued = session_id in self._turns or session_id in self._awaiting_approval
        self._pending.append(
            PendingWork(
                session_id=session_id,
                text=text,
                kind="user",
                forced_skill=forced_skill,
            )
        )
        return queued

    def submit_event(self, session_id: str, text: str) -> None:
        """A background completion reporting in — a finished process, a job
        that reached a terminal state. Queued exactly like a typed message so
        it cannot land in the middle of a turn.

        Schedules its own drain, because the caller is a poller and a poll's
        cadence must not decide a turn's timing. In the old code the pollers
        called ``drain_work()`` themselves; that coupled "how often we ask
        sacct" to "how soon the agent hears about it", and a submitter that
        forgot the call left the completion sitting in the queue until
        something unrelated happened to drain it.
        """
        self._pending.append(
            PendingWork(session_id=session_id, text=text, kind="event")
        )
        self._schedule_drain()

    def _schedule_drain(self) -> None:
        """Drain soon, once, however many callers ask.

        Coalesced rather than one task per submission: a poll can deliver a
        dozen finished processes at once, and a dozen concurrent drains would
        each walk the same queue. Re-entrant by construction anyway — drain
        skips busy sessions and removes items by identity — so the coalescing
        is about waste, not correctness.
        """
        if self._shutting_down:
            return
        if self._drain_task is not None and not self._drain_task.done():
            return
        try:
            self._drain_task = asyncio.ensure_future(self.drain())
        except RuntimeError:
            # No running loop: a caller queued work outside the core's loop
            # (tests do this). The next explicit drain picks it up.
            self._drain_task = None

    # ---------------------------------------------------------------- drain

    async def drain(self) -> None:
        """Start the next waiting item for every session free to run.

        Walks the queue rather than bailing on "anything is busy": a session
        with a turn in flight, one parked on an approval, or one already
        started in this pass is skipped, and every other session starts at
        once. Same-session serialisation is what the skip preserves.
        """
        if self._shutting_down:
            return
        started: set[str] = set()
        # Snapshot: starting a turn mutates _turns and delivering an event
        # awaits, so iterate a copy and remove drained items by identity.
        for item in list(self._pending):
            if item not in self._pending:
                continue
            sid = item.session_id
            if (
                sid in self._turns
                or sid in started
                or sid in self._awaiting_approval
            ):
                continue
            self._pending.remove(item)
            started.add(sid)
            try:
                if item.kind == "user":
                    session = self._session_for(sid)
                    if session is not None:
                        self.start_turn(
                            session,
                            user_text=item.text,
                            forced_skill=item.forced_skill,
                        )
                else:
                    await self._deliver(sid, item.text)
            except Exception as e:  # one bad item must not stall the queue
                logger.exception("queued work failed")
                self._deps.emit(
                    Notify(severity="error", text=f"Queued work failed: {e}")
                )

    async def _deliver(self, session_id: str, text: str) -> None:
        """React now if the user is looking at this session, otherwise leave
        the message in the thread for the next turn to find.

        Both are the same LangGraph primitive — new input on an existing
        thread_id. The difference is only whether the model runs immediately
        (one call, the agent speaks unprompted) or the message simply waits
        (free). This is the one behaviour that genuinely depends on what the
        user is looking at, which is why `focused_session_id` exists at all
        and why it arrives as an explicit command rather than being read off a
        widget (§4.4).
        """
        session = self._session_for(session_id)
        if session is None:
            await deliver_event(self._graph, session_id=session_id, text=text)
            return
        if self._deps.focused_session_id == session_id:
            self.start_turn(session, user_text=text)
        else:
            await deliver_event(self._graph, session_id=session_id, text=text)
            self._deps.emit(
                Notify(
                    severity="information",
                    text=f"{getattr(session, 'title', session_id)}: "
                    "background work finished",
                )
            )

    # ----------------------------------------------------------- turn running

    def start_turn(
        self,
        session: Any,
        *,
        user_text: str | None = None,
        resume: Command | None = None,
        forced_skill: Any = None,
    ) -> asyncio.Task:
        """Begin a turn for one session.

        It stays that session's turn even if the user switches away while the
        model works: everything it touches is captured in its TurnPlan here,
        not read from whatever session happens to be open when the reply lands.
        """
        plan = self._prepare(session, user_text=user_text, forced_skill=forced_skill)
        ts = TurnState(session=session, plan=plan, user_text=user_text)
        self._turns[session.session_id] = ts
        self._deps.emit(TurnStarted(session_id=session.session_id))
        self._emit_activity(ts)
        ts.task = asyncio.ensure_future(self._run(session, ts, resume=resume))
        return ts.task

    def report_activity(self, session_id: str, activity: str) -> None:
        """What a session's turn is doing right now.

        Recorded on that turn's own state so it survives the user leaving and
        coming back, and emitted so a renderer can say it. The core does not
        know or care whether anyone is looking.
        """
        ts = self._turns.get(session_id)
        if ts is None:
            return
        ts.activity = activity
        self._emit_activity(ts)

    def _emit_activity(self, ts: TurnState) -> None:
        self._deps.emit(
            TurnActivity(
                session_id=ts.session.session_id,
                activity=ts.activity,
                started_at=ts.started_at,
            )
        )

    async def _run(self, session: Any, ts: TurnState, *, resume: Command | None) -> None:
        session_id = session.session_id
        if ts.user_text is not None:
            # Where to roll back to if this turn is interrupted: captured
            # before run_turn appends the user message.
            try:
                ts.interrupt_keep = await thread_message_count(
                    self._graph, session_id=session_id
                )
            except Exception:
                ts.interrupt_keep = None
        try:
            result = await run_turn(
                self._graph,
                session_id=session_id,
                user_text=ts.user_text,
                resume=resume,
                api_content=ts.plan.api_content,
            )
        except asyncio.CancelledError:
            raise  # an interrupt; _interrupt owns the cleanup
        except Exception as e:
            logger.exception("turn failed")
            self._deps.emit(TurnFailed(session_id=session_id, error=str(e)))
            return
        finally:
            # Drop this session's turn wholesale, and stop the clock.
            self._turns.pop(session_id, None)
            self._deps.emit(TurnActivity(session_id=session_id, activity=""))
            # Whatever queued behind this turn starts as soon as this unwinds.
            asyncio.ensure_future(self.drain())

        if self._on_turn_result is not None:
            try:
                await self._on_turn_result(session, result, ts.plan)
            except Exception:  # logging and titling are never worth a turn
                logger.exception("post-turn handling failed")

        if result.interrupt is not None:
            # Parked on an approval: the thread cannot move without an answer.
            self._awaiting_approval.add(session_id)
            self._decisions[session_id] = dict(result.interrupt)
            self._deps.emit(
                DecisionRequested(session_id=session_id, payload=dict(result.interrupt))
            )
            return
        self._deps.emit(TurnFinished(session_id=session_id, reply=result.reply))

    # ------------------------------------------------------------- approvals

    def resolve_decision(
        self, session_id: str, *, approved: bool, reason: str = ""
    ) -> bool:
        """Answer a parked decision and resume its turn.

        Resumed on the thread it belongs to — the user may have switched
        sessions since the prompt appeared. ``reason`` is what they typed when
        refusing; it rides back with the refusal so the next attempt can be a
        corrected one rather than the same call in another shape.
        """
        if session_id not in self._decisions:
            return False  # stale answer: already resolved, or never parked
        self._decisions.pop(session_id, None)
        self._awaiting_approval.discard(session_id)
        self._deps.emit(DecisionCleared(session_id=session_id))
        session = self._session_for(session_id)
        if session is None:
            return False
        self.start_turn(
            session,
            resume=Command(resume={"approved": bool(approved), "reason": reason}),
        )
        return True

    # ------------------------------------------------------------- interrupt

    def can_interrupt(self, session_id: str) -> bool:
        """Only while a session's own *user* turn is parked on the model — the
        one phase where telling the backend to stop makes sense, and the only
        case with a prompt to hand back."""
        ts = self._turns.get(session_id)
        return (
            ts is not None
            and ts.activity == LLM_WAIT_ACTIVITY
            and ts.user_text is not None
            and ts.interrupt_keep is not None
        )

    async def interrupt(self, session_id: str) -> str | None:
        """Abort the in-flight request and drop the aborted turn from the
        thread. Returns the message to hand back for editing, or None.

        The rollback is what makes the re-edited prompt start from a clean
        history: the interrupted user message and any partial tool traffic
        must leave the thread, or the model sees the abandoned attempt twice.
        """
        ts = self._turns.get(session_id)
        if ts is None or ts.interrupt_keep is None or ts.user_text is None:
            return None
        text, keep, task = ts.user_text, ts.interrupt_keep, ts.task
        if task is not None:
            task.cancel()
            try:
                await task  # let the cancellation unwind before editing
            except (Exception, asyncio.CancelledError):
                pass
        self._turns.pop(session_id, None)
        self._deps.emit(TurnActivity(session_id=session_id, activity=""))
        try:
            await rollback_thread(self._graph, session_id=session_id, keep=keep)
        except Exception as e:
            self._deps.emit(
                Notify(severity="error", text=f"Interrupt cleanup failed: {e}")
            )
        asyncio.ensure_future(self.drain())
        return text

    # -------------------------------------------------------------- shutdown

    def forget_session(self, session_id: str) -> None:
        """Drop everything a deleted session owned, so nothing queued for it
        starts against a thread that no longer exists."""
        self._pending = [w for w in self._pending if w.session_id != session_id]
        self._decisions.pop(session_id, None)
        self._awaiting_approval.discard(session_id)

    async def shutdown(self) -> None:
        """Stop accepting work and let in-flight turns unwind.

        Queued work is dropped rather than run: the databases and the
        checkpointer are about to close under it.
        """
        self._shutting_down = True
        self._pending.clear()
        tasks = [ts.task for ts in self._turns.values() if ts.task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (Exception, asyncio.CancelledError):
                pass
        self._turns.clear()
