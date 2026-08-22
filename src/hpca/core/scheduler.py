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

## Drawing a turn as it happens

This is also where a turn becomes chat. A front-end that has a session open
must see the conversation happen without ever asking for a snapshot (§4.2
property 1), and everything it sees comes from here: the message the turn
started with, the tool calls as they are made, the results filling into the
rows those calls drew, and the reply.

The hard part is not emitting them; it is emitting them so that a session
re-opened afterwards looks *identical*. A `chat.reset` is
`transcript.build_entries` over the checkpointed thread, and that function
**folds** a turn's reasoning and tool exchanges into a single ``thinking``
entry. Live rows that were shaped differently would make the screen rearrange
itself the moment the user re-opened the session — the bug the old UI had, and
paid for by rebuilding the whole log every turn.

So the two are reconciled by construction, on three rules:

1. **The same function shapes both.** The working box is
   `transcript.thinking_entry` in the live path and in the fold; the turn's
   message is `build_entries` over that one message, so a recalled-memory row
   appears live exactly where the fold puts it.
2. **A row is named by its position in the turn.** `build_entries` walks
   messages in order and only ever extends its output or revises its last
   entry, so entry *n* of a turn keeps meaning the same row as the turn grows.
   That is what makes the position a safe row identity — and it is why a call
   and its result share one `Entry.seq` rather than becoming two rows.
3. **The fold has the last word.** When the graph returns, the turn's rows are
   rebuilt with `build_entries` over the state it produced and re-emitted
   against the seqs already drawn: same position, `chat.update`; new position,
   `chat.append`. Anything the live path could not know — above all the
   model's reasoning, which arrives in the checkpointed state and not through
   a callback — appears then, in the place the fold gives it. Only the turn's
   own rows cross, never the transcript, so this is a delta and not the
   per-turn rebuild this protocol exists to delete.

A turn parked on an approval keeps its live record: the resume is the same
*logical* turn, its rows continue the same box, and the fold agrees because
nothing flushed that box in between.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Awaitable, Callable, Protocol

from langgraph.types import Command

from hpca.agent.graph import (
    deliver_event,
    run_turn,
    stop_thread,
    thread_message_count,
)
from hpca.core.deps import CoreDeps
from hpca.protocol import (
    ChatAppend,
    ChatUpdate,
    DecisionCleared,
    DecisionRequested,
    Entry,
    Notify,
    TurnActivity,
    TurnFailed,
    TurnFinished,
    TurnStarted,
)
from hpca.transcript import (
    EVENT,
    THINKING,
    USER,
    Step,
    build_entries,
    live_step,
    thinking_entry,
)

logger = logging.getLogger("hpca.core.scheduler")


def wire_entry(entry, seq: int) -> Entry:
    """One `transcript.Entry` as the protocol's twin of it, named ``seq``.

    Converted by field name rather than by hand: the two classes are asserted
    to have the same shape (`test_protocol`), and ``extra="forbid"`` turns a
    drift between them into a failure here — at the boundary, on the first
    entry — instead of a field that is quietly missing from every chat.

    Lives here rather than in `core.service` because this module is what names
    rows, and the reset and the deltas have to shape them the same way.
    """
    return Entry.model_validate({**asdict(entry), "seq": seq})


def _tail_exchange(entries: list[Entry]) -> list[Entry]:
    """The last exchange in a `chat.reset`: its final message row and
    everything drawn after it.

    "Message row" is a `user` or `event` entry that is a thread message of its
    own (``index`` >= 0) — the row `build_entries` opens a turn with. A
    ``queued`` row is neither: it is not in the thread and has no index, which
    is what keeps a typed-ahead message from being mistaken for the start of
    the exchange a resume is about to continue.
    """
    for position in range(len(entries) - 1, -1, -1):
        entry = entries[position]
        if entry.kind in (USER, EVENT) and entry.index >= 0:
            return list(entries[position:])
    return []


def _as_steps(parts) -> list[Step]:
    """Working parts as the render model, whichever twin they arrived as.

    A `chat.reset` hands them back as `protocol.Part` and the fold produces
    `transcript.Step`; the two have the same fields by construction (asserted
    in `test_protocol`) and only the second knows how to take a result. One
    conversion here beats two shapes of "the open box" downstream.
    """
    return [
        part
        if isinstance(part, Step)
        else Step(**(part.model_dump() if hasattr(part, "model_dump") else part))
        for part in parts
    ]


@dataclass
class LiveTurn:
    """The rows one *logical* turn has drawn, and the box it is still filling.

    Logical, not per graph invocation: a turn parked on an approval and resumed
    is one conversation exchange and one working box, which is how
    `build_entries` folds it, so this record has to survive the park (see the
    module docstring's rule 3).

    ``rows`` is the whole point — the ``Entry.seq`` of each row drawn for this
    turn, in the order the fold produces them, so the reconcile can address
    row *n* rather than guess at it.
    """

    # The index of the message this turn began with. What `build_entries` is
    # given as ``start`` to produce this turn's rows and nobody else's, and
    # what tells a `chat.reset` which of its entries belong to this turn.
    start: int | None = None
    rows: list[int] = field(default_factory=list)
    # The working box's parts, accumulated as the graph reports them, and the
    # row they are drawn as (0 = not drawn yet).
    parts: list[Step] = field(default_factory=list)
    thinking_seq: int = 0


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
    # The chat row this message is drawn as while it waits (`Entry.seq`), or 0
    # for work that was never drawn: anything that started at once, and every
    # background completion. It is what lets the promotion land in the row the
    # user is already looking at, and what `unqueue` names a message by.
    entry_seq: int = 0


@dataclass
class Stopped:
    """What :meth:`TurnScheduler.interrupt` did, for the caller to answer to.

    A bare ``str | None`` return said both "a turn was stopped" and "here is
    its message"; those came apart the moment a stop stopped withdrawing the
    message, since the usual stop now has a chat to re-state and no text to
    give back. Distinguishing them is the whole reason this exists: ``None``
    from ``interrupt`` still means nothing was running.
    """

    # The message to put back in the entry box, and only when the exchange
    # left no trace to keep — otherwise it is in the conversation, where
    # handing it back too would get it sent twice.
    text: str | None = None


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
    # Where the *exchange* this turn belongs to begins in the thread, measured
    # before its message was appended. It was the rollback point back when
    # stopping a turn withdrew it; a stop now keeps the work, and what the
    # number answers is "did any of this reach the thread at all" — which is
    # what decides whether the message is the conversation's or is handed back
    # (see :meth:`TurnScheduler.interrupt`). Shared by every turn of one
    # exchange, so the half after an approval measures nothing of its own.
    exchange_start: int | None = None
    # Where this turn's own messages begin in the thread — what `TurnResult`
    # calls ``first_new``, measured before the turn ran so a turn that never
    # produces a result still knows which tail is its own. The same number as
    # ``exchange_start`` for a turn that brings its own message, and not for
    # the resumed half of one. Read by the failure and the stop hooks; a turn
    # that finishes carries the same number back itself.
    first_new: int | None = None
    # The user message this turn is running (None for a resume or an event).
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
        on_turn_error: (
            Callable[[Any, TurnPlan, Exception, int | None], Awaitable[None]] | None
        ) = None,
        on_turn_stopped: (
            Callable[[Any, TurnPlan, int | None], Awaitable[None]] | None
        ) = None,
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
        # The same door for a turn that produced no result: (session, plan,
        # error, first_new). A failure is still something that happened, and
        # the transcript is the record of what happened — a run whose backend
        # died mid-turn otherwise leaves a log file with the question missing
        # and no reason for the silence. ``first_new`` is where this turn's
        # messages start in the thread, so the same tail can be read back out
        # of the checkpoint; None when it could not be measured.
        self._on_turn_error = on_turn_error
        # And the third ending: (session, plan, first_new) for a turn the user
        # stopped. It gets a door of its own rather than borrowing the failure
        # one because a stop is not a failure — there is no error to write
        # under it — and rather than borrowing the result one because there is
        # no result. What it has in common with both is that it happened, and
        # what happened belongs in the log and the index: the work stays in
        # the thread now, and a conversation the user can still read but
        # `session_search` cannot find would be a record with a hole in it.
        self._on_turn_stopped = on_turn_stopped
        self._turns: dict[str, TurnState] = {}
        self._pending: list[PendingWork] = []
        self._awaiting_approval: set[str] = set()
        # session_id -> the graph interrupt payload it is parked on. See the
        # module docstring: this used to live in the UI and die with it.
        self._decisions: dict[str, dict] = {}
        # What an interrupt needs — (the thread length before the message, the
        # message) — kept per session for as long as the *exchange* lasts and
        # not just for one turn. An approval ends the turn it parked; the
        # resume starts a fresh one carrying no user message of its own, and
        # without this it would be a turn nobody could stop
        # (`tui/app.py:_interrupt_anchor`). It outlived the rollback it was
        # first written for: the length still says whether the exchange ever
        # reached the thread, and the message is still what a stop with
        # nothing to keep gives back.
        self._anchors: dict[str, tuple[int, str]] = {}
        self._shutting_down = False
        # session_id -> the last chat row name handed out for it. See
        # _next_entry_seq; every row the core draws is minted here.
        self._entry_seqs: dict[str, int] = {}
        # session_id -> the rows its current turn has drawn. See LiveTurn.
        self._live: dict[str, LiveTurn] = {}
        # session_id -> the rows the last `chat.reset` drew for the exchange it
        # ended on, as it named them. The one thing a resume can be re-bound to
        # when the turn it continues was never drawn by this process. See
        # _adopt_parked_turn.
        self._last_exchange: dict[str, list[Entry]] = {}
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

    def rewind_blocker(self, session_id: str) -> str | None:
        """Why this session's thread cannot be *truncated* right now, or None.

        For `session.rollback` only. Everything named here either writes to
        the thread the cut is about to shorten or is parked inside it: a turn
        in flight appends as it works, an unanswered approval holds a graph
        interrupt whose resume would land on indices the cut removed, and a
        queued message becomes a turn the moment the session is free.

        `session.fork` is deliberately NOT gated by this, and re-gating it
        would be a regression. Both can be aimed at a stale cut point — that
        risk does not tell them apart — but what happens next does: a fork at
        a stale index leaves an extra session the user deletes, while a
        rollback at one destroys messages that cannot be recovered. Gate the
        destructive half, leave the recoverable one open. Branching off while
        the agent is mid-turn is also the case forking is most useful for.

        The reason is returned rather than a bool because it is meant to be
        read: it reaches the user as the text of a warning `notify`.
        """
        if session_id in self._turns:
            return "a turn is running — wait, or interrupt it first"
        if session_id in self._awaiting_approval or session_id in self._decisions:
            return "a decision is pending — answer it first"
        if self.queued_texts_for(session_id):
            return "queued messages are waiting to run"
        return None

    def pending_decisions(self) -> dict[str, dict]:
        """Every parked decision, for re-emitting when a client subscribes.

        A copy: a caller iterating this while a turn resolves one would
        otherwise mutate under itself.
        """
        return dict(self._decisions)

    def busy_sessions(self) -> set[str]:
        return set(self._turns)

    def busy_profiles(self) -> set[str]:
        """Which profiles have a turn in flight.

        The memory service's half of "may this profile be deleted": deleting
        one out from under a running turn would strand it under a profile
        whose memories and skills no longer exist. Answered from the live
        turns' own session copies rather than from the store, because that is
        the profile the turn is actually running as.
        """
        return {
            ts.session.profile
            for ts in self._turns.values()
            if getattr(ts.session, "profile", "")
        }

    def tool_context(self, session_id: str) -> Any:
        """The tool context of the turn running on this session, or None.

        Exposed because two things outside the scheduler legitimately want the
        *live* one rather than a rebuilt copy: the graph's per-turn resolver,
        and killing a subprocess by pid — the runner that started it is the one
        with a monitor that will record how it ended.
        """
        ts = self._turns.get(session_id)
        return ts.plan.ctx if ts is not None else None

    # --------------------------------------------------------------- intake

    def submit_user(
        self, session_id: str, text: str, *, forced_skill: Any = None
    ) -> bool:
        """Accept a typed message. Returns whether it has to wait its turn.

        Accepted either way — only its turn may have to queue, which is what
        makes typing ahead look like it worked.
        """
        queued = session_id in self._turns or session_id in self._awaiting_approval
        work = PendingWork(
            session_id=session_id,
            text=text,
            kind="user",
            forced_skill=forced_skill,
        )
        if queued:
            # Drawn now, because "it looks like it worked" is the whole point
            # of typing ahead — and drawn by the core, because the queue it is
            # waiting in is state a front-end may not read (§4.2 rule 2). An
            # ordinary chat row rather than an event of its own: it *is* a chat
            # row, and the same row becomes the message that ran (see drain).
            work.entry_seq = self._next_entry_seq(session_id)
            self._deps.emit(
                ChatAppend(
                    session_id=session_id,
                    entry=Entry(kind="queued", text=text, seq=work.entry_seq),
                )
            )
        self._pending.append(work)
        return queued

    def _next_entry_seq(self, session_id: str) -> int:
        """The next chat row name for one session (`protocol.Entry.seq`).

        Monotonic per session and never reused *within a reset generation*,
        so an update can only ever find the row it means; `rebase_rows` is the
        one thing allowed to restart it, because a `chat.reset` is exactly the
        frame that tells the UI to forget the names it held.

        It lives here because the queue was the first thing to mint rows, and
        it stayed here when turns began drawing their own: two allocators
        would hand the same name to two rows, so there is one, and it is the
        scheduler's because every row the core draws belongs to a turn or to
        the queue in front of one.
        """
        nxt = self._entry_seqs.get(session_id, 0) + 1
        self._entry_seqs[session_id] = nxt
        return nxt

    def unqueue(self, session_id: str, seq: int) -> str | None:
        """Take one typed-ahead message back out. Returns its text, or None.

        ``seq`` is the row it was drawn as — the `Entry.seq` on the ``queued``
        `chat.append` this scheduler sent when it accepted the message. Named
        rather than counted on purpose: the turn ahead can finish while the
        user is deciding, and a queue *position* would then silently point at
        the neighbour, cancelling a message nobody asked to cancel. A row name
        either still refers to something waiting or it does not.

        None means it no longer does — it started while the user was deciding.
        A background completion can never be hit: it was never drawn, so it
        carries no row name (and 0, "unnamed", matches nothing).
        """
        if seq <= 0:
            return None
        work = next(
            (
                w
                for w in self._pending
                if w.kind == "user"
                and w.session_id == session_id
                and w.entry_seq == seq
            ),
            None,
        )
        if work is None:
            return None
        self._pending.remove(work)
        return work.text

    def rebase_rows(self, session_id: str, *, entries: list[Entry]) -> list[Entry]:
        """Restart a session's row names after a `chat.reset`, and hand back
        the rows that reset cannot contain.

        A reset renumbers from 1 (see `protocol.Entry.seq`), so the counter has
        to be told where the new numbering ended: ``entries`` is what the reset
        carries. Skipping that would leave the queue holding names from a
        generation the UI has just dropped, and the promotion or the unqueue
        that follows would address a row nobody has.

        Two jobs, and both exist because a reset renames every row on screen.

        **The unfinished exchange is re-bound.** Its record
        (:class:`LiveTurn`) holds the names it drew, and those names are gone.
        If the copy just read already contains the turn's own message — found
        by the index the turn started at, not by counting — the reset has drawn
        its rows for us and the record adopts them, working box included. If it
        does not, the message is a moment away from being checkpointed and the
        reset would show a session with the user's own sentence missing, so it
        is re-drawn here along with whatever the turn has done since.

        Unfinished, not running: a turn parked on an approval has ended (there
        is no `TurnState` while it waits) and its record is still open, because
        the answer resumes the same exchange into the same working box. Binding
        only what was running would leave the resume revising rows from a
        generation the UI has already dropped.

        **What was never in the thread comes back:** every message still
        queued behind that turn. The UI used to re-add all of this itself
        (`tui/app.py:4585`) out of state it owned. It owns none of it now — the
        gap specs-ui-replacement.md §4.2 records.
        """
        self._entry_seqs[session_id] = max(len(entries), 0)
        self._last_exchange[session_id] = _tail_exchange(entries)
        rows: list[Entry] = []
        ts = self._turns.get(session_id)
        live = self._live.get(session_id)
        if live is not None and (live.start is not None or ts is not None):
            rows += self._rebase_turn(session_id, ts, live, entries)
        for work in self._pending:
            if work.kind != "user" or work.session_id != session_id:
                continue
            work.entry_seq = self._next_entry_seq(session_id)
            rows.append(
                Entry(kind="queued", text=work.text, seq=work.entry_seq)
            )
        return rows

    def _rebase_turn(
        self,
        session_id: str,
        ts: TurnState | None,
        live: LiveTurn,
        entries: list[Entry],
    ) -> list[Entry]:
        """Re-bind an unfinished exchange's row record to a `chat.reset`'s
        numbering.

        See :meth:`rebase_rows`. The turn's message is found by its index — the
        thread position it took, which the turn recorded before it ran — rather
        than by comparing lengths: an index either is in the copy or is not,
        and the answer does not depend on counting the same thing twice.

        ``ts`` is None for an exchange parked on an approval: the turn ended
        when it parked. Only the re-draw below needs it, and a parked exchange
        never reaches it — its message has been checkpointed since, or the
        graph could not have parked at all.
        """
        start = live.start
        position = (
            next(
                (i for i, entry in enumerate(entries) if entry.index == start), None
            )
            if start is not None
            else None
        )
        if position is not None:
            # The copy holds this turn already, so the reset has drawn it. Its
            # own account of the working box supersedes the live one: it is
            # the same fold a re-open would show, and it carries the reasoning
            # the live path never sees.
            live.rows = [entry.seq for entry in entries[position:]]
            self._adopt_working(live, entries[position:])
            return []
        if ts is None or ts.user_text is None:
            # Nothing to re-draw the exchange from. Only reachable if the
            # thread lost the message a parked turn asked about, which is not
            # a state the graph can produce; the next reset says what is real.
            return []
        # The copy predates the turn's own message by a moment. Re-draw what
        # is on screen and nowhere else: the message, and the working done
        # since — the parts, unlike the message, may genuinely not be
        # checkpointed yet (a call is announced before its result lands).
        live.rows = []
        rows: list[Entry] = []
        for entry in self._message_entries(ts.user_text or "", ts.plan.api_content):
            seq = self._next_entry_seq(session_id)
            live.rows.append(seq)
            rows.append(wire_entry(entry, seq))
        if live.parts:
            live.thinking_seq = self._next_entry_seq(session_id)
            live.rows.append(live.thinking_seq)
            rows.append(wire_entry(thinking_entry(live.parts), live.thinking_seq))
        else:
            live.thinking_seq = 0
        return rows

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
                        # ``entry_seq`` is the row it was drawn as while it
                        # waited, if it waited at all. Handed to the turn
                        # rather than promoted here: one place draws a turn's
                        # opening rows (`_open_turn_rows`), so the row that was
                        # queued and the row that never had to be cannot end
                        # up shaped differently.
                        self.start_turn(
                            session,
                            user_text=item.text,
                            forced_skill=item.forced_skill,
                            entry_seq=item.entry_seq,
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
        entry_seq: int = 0,
    ) -> asyncio.Task:
        """Begin a turn for one session.

        It stays that session's turn even if the user switches away while the
        model works: everything it touches is captured in its TurnPlan here,
        not read from whatever session happens to be open when the reply lands.

        ``entry_seq`` names the ``queued`` row this message was already drawn
        as, when it had to wait; 0 when it starts at once. Either way its rows
        are drawn here, before the turn is announced, so nothing is ever on
        screen as still waiting behind a turn that is already itself.

        A resume — an answered approval — is the second half of an exchange a
        user message started, so it inherits that message's anchor: stopping it
        rolls the thread back to the same point and hands the same text back.
        Without that the resumed turn is a spinner nothing can answer.
        """
        plan = self._prepare(session, user_text=user_text, forced_skill=forced_skill)
        anchor = self._anchors.get(session.session_id) if resume is not None else None
        ts = TurnState(
            session=session,
            plan=plan,
            user_text=user_text if anchor is None else anchor[1],
            # Filled once the pre-turn count is read — already known here for
            # a resume, since the message it belongs to ran before it.
            exchange_start=None if anchor is None else anchor[0],
        )
        self._turns[session.session_id] = ts
        if user_text is not None:
            self._open_turn_rows(
                session.session_id,
                user_text,
                plan.api_content,
                entry_seq=entry_seq,
            )
        self._deps.emit(
            TurnStarted(session_id=session.session_id, started_at=ts.started_at)
        )
        self._emit_activity(ts)
        ts.task = asyncio.ensure_future(self._run(session, ts, resume=resume))
        return ts.task

    # ------------------------------------------------------- drawing the turn

    def _message_entries(self, text: str, api_content: str | None) -> list[Any]:
        """The rows one message opens a turn with, as the fold will draw them.

        Through `build_entries` rather than by hand, over the one message the
        turn is about, because the fold makes decisions this would otherwise
        have to repeat and eventually get wrong: a completion reporting in is
        an ``event`` row and not a ``user`` one, and a message that recalled
        something from memory is followed by a ``recall`` row saying what.

        The index the fold gives here is meaningless — this list holds one
        message, not the thread — so it is cleared. The row is not addressable
        for a rewind while its own turn runs anyway (``rewind_blocker``), and
        the reconcile at the end of the turn fills in the real one.
        """
        message: dict = {"role": "user", "content": text}
        if api_content is not None:
            message["api_content"] = api_content
        entries = build_entries([message])
        for entry in entries:
            entry.index = -1
        return entries

    def _open_turn_rows(
        self, session_id: str, text: str, api_content: str | None, *, entry_seq: int
    ) -> None:
        """Draw the message a turn starts from, and begin its row record.

        A queued message is *revised* into its ``user`` row rather than drawn
        again below the one the user is already looking at — the promotion
        `chat.update` exists for (§4.2). Anything the message brought with it
        (a recall row) is appended after it, in the fold's order.
        """
        live = LiveTurn()  # a new logical turn: last turn's rows are settled
        self._live[session_id] = live
        for position, entry in enumerate(self._message_entries(text, api_content)):
            if position == 0 and entry_seq:
                live.rows.append(entry_seq)
                self._deps.emit(
                    ChatUpdate(
                        session_id=session_id, entry=wire_entry(entry, entry_seq)
                    )
                )
                continue
            seq = self._next_entry_seq(session_id)
            live.rows.append(seq)
            self._deps.emit(
                ChatAppend(session_id=session_id, entry=wire_entry(entry, seq))
            )

    def report_step(self, session_id: str, payload: dict) -> None:
        """One tool exchange as it happens (the graph's ``on_step``).

        The call the moment it is made, and the result filled into that same
        row when it lands — not a second row below it, which is why
        `Part.done` and `chat.update` exist. A turn can spend minutes in
        tools, and an activity line saying "running run_bash" does not say
        what it is running.

        The result finds its call the way `build_entries` does: the earliest
        one still waiting, so a model that made two calls before either
        answered gets them back in the order it asked. A result with no call
        to land on stands as its own part — that happens when a `chat.reset`
        adopted the checkpoint's version of this box between the two halves,
        and the reconcile puts it right when the turn ends.
        """
        live = self._live.get(session_id)
        if live is None:
            # Nothing is drawing rows for this session — a turn that started
            # before the last reset, or one already reconciled. Its work is in
            # the state, and the next reset shows it.
            return
        if payload.get("kind") == "call":
            live.parts.append(live_step(payload))
        else:
            call = next(
                (p for p in live.parts if p.kind == "call" and not p.done), None
            )
            if call is not None:
                call.attach(str(payload.get("text", "")))
            else:
                live.parts.append(live_step(payload))
        self._draw_working(session_id, live)

    def _draw_working(self, session_id: str, live: LiveTurn) -> None:
        """The working box, drawn the first time and revised after that."""
        entry = thinking_entry(live.parts)
        if live.thinking_seq:
            self._deps.emit(
                ChatUpdate(
                    session_id=session_id,
                    entry=wire_entry(entry, live.thinking_seq),
                )
            )
            return
        live.thinking_seq = self._next_entry_seq(session_id)
        live.rows.append(live.thinking_seq)
        self._deps.emit(
            ChatAppend(
                session_id=session_id, entry=wire_entry(entry, live.thinking_seq)
            )
        )

    def _reconcile(self, session_id: str, result: Any) -> None:
        """Re-state this turn's rows from the state the graph produced.

        The fold has the last word (module docstring, rule 3). Everything the
        live path could not know arrives here — the model's reasoning above
        all, which is checkpointed state and reaches nobody through a callback
        — and it arrives in the position `build_entries` gives it, which is
        the position a re-opened session will give it too.

        Positional addressing is safe because `build_entries` walks messages
        in order: as a turn grows it only extends its entry list or revises
        the last entry, so entry *n* of a turn keeps naming the same row. A
        position that already has a name is revised; one that does not is
        appended and named.

        Only this turn's own rows are rebuilt — ``start`` is the message it
        began with — so this is a delta, not the per-turn transcript rebuild
        the protocol exists to delete.
        """
        live = self._live.get(session_id)
        if live is None:
            # A turn nobody drew, and nothing to bind it to: the resume's
            # parked half belongs to a core that has since restarted and this
            # session was never re-opened here, so there are no row names on
            # screen to revise (`_adopt_parked_turn` is what recovers them
            # when there are). Emitting nothing leaves the reply to the next
            # `chat.reset`, which is late but never wrong — rows guessed at
            # from this invocation alone would be a second working box under
            # the one already drawn.
            return
        if live.start is None:
            # A turn whose thread length could not be read before it ran; the
            # graph reports where its own messages began.
            live.start = result.first_new
        self._fold_rows(
            session_id,
            live,
            list(result.messages),
            list(result.thinking),
            list(result.calls),
        )

    def _fold_rows(
        self,
        session_id: str,
        live: LiveTurn,
        messages: list[Any],
        thinking: list[dict],
        calls: list[dict],
    ) -> None:
        """This exchange's rows, re-stated from a thread state — the fold both
        the end of a turn and the failure of one go through."""
        entries = build_entries(messages, thinking, calls, start=live.start or 0)
        for position, entry in enumerate(entries):
            if position < len(live.rows):
                self._deps.emit(
                    ChatUpdate(
                        session_id=session_id,
                        entry=wire_entry(entry, live.rows[position]),
                    )
                )
                continue
            seq = self._next_entry_seq(session_id)
            live.rows.append(seq)
            self._deps.emit(
                ChatAppend(session_id=session_id, entry=wire_entry(entry, seq))
            )
        self._adopt_working(live, entries)

    def _adopt_working(self, live: LiveTurn, entries: list[Any]) -> None:
        """Re-point the open box at what the fold just said it contains.

        Only matters when the turn parked on an approval: the resume is the
        same logical turn and its next tool call has to land in the same box,
        which by then holds parts the live path never saw (reasoning) and
        parts it saw in a different shape. Taking the fold's copy is what
        keeps the two from diverging over a long, repeatedly-gated turn.
        """
        for position, entry in enumerate(entries):
            if entry.kind == THINKING:
                live.parts = _as_steps(entry.parts)
                live.thinking_seq = live.rows[position]
                return
        live.parts, live.thinking_seq = [], 0

    async def _reconcile_failed(self, session_id: str) -> None:
        """Settle the rows of a turn that broke instead of finishing.

        There is no result to fold, so without this the rows settle exactly as
        the live path left them — including a working box whose last call is
        still spinning for a result that is never coming, which is not how a
        re-opened session would draw it. The thread is the answer: whatever
        the graph checkpointed before it broke is what the next `chat.reset`
        shows, so the same fold is run over it and this turn's rows are
        re-stated to match. After it, what the user is looking at and what
        they would get back are the same rows.

        The failure itself is not a row here. It never entered the thread —
        the model did not see it and the next turn must not — so it travels as
        `turn.failed`, which is also what stops the spinner, and a front-end
        draws it as an ephemeral `error` entry of its own
        (specs-ui-replacement.md §3.2). Emitting one from here as well would
        put two of them on screen.
        """
        live = self._live.get(session_id)
        if live is None or live.start is None:
            return  # nothing was drawn for this turn, or nothing it can find
        try:
            values = await self._thread_values(session_id)
        except Exception:  # a backend that died may have taken more with it
            logger.exception("could not re-state the rows of a failed turn")
            return
        self._fold_rows(
            session_id,
            live,
            list(values.get("messages", [])),
            list(values.get("thinking", []) or []),
            list(values.get("calls", []) or []),
        )

    async def _thread_values(self, session_id: str) -> dict:
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        return snapshot.values or {}

    def _adopt_parked_turn(self, session_id: str) -> None:
        """Re-bind a resume to rows this process never drew.

        The case: the turn that parked on the approval belonged to a core that
        has since restarted, so its :class:`LiveTurn` is gone. The rows are
        not — the session was re-opened to answer the prompt, and that
        `chat.reset` drew the whole parked exchange and named every row of it.
        Adopting those names is what lets the resume revise the working box
        that is already on screen instead of the alternatives, both wrong: a
        second working box appended below the first, or the silence
        `_reconcile` falls back to, where the reply appears only on the next
        reset.

        The exchange is the reset's tail — its last user (or event) row and
        everything after it — which is exactly the turn a parked approval
        belongs to, since a thread parked at `interrupt()` cannot have moved
        since. Nothing is emitted here; the fold at the end of the turn does
        the drawing.
        """
        rows = self._last_exchange.get(session_id)
        if not rows:
            return
        live = LiveTurn(start=rows[0].index, rows=[row.seq for row in rows])
        self._live[session_id] = live
        self._adopt_working(live, rows)

    def _close_turn(self, session_id: str) -> None:
        """This turn's rows are settled; the next one starts a new record."""
        self._live.pop(session_id, None)

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
        if resume is None and ts.user_text is not None:
            # Where this exchange begins: captured before run_turn appends the
            # user message, so a stop can tell an exchange that reached the
            # thread from one that never did. It is also the index that
            # message is about to take, which is what tells a `chat.reset`
            # which of its entries belong to this turn (`rebase_rows`) and
            # where the reconcile starts folding.
            #
            # Read only for a turn that brings its own message: a resume was
            # handed both by the anchor, and re-reading the count here would
            # measure a thread that already holds the message and put the
            # start of the exchange halfway through itself.
            try:
                ts.exchange_start = await thread_message_count(
                    self._graph, session_id=session_id
                )
            except Exception:
                ts.exchange_start = None
            ts.first_new = ts.exchange_start
            live = self._live.get(session_id)
            if live is not None:
                live.start = ts.exchange_start
            if ts.exchange_start is not None:
                # Outlives this turn: an approval splits one exchange into
                # several, and each of them has to stay stoppable.
                self._anchors[session_id] = (ts.exchange_start, ts.user_text)
        elif resume is not None:
            # Measured, not inherited: the anchor points at the message that
            # opened the whole exchange, whose half has already been logged.
            # What this half adds starts where the parked thread stopped. The
            # count is safe to read here precisely because it is not the start
            # of the exchange — see the anchor above.
            try:
                ts.first_new = await thread_message_count(
                    self._graph, session_id=session_id
                )
            except Exception:
                ts.first_new = None
            if session_id not in self._live:
                self._adopt_parked_turn(session_id)
        try:
            result = await run_turn(
                self._graph,
                session_id=session_id,
                user_text=ts.user_text,
                resume=resume,
                api_content=ts.plan.api_content,
            )
        except asyncio.CancelledError:
            raise  # a stop; `interrupt` owns the cleanup
        except Exception as e:
            logger.exception("turn failed")
            await self._reconcile_failed(session_id)
            self._close_turn(session_id)
            # Nothing of this attempt is stoppable or continuable any more:
            # the exchange ended where it broke.
            self._anchors.pop(session_id, None)
            if self._on_turn_error is not None:
                try:
                    await self._on_turn_error(session, ts.plan, e, ts.first_new)
                except Exception:  # a log is never worth a second failure
                    logger.exception("post-turn handling of a failure failed")
            # After the record, as on the finished path: what a client is told
            # about is what has already been written down.
            self._deps.emit(TurnFailed(session_id=session_id, error=str(e)))
            return
        finally:
            # Drop this session's turn wholesale, and stop the clock.
            self._turns.pop(session_id, None)
            self._deps.emit(TurnActivity(session_id=session_id, activity=""))
            # Whatever queued behind this turn starts as soon as this unwinds.
            asyncio.ensure_future(self.drain())

        # Before anything else that could fail: what happened is what the user
        # is waiting to see, and a titler blowing up must not cost them the
        # reply. Synchronous, so these frames are queued ahead of the drain the
        # `finally` above just scheduled — the next turn's rows can only follow
        # this one's.
        self._reconcile(session_id, result)

        if self._on_turn_result is not None:
            try:
                await self._on_turn_result(session, result, ts.plan)
            except Exception:  # logging and titling are never worth a turn
                logger.exception("post-turn handling failed")

        if result.interrupt is not None:
            # Parked on an approval: the thread cannot move without an answer.
            # The row record stays — the resume continues this same turn, and
            # its next tool call belongs in the box already on screen.
            self._awaiting_approval.add(session_id)
            self._decisions[session_id] = dict(result.interrupt)
            self._deps.emit(
                DecisionRequested(session_id=session_id, payload=dict(result.interrupt))
            )
            return
        self._close_turn(session_id)
        # Answered for good: there is no exchange left to stop, and the next
        # turn must not inherit this one's message.
        self._anchors.pop(session_id, None)
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
        """Whether this session's turn can be stopped right now.

        Any phase of it — waiting on the model, or running a tool. It was once
        only the wait on the model, on the reasoning that stopping the backend
        is the only thing an abort really does. That is not how a turn spends
        its time: a script that runs for minutes, or a chain of tool rounds
        gone astray, is exactly what a user wants to stop, and refusing there
        left them watching a spinner they could not answer
        (`tui/app.py:_can_interrupt`, and specs-ui-acceptance.md's "a turn
        currently running a tool is interruptible").

        What cancelling mid-tool does NOT do is stop what the tool started: a
        script runs on under its own monitor — a task of the core's, not of the
        turn's — until it exits. The abort ends the *turn*, not the work
        already in flight.

        The two conditions that remain are what a stop needs to leave the
        session in a state someone can carry on from: a message of the user's
        own, and the point in the thread the exchange began at — which is what
        tells the stop whether any of it got as far as being written down
        (:meth:`interrupt`). A turn a fraction of a second old has only the
        first, and is stoppable a moment later.
        """
        ts = self._turns.get(session_id)
        return (
            ts is not None
            and ts.user_text is not None
            and ts.exchange_start is not None
        )

    async def interrupt(self, session_id: str) -> Stopped | None:
        """Stop the agent and keep what it has already done. None if there was
        no turn to stop.

        This used to throw the turn away: the message and every tool exchange
        under it were rolled out of the thread and the text handed back to be
        re-typed, on the reasoning that the model must not meet an abandoned
        attempt twice. The user who asked for the change put it better than
        the reasoning did — *"not the whole turn should be thrown away, the
        agent should just be stopped"*. The steps cost real cluster time, and
        half a turn is very often exactly the half they wanted to read.

        So the thread keeps its work and ``stop_thread`` writes the one thing
        that makes it safe to keep: a note saying a person stopped this, which
        is what stands between the next turn and a model that reads an
        unfinished history as an instruction to finish it.

        **The message is not handed back** when the exchange survives — it is
        in the conversation now, and putting it into the entry box as well
        would have the user send it twice without meaning to. The hand-back
        remains for the one case where there is nothing to keep: a stop that
        lands in the window between the turn being announced and its message
        reaching the thread. Nothing of that turn was ever written down, so
        the sentence would be lost outright, and it goes back to the box.
        Taking a message back on purpose is what the chat rewind is for.

        It ends with a `turn.finished` because a stopped turn is a turn that
        is over, and `turn.finished`/`turn.failed` is the only thing any
        front-end reads as one ending. Without it the spinner ran on forever:
        `chat.reset` restates rows and `turn.interrupted` hands a message
        back, and neither says the turn is done, so the working row kept
        drawing — spinning on a turn that no longer existed and offering to
        stop it. Not `turn.failed`: nothing broke, and that event puts an
        error row in the transcript.
        """
        ts = self._turns.get(session_id)
        if ts is None or ts.exchange_start is None or ts.user_text is None:
            return None
        text, start, task = ts.user_text, ts.exchange_start, ts.task
        if task is not None:
            task.cancel()
            try:
                await task  # let the cancellation unwind before anything reads
            except (Exception, asyncio.CancelledError):
                pass
        self._turns.pop(session_id, None)
        # The exchange is over, however many turns it took: nothing after this
        # continues it, so the next turn must not inherit its message.
        self._anchors.pop(session_id, None)
        # The rows this turn drew include a call still waiting for a result
        # that is never coming. Nothing may revise them again; what re-states
        # them is the `chat.reset` the caller sends once the thread has
        # settled, which is the only frame that can take a row off the screen.
        self._close_turn(session_id)
        self._deps.emit(TurnActivity(session_id=session_id, activity=""))
        kept = await self._keep_stopped_work(session_id, start)
        if kept and self._on_turn_stopped is not None:
            try:
                await self._on_turn_stopped(ts.session, ts.plan, ts.first_new)
            except Exception:  # a log is never worth a second failure
                logger.exception("post-turn handling of a stop failed")
        # Last, as on both the finished and the failed paths: what a client is
        # told about is what has already been written down.
        self._deps.emit(TurnFinished(session_id=session_id))
        asyncio.ensure_future(self.drain())
        return Stopped(text=None if kept else text)

    async def _keep_stopped_work(self, session_id: str, start: int) -> bool:
        """Close off a stopped turn's work in the thread, and say whether
        there was any to close off.

        The one question that cannot be answered from the scheduler's own
        bookkeeping: a turn is announced before the graph has written
        anything, so between ``start`` and the thread's length now is the
        difference between an exchange that happened and one that never got
        off the ground. An empty one is left completely alone — a lone
        "[stopped]" note under nothing at all would be the only trace of a
        turn nobody can see.
        """
        try:
            count = await thread_message_count(self._graph, session_id=session_id)
            if count <= start:
                return False
            await stop_thread(self._graph, session_id=session_id)
        except Exception as e:
            # The turn is stopped either way; what failed is the tidying. Said
            # rather than logged, because the next turn runs on this thread.
            # Falling through to "kept" is the answer that cannot do damage:
            # handing a message back that is in fact in the thread would have
            # the user send it twice, and the reset shows them either way.
            self._deps.emit(
                Notify(severity="error", text=f"Interrupt cleanup failed: {e}")
            )
        return True

    # -------------------------------------------------------------- shutdown

    def forget_session(self, session_id: str) -> None:
        """Drop everything a deleted session owned, so nothing queued for it
        starts against a thread that no longer exists."""
        self._pending = [w for w in self._pending if w.session_id != session_id]
        self._decisions.pop(session_id, None)
        self._awaiting_approval.discard(session_id)
        self._anchors.pop(session_id, None)
        # Its rows went with it; nothing can address them again.
        self._entry_seqs.pop(session_id, None)
        self._live.pop(session_id, None)
        self._last_exchange.pop(session_id, None)

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
