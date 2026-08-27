"""Everything hpca does on a timer, with nothing to paint it (§5, §7).

Four sweeps that used to be ``set_interval`` callbacks on the Textual app: the
sacct poll over the jobs hpca submitted, the two watch polls behind the right
column's boxes, and the pass that notices a background subprocess has ended.
None of them was ever *about* the UI — they read the database and the cluster
and then told somebody — which is why they are the easiest domain to lift out
whole.

Two shapes differ from the code they came from.

**The panel is pushed, not pulled.** ``refresh_watchers`` queried a widget and
painted into it; here a poll produces `PanelRow` values and emits one
`PanelUpdate`, and what draws them is the renderer's business. The row *order*
crosses unchanged, because it is load-bearing for the UI's cursor — and it is
the user's own order now, so a poll must never reach for a different one.

**A finished background process reaches the agent through a callback.**
``watch_processes`` and ``poll_jobs`` appended to the app's ``_pending_work``
list directly. That queue, and the rule for when a session is free to run,
belong to the turn scheduler; a poller reaching into them would make a poll's
cadence decide a turn's timing, and would tie the timers to the one module
most likely to be rewritten next. So the text is handed over and forgotten.
It is deliberately not `CoreDeps.emit`: an event is for whoever is rendering
and may be dropped when nobody is, while a completion the agent promised to
check back on must not be.

Nothing here starts a timer. The coroutines are exposed one by one so the
caller schedules them — which is also what lets a test call them without
waiting on a wall clock — and :meth:`Pollers.timers` says what the core thinks
the cadence should be, since that is a property of how often sacct and an NFS
stat may be asked, not of how fast something draws.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

from hpca.agent.explainer import ProposedSignature, explain_process_failure
from hpca.core.deps import CoreDeps
from hpca.jobs import JobStore, apply_statuses
from hpca.protocol import (
    PANEL_WATCH,
    Notify,
    PanelRow,
    PanelUpdate,
)
from hpca.runner import (
    ProcessChange,
    analyse_process_failure,
    format_process_event,
    poll_processes,
)
from hpca.sessions import SessionStore
from hpca.slurm import TERMINAL_STATES as SLURM_TERMINAL_STATES
from hpca.triage import LogFinding, Signature, append_user_signature
from hpca.watches import (
    KIND_JOB,
    KIND_LOG,
    LOG_GONE,
    Watch,
    WatchStore,
    apply_job_details,
    is_settled,
    poll_log_watches,
    watch_class,
    watch_lines,
)

# How often the column is rebuilt. Also the resolution of the clock inside a
# watch box, which counts up from the last write on its own.
PANEL_SECONDS = 2.0
# ...and how often ended subprocesses are collected. The same cadence, because
# it is the same table write being noticed.
PROCESS_WATCH_SECONDS = 2.0
# How often a watched log is stat'ed. Cheap (one stat per box) and worth being
# responsive about: the whole point of a log box is that "still writing" and
# "stopped a minute ago" are visibly different.
LOG_WATCH_SECONDS = 5.0
# ...and how often watched jobs are refreshed. Slower: each sweep is an squeue
# call to the controller, and a job's state does not change on the second.
JOB_WATCH_SECONDS = 15.0
# Floor under the configured sacct cadence. Accounting is a shared database on
# a login node, and asking it faster than this tells nobody anything new.
MIN_JOB_POLL_SECONDS = 5.0


def _read_process_logs(
    change: ProcessChange,
) -> tuple[LogFinding | None, str]:
    """The finding and the event text for one finished process.

    The two steps are held together in one function so that the whole of the
    log reading — the triage scan and the tail that `format_process_event`
    takes when triage found nothing — crosses to a worker thread once, rather
    than the poller hopping threads twice per process.
    """
    finding = (
        analyse_process_failure(change) if change.state != "finished" else None
    )
    return finding, format_process_event(change, finding)


def panel_profile(
    conn: sqlite3.Connection, *, session_id: str | None, default: str
) -> str:
    """Whose watches the right column shows: the focused session's profile, or
    the core's own when no session is focused.

    Resolved inside the same database call as the rows themselves, so the
    watches and the profile they were selected by come from one snapshot —
    and so that reading it costs no extra round trip on an NFS home.
    """
    if session_id is None:
        return default
    session = SessionStore(conn).get(session_id)
    return session.profile if session is not None else default


class Confirm(Protocol):
    """Put a yes/no question about one session and run ``on_yes`` if accepted.

    Returns immediately: the poll that raised the question must not wait for a
    human to answer it. ``on_yes`` is the coroutine *function*, not a started
    coroutine, so a hook that decides to drop the question leaves nothing
    un-awaited behind.

    ``session_id`` is which conversation the question is about. A poll knows
    it and a person does not — by the time a background job fails, whoever
    started it is as likely as not reading another session — so it is asked
    for here rather than left to whoever draws the question to guess.
    """

    def __call__(
        self,
        session_id: str,
        question: str,
        on_yes: Callable[[], Awaitable[None]],
    ) -> None: ...


class Pollers:
    """The timed half of the runtime: what changed, and who needs to know.

    Holds no widget and never asks what is on screen. The one piece of UI
    state it reads is ``deps.focused_session_id``, which arrived as a
    `session.focus` command and decides whose processes and jobs the panel
    lists — the same question the old ``_panel_profile`` answered by reaching
    into a widget, made explicit.
    """

    def __init__(
        self,
        deps: CoreDeps,
        *,
        submit_event: Callable[[str, str], None],
        llm: Callable[[], Any | None] | None = None,
        confirm: Confirm | None = None,
    ) -> None:
        self._deps = deps
        # See the module docstring: the scheduler owns the queue, so finished
        # work is handed over rather than queued from here.
        self._submit_event = submit_event
        # A getter rather than a client: the user can switch backend between
        # two polls, and the code this came from read ``self._llm`` fresh on
        # every call. Optional because the explainer is a luxury — without it
        # a tier-2 finding is still delivered, just undiagnosed.
        self._llm = llm
        self._confirm = confirm
        # Overlap guard: a fetch outlasting the tick (NFS again) would let two
        # passes interleave. Skipping is safe — the next tick repaints.
        self._refreshing = False
        # The last rows emitted, so an unchanged column is not re-sent every
        # two seconds. Starts as the empty list rather than None because an
        # empty column is what the UI already shows before anything is said.
        self._last_rows: list[PanelRow] = []

    # ------------------------------------------------------------- the panel

    async def refresh_panel(self, *, force: bool = False) -> None:
        """Emit the right column, read from the *table* and not a live runner.

        A ``ProcessRunner`` only knows the processes it started and a fresh one
        is built per turn, so reading it emptied the panel the moment a session
        was reopened. The table outlives all of that, which is what makes the
        history still be there after a restart.

        ``force`` re-sends rows that have not changed — for a client that has
        just connected and has nothing on screen yet.
        """
        if self._refreshing:
            return
        self._refreshing = True
        try:
            await self._refresh_inner(force=force)
        finally:
            self._refreshing = False

    async def _refresh_inner(self, *, force: bool) -> None:
        deps = self._deps
        session_id = deps.focused_session_id
        default_profile = deps.profile

        def _gather(conn: sqlite3.Connection):
            # A watch belongs to the session that registered it, so the column
            # describes the conversation being read and nothing else. With no
            # session focused there is nothing of anyone's to show.
            profile = panel_profile(
                conn, session_id=session_id, default=default_profile
            )
            watches = (
                WatchStore(conn).list(session_id=session_id)
                if session_id is not None
                else []
            )
            return profile, watches

        try:
            profile, watches = await deps.db(_gather)
        except Exception:
            # A column that failed to read is one missing repaint and the next
            # tick fixes it. A toast twice a second out of a database being
            # torn down at shutdown would be worse than stale rows.
            return
        if session_id != deps.focused_session_id:
            return  # focus moved mid-fetch; the next tick paints the new one
        rows = self.panel_rows(watches)
        if rows == self._last_rows and not force:
            return
        self._last_rows = rows
        deps.emit(
            PanelUpdate(profile=profile, session_id=session_id, rows=rows)
        )

    def panel_rows(
        self, watches: list[Watch], *, now: datetime | None = None
    ) -> list[PanelRow]:
        """The whole right column, top to bottom: one box per watch.

        It used to carry the session's own run history underneath, under a
        "── this session ──" heading. That is gone. Every one of those calls is
        already in the chat log a column to the left, so the history was a
        second copy of something the user had just read — and it grew without
        bound while the boxes they had actually asked to be shown were pushed
        off the bottom of a short terminal. The column now holds only what was
        asked for, in the order it was asked for.

        In ``watches`` order, which is the user's own (``WatchStore.move``).
        """
        now = now or datetime.now(timezone.utc)
        return [
            PanelRow(
                key=f"w{watch.id}",
                text="\n".join(watch_lines(watch, now=now)),
                classes=f"watch {watch_class(watch)}",
                title=watch.title,
                kind=PANEL_WATCH,
                ref=str(watch.id),
            )
            for watch in watches
        ]

    # -------------------------------------------------------------- the jobs

    async def poll_jobs(self) -> None:
        """The sacct sweep over the jobs hpca submitted (§5.4).

        Both store halves run wherever ``deps.db`` puts sqlite; only the sacct
        subprocess is awaited here.
        """
        deps = self._deps
        if deps.slurm is None:
            return  # no cluster on this machine, so nothing to ask
        try:
            active = await deps.db(lambda conn: JobStore(conn).active())
            if not active:
                return
            statuses = await deps.slurm.status([j.job_id for j in active])
            changes = await deps.db(
                lambda conn: apply_statuses(JobStore(conn), active, statuses)
            )
        except Exception as e:
            deps.emit(Notify(severity="warning", text=f"Job polling failed: {e}"))
            return
        for change in changes:
            deps.emit(
                Notify(
                    text=(
                        f"Job {change.job_id}: "
                        f"{change.old_state} → {change.new_state}"
                    )
                )
            )
            # Only a terminal state is worth waking the agent for. PENDING →
            # RUNNING is news for the person watching; there is nothing to do
            # about it, and a turn spent saying so costs a real model call.
            if change.new_state in SLURM_TERMINAL_STATES and change.session_id:
                self._submit_event(
                    change.session_id,
                    f"[job {change.new_state.lower()}] cluster job "
                    f"{change.job_id} is now {change.new_state}. Check its "
                    "logs with triage_job and act on the result.",
                )
        if changes:
            await self.refresh_panel()

    # ----------------------------------------------------------- the watches

    async def poll_watched_logs(self) -> None:
        """Stat every watched log — where the "last write …" clock comes from.

        Stat and write happen in one database call: on an NFS home the stat is
        the slow half, and a stalled filesystem must cost the panel a late
        repaint, never the loop everything else runs on.
        """
        deps = self._deps
        session_id = deps.focused_session_id
        default_profile = deps.profile

        def _poll(conn: sqlite3.Connection):
            # Store-wide, not the focused session's: scoping decides what is
            # *shown*, never what stays true. A watch left behind in another
            # session must still be current when the user goes back to it.
            store = WatchStore(conn)
            logs = [w for w in store.list() if w.kind == KIND_LOG]
            return poll_log_watches(store, logs)

        try:
            changes = await deps.db(_poll)
        except Exception as e:
            deps.emit(Notify(severity="warning", text=f"Log watch failed: {e}"))
            return
        for change in changes:
            # A vanished file is the only thing a log poll can report, and the
            # only one worth interrupting for. There used to be a "no new
            # output" toast as well; it fired whenever a log had simply not
            # been written to for a while, which is not an event — the box
            # already says when the last write was, and a job between log
            # lines is not news.
            if change.old_state and change.new_state == LOG_GONE:
                deps.emit(
                    Notify(
                        severity="warning",
                        text=f"{change.watch.title}: the file is gone",
                    )
                )
        if changes:
            await self.refresh_panel()

    async def poll_watched_jobs(self) -> None:
        """Refresh watched Slurm jobs from squeue, finished ones from sacct.

        Only jobs that can still move are asked about: a box that already says
        COMPLETED costs nothing from here on.
        """
        deps = self._deps
        if deps.slurm is None:
            return
        session_id = deps.focused_session_id
        default_profile = deps.profile

        def _due(conn: sqlite3.Connection) -> list[Watch]:
            # Store-wide; see poll_watched_logs.
            return [
                w
                for w in WatchStore(conn).list()
                if w.kind == KIND_JOB and not is_settled(w)
            ]

        try:
            watches = await deps.db(_due)
        except Exception as e:
            deps.emit(Notify(severity="warning", text=f"Job watch failed: {e}"))
            return
        if not watches:
            return
        job_ids = [w.target for w in watches]
        try:
            details = await deps.slurm.job_details(job_ids)
            gone = [i for i in job_ids if i not in details]
            finished = await deps.slurm.status(gone) if gone else {}
            changes = await deps.db(
                lambda conn: apply_job_details(
                    WatchStore(conn), watches, details, finished
                )
            )
        except Exception as e:
            deps.emit(Notify(severity="warning", text=f"Job watch failed: {e}"))
            return
        for change in changes:
            # Same rule as a log: arriving in a state is not a change of one.
            if change.old_state:
                deps.emit(
                    Notify(
                        text=(
                            f"{change.watch.title}: {change.old_state} → "
                            f"{change.new_state}"
                        )
                    )
                )
        if changes:
            await self.refresh_panel()

    # --------------------------------------------------- local subprocesses

    async def watch_processes(self) -> None:
        """Turn finished background subprocesses into agent-visible events.

        The sibling of poll_jobs for local work. A panel repaint only tells
        the *user*, so before this nothing ever told the agent that the script
        it started had exited — it promised to check back and had no way to
        keep the promise.

        No panel update follows: the row's new state is picked up by the next
        ordinary refresh, and emitting from here would only make the column
        repaint on a schedule nobody reads.
        """
        deps = self._deps
        try:
            changes = await deps.db(poll_processes)
        except Exception as e:
            deps.emit(Notify(severity="warning", text=f"Process watch failed: {e}"))
            return
        for change in changes:
            # Off the loop: both of these read the process's logs. The reads
            # are bounded (hpca.filetail), but the logs sit on the cluster
            # filesystem, and this loop has one socket with every session's
            # commands behind it.
            finding, text = await asyncio.to_thread(_read_process_logs, change)
            if finding is not None and finding.tier == 2:
                # Tier 3: the keyword scan found candidates but cannot say
                # which one caused it. That judgment is worth a model call.
                text = await self._explain_candidates(change, finding, text)
            self._submit_event(change.session_id, text)

    async def _explain_candidates(
        self, change: ProcessChange, finding: LogFinding, fallback: str
    ) -> str:
        """Ask the explainer to pick the causing line, and learn from it."""
        llm = self._llm() if self._llm is not None else None
        if llm is None:
            return fallback
        try:
            explanation = await explain_process_failure(
                llm,
                name=change.name,
                exit_code=change.exit_code,
                candidates=finding.candidates,
            )
        except Exception as e:
            self._deps.emit(
                Notify(severity="warning", text=f"Log explainer failed: {e}")
            )
            return fallback
        if not explanation.conclusive:
            return fallback  # it said it could not tell; do not dress that up
        if explanation.proposed_signature is not None:
            self._offer_signature(change, explanation.proposed_signature)
        head = fallback.split("\n\n", 1)[0]
        return f"{head}\n\n{explanation.render()}\n\nlog:\n{finding.excerpt}"

    def _offer_signature(
        self, change: ProcessChange, proposed: ProposedSignature
    ) -> None:
        """A tier-3 diagnosis means a signature was missing; offer to keep it.

        Offered and not simply written: the library is the user's, one model
        call is thin evidence for adding to it, and a careless regex here
        widens into every later triage. With nothing wired to ask with, the
        proposal is dropped — the diagnosis still reaches the agent, and only
        the learning is skipped, which is the safe half to lose.

        ``change`` is here for the question rather than for the saving: what
        goes in the library is global, but *why it is being offered* is one
        job in one session, and both belong in front of whoever answers. The
        diagnosis this came out of has already gone to that session's chat
        (`_submit_event`), so the question and its evidence end up in the same
        conversation.
        """
        if self._confirm is None:
            return
        try:
            signature = Signature(
                id=proposed.id,
                title=proposed.title,
                patterns=list(proposed.patterns),
                hint=proposed.hint,
            )
        except re.error as e:
            # `Signature` compiles its patterns on construction, so a regex
            # the model got wrong surfaces here rather than in `save` — where
            # the old code expected it, and where a timer callback had nothing
            # to catch it. There is nothing to ask the user about, but a
            # backend proposing patterns that never compile is worth a word.
            self._deps.emit(
                Notify(
                    severity="warning",
                    text=f"Ignored a proposed error signature: {e}",
                )
            )
            return

        async def save() -> None:
            try:
                path = append_user_signature(signature)
            except Exception as e:
                self._deps.emit(
                    Notify(
                        severity="error",
                        text=f"Could not save signature: {e}",
                    )
                )
                return
            self._deps.emit(
                Notify(
                    text=(
                        f"Saved error signature {signature.id!r} "
                        f"to {path.name}"
                    )
                )
            )

        self._confirm(
            change.session_id,
            f"{change.name} failed. Remember this as {signature.id!r} "
            f"({signature.title}) so it is recognised next time?",
            save,
        )

    # ------------------------------------------------------------- schedule

    def timers(self) -> list[tuple[float, Callable[[], Awaitable[None]]]]:
        """``(interval, coroutine)`` for every sweep, for whoever schedules.

        The cadence belongs here — it is a property of how often sacct and an
        NFS stat may be asked — but the loop that runs them does not, so this
        hands over a list instead of starting tasks. The two cluster sweeps
        are absent without Slurm, exactly as their timers were never installed.
        """
        every: list[tuple[float, Callable[[], Awaitable[None]]]] = [
            (PANEL_SECONDS, self.refresh_panel),
            (PROCESS_WATCH_SECONDS, self.watch_processes),
            (LOG_WATCH_SECONDS, self.poll_watched_logs),
        ]
        if self._deps.slurm is not None:
            configured = float(self._deps.settings.cluster.job_poll_seconds)
            every.append((max(MIN_JOB_POLL_SECONDS, configured), self.poll_jobs))
            every.append((JOB_WATCH_SECONDS, self.poll_watched_jobs))
        return every
