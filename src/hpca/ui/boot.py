"""Standing the core up in the UI's own process, and taking it down in order.

`run.py` owns the terminal and `client.py` owns the protocol; this owns the
*runtime* — the databases, the checkpointer, the `AgentService` — and the
`InProcessConnection` the other two talk over. It is the half of
`tui/app.py`'s `on_mount`/`on_unmount` that is not about widgets, and it is
deliberately the only place in `hpca.ui` that knows `hpca.core` exists.

**Why in-process, and why that is not a shortcut.** `specs-ui-replacement.md`
§2 puts the subprocess split (`--serve`) last on purpose: it buys crash
isolation, not correctness. `InProcessConnection.pair()` is the same
`Connection` surface the socket presents, down to close ending the peer's
iteration, so the day the core moves out only `Core.start` changes.

**The shutdown order is load-bearing, not tidiness.** Checkpointer, then DbIO,
then sqlite, then the RAG store, and only then the final sync home — because
the sync must copy a quiesced database, and nothing may open a file inside the
working dir after it has been removed. `specs-ui-acceptance.md` records it
under "Node-local databases" and `tests/test_tui_dbcache.py` asserts it of the
Textual app; the same guarantees hold here, and `tests/test_ui_boot.py`
asserts them of this one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import signal
from pathlib import Path
from typing import Any

from hpca import protocol
from hpca.transport import Connection, InProcessConnection

logger = logging.getLogger("hpca.ui.boot")


# Printed on the real terminal while the final sync runs. Until it finishes the
# app looks hung — the UI is gone and the shell prompt is not back yet — and
# copying three databases home over NFS is seconds to minutes.
#
# The same three strings as `tui/app.py`'s, and copied rather than imported:
# importing them would drag Textual into the front-end that exists to replace
# it. M9 deletes that module and leaves these as the only copies.
DB_SYNC_WAIT_MESSAGE = (
    "please WAIT a moment while the chat log databases are being copied ..."
)

# What one impatient Ctrl+C gets instead of killing the copy. A second press
# still aborts: the working dir and its lease survive a kill, and the next
# start recovers from them, so no chat log is lost either way.
DB_SYNC_INTERRUPT_MESSAGE = (
    "still copying — press Ctrl+C again to abort "
    "(the next start then finishes the copy)"
)

DB_SYNC_DONE_MESSAGE = "chat log databases copied."


def file_logger(name: str, filename: str) -> logging.Logger:
    """A logger writing to <app_dir>/<filename>, and only there.

    The handler is attached on first use (idempotent) and does not propagate:
    this UI owns the terminal in raw mode, so a record reaching the root
    logger — whose last-resort handler writes to stderr — would land in the
    middle of a frame and stay there until the next full paint.

    The app dir is created here rather than assumed: on a first run nothing has
    written settings yet, and these loggers are the first thing startup
    touches.
    """
    from hpca.config import app_dir

    log = logging.getLogger(name)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        app_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(app_dir() / filename)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


@contextlib.contextmanager
def quiet_terminal():
    """Every log record into a file, and none of them onto the terminal.

    Without this a single WARNING from anywhere in `hpca` reaches logging's
    last-resort handler, which writes to stderr — into the middle of a frame,
    where it stays until the next full paint. Disabling propagation on the
    package logger is what stops that.

    Given back on the way out, because a UI is a guest in someone's process:
    the test suite reads records off the root logger, and a front-end that
    permanently silenced `hpca` would take those assertions with it.
    """
    from hpca.config import app_dir

    log = logging.getLogger("hpca")
    before = (list(log.handlers), log.propagate, log.level)
    handler: logging.Handler | None = None
    try:
        app_dir().mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(app_dir() / "ui.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    except OSError:  # an unwritable app dir is not a reason not to start
        logger.warning("no log file; records stay wherever they were going")
    try:
        yield
    finally:
        log.handlers, log.propagate, log.level = before
        if handler is not None:
            handler.close()


def _say(text: str) -> None:
    """Straight to the terminal, outside any frame. Best effort.

    Nothing said here is worth failing a shutdown over, and under a test there
    may be nothing on the other end of stdout at all.
    """
    import sys

    with contextlib.suppress(Exception):
        sys.stdout.write(text)
        sys.stdout.flush()


class Core:
    """An `AgentService`, the databases under it, and the wire to the UI.

    Built by `Core.start`, which is the order `tui/app.py`'s `on_mount` proved
    out: the database cache decides *where* the databases are, so it comes
    before anything opens one.
    """

    def __init__(
        self,
        service,
        wire: Connection,
        *,
        dbcache=None,
        dbio=None,
        db=None,
        saver_ctx=None,
        rag=None,
        sync_interval: float = 0.0,
        notices: list[str] | None = None,
        profile: str = "default",
    ) -> None:
        self.service = service
        self.wire = wire
        self.dbcache = dbcache
        self.dbio = dbio
        self.db = db
        self.saver_ctx = saver_ctx
        self.rag = rag
        self.profile = profile
        self.sync_interval = sync_interval
        # Things the user should be told that happened before there was a UI to
        # tell — a declined local cache, a quarantined corrupt copy. Delivered
        # as `notify` events so they arrive as toasts like everything else.
        self.notices = list(notices or [])
        self._tasks: list[asyncio.Task] = []
        self._syncing = False
        self._sync_failing = False
        self._stopped = False

    # ------------------------------------------------------------- lifecycle

    @classmethod
    async def start(
        cls,
        *,
        settings=None,
        profile: str = "default",
        llm: Any = None,
        tools=None,
        slurm=None,
        app_dir: Path | None = None,
        wire: Connection | None = None,
    ) -> "Core":
        """Everything the runtime needs, opened in the one order that works."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from hpca.config import Settings, app_dir as default_app_dir
        from hpca.core.service import build_service
        from hpca.db import DbIO, connect, init_db
        from hpca.rag import RagStore
        from hpca.runner import reconcile_orphans
        from hpca.sessions import SessionStore

        settings = settings or Settings.load()
        root = Path(app_dir) if app_dir is not None else default_app_dir()
        cache, notices = _open_db_cache(settings, root)

        db = connect(cache.path_for("hpca.db"))
        init_db(db)
        # Processes the previous run was watching when it exited would claim to
        # be running forever.
        reconcile_orphans(db)
        dbio = DbIO(cache.path_for("hpca.db"))
        saver_ctx = AsyncSqliteSaver.from_conn_string(
            str(cache.path_for("checkpoints.db"))
        )
        checkpointer = await saver_ctx.__aenter__()
        rag = RagStore(cache.path_for("rag.db"))

        service = build_service(
            settings=settings,
            app_dir=root,
            db=dbio.run,
            conn=db,
            checkpointer=checkpointer,
            profile=profile,
            slurm=slurm if slurm is not None else _detect_slurm(settings),
            tools=tools,
            llm=llm,
            session_store=SessionStore(db),
        )
        # The RAG store is the one database whose location `build_service` has
        # no parameter for — it falls back to `<app_dir>/rag.db`, which is
        # home, which is the NFS mount this whole cache exists to keep off the
        # hot path. `extras` is the sanctioned way to hand a service something
        # without widening `CoreDeps`; a `rag=` argument on `build_service`
        # would be tidier and belongs to whoever next edits `hpca.core`.
        service._deps.extras["rag"] = rag

        core = cls(
            service,
            wire if wire is not None else InProcessConnection(),
            dbcache=cache,
            dbio=dbio,
            db=db,
            saver_ctx=saver_ctx,
            rag=rag,
            sync_interval=settings.database.sync_interval_s,
            notices=notices,
            profile=profile,
        )
        return core

    def run(self) -> None:
        """Start the pumps. Returns immediately; the tasks live on the loop.

        Three of them, and each is one direction of one thing: events out,
        commands in, databases home. Started together because the handshake is
        the first event out and a UI that never receives it never asks for a
        session list.
        """
        self.service.start_timers()
        self._tasks.append(asyncio.ensure_future(self._events_out()))
        self._tasks.append(asyncio.ensure_future(self._commands_in()))
        if self.dbcache is not None and self.dbcache.active and self.sync_interval > 0:
            self._tasks.append(asyncio.ensure_future(self._sync_timer()))

    async def _events_out(self) -> None:
        """Everything the core says, onto the wire, in the order it said it."""
        queue = self.service.subscribe()
        await self.wire.send(
            protocol.Hello(profile=self.profile, settings_digest="")
        )
        for notice in self.notices:
            await self.wire.send(
                protocol.Notify(severity="warning", text=f"Databases: {notice}")
            )
        try:
            while True:
                event = await queue.get()
                await self.wire.send(event)
        except asyncio.CancelledError:
            raise
        finally:
            self.service.unsubscribe(queue)

    async def _commands_in(self) -> None:
        """Everything the UI asks for, into the service.

        A frame that is not a command is dropped rather than raised: the far
        end of this is a socket one day, and a peer sending junk down it must
        not be able to end the session (`transport` makes the same bargain).
        """
        async for env in self.wire:
            try:
                message = protocol.parse(env)
            except protocol.ProtocolError as e:
                logger.warning("dropping a frame the core cannot read: %s", e)
                continue
            if not isinstance(message, protocol.Command):
                logger.warning("dropping %s: not a command", env.type)
                continue
            await self.service.handle(message)

    # ----------------------------------------------------------- the db sync

    async def _sync_timer(self) -> None:
        """Write the node-local databases back to home, every interval.

        Skipped while a previous sync is still running: on NFS one can outlast
        its interval, and two backups of the same file at once is pointless
        work. Failures are reported and retried on the next tick — home being
        briefly unreachable must not take the app down.
        """
        while True:
            await asyncio.sleep(self.sync_interval)
            await self.sync()

    async def sync(self) -> None:
        cache = self.dbcache
        if cache is None or not cache.active or self._syncing:
            return
        self._syncing = True
        try:
            complete = await asyncio.to_thread(cache.sync)
            for warning in cache.drain_warnings():
                self._notify(f"Databases: {warning}", "warning")
            # Said once when syncing starts failing, not on every tick: a
            # database that cannot reach home for long means the app dir's copy
            # is quietly falling behind, and dbcache.log alone proved too easy
            # to miss.
            if not complete and not self._sync_failing:
                self._notify(
                    "Database sync to home is failing; see dbcache.log", "warning"
                )
            self._sync_failing = not complete
        except Exception as e:
            self._notify(f"Database sync to home failed: {e}", "warning")
        finally:
            self._syncing = False

    def _notify(self, text: str, severity: str = "information") -> None:
        """Say something through the same fan-out the services use, so it
        reaches the UI as a toast rather than onto a terminal in raw mode."""
        with contextlib.suppress(Exception):
            self.service._deps.emit(protocol.Notify(severity=severity, text=text))

    # ------------------------------------------------------------- shutdown

    async def stop(self, *, say=_say) -> None:
        """Quiesce, close, then copy home — in that order, which is the point.

        Every close before the sync is what makes the copied database a
        quiesced one; the sync last is what makes it safe to remove the working
        dir. The message is printed by the caller's `say`, which reaches a
        terminal that has already left the alternate screen — otherwise it
        would be wiped out along with it.
        """
        if self._stopped:
            return
        self._stopped = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
        self._tasks.clear()
        # Timers, live turns, backend clients. First, because a poll firing
        # after the databases close would raise into a dead loop.
        with contextlib.suppress(Exception):
            await self.service.stop()
        if self.saver_ctx is not None:
            with contextlib.suppress(Exception):
                await self.saver_ctx.__aexit__(None, None, None)
            self.saver_ctx = None
        if self.dbio is not None:
            with contextlib.suppress(Exception):
                await self.dbio.close()
            self.dbio = None
        if self.db is not None:
            with contextlib.suppress(Exception):
                self.db.close()
            self.db = None
        if self.rag is not None:
            with contextlib.suppress(Exception):
                self.rag.close()
            self.rag = None
        await self._final_sync(say)
        with contextlib.suppress(Exception):
            await self.wire.close()

    async def _final_sync(self, say) -> None:
        """The copy home, and the only thing the user is left looking at.

        Nothing is said when there is nothing to wait for: without the local
        cache the databases are already home and `release()` returns at once.
        """
        cache = self.dbcache
        if cache is None:
            return
        announced = cache.active
        if announced:
            say(f"\n{DB_SYNC_WAIT_MESSAGE}\n")
        with _sync_interrupt_guard(announced, say):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(cache.release)
        if announced:
            say(f"{DB_SYNC_DONE_MESSAGE}\n")
        self.dbcache = None


@contextlib.contextmanager
def _sync_interrupt_guard(announced: bool, say):
    """Answer the first Ctrl+C during the final sync instead of dying on it.

    While the UI runs, the terminal is in raw mode and Ctrl+C is not a signal
    at all — it is the byte `keys.py` calls "quit". By the time this runs the
    terminal is back to normal, so the impatient press that the wait message
    exists to prevent really would kill the process mid-copy. The first press
    gets an answer; a second restores the default handler and aborts, because a
    hung NFS mount must not become a trap and a kill costs nothing permanent —
    the working dir survives it and the next start recovers from there.
    """
    if not announced:
        yield
        return

    def on_interrupt(signum: int, frame: object) -> None:
        signal.signal(signal.SIGINT, previous)
        say(f"{DB_SYNC_INTERRUPT_MESSAGE}\n")

    try:
        previous = signal.signal(signal.SIGINT, on_interrupt)
    except ValueError:
        # Not the main thread; nobody's signal handling to borrow.
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, previous)


def _open_db_cache(settings, root: Path):
    """Decide where the databases run, before any of them is opened.

    Local mode is the fast path; declining it (disabled in settings, another
    instance holding the lease, an unusable working dir) means running straight
    from home, which is slower but always correct — so the only thing to do
    about it is say so.
    """
    from hpca.dbcache import DbCache, local_dir_for

    file_logger("hpca.dbcache", "dbcache.log")  # before acquire(): recovery logs here
    cache = DbCache(
        root,
        local_dir=local_dir_for(root, configured=settings.database.local_dir),
        enabled=settings.database.local_cache,
    )
    notices: list[str] = []
    if not cache.acquire() and settings.database.local_cache:
        notices.append(cache.reason)
    notices.extend(cache.drain_warnings())
    return cache, notices


def _detect_slurm(settings):
    """Job tools are available when sbatch exists or a submit host is set."""
    from hpca.slurm import SlurmClient

    submit_host = settings.cluster.submit_host
    if submit_host or shutil.which("sbatch"):
        return SlurmClient(submit_host=submit_host)
    return None


async def start(
    *,
    settings=None,
    profile: str = "default",
    llm: Any = None,
    tools=None,
    slurm=None,
    app_dir: Path | None = None,
    screen=None,
    say=_say,
) -> int:
    """The whole `hpca --new-ui` run: build, draw, shut down.

    The nesting is the shutdown order. `Screen` restores the terminal on the
    way out of its `with`, including out of a traceback, and the core is
    stopped *after* that — so the wait message lands on a terminal the user can
    read rather than on the alternate screen a moment before it disappears.
    """
    from hpca.ui.app import RowUI
    from hpca.ui.client import UIClient
    from hpca.ui.run import drive

    ui_end, core_end = InProcessConnection.pair()
    core = await Core.start(
        settings=settings,
        profile=profile,
        llm=llm,
        tools=tools,
        slurm=slurm,
        app_dir=app_dir,
        wire=core_end,
    )
    ui = RowUI()
    client = UIClient(ui, ui_end)
    try:
        core.run()
        with quiet_terminal():
            return await drive(ui, client=client, conn=ui_end, screen=screen)
    finally:
        with contextlib.suppress(Exception):
            await ui_end.close()
        await core.stop(say=say)
