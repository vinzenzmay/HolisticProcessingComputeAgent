"""Node-local working copies of the sqlite databases (see specs-db-local-cache.md).

On a cluster node ``$HOME`` is NFS, where every sqlite call is network
round-trips plus a remote fsync. HPCA hits sqlite constantly — the checkpointer
writes on every graph step, tool handlers read and write the app tables, the
poll timers tick every two seconds — so the TUI lags whenever the agent works.

``hpca.db.DbIO`` moved the poll timers off the event loop; it did not make
sqlite any faster. This module attacks the latency itself: the databases are
copied to node-local storage at startup (a Slurm job's ``/tmp`` is on the node
running the job), used from there, and synced back to home periodically and on
exit. Both mechanisms stay — one decides who waits, the other how long.

The trade is durability: a hard kill loses at most one sync interval, and only
if the node's ``/tmp`` is gone too — a surviving working dir is recovered on the
next start.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("hpca.dbcache")

# The databases this module manages. Everything else in the app dir
# (settings.json, profiles/, skills/, chatlogs/, trash/, ...) stays in home:
# none of it is on the hot path.
#
# checkpoints.db is a separate file from hpca.db by design: the LangGraph
# checkpointer writes through its own aiosqlite connection during graph
# execution, and sharing one file produced writer contention ("database is
# locked") with tool code updating the app tables mid-turn.
DB_NAMES = ("hpca.db", "checkpoints.db", "rag.db")

LEASE_NAME = "db.lease"

# How long another host's lease is honoured without a heartbeat. Three sync
# intervals, so a live instance whose sync is merely slow is never evicted.
DEFAULT_STALE_AFTER_S = 180.0


def local_root(configured: str | None = None) -> Path:
    """The node-local scratch root.

    ``$HPCA_LOCAL_DIR`` wins (tests, and users who know their site), then the
    configured override, then ``$TMPDIR`` — the per-job directory Slurm sets,
    which is what makes this node-local — then the system temp dir.

    ``$TMPDIR`` is read directly rather than through ``tempfile.gettempdir()``,
    which memoises its answer on first use and discards a dir that does not
    exist yet. Here it is only a location to create.
    """
    override = os.environ.get("HPCA_LOCAL_DIR")
    if override:
        return Path(os.path.expanduser(override))
    if configured:
        return Path(os.path.expanduser(configured))
    job_tmp = os.environ.get("TMPDIR")
    if job_tmp:
        return Path(os.path.expanduser(job_tmp))
    return Path(tempfile.gettempdir())


def local_dir_for(
    home: Path, *, root: Path | None = None, configured: str | None = None
) -> Path:
    """The working dir for the app dir ``home``.

    Keyed to the app dir, so two ``$HPCA_HOME``s never share one; and stable
    for a given app dir, which is what lets the next start find the copies a
    crash left behind. The uid keeps users apart in a shared ``/tmp``.
    """
    base = Path(root) if root is not None else local_root(configured)
    digest = hashlib.sha256(str(Path(home).resolve()).encode()).hexdigest()[:12]
    return base / f"hpca-{os.getuid()}-{digest}"


def copy_database(src: Path, dst: Path) -> None:
    """Copy one sqlite database with the online-backup API.

    Not ``shutil.copy``: the backup API is page-level and transactionally
    consistent even while another connection writes the source, so the live
    checkpointer and RagStore connections need not be quiesced. It also gets
    WAL right — the destination is one self-contained file, with no
    ``-wal``/``-shm`` sidecars to copy along — and it carries ``rag.db``'s
    ``vec0`` virtual tables without the sqlite-vec extension being loaded,
    because it copies pages rather than rows.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(src)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
            # Leave the destination standalone: another node may read it.
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            target.close()
    finally:
        source.close()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process — running, just not ours to signal
    except OSError:
        return True  # unknown: assume alive, the safe direction
    return True


@dataclass(frozen=True)
class Lease:
    """Who currently owns the local copies. Lives in ``<app_dir>/db.lease``."""

    host: str
    pid: int
    started_at: float
    heartbeat: float
    local_dir: str

    @classmethod
    def read(cls, path: Path) -> "Lease | None":
        """The lease at ``path``, or None if absent, unreadable or malformed.

        A lease we cannot parse must not lock anyone out — the file is a hint
        about another process, not a source of truth about our own data.
        """
        try:
            data = json.loads(path.read_text())
            return cls(
                host=str(data["host"]),
                pid=int(data["pid"]),
                started_at=float(data.get("started_at", 0.0)),
                heartbeat=float(data.get("heartbeat", 0.0)),
                local_dir=str(data.get("local_dir", "")),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)))

    def is_live(self, *, host: str, now: float, stale_after_s: float) -> bool:
        """Whether the holder is still running.

        On this host that is knowable exactly, from the pid. From another host
        it is not, so the heartbeat decides — and a heartbeat is only refreshed
        by a running sync timer.
        """
        if self.host == host:
            return _pid_alive(self.pid)
        return (now - self.heartbeat) < stale_after_s

    def held_by(self, *, host: str, pid: int) -> bool:
        return self.host == host and self.pid == pid


class DbCache:
    """The databases' working location, and the syncing that keeps home current.

    ``acquire()`` decides between local mode and direct-home mode; ``path_for``
    then hands out whichever paths apply, so callers never branch on it.
    Failure is always a fall back to direct-home mode — today's behaviour, which
    is correct, just slow — never a crash and never a lost database.
    """

    def __init__(
        self,
        home: Path,
        *,
        local_dir: Path | None = None,
        enabled: bool = True,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        names: Sequence[str] = DB_NAMES,
    ) -> None:
        self.home = Path(home)
        self.names = tuple(names)
        self.enabled = enabled
        self.stale_after_s = stale_after_s
        self.host = socket.gethostname()
        self.active = False
        # Why local mode was declined, for the toast. Empty while it is on.
        self.reason = ""
        self._configured_dir = Path(local_dir) if local_dir is not None else None
        self._local: Path | None = None
        self._holds_lease = False
        # sync() and release() both run on worker threads, and shutdown can
        # start while a periodic sync is still copying. Reentrant because
        # release() syncs before it lets go.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths

    @property
    def local_dir(self) -> Path:
        if self._configured_dir is not None:
            return self._configured_dir
        return local_dir_for(self.home)

    @property
    def lease_path(self) -> Path:
        return self.home / LEASE_NAME

    def path_for(self, name: str) -> Path:
        """Where ``name`` should be opened right now."""
        if name not in self.names:
            raise KeyError(name)
        if self.active and self._local is not None:
            return self._local / name
        return self.home / name

    # -------------------------------------------------------------- lifecycle

    def acquire(self) -> bool:
        """Try to take local mode. Returns whether it is on; see ``reason``."""
        if not self.enabled:
            self.reason = "local database cache is disabled in settings"
            return False

        local = self.local_dir
        if local.resolve() == self.home.resolve():
            self.reason = (
                "local database dir is the same as the app dir; "
                "running databases directly from home"
            )
            return False

        if not self._take_lease(local):
            return False
        try:
            local.mkdir(parents=True, exist_ok=True)
            # /tmp is shared on a cluster node and these databases hold whole
            # conversations. Set explicitly rather than via mkdir's mode: the
            # dir may be inherited from a crashed run, and umask can only
            # remove bits from what mkdir was asked for.
            local.chmod(0o700)
        except OSError as e:
            self._drop_lease()
            self.reason = f"cannot use {local}: {e}"
            return False

        self._local = local
        self.active = True
        self.reason = ""
        recovered = self._recover()
        self._seed(skip=recovered)
        return True

    def sync(self) -> bool:
        """Write the local databases back to home.

        Never raises — home being briefly unreachable is a reason to try again
        next tick, not to take the app down. Returns whether *everything* got
        there, which is what ``release`` needs to know before it deletes the
        only other copy.
        """
        with self._lock:
            if not self.active or self._local is None:
                return False
            complete = True
            for name in self.names:
                src = self._local / name
                if not src.exists():
                    continue
                try:
                    copy_database(src, self.home / name)
                except Exception:
                    # One unreadable database must not cost the others their
                    # sync; the next tick tries again.
                    logger.exception("sync back failed for %s", name)
                    complete = False
            try:
                self._write_lease()
            except OSError:
                logger.exception("could not refresh the lease heartbeat")
                complete = False
            return complete

    def release(self) -> None:
        """Final sync, then hand local mode back. Idempotent."""
        with self._lock:
            if not self.active:
                return
            try:
                synced = self.sync()
            except Exception:  # pragma: no cover - sync is already total
                logger.exception("final sync back failed")
                synced = False
            self.active = False
            local, self._local = self._local, None
            if local is not None and synced:
                # After a clean, complete sync home is authoritative, so
                # leaving the copies behind would only make the next start
                # recover them pointlessly. If the sync did NOT get everything
                # home, the working dir is the only copy of what is missing —
                # keep it, and let the next start recover it as after a crash.
                shutil.rmtree(local, ignore_errors=True)
            self._drop_lease()

    # ------------------------------------------------------------ seed/recover

    def _recover(self) -> set[str]:
        """Sync back databases a previous run left behind, before anything
        overwrites them. Returns the names recovered."""
        recovered: set[str] = set()
        assert self._local is not None
        for name in self.names:
            src = self._local / name
            if not src.exists():
                continue
            try:
                copy_database(src, self.home / name)
                recovered.add(name)
                logger.info("recovered %s from %s", name, src)
            except Exception:
                logger.exception("could not recover %s from %s", name, src)
        return recovered

    def _seed(self, *, skip: set[str]) -> None:
        """Copy home's databases into the working dir. Names in ``skip`` were
        just recovered, so the two sides already agree."""
        assert self._local is not None
        for name in self.names:
            if name in skip:
                continue
            src = self.home / name
            if not src.exists():
                continue  # first run for this database; the app creates it
            try:
                copy_database(src, self._local / name)
            except Exception:
                logger.exception("could not seed %s from %s", name, src)

    # ------------------------------------------------------------------ lease

    def _take_lease(self, local: Path) -> bool:
        held = Lease.read(self.lease_path)
        if held is not None and held.is_live(
            host=self.host, now=time.time(), stale_after_s=self.stale_after_s
        ):
            # Syncing back is a whole-file overwrite, so two instances with
            # private copies would erase each other's sessions. Sharing one
            # file on NFS does not have that problem — sqlite's own locking
            # handles concurrent writers — so falling back is the safe answer.
            self.reason = (
                f"another HPCA instance holds the local database cache "
                f"(host {held.host}, pid {held.pid}); running databases "
                f"directly from home"
            )
            return False
        try:
            self._write_lease(local)
        except OSError as e:
            self.reason = f"cannot write {self.lease_path}: {e}"
            return False
        self._holds_lease = True
        return True

    def _write_lease(self, local: Path | None = None) -> None:
        target = local if local is not None else self._local
        if target is None:
            return
        now = time.time()
        Lease(
            host=self.host,
            pid=os.getpid(),
            started_at=now,
            heartbeat=now,
            local_dir=str(target),
        ).write(self.lease_path)

    def _drop_lease(self) -> None:
        if not self._holds_lease:
            return
        self._holds_lease = False
        held = Lease.read(self.lease_path)
        if held is not None and not held.held_by(host=self.host, pid=os.getpid()):
            return  # someone else took over; not ours to remove
        self.lease_path.unlink(missing_ok=True)
