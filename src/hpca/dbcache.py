"""Node-local working copies of the sqlite databases
(see specs/specs-db-local-cache.md).

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

Two things keep the copies themselves cheap. The databases that churn are
*rebuilt* rather than duplicated, so a file that is mostly free list does not
cross NFS as one (``COMPACT_DB_NAMES``); and a periodic sync skips any database
nothing has written to since it last went home (``fingerprint``). Neither
applies to the final sync, which always copies everything.
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
DB_NAMES = ("hpca.db", "checkpoints.db")

# Where the per-profile document indexes live, one ``<profile>.db`` each
# (`hpca.rag.RagStores`). Not in ``DB_NAMES``: a user with ten profiles would
# copy ten indexes of a few hundred megabytes to the node at every start to use
# one. A profile's index joins the cache when it is first opened (``adopt``).
RAG_DIR = "rag"

# Databases an older HPCA kept here. Recovered from a crashed run's working
# dir like the rest, and then never seeded or synced again: ``rag.db`` was the
# one document index every profile shared, and boot moves it into ``RAG_DIR``
# (`hpca.rag.migrate_shared_index`).
LEGACY_NAMES = ("rag.db",)

# The databases every copy of which is written *compacted* — rebuilt with
# ``VACUUM INTO`` rather than page-copied with the backup API.
#
# sqlite never shrinks a file on its own: ``auto_vacuum`` is NONE, deleted
# pages go on the free list, and the backup API reproduces that free list
# page for page. checkpoints.db is where that bites — LangGraph writes a full
# state snapshot per super-step and a deleted session frees every page it
# ever wrote, so the file measured 94% free pages (150 MB of air around
# 9 MB of live checkpoints) and all 160 MB of it crossed NFS at every seed,
# every sync and every exit. Rebuilding copies only the live pages, which
# makes the copy both smaller *and* cheaper to make.
#
# The document indexes (rag/*.db, formerly rag.db) are deliberately absent, and not because a rebuild would fail — VACUUM
# copies a virtual table's shadow tables and carries its schema row across
# without ever instantiating the module, so vec0 survives it untouched. It is
# absent because it has nothing to reclaim: an embedding index grows, it does
# not churn, and it was measured at zero free pages beside checkpoints.db's
# 36,767. Rebuilding it would re-create every index on a hundred thousand
# chunks, every sync, to save nothing. ``_write_copy`` falls back on its own
# if a rebuild fails anyway, so this list is the default and not a promise.
COMPACT_DB_NAMES = frozenset({"hpca.db", "checkpoints.db"})

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


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_sqlite(path: Path) -> bool:
    """Whether ``path`` could be a sqlite database at all.

    Empty counts as yes — sqlite treats a zero-byte file as a database with
    no tables yet. Unreadable counts as yes too: failing to read the header
    says nothing about the content, and quarantining must fire only on
    evidence.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return True
    return head == b"" or head == _SQLITE_MAGIC


def quarantine_corrupt(path: Path) -> Path:
    """Move a file that is not a sqlite database out of the database's name.

    Renamed, not deleted: the corruption seen in the field zeroed exactly the
    first page and left every later one intact, so the bytes are worth an
    offline ``.recover``. The ERROR is deliberate — this is the moment a
    database was found destroyed, and it must not read like routine sync
    noise.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    aside = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 0
    while aside.exists():
        n += 1
        aside = path.with_name(f"{path.name}.corrupt-{stamp}.{n}")
    os.replace(path, aside)
    logger.error(
        "%s is not a sqlite database (torn write, or an uncoordinated "
        "writer); moved it to %s so a fresh copy can take its place",
        path,
        aside,
    )
    return aside


# Sources a rebuild has already failed for, so the fallback is taken directly
# rather than after wasting another failed VACUUM on every sync. Keyed by the
# source path as given — which is stable here, since both sides of the cache
# ask for a database by a path this module composed itself.
_COMPACT_UNSUPPORTED: set[str] = set()


def _vacuum_into(source: sqlite3.Connection, tmp: Path) -> None:
    """Rebuild ``source`` into the not-yet-existing file ``tmp``.

    Its own function so the fallback in ``_write_copy`` has a seam to be
    tested through: the rebuild is hard to make fail on purpose, which is
    rather the point of keeping a fallback at all.
    """
    # A parameter, not an f-string: the path is a value here and sqlite takes
    # it as one.
    source.execute("VACUUM INTO ?", (str(tmp),))


def _write_copy(src: Path, tmp: Path, *, compact: bool) -> None:
    """Write a standalone copy of ``src`` to the fresh path ``tmp``.

    ``compact`` rebuilds rather than duplicates: ``VACUUM INTO`` writes only
    the live pages, so the copy carries none of the source's free list. It is
    still one statement inside a read transaction, so it is as safe under a
    live writer as the backup API is.

    The fallback is for a rebuild that fails for a reason this code cannot
    anticipate. A database that can be page-copied is not one a sync should
    give up on, so the failure is logged, the page copy is taken instead, and
    the source is remembered — the answer is a property of the database, and
    retrying it every sync would waste a whole failed rebuild each time.
    """
    source = sqlite3.connect(src)
    try:
        if compact and str(src) not in _COMPACT_UNSUPPORTED:
            try:
                _vacuum_into(source, tmp)
                return
            except sqlite3.Error as e:
                _COMPACT_UNSUPPORTED.add(str(src))
                logger.warning(
                    "cannot rebuild %s compacted (%s); copying its pages "
                    "instead, free list and all",
                    src,
                    e,
                )
                tmp.unlink(missing_ok=True)  # a partial rebuild may remain
        target = sqlite3.connect(tmp)
        try:
            source.backup(target)
            # Leave the destination standalone: another node may read it.
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            target.close()
    finally:
        source.close()


def copy_database(src: Path, dst: Path, *, compact: bool = False) -> Path | None:
    """Copy one sqlite database, page for page or rebuilt compact.

    Not ``shutil.copy``: both mechanisms are transactionally consistent even
    while another connection writes the source, so the live checkpointer and
    RagStore connections need not be quiesced, and both get WAL right — the
    destination is one self-contained file, with no ``-wal``/``-shm``
    sidecars to copy along.

    The default is the page-level backup API, which carries ``rag.db``'s
    ``vec0`` virtual tables without the sqlite-vec extension being loaded
    because it copies pages rather than rows. ``compact`` asks instead for a
    rebuild that leaves the source's free pages behind — see
    ``COMPACT_DB_NAMES`` for which databases want that and why, and
    ``_write_copy`` for what happens when a rebuild fails.

    The copy lands in a sibling temp file that is renamed over ``dst`` only
    once complete. Writing straight into ``dst`` proved able to destroy
    it: an interrupted write on a network filesystem left a home copy with
    its first page zeroed, and the backup API then refused that file in both
    directions ("file is not a database") — one torn file blocked seed,
    recovery and every later sync-back. The rename is atomic, so ``dst`` is
    only ever its old self or the finished copy. If ``dst`` already is such
    a torn file it is moved aside (see ``quarantine_corrupt``) rather than
    silently buried under the fresh copy; the quarantined path is returned
    so callers can tell the user.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".backup-tmp")
    tmp.unlink(missing_ok=True)  # a killed copy may have left one behind
    try:
        _write_copy(src, tmp, compact=compact)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    quarantined = None
    if dst.exists() and not _is_sqlite(dst):
        quarantined = quarantine_corrupt(dst)
    os.replace(tmp, dst)
    return quarantined


def fingerprint(path: Path) -> tuple:
    """What ``path`` and its WAL sidecar look like from the outside.

    The cheap answer to "has anything been written since the last sync": two
    stats on node-local storage, against a whole-file copy over NFS. Size and
    mtime of both files, because a WAL-mode commit lands in the ``-wal``
    sidecar and may leave the main file untouched for a long time.

    Deliberately conservative in one direction only. A stat that changes
    without the content changing costs one needless copy; the opposite —
    content changing without either stat moving — is what would lose data, so
    the only place this is trusted is a periodic sync, never the final one.
    A missing file is a value like any other (None), so a WAL appearing or
    being checkpointed away both read as a change.
    """
    out = []
    for p in (path, path.with_name(path.name + "-wal")):
        try:
            st = p.stat()
            out.append((st.st_size, st.st_mtime_ns))
        except OSError:
            out.append(None)
    return tuple(out)


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
        # Things the user should be told, not just the log: today, that a
        # corrupt home copy was quarantined. Drained by the app for toasts.
        self.warnings: list[str] = []
        self._configured_dir = Path(local_dir) if local_dir is not None else None
        self._local: Path | None = None
        self._holds_lease = False
        # What each database looked like when it was last successfully synced
        # home (see ``fingerprint``). A name absent from here has never been
        # synced by this run and is always copied — which is what gives the
        # first sync of a run the chance to replace a bloated home copy with
        # a compacted one even if nothing has written to it yet.
        self._synced: dict[str, tuple] = {}
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

    @staticmethod
    def adoptable(name: str) -> bool:
        """Whether ``name`` is a database ``adopt`` may take on: one file
        directly inside ``RAG_DIR``, so a name can never reach outside it."""
        path = Path(name)
        return (
            len(path.parts) == 2
            and path.parts[0] == RAG_DIR
            and path.suffix == ".db"
            and not path.name.startswith(".")
        )

    def adopt(self, name: str) -> Path:
        """Take ``name`` into the cache, and say where to open it.

        In local mode a home copy is seeded into the working dir the first
        time, and from then on it is synced like the fixed databases. Blocking
        — a seed can be hundreds of megabytes over NFS — so call it off the
        loop.
        """
        if not self.adoptable(name):
            raise ValueError(f"not a database the cache can take on: {name!r}")
        with self._lock:
            if name not in self.names:
                self.names = (*self.names, name)
                if self.active and self._local is not None:
                    self._seed_one(name)
            return self.path_for(name)

    def forget(self, name: str) -> None:
        """Stop caching ``name`` and delete its working copy. Home is the
        caller's: this is half of deleting a database, not all of it."""
        with self._lock:
            self.names = tuple(n for n in self.names if n != name)
            self._synced.pop(name, None)
            if self._local is not None:
                for suffix in ("", "-wal", "-shm", ".backup-tmp"):
                    (self._local / (name + suffix)).unlink(missing_ok=True)

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

    def sync(self, *, force: bool = False) -> bool:
        """Write the local databases back to home.

        A database nothing has written to since its last sync is skipped:
        every tick otherwise pushed all three whole files across NFS, and
        ``rag.db`` alone is hundreds of megabytes that only change when
        something is indexed. ``fingerprint`` is what "written to" is read
        from, and it can only err towards copying too often.

        ``force`` syncs regardless, and ``release`` uses it. A skipped sync
        is only ever safe because the local copy is still there: a hard kill
        in the window leaves the working dir behind and the next start
        recovers it (``_recover``). The final sync has no such window — the
        working dir is about to be deleted — so it never skips.

        Never raises — home being briefly unreachable is a reason to try again
        next tick, not to take the app down. Returns whether *everything* got
        there, which is what ``release`` needs to know before it deletes the
        only other copy; a database that was already there counts as arrived.
        """
        with self._lock:
            if not self.active or self._local is None:
                return False
            complete = True
            for name in self.names:
                src = self._local / name
                if not src.exists():
                    continue
                # Read before the copy, not after: opening and closing a
                # connection to a WAL database can checkpoint it and remove
                # the sidecar, which would move the stamp under us. Stale in
                # that direction only costs one extra copy next tick.
                stamp = fingerprint(src)
                if not force and self._synced.get(name) == stamp:
                    continue
                try:
                    aside = copy_database(
                        src, self.home / name, compact=name in COMPACT_DB_NAMES
                    )
                    self._synced[name] = stamp
                    if aside is not None:
                        self.warnings.append(
                            f"home copy of {name} was corrupt; moved it to "
                            f"{aside.name} and synced a fresh copy"
                        )
                except Exception:
                    # One unreadable database must not cost the others their
                    # sync; the next tick tries again — which it only will if
                    # this name is not left looking already-synced.
                    self._synced.pop(name, None)
                    logger.exception("sync back failed for %s", name)
                    complete = False
            try:
                self._write_lease()
            except OSError:
                logger.exception("could not refresh the lease heartbeat")
                complete = False
            return complete

    def drain_warnings(self) -> list[str]:
        """Notices the user should see, cleared on read.

        The log has the tracebacks; these are the one-line versions the app
        can toast. Under the lock because ``sync()`` appends from a worker
        thread while the UI drains.
        """
        with self._lock:
            out, self.warnings = self.warnings, []
            return out

    def release(self) -> None:
        """Final sync, then hand local mode back. Idempotent."""
        with self._lock:
            if not self.active:
                return
            try:
                synced = self.sync(force=True)
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
        # Beside the fixed names, whatever a crashed run had adopted — which
        # this run has not, yet — and what an older HPCA left under a name no
        # longer in use.
        left = sorted(
            f"{RAG_DIR}/{path.name}"
            for path in (self._local / RAG_DIR).glob("*.db")
            if self.adoptable(f"{RAG_DIR}/{path.name}")
        )
        for name in dict.fromkeys([*self.names, *LEGACY_NAMES, *left]):
            src = self._local / name
            if not src.exists():
                continue
            try:
                aside = copy_database(
                    src, self.home / name, compact=name in COMPACT_DB_NAMES
                )
                recovered.add(name)
                logger.info("recovered %s from %s", name, src)
                if aside is not None:
                    self.warnings.append(
                        f"home copy of {name} was corrupt; moved it to "
                        f"{aside.name} and recovered the local copy in its "
                        f"place"
                    )
            except Exception:
                logger.exception("could not recover %s from %s", name, src)
        return recovered

    def _seed(self, *, skip: set[str]) -> None:
        """Copy home's databases into the working dir. Names in ``skip`` were
        just recovered, so the two sides already agree.

        Compacted like the sync back, and this is the direction where it pays
        most: a rebuild reads only the live pages, so seeding a home copy
        that is mostly free list moves a fraction of the bytes over NFS *and*
        starts the run from a small working copy.
        """
        assert self._local is not None
        for name in self.names:
            if name in skip:
                continue
            self._seed_one(name)

    def _seed_one(self, name: str) -> None:
        """Copy one database from home into the working dir, if home has it."""
        assert self._local is not None
        src = self.home / name
        if not src.exists():
            return  # first run for this database; the app creates it
        try:
            copy_database(src, self._local / name, compact=name in COMPACT_DB_NAMES)
        except Exception:
            logger.exception("could not seed %s from %s", name, src)
            if not _is_sqlite(src):
                # The home copy itself is destroyed and there is no local
                # copy to prefer. Move it aside so the app can start this
                # database afresh — leaving it would also block every
                # sync-back for the rest of the run.
                aside = quarantine_corrupt(src)
                self.warnings.append(
                    f"home copy of {name} is corrupt; moved it to "
                    f"{aside.name} and starting this database afresh"
                )

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
