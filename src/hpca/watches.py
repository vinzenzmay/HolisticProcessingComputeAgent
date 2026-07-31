"""Registered watches: what the user asked hpca to keep an eye on (§3.3).

The right column used to show only what hpca itself started — its own
subprocesses and its own sbatch submissions — which is a thin slice of what
actually runs on a cluster, and a slice already visible in the chat log. The
work that matters is usually somebody else's: an sbatch script submitted by
hand, a snakemake run spawning one tool after another, a log some long-running
program appends to. Checking on those means ``squeue``, then ssh to the node,
then ``tail``. A watch pins one of them to the panel instead.

Two kinds, both cheap enough to poll on a timer:

* ``log`` — a file. Its mtime answers the one question worth asking while a
  tool runs: is anything still being written? "last write 4s ago" is alive,
  "last write 40m ago" is dead or wedged, and the difference is legible at a
  glance without leaving the TUI.
* ``job`` — a Slurm job id, refreshed from ``squeue``. When it drops out of
  the queue ``sacct`` supplies the final state, so the box settles on
  COMPLETED or FAILED instead of quietly vanishing.

A watch is bound to a *profile*, not a session: it describes the machine, not
the conversation, and the user wants it on screen whichever session they are
reading. The originating session is recorded for provenance only.

Everything here is pure or sqlite-only. The subprocess half (squeue, sacct)
lives in :mod:`hpca.slurm` and is handed in, so the whole module tests without
a cluster.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hpca.slurm import JobDetail, JobStatus, TERMINAL_STATES

KIND_LOG = "log"
KIND_JOB = "job"
KINDS = (KIND_LOG, KIND_JOB)

# Log states.
LOG_WRITING = "writing"
LOG_IDLE = "idle"
LOG_GONE = "gone"

# A job squeue no longer lists and sacct cannot account for either.
JOB_GONE = "GONE"

# How long after its last write a log still reads as "writing". Generous on
# purpose: plenty of tools flush per output chunk rather than per line, and a
# box that flickers between writing and idle every few seconds is worse than
# useless — it teaches the user to ignore it.
FRESH_SECONDS = 120

# The Enter peek: enough tail to carry a traceback's last line or a "Done.",
# short enough to stay a toast rather than a wall of text.
PEEK_CHARS = 300


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.isoformat()


def default_label(kind: str, target: str) -> str:
    """What to call a watch the user did not name."""
    if kind == KIND_LOG:
        return Path(target).name or target
    return f"job {target}"


@dataclass
class Watch:
    id: int
    kind: str
    target: str
    label: str = ""
    profile: str = ""
    session_id: str = ""
    created_at: str = ""
    state: str = ""
    # The two rendered halves of the box: ``head`` follows the state on the
    # first line, ``detail`` is the whole second line (jobs only — a log's
    # second line is its freshness, which has to be recomputed every repaint).
    head: str = ""
    detail: str = ""
    # When the watched thing itself last moved: a log's mtime, a job's last
    # state change. The panel counts up from this, so it must be the real
    # event time and not the time of the poll that noticed it.
    changed_at: str = ""
    checked_at: str = ""

    @property
    def title(self) -> str:
        return self.label or default_label(self.kind, self.target)


_COLUMNS = (
    "id",
    "profile",
    "session_id",
    "kind",
    "target",
    "label",
    "created_at",
    "state",
    "head",
    "detail",
    "changed_at",
    "checked_at",
)


def _to_watch(row: sqlite3.Row) -> Watch:
    return Watch(**{name: row[name] for name in _COLUMNS})


class WatchStore:
    """The ``watches`` table. Every method commits; see DbIO for who calls it."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        *,
        kind: str,
        target: str,
        label: str = "",
        profile: str = "",
        session_id: str = "",
    ) -> Watch:
        """Register a watch, or re-label the one that is already on this target.

        Registering the same log twice is the normal way to rename its box, not
        a mistake worth an error — and never a second box for one file.
        """
        if kind not in KINDS:
            raise ValueError(f"Unknown watch kind {kind!r}; expected one of {KINDS}")
        existing = self.find(kind=kind, target=target, profile=profile)
        if existing is not None:
            if label and label != existing.label:
                self._conn.execute(
                    "UPDATE watches SET label = ? WHERE id = ?", (label, existing.id)
                )
                self._conn.commit()
                existing.label = label
            return existing
        cursor = self._conn.execute(
            "INSERT INTO watches (profile, session_id, kind, target, label, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (profile, session_id, kind, target, label, _stamp(_now())),
        )
        self._conn.commit()
        watch = self.get(int(cursor.lastrowid or 0))
        assert watch is not None
        return watch

    def get(self, watch_id: int) -> Watch | None:
        row = self._conn.execute(
            "SELECT * FROM watches WHERE id = ?", (watch_id,)
        ).fetchone()
        return _to_watch(row) if row else None

    def find(self, *, kind: str, target: str, profile: str = "") -> Watch | None:
        row = self._conn.execute(
            "SELECT * FROM watches WHERE profile = ? AND kind = ? AND target = ?",
            (profile, kind, target),
        ).fetchone()
        return _to_watch(row) if row else None

    def list(self, *, profile: str | None = None) -> list[Watch]:
        """Every watch, oldest first.

        Insertion order, not freshness order: the panel is navigated with the
        arrow keys, and a list that reorders itself under the cursor every poll
        cannot be navigated at all.
        """
        sql = "SELECT * FROM watches"
        params: tuple = ()
        if profile is not None:
            sql += " WHERE profile = ?"
            params = (profile,)
        sql += " ORDER BY id"
        return [_to_watch(row) for row in self._conn.execute(sql, params)]

    def remove(self, watch_id: int) -> bool:
        cursor = self._conn.execute("DELETE FROM watches WHERE id = ?", (watch_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def update(
        self,
        watch_id: int,
        *,
        state: str,
        head: str = "",
        detail: str = "",
        changed_at: str | None = None,
        checked_at: str | None = None,
    ) -> None:
        """Fold one poll's result into a row.

        ``changed_at`` is left alone when None, so a poll that finds nothing
        new does not reset the clock the user is reading.
        """
        sets = ["state = ?", "head = ?", "detail = ?", "checked_at = ?"]
        params: list = [state, head, detail, checked_at or _stamp(_now())]
        if changed_at is not None:
            sets.append("changed_at = ?")
            params.append(changed_at)
        params.append(watch_id)
        self._conn.execute(
            f"UPDATE watches SET {', '.join(sets)} WHERE id = ?", params
        )
        self._conn.commit()


# --------------------------------------------------------------- formatting


def format_size(num_bytes: int) -> str:
    """Bytes as the panel shows them: three significant figures, one unit."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    value = float(num_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
    return f"{value:.0f} PB"


def format_age(seconds: float) -> str:
    """A duration in one or two units, never longer than six characters.

    Rounds down: "59s" only becomes "1m" once a minute has actually passed, so
    the number never claims more time than has elapsed.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        hours, rest = divmod(int(seconds), 3600)
        return f"{hours}h{rest // 60:02d}m"
    days, rest = divmod(int(seconds), 86400)
    return f"{days}d{rest // 3600:02d}h"


def age_seconds(stamp: str, now: datetime | None = None) -> float | None:
    """Seconds since an ISO stamp, or None when it is missing or unparseable."""
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return ((now or _now()) - moment).total_seconds()


def format_elapsed(seconds: int | None) -> str:
    """Seconds as Slurm writes them: ``[D-]HH:MM:SS``."""
    if seconds is None:
        return ""
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    body = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{body}" if days else body


# Live, waiting, finished. Plain enough to survive a terminal without a font
# for anything cleverer, distinct enough to read as a status column.
GLYPH_LIVE = "●"
GLYPH_IDLE = "○"
GLYPH_DONE = "✓"
GLYPH_DEAD = "✗"


def watch_glyph(watch: Watch) -> str:
    if watch.kind == KIND_LOG:
        return {
            LOG_WRITING: GLYPH_LIVE,
            LOG_IDLE: GLYPH_IDLE,
            LOG_GONE: GLYPH_DEAD,
        }.get(watch.state, GLYPH_IDLE)
    if watch.state == "RUNNING":
        return GLYPH_LIVE
    if watch.state == "COMPLETED":
        return GLYPH_DONE
    if watch.state in TERMINAL_STATES or watch.state == JOB_GONE:
        return GLYPH_DEAD
    return GLYPH_IDLE


def watch_class(watch: Watch) -> str:
    """CSS class for the box: colour carries the state, so a glance is enough."""
    if watch.kind == KIND_LOG:
        return {
            LOG_WRITING: "watch-live",
            LOG_IDLE: "watch-idle",
            LOG_GONE: "watch-dead",
        }.get(watch.state, "watch-idle")
    if watch.state == "RUNNING":
        return "watch-live"
    if watch.state == "COMPLETED":
        return "watch-done"
    if watch.state in TERMINAL_STATES or watch.state == JOB_GONE:
        return "watch-dead"
    return "watch-idle"


def _log_freshness(watch: Watch, now: datetime | None = None) -> str:
    if watch.state == LOG_GONE:
        return "no such file"
    age = age_seconds(watch.changed_at, now)
    if age is None:
        return "not polled yet"
    return f"last write {format_age(age)} ago"


def watch_lines(watch: Watch, *, now: datetime | None = None) -> list[str]:
    """The two body lines of a watch box.

    Kept to two so several watches fit a short terminal beside the process
    history — a monitor nobody can see all of monitors nothing.
    """
    first = f"{watch_glyph(watch)} {watch.state or 'not polled yet'}"
    if watch.head:
        first += f" · {watch.head}"
    if watch.kind == KIND_LOG:
        return [first, _log_freshness(watch, now)]
    return [first, watch.detail or f"id {watch.target}"]


# ------------------------------------------------------------------ polling


@dataclass
class WatchChange:
    """A watch whose state moved between two polls — what a toast is worth."""

    watch: Watch
    old_state: str
    new_state: str


def is_settled(watch: Watch) -> bool:
    """Whether polling this watch again could still tell us anything.

    A finished job cannot un-finish, so it stops costing an squeue call. A
    missing log is *not* settled: a job that has not created its log yet is
    the most common reason to be watching one at all.
    """
    if watch.kind != KIND_JOB:
        return False
    return watch.state in TERMINAL_STATES or watch.state == JOB_GONE


def log_fields(
    path: str | Path, *, now: datetime | None = None
) -> tuple[str, str, str]:
    """Stat one log: ``(state, head, changed_at)``.

    The mtime is the whole point, so it is read straight from the filesystem
    rather than inferred from size deltas between polls: a tool that rewrites
    in place keeps its size and still counts as alive.
    """
    now = now or _now()
    try:
        stat = Path(path).stat()
    except OSError:
        return LOG_GONE, "", ""
    written = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    age = (now - written).total_seconds()
    state = LOG_WRITING if age <= FRESH_SECONDS else LOG_IDLE
    return state, format_size(stat.st_size), _stamp(written)


def poll_log_watches(
    store: WatchStore, watches: list[Watch], *, now: datetime | None = None
) -> list[WatchChange]:
    """Stat every log watch and write the result back; returns state changes.

    Runs whole on the DB thread — the stat is the slow half on an NFS home,
    so keeping it next to the write is one thread hop instead of two.
    """
    now = now or _now()
    changes: list[WatchChange] = []
    for watch in watches:
        if watch.kind != KIND_LOG:
            continue
        state, head, changed_at = log_fields(watch.target, now=now)
        store.update(
            watch.id,
            state=state,
            head=head,
            changed_at=changed_at or None,
            checked_at=_stamp(now),
        )
        if state != watch.state:
            changes.append(WatchChange(watch, watch.state, state))
    return changes


def job_fields(detail: JobDetail) -> tuple[str, str, str]:
    """One squeue row as ``(state, head, detail)``.

    A queued job's second line is why it is queued — the answer to the only
    question a PENDING box provokes.
    """
    def joined(*parts: str) -> str:
        return " · ".join(p for p in parts if p)

    if detail.state == "RUNNING":
        left = f"{detail.time_left} left" if detail.time_left else ""
        return detail.state, detail.elapsed, joined(detail.nodes, left)
    if detail.state == "PENDING":
        queued = f"queued {detail.elapsed}" if detail.elapsed else ""
        return detail.state, "", joined(detail.reason, queued)
    return detail.state, "", joined(detail.nodes, detail.elapsed)


def final_job_fields(status: JobStatus | None) -> tuple[str, str, str]:
    """What to show once a job has left the queue.

    sacct is asked only for jobs squeue has dropped, so a finished box reads
    COMPLETED or FAILED rather than simply disappearing — the moment the user
    opened the panel for in the first place.
    """
    if status is None:
        return JOB_GONE, "", "left the queue; sacct has no record"
    head = format_elapsed(status.elapsed_s)
    parts: list[str] = []
    if status.exit_code is not None:
        parts.append(f"exit {status.exit_code}")
    if status.signal:
        parts.append(f"signal {status.signal}")
    if status.max_rss_bytes:
        parts.append(f"peak {format_size(status.max_rss_bytes)}")
    return status.state, head, " · ".join(parts)


def apply_job_details(
    store: WatchStore,
    watches: list[Watch],
    details: dict[str, JobDetail],
    finished: dict[str, JobStatus],
    *,
    now: datetime | None = None,
) -> list[WatchChange]:
    """Fold one squeue (plus sacct fallback) sweep into the store.

    Split from the fetch so the TUI can await the subprocesses on the loop and
    run the sqlite half on its DB thread.
    """
    now = now or _now()
    changes: list[WatchChange] = []
    for watch in watches:
        if watch.kind != KIND_JOB:
            continue
        detail = details.get(watch.target)
        if detail is not None:
            state, head, line = job_fields(detail)
        else:
            state, head, line = final_job_fields(finished.get(watch.target))
        changed = state != watch.state
        store.update(
            watch.id,
            state=state,
            head=head,
            detail=line,
            changed_at=_stamp(now) if changed else None,
            checked_at=_stamp(now),
        )
        if changed:
            changes.append(WatchChange(watch, watch.state, state))
    return changes


# --------------------------------------------------------------------- peek


def peek(path: str | Path, chars: int = PEEK_CHARS) -> str:
    """The tail of a file, for the flash-up notice bound to Enter.

    Reads only the last few KB: these are job logs, and one of them being a
    gigabyte of progress bars is entirely normal.
    """
    path = Path(path)
    window = max(chars * 8, 4096)
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - window))
            raw = handle.read()
    except OSError as e:
        return f"(could not read {path}: {e})"
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return "(empty)"
    tail = text[-chars:]
    return tail if len(text) <= chars else f"…{tail}"
