"""Registered watches: what the user asked hpca to keep an eye on (§3.3).

The right column used to show only what hpca itself started — its own
subprocesses and its own sbatch submissions — which is a thin slice of what
actually runs on a cluster, and a slice already visible in the chat log. The
work that matters is usually somebody else's: an sbatch script submitted by
hand, a snakemake run spawning one tool after another, a log some long-running
program appends to. Checking on those means ``squeue``, then ssh to the node,
then ``tail``. A watch pins one of them to the panel instead.

Two kinds, both cheap enough to poll on a timer:

* ``log`` — a file. The box reports its size and its mtime: "last write 4s
  ago", legible at a glance without leaving the TUI. It reports nothing beyond
  that, deliberately. The mtime used to be turned into a verdict — "writing"
  under a couple of minutes, "idle" over — but that reads a fact about a file
  as a claim about the work behind it, and the two are not the same: a job can
  be entirely healthy and silent for an hour between checkpoints. The number is
  shown; the conclusion is the reader's.
* ``job`` — a Slurm job id, refreshed from ``squeue``. When it drops out of
  the queue ``sacct`` supplies the final state, so the box settles on
  COMPLETED or FAILED instead of quietly vanishing.

A watch belongs to the *session* that registered it. It was originally scoped
to the profile, on the reasoning that a watch describes the machine rather than
the conversation — but in use that is backwards: sessions on one profile are
the normal case, so every session showed every other session's boxes, and the
right column stopped describing the conversation the user was reading.

Polling is deliberately *not* scoped the same way. State is refreshed for every
watch in the store, so a session returned to shows a current clock rather than
one frozen at the moment the user switched away. Scoping decides what is shown;
it must not decide what stays true.

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

# Log states. Only two, and both are observed rather than inferred: the file
# is there, or it is not.
#
# There used to be a third — the file's mtime was compared against a freshness
# window and the box read "writing" or "idle" accordingly. That was a guess
# dressed as a status. A job can be very much alive and not writing: buffered
# output, a long compute phase between log lines, a rank that only reports at
# checkpoints. Whether the work is alive cannot be read off the age of its last
# write, so the box no longer claims to know. It shows when the file was last
# written and lets the reader draw the conclusion, which is the one thing the
# mtime actually supports.
LOG_PRESENT = "present"
LOG_GONE = "gone"

# A job squeue no longer lists and sacct cannot account for either.
JOB_GONE = "GONE"

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
    # Where the box sits in the column. Assigned on insert and only ever
    # rewritten by ``WatchStore.move``; see there for why it is dense.
    position: int = 0

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
    "position",
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
        existing = self.find(kind=kind, target=target, session_id=session_id)
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
            "created_at, position) VALUES (?, ?, ?, ?, ?, ?, "
            # One past the highest anywhere, so a new box lands at the bottom
            # of its own column. Store-wide rather than per-session because
            # ``move`` renumbers a session densely from 1: a per-session
            # maximum would then hand out a number that session is already
            # using, and the new box would land in the middle of the list.
            "(SELECT COALESCE(MAX(position), 0) + 1 FROM watches))",
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

    def find(self, *, kind: str, target: str, session_id: str = "") -> Watch | None:
        row = self._conn.execute(
            "SELECT * FROM watches WHERE session_id = ? AND kind = ? AND target = ?",
            (session_id, kind, target),
        ).fetchone()
        return _to_watch(row) if row else None

    def list(
        self, *, session_id: str | None = None, profile: str | None = None
    ) -> list[Watch]:
        """Every watch in column order; ``session_id`` narrows to one session's.

        The order is the user's, falling back to insertion — never freshness.
        The panel is navigated with the arrow keys, and a list that reorders
        itself under the cursor every poll cannot be navigated at all, so
        nothing a poll learns is allowed to move a box. ``id`` breaks ties so
        rows that predate ``position`` still come out in the order they were
        registered.

        Passing neither returns the whole store, which is what the pollers
        want — see the module docstring on why refreshing is not scoped the way
        displaying is. ``profile`` is still accepted because a profile-wide
        sweep is the right question when a profile is being deleted.
        """
        sql = "SELECT * FROM watches"
        clauses: list[str] = []
        params: list = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if profile is not None:
            clauses.append("profile = ?")
            params.append(profile)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY position, id"
        return [_to_watch(row) for row in self._conn.execute(sql, tuple(params))]

    def move(self, watch_id: int, delta: int) -> bool:
        """Shift a box one step up (``-1``) or down (``+1``) in its column.

        Swapping with the neighbour rather than assigning an absolute slot,
        because that is the whole gesture the user has: alt+↑ pressed twice
        should walk a box past two others, and there is no way to say "third
        from the top".

        Scoped to the moving watch's own session, since that is the only list
        anyone sees — a store-wide swap could put it next to a box belonging to
        a conversation the user is not even looking at.

        Returns whether anything moved: at the top or the bottom there is no
        neighbour to trade with, and that is an ordinary outcome of holding the
        key down, not a failure worth a message.
        """
        watch = self.get(watch_id)
        if watch is None:
            return False
        column = self.list(session_id=watch.session_id)
        index = next(
            (i for i, other in enumerate(column) if other.id == watch_id), None
        )
        if index is None:
            return False
        target = index + delta
        if not 0 <= target < len(column):
            return False
        column[index], column[target] = column[target], column[index]
        # Renumber the whole column rather than swapping the two positions.
        # Rows that predate the column all share position 0, and swapping two
        # zeroes changes nothing at all; numbering from 1 also keeps 0 meaning
        # "never assigned", which is what the backfill in db.py keys on.
        self._conn.executemany(
            "UPDATE watches SET position = ? WHERE id = ?",
            [(position, other.id) for position, other in enumerate(column, 1)],
        )
        self._conn.commit()
        return True

    def remove(self, watch_id: int) -> bool:
        cursor = self._conn.execute("DELETE FROM watches WHERE id = ?", (watch_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def forget_session(self, session_id: str) -> int:
        """Drop a deleted session's watches; returns how many.

        Necessary because watches are session-scoped: without this a deleted
        conversation's boxes become invisible — no session will ever match
        them again — while the pollers go on stat-ing their files and asking
        squeue about their jobs every few seconds, forever.
        """
        cursor = self._conn.execute(
            "DELETE FROM watches WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()
        return cursor.rowcount

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
        # Never GLYPH_LIVE: a log file existing says nothing about whether
        # anything is still writing to it.
        return GLYPH_DEAD if watch.state == LOG_GONE else GLYPH_IDLE
    if watch.state == "RUNNING":
        return GLYPH_LIVE
    if watch.state == "COMPLETED":
        return GLYPH_DONE
    if watch.state in TERMINAL_STATES or watch.state == JOB_GONE:
        return GLYPH_DEAD
    return GLYPH_IDLE


def watch_class(watch: Watch) -> str:
    """CSS class for the box: colour carries the state, so a glance is enough.

    A log is never "watch-live" — see the note on the log states. Colouring a
    box green because the file was touched recently is the same claim in
    another form, and it is the claim that cannot be supported.
    """
    if watch.kind == KIND_LOG:
        return "watch-dead" if watch.state == LOG_GONE else "watch-idle"
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
    if watch.kind == KIND_LOG:
        # No state word: "present" is not news, and the words it replaced
        # ("writing", "idle") were a guess about the job, not a fact about the
        # file. What the box has to say is the size and the last write.
        first = f"{watch_glyph(watch)} {watch.head}" if watch.head else (
            f"{watch_glyph(watch)} no such file" if watch.state == LOG_GONE
            else f"{watch_glyph(watch)} not polled yet"
        )
        return [first, _log_freshness(watch, now)]
    first = f"{watch_glyph(watch)} {watch.state or 'not polled yet'}"
    if watch.head:
        first += f" · {watch.head}"
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
    return LOG_PRESENT, format_size(stat.st_size), _stamp(written)


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
