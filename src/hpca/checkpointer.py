"""Log-structured checkpoints: store the conversation once, not once per step.

LangGraph writes a checkpoint per super-step, and `AsyncSqliteSaver` puts the
whole `channel_values` inline in that row — so a thread's storage cost is the
size of the conversation times the number of steps. On a measured thread that
was 7.73 MB of rows describing 150 KB of conversation.

`CheckpointLogSaver` keeps the same contract and splits the storage in two:

* `channel_items` — every item ever appended to a *log channel*
  (`LOG_CHANNELS`), stored exactly once, keyed by its index.
* `channel_blobs` — everything else, stored whole but keyed by the channel's
  LangGraph version, so a channel nobody touched shares its row.
* `checkpoints` — a manifest naming, per channel, either how many items of the
  log are live (`{"log": n, "chain": ...}`) or which blob version to read
  (`{"blob": v}`). Roughly a kilobyte whatever the conversation weighs.

The append-only claim is **verified on every put, never assumed** (§3.3 of
`specs/specs-checkpoint-log.md`). `_plan_channel` starts from "write it whole"
and only returns a tail-append when the stored prefix is proven intact; every
other outcome — a shrinking list, a cold cache after a restart, a reducer that
stopped being append-only, an exception anywhere in the check — falls through
to the snapshot it started with. A wrong check costs a bigger row; it can
never produce a wrong restore.

The one property this deliberately gives up: **the log is the current
conversation, not its history.** A rewind rewrites the items it rolled past, so
a checkpoint older than the rewind can no longer be restored. Nothing in HPCA
ever reads a historical checkpoint (every read is `aget_state(config)` with no
`checkpoint_id`), and "the database reflects the current state" is the
requirement this exists to satisfy — but it is a property, not an accident.

One assumption is worth naming because the fast path rests on it: an item that
has been appended to a log channel is never mutated in place. The reducers
(`agent.graph._append`, `_append_messages`) only ever build new lists out of
the same item objects, so a prefix can be verified by object identity; when
identity does not hold the rolling hash chain is recomputed instead, and when
*that* does not hold the channel is snapshotted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

# Which channels are stored as an append log. A property of the *saver*,
# declared once — a saver that inferred this from reducer identity would be
# reading the graph's internals to decide how to store its data, and would
# change behaviour silently the day someone edits a reducer. Declared, plus
# verified per put, plus a fallback that is always correct.
#
# A name in here that no longer exists in the state costs nothing (the channel
# simply never appears in a checkpoint), and a state channel not named in here
# is stored as a blob. Neither is an error.
LOG_CHANNELS: tuple[str, ...] = ("messages", "thinking", "calls")

# How many checkpoint rows to keep per thread. Superseded rows are unreachable
# through any HPCA code path, they are what pins `channel_blobs` versions
# alive, and a bounded count is what makes the file's size a function of the
# conversation rather than of how long it has been open. Comfortably more than
# the two LangGraph needs to resume an interrupt.
KEEP_CHECKPOINTS = 32

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    meta_type TEXT,
    meta BLOB,
    manifest TEXT,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS channel_items (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    idx INTEGER NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, idx)
);
CREATE TABLE IF NOT EXISTS channel_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
"""


# --------------------------------------------------------------- the chain

CHAIN_ZERO = hashlib.blake2b(b"", digest_size=16).digest()


def chain_step(chain: bytes, serialized: tuple[str, bytes]) -> bytes:
    """`h_i = H(h_{i-1} ‖ serialized(item_i))`."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(chain)
    digest.update(serialized[0].encode("utf-8"))
    digest.update(b"\x00")
    digest.update(serialized[1])
    return digest.digest()


def chain_over(serialized: Iterable[tuple[str, bytes]]) -> bytes:
    chain = CHAIN_ZERO
    for one in serialized:
        chain = chain_step(chain, one)
    return chain


@dataclass
class _LogState:
    """What this process knows about one `(thread, ns, channel)` log.

    `items` is held for the identity fast path — the same objects the graph's
    channel holds, so keeping them costs references, not copies.
    """

    length: int
    chain: bytes
    items: list[Any] = field(default_factory=list)


@dataclass
class _ChannelPlan:
    """How one channel of one put is going to be written.

    Constructed as a snapshot; `_plan_channel` only ever narrows it to an
    append after the prefix is proven.
    """

    kind: str  # "log" or "blob"
    manifest: dict[str, Any]
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    truncate_from: int | None = None  # delete log rows at or above this index
    state: _LogState | None = None  # cache entry to install once committed


# ------------------------------------------------------------- the migration


@dataclass
class MigrationResult:
    """What `migrate_inline_format` did, in a shape a caller can put on the
    wire. Never raises: a migration that fails must not stop the app."""

    ran: bool = False
    threads: int = 0
    seconds: float = 0.0
    bytes_before: int = 0
    bytes_after: int = 0
    error: str | None = None

    def notices(self) -> list[str]:
        if self.error is not None:
            return [
                "converting checkpoints.db to the checkpoint log failed "
                f"({self.error}); conversations before this run may not be "
                "restored. The old rows are kept in `checkpoints_inline` and "
                "the next start will try again."
            ]
        if not self.ran:
            return []
        return [
            f"checkpoints.db converted to the checkpoint log — {self.threads} "
            f"thread(s), {_mb(self.bytes_before)} -> {_mb(self.bytes_after)}, "
            f"{self.seconds:.1f}s. Only the latest state of each conversation "
            "was carried over."
        ]


def _mb(n: int) -> str:
    """The size, in whichever unit does not read as zero — this is the same
    line for a 20 KB dev database and a 1.5 GB one."""
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def migrate_inline_format(
    path: str | Path, *, serde: SerializerProtocol | None = None
) -> MigrationResult:
    """Convert a `checkpoints.db` written by `AsyncSqliteSaver` in place.

    The old format is a `checkpoints` table carrying a `checkpoint BLOB`
    column. For each thread the latest row is read, written out as a log plus
    a manifest, and the old table dropped — which collapses the 160 MB / 1.5 GB
    files this design exists to prevent, on the spot.

    Synchronous on purpose: it runs off the event loop, before the graph is
    built, and on a large file it is a minute of sqlite. It never raises —
    every failure is reported in the result so the caller can say so and carry
    on with an app that still starts.

    Resumable across a failed attempt: the rename to `checkpoints_inline`
    commits on its own, so a crash anywhere after it leaves a file the next
    start picks straight back up.
    """
    result = MigrationResult()
    path = Path(path)
    if not path.exists():
        return result
    started = time.monotonic()
    serde = serde or JsonPlusSerializer()
    conn: sqlite3.Connection | None = None
    try:
        result.bytes_before = path.stat().st_size
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode=WAL")
        tables = _tables(conn)
        if (
            "checkpoints" in tables
            and "checkpoint" in _columns(conn, "checkpoints")
            and "manifest" not in _columns(conn, "checkpoints")
        ):
            conn.execute("ALTER TABLE checkpoints RENAME TO checkpoints_inline")
            conn.commit()
            tables = _tables(conn)
        if "checkpoints_inline" not in tables:
            return result
        result.ran = True
        conn.executescript(SCHEMA)
        conn.commit()

        threads = conn.execute(
            "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints_inline"
        ).fetchall()
        for thread_id, checkpoint_ns in threads:
            checkpoint_ns = checkpoint_ns or ""
            already = conn.execute(
                "SELECT 1 FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                "LIMIT 1",
                (thread_id, checkpoint_ns),
            ).fetchone()
            if already:
                continue
            row = conn.execute(
                "SELECT checkpoint_id, type, checkpoint, metadata "
                "FROM checkpoints_inline WHERE thread_id=? AND checkpoint_ns=? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id, checkpoint_ns),
            ).fetchone()
            if row is None:
                continue
            checkpoint_id, type_, blob, metadata = row
            # One unreadable thread must not cost the user every other one, and
            # "unreadable" is wider than a serializer that raises: a corrupt
            # blob can also decode to something that is not a checkpoint at
            # all. So the whole conversion of one thread is the unit that is
            # allowed to fail.
            try:
                checkpoint = serde.loads_typed((type_, blob))
                if not isinstance(checkpoint, dict):
                    raise TypeError(f"not a checkpoint: {type(checkpoint).__name__}")
                _write_migrated(
                    conn,
                    serde,
                    thread_id=str(thread_id),
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    checkpoint=checkpoint,
                    metadata=metadata,
                )
            except Exception:
                logger.exception(
                    "checkpoint log migration: thread %s is unreadable, skipped",
                    thread_id,
                )
                continue
            result.threads += 1
        # Writes belonging to checkpoints that no longer exist.
        conn.execute(
            "DELETE FROM writes WHERE NOT EXISTS ("
            "  SELECT 1 FROM checkpoints c WHERE c.thread_id = writes.thread_id"
            "   AND c.checkpoint_ns = writes.checkpoint_ns"
            "   AND c.checkpoint_id = writes.checkpoint_id)"
        )
        conn.commit()
        conn.execute("DROP TABLE checkpoints_inline")
        conn.commit()
    except Exception as exc:  # never take the app down with us
        logger.exception("checkpoint log migration failed")
        result.error = f"{type(exc).__name__}: {exc}"
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
    if conn is not None:
        # Reclaiming the space is the point of doing this at startup at all,
        # but it is also the step most likely to run out of room — a failure
        # here leaves a correct, merely large, database.
        if result.ran and result.error is None:
            try:
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                conn.execute("VACUUM")
            except Exception:
                logger.exception("checkpoint log migration: VACUUM failed")
        try:
            conn.close()
        except Exception:
            pass
    result.seconds = time.monotonic() - started
    try:
        result.bytes_after = path.stat().st_size
    except OSError:
        pass
    return result


def _write_migrated(
    conn: sqlite3.Connection,
    serde: SerializerProtocol,
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    checkpoint: Checkpoint,
    metadata: bytes | None,
) -> None:
    """One inline checkpoint, written out as an initial log plus manifest."""
    values = dict(checkpoint.get("channel_values") or {})
    versions = checkpoint.get("channel_versions") or {}
    manifest: dict[str, Any] = {}
    for channel, value in values.items():
        if channel in LOG_CHANNELS and isinstance(value, list):
            serialized = [serde.dumps_typed(item) for item in value]
            conn.executemany(
                "INSERT OR REPLACE INTO channel_items "
                "(thread_id, checkpoint_ns, channel, idx, type, value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (thread_id, checkpoint_ns, channel, idx, one[0], one[1])
                    for idx, one in enumerate(serialized)
                ],
            )
            manifest[channel] = {
                "log": len(serialized),
                "chain": chain_over(serialized).hex(),
            }
        else:
            version = _blob_version(versions, channel, checkpoint_id)
            type_, blob = serde.dumps_typed(value)
            conn.execute(
                "INSERT OR REPLACE INTO channel_blobs "
                "(thread_id, checkpoint_ns, channel, version, type, value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_ns, channel, version, type_, blob),
            )
            manifest[channel] = {"blob": version}
    meta_type, meta = serde.dumps_typed(_meta_of(checkpoint))
    conn.execute(
        "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, "
        "checkpoint_id, parent_checkpoint_id, meta_type, meta, manifest, "
        "metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            # The parent did not survive: this row is the thread's new head,
            # and nothing should walk back off it.
            None,
            meta_type,
            meta,
            json.dumps(manifest),
            metadata,
        ),
    )


def _meta_of(checkpoint: Checkpoint) -> dict[str, Any]:
    """The checkpoint minus its channel values — v, ts, id, versions, and
    whatever else a future LangGraph puts there."""
    return {k: v for k, v in checkpoint.items() if k != "channel_values"}


def _blob_version(versions: ChannelVersions, channel: str, checkpoint_id: str) -> str:
    """The key a blob channel's row is stored under.

    LangGraph bumps a channel's version whenever it is written, so
    `(channel, version)` names one value for all time — which is what lets a
    checkpoint that did not touch a channel point at the row an earlier one
    wrote. A channel with no version (which should not happen) falls back to a
    key unique to this checkpoint, which is merely wasteful.
    """
    version = versions.get(channel)
    return str(version) if version is not None else f"cp:{checkpoint_id}"


# ------------------------------------------------------------------ the saver


class CheckpointLogSaver(BaseCheckpointSaver[str]):
    """`BaseCheckpointSaver` over sqlite, storing list channels as a log.

    Async-only, like every way HPCA drives the graph. The sync methods the
    base class leaves as `NotImplementedError` stay that way — that is a fact
    about how this app uses langgraph, not a promise about langgraph.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: SerializerProtocol | None = None,
        log_channels: Sequence[str] = LOG_CHANNELS,
        keep_checkpoints: int = KEEP_CHECKPOINTS,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False
        self.log_channels = tuple(log_channels)
        self.keep_checkpoints = keep_checkpoints
        # (thread_id, ns, channel) -> what we last wrote for it.
        self._logs: dict[tuple[str, str, str], _LogState] = {}
        # (thread_id, ns, channel, version) already on disk in this process.
        self._blobs: set[tuple[str, str, str, str]] = set()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str, **kwargs: Any
    ) -> AsyncIterator["CheckpointLogSaver"]:
        async with aiosqlite.connect(conn_string) as conn:
            yield cls(conn, **kwargs)

    # ------------------------------------------------------------ lifecycle

    async def setup(self) -> None:
        async with self.lock:
            if self.is_setup:
                return
            await _ensure_connected(self.conn)
            # Before the tables exist, so it takes on an empty file; on a file
            # that already has pages this is a no-op until the next VACUUM.
            async with self.conn.executescript(
                "PRAGMA journal_mode=WAL;\nPRAGMA auto_vacuum=INCREMENTAL;\n"
                + await self._schema_for(self.conn)
            ):
                await self.conn.commit()
            self.is_setup = True

    async def _schema_for(self, conn: aiosqlite.Connection) -> str:
        """`SCHEMA`, unless an un-migrated inline `checkpoints` table is in the
        way — in which case move it aside first.

        This is the last line of the migration's defence: if
        `migrate_inline_format` failed or was never called, `CREATE TABLE IF
        NOT EXISTS checkpoints` would quietly leave the old shape in place and
        the first put would fail on a missing column. Renaming loses no data
        and leaves the file in exactly the state the migration knows how to
        resume from.
        """
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ) as cur:
            exists = await cur.fetchone()
        if not exists:
            return SCHEMA
        async with conn.execute("PRAGMA table_info(checkpoints)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        if "manifest" in columns:
            return SCHEMA
        logger.warning(
            "checkpoints.db is still in the inline format; moving it aside to "
            "`checkpoints_inline` so the app can start"
        )
        return "ALTER TABLE checkpoints RENAME TO checkpoints_inline;\n" + SCHEMA

    # ------------------------------------------------------------ read path

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await self.setup()
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") or ""
        async with self.lock, self.conn.cursor() as cur:
            if checkpoint_id := get_checkpoint_id(config):
                await cur.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, meta_type, meta, "
                    "manifest, metadata FROM checkpoints WHERE thread_id=? AND "
                    "checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                await cur.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, meta_type, meta, "
                    "manifest, metadata FROM checkpoints WHERE thread_id=? AND "
                    "checkpoint_ns=? ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                )
            row = await cur.fetchone()
            if row is None:
                return None
            return await self._tuple_from_row(cur, thread_id, checkpoint_ns, row)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        from langgraph.checkpoint.sqlite.utils import search_where

        where, params = search_where(config, filter, before)
        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
            f"meta_type, meta, manifest, metadata FROM checkpoints {where} "
            "ORDER BY checkpoint_id DESC"
        )
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        async with self.lock, self.conn.cursor() as cur:
            await cur.execute(query, params)
            rows = await cur.fetchall()
            for thread_id, checkpoint_ns, *rest in rows:
                yield await self._tuple_from_row(
                    cur, str(thread_id), checkpoint_ns or "", tuple(rest)
                )

    async def _tuple_from_row(
        self,
        cur: aiosqlite.Cursor,
        thread_id: str,
        checkpoint_ns: str,
        row: tuple,
    ) -> CheckpointTuple:
        checkpoint_id, parent_checkpoint_id, meta_type, meta, manifest, metadata = row
        checkpoint: Checkpoint = self.serde.loads_typed((meta_type, meta))
        checkpoint["channel_values"] = await self._values_for(
            cur, thread_id, checkpoint_ns, json.loads(manifest or "{}")
        )
        await cur.execute(
            "SELECT task_id, channel, type, value FROM writes WHERE thread_id=? "
            "AND checkpoint_ns=? AND checkpoint_id=? ORDER BY task_id, idx",
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        pending = [
            (task_id, channel, self.serde.loads_typed((type_, value)))
            for task_id, channel, type_, value in await cur.fetchall()
        ]
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint,
            metadata=json.loads(metadata) if metadata is not None else {},
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending,
        )

    async def _values_for(
        self,
        cur: aiosqlite.Cursor,
        thread_id: str,
        checkpoint_ns: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for channel, entry in manifest.items():
            if "log" in entry:
                length = entry["log"]
                await cur.execute(
                    "SELECT type, value FROM channel_items WHERE thread_id=? AND "
                    "checkpoint_ns=? AND channel=? AND idx < ? ORDER BY idx",
                    (thread_id, checkpoint_ns, channel, length),
                )
                rows = await cur.fetchall()
                if len(rows) != length:
                    # Only reachable by asking for a checkpoint older than a
                    # rewind, which nothing in HPCA does (§3.3). Say so rather
                    # than hand back a silently short conversation.
                    logger.warning(
                        "checkpoint log: %s/%s wanted %d items of %r but %d "
                        "survive; an older checkpoint was rolled past",
                        thread_id,
                        checkpoint_ns,
                        length,
                        channel,
                        len(rows),
                    )
                values[channel] = [
                    self.serde.loads_typed((type_, value)) for type_, value in rows
                ]
            else:
                await cur.execute(
                    "SELECT type, value FROM channel_blobs WHERE thread_id=? AND "
                    "checkpoint_ns=? AND channel=? AND version=?",
                    (thread_id, checkpoint_ns, channel, str(entry["blob"])),
                )
                blob = await cur.fetchone()
                if blob is None:
                    logger.warning(
                        "checkpoint log: %s/%s has no blob for %r at version %r",
                        thread_id,
                        checkpoint_ns,
                        channel,
                        entry["blob"],
                    )
                    continue
                values[channel] = self.serde.loads_typed((blob[0], blob[1]))
        return values

    # ----------------------------------------------------------- write path

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await self.setup()
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") or ""
        checkpoint_id = checkpoint["id"]
        values = checkpoint.get("channel_values") or {}
        versions = checkpoint.get("channel_versions") or {}

        plans: dict[str, _ChannelPlan] = {}
        for channel, value in values.items():
            plans[channel] = self._plan_channel(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                channel=channel,
                value=value,
                versions=versions,
                checkpoint_id=checkpoint_id,
            )
        manifest = {channel: plan.manifest for channel, plan in plans.items()}
        meta_type, meta = self.serde.dumps_typed(_meta_of(checkpoint))
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")

        async with self.lock, self.conn.cursor() as cur:
            for channel, plan in plans.items():
                if plan.kind == "log":
                    if plan.truncate_from is not None:
                        await cur.execute(
                            "DELETE FROM channel_items WHERE thread_id=? AND "
                            "checkpoint_ns=? AND channel=? AND idx >= ?",
                            (
                                thread_id,
                                checkpoint_ns,
                                channel,
                                plan.truncate_from,
                            ),
                        )
                    if plan.rows:
                        await cur.executemany(
                            "INSERT OR REPLACE INTO channel_items (thread_id, "
                            "checkpoint_ns, channel, idx, type, value) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            plan.rows,
                        )
                elif plan.rows:
                    await cur.executemany(
                        "INSERT OR REPLACE INTO channel_blobs (thread_id, "
                        "checkpoint_ns, channel, version, type, value) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        plan.rows,
                    )
            await cur.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, "
                "checkpoint_id, parent_checkpoint_id, meta_type, meta, manifest, "
                "metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    config["configurable"].get("checkpoint_id"),
                    meta_type,
                    meta,
                    json.dumps(manifest),
                    serialized_metadata,
                ),
            )
            await self._prune(cur, thread_id, checkpoint_ns)
            await self.conn.commit()

        # Only once the rows are on disk: a cache that ran ahead of a failed
        # commit would let the next put append onto a prefix that is not there.
        for channel, plan in plans.items():
            key = (thread_id, checkpoint_ns, channel)
            if plan.kind == "log" and plan.state is not None:
                self._logs[key] = plan.state
            elif plan.kind == "blob":
                self._blobs.add((*key, str(plan.manifest["blob"])))
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def _plan_channel(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        channel: str,
        value: Any,
        versions: ChannelVersions,
        checkpoint_id: str,
    ) -> _ChannelPlan:
        """Decide how one channel is written, starting from "write it whole".

        The snapshot is the default and every path that cannot *prove* an
        append returns it. `_verified_append` is the only way out, and it is
        wrapped so that a serializer that raises, a channel whose items stopped
        being comparable, or anything else unforeseen produces a larger row
        rather than a wrong one.
        """
        key = (thread_id, checkpoint_ns, channel)
        if channel in self.log_channels and isinstance(value, list):
            try:
                append = self._verified_append(key, value)
            except Exception:
                logger.exception(
                    "checkpoint log: verifying %r failed; writing it whole", channel
                )
                append = None
            return append if append is not None else self._snapshot_log(key, value)
        # Not a log channel — or a log channel that is not a list, which is
        # what a parallel branch dropping `thinking` from the state, or a
        # reducer that stopped producing lists, looks like from here.
        version = _blob_version(versions, channel, checkpoint_id)
        plan = _ChannelPlan(kind="blob", manifest={"blob": version})
        if (*key, version) not in self._blobs:
            type_, blob = self.serde.dumps_typed(value)
            plan.rows = [
                (thread_id, checkpoint_ns, channel, version, type_, blob)
            ]
        return plan

    def _verified_append(
        self, key: tuple[str, str, str], value: list
    ) -> _ChannelPlan | None:
        """A tail-insert plan, or `None` when the stored prefix is not proven.

        `None` for: nothing known about this log (a cold cache after a restart
        is exactly this), a list that shrank, or a prefix that does not hash to
        what was written.
        """
        state = self._logs.get(key)
        if state is None or len(value) < state.length:
            return None
        head = value[: state.length]
        if not self._prefix_intact(state, head):
            return None
        tail = [self.serde.dumps_typed(item) for item in value[state.length :]]
        chain = state.chain
        rows = []
        for offset, one in enumerate(tail):
            chain = chain_step(chain, one)
            rows.append((*key, state.length + offset, one[0], one[1]))
        return _ChannelPlan(
            kind="log",
            manifest={"log": len(value), "chain": chain.hex()},
            rows=rows,
            state=_LogState(length=len(value), chain=chain, items=list(value)),
        )

    def _prefix_intact(self, state: _LogState, head: list) -> bool:
        """Is `head` what we last wrote for this log?

        Identity first: the reducers build `left + right`, so an ordinary step
        hands back the very objects already stored and the check is a walk of
        pointers. When identity does not hold — items rebuilt by a
        deserialization, a fork, a reducer that copies — the rolling chain is
        recomputed over the prefix and compared, which is slower but decides
        the same question on content.
        """
        if len(state.items) == state.length and all(
            a is b for a, b in zip(state.items, head)
        ):
            return True
        return chain_over(self.serde.dumps_typed(item) for item in head) == state.chain

    def _snapshot_log(self, key: tuple[str, str, str], value: list) -> _ChannelPlan:
        """Write the whole log out again from index 0, and drop what is above
        it. The re-seed that lets the next put go back to tailing."""
        serialized = [self.serde.dumps_typed(item) for item in value]
        rows = [
            (*key, idx, one[0], one[1]) for idx, one in enumerate(serialized)
        ]
        chain = chain_over(serialized)
        return _ChannelPlan(
            kind="log",
            manifest={"log": len(value), "chain": chain.hex()},
            rows=rows,
            truncate_from=len(value),
            state=_LogState(length=len(value), chain=chain, items=list(value)),
        )

    async def _prune(
        self, cur: aiosqlite.Cursor, thread_id: str, checkpoint_ns: str
    ) -> None:
        """Keep the last `keep_checkpoints` manifests of this thread, and the
        `channel_blobs` versions they still name.

        `channel_items` is never pruned by age: it *is* the current
        conversation.
        """
        if self.keep_checkpoints <= 0:
            return
        await cur.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? AND "
            "checkpoint_ns=? ORDER BY checkpoint_id DESC LIMIT 1 OFFSET ?",
            (thread_id, checkpoint_ns, self.keep_checkpoints),
        )
        row = await cur.fetchone()
        if row is None:
            return
        cutoff = row[0]
        await cur.execute(
            "DELETE FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND "
            "checkpoint_id <= ?",
            (thread_id, checkpoint_ns, cutoff),
        )
        await cur.execute(
            "DELETE FROM writes WHERE thread_id=? AND checkpoint_ns=? AND "
            "checkpoint_id <= ?",
            (thread_id, checkpoint_ns, cutoff),
        )
        await cur.execute(
            "SELECT manifest FROM checkpoints WHERE thread_id=? AND checkpoint_ns=?",
            (thread_id, checkpoint_ns),
        )
        live: set[tuple[str, str]] = set()
        for (manifest,) in await cur.fetchall():
            for channel, entry in json.loads(manifest or "{}").items():
                if "blob" in entry:
                    live.add((channel, str(entry["blob"])))
        await cur.execute(
            "SELECT channel, version FROM channel_blobs WHERE thread_id=? AND "
            "checkpoint_ns=?",
            (thread_id, checkpoint_ns),
        )
        dead = [
            (channel, version)
            for channel, version in await cur.fetchall()
            if (channel, str(version)) not in live
        ]
        if dead:
            await cur.executemany(
                "DELETE FROM channel_blobs WHERE thread_id=? AND checkpoint_ns=? "
                "AND channel=? AND version=?",
                [(thread_id, checkpoint_ns, c, v) for c, v in dead],
            )
            for channel, version in dead:
                self._blobs.discard(
                    (thread_id, checkpoint_ns, channel, str(version))
                )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self.setup()
        query = (
            "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id,"
            " task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else "INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, "
            "checkpoint_id, task_id, idx, channel, type, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        async with self.lock, self.conn.cursor() as cur:
            await cur.executemany(
                query,
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"].get("checkpoint_ns", "") or ""),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )
            await self.conn.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        await self.setup()
        thread_id = str(thread_id)
        async with self.lock, self.conn.cursor() as cur:
            for table in ("checkpoints", "writes", "channel_items", "channel_blobs"):
                await cur.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
                )
            await self.conn.commit()
        for key in [k for k in self._logs if k[0] == thread_id]:
            del self._logs[key]
        self._blobs = {b for b in self._blobs if b[0] != thread_id}

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        """The same scheme `AsyncSqliteSaver` used, on purpose.

        The base class's default is an integer, which is tidier — and would
        mean rewriting every `channel_versions` and `versions_seen` entry of
        every migrated thread, since LangGraph compares versions with `<` and a
        thread holding both shapes raises. Keeping the string scheme makes the
        migration a pure re-shaping of storage.
        """
        import random

        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"


async def _ensure_connected(conn: aiosqlite.Connection) -> None:
    started: Callable[[aiosqlite.Connection], bool] | None = getattr(
        aiosqlite.Connection, "is_alive", None
    )
    if callable(started):
        if not conn.is_alive():  # type: ignore[attr-defined]
            await conn
        return
    thread = getattr(conn, "_thread", None)
    if thread is None or not thread.is_alive():
        await conn


__all__ = [
    "CheckpointLogSaver",
    "KEEP_CHECKPOINTS",
    "LOG_CHANNELS",
    "MigrationResult",
    "migrate_inline_format",
]
