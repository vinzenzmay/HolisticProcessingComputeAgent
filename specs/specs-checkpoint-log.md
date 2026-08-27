# Spec: log-structured checkpoints ("checkpoint log")

**Status:** implemented (2026-08-27), `src/hpca/checkpointer.py`, wired in
`src/hpca/ui/boot.py`, covered by `tests/test_checkpointer.py`. Lever C of §5;
A and B were not needed separately and were not done. Where building it
contradicted the design, this document has been corrected to describe what
exists — the places that moved are §3.1 (columns), §3.2 (how a channel that
did not change is deduplicated), §3.3 (one snapshot mechanism, not two), §3.5
(no incremental_vacuum from the sync worker), §6 (version scheme), §8
(measured, not projected) and §10 (resumable).

**One-line purpose:** stop storing a full copy of the conversation on every
graph step. Store each message, thought and call exactly once, and reduce the
checkpoint row to a manifest that names how much of each log is live — so
`checkpoints.db` reflects the current state of a conversation instead of every
state it ever passed through.

---

## 1. Context — why

`ui/boot.py:229` opens LangGraph's `AsyncSqliteSaver` (langgraph 1.2.9,
langgraph-checkpoint-sqlite 3.1.0) over `checkpoints.db`. That saver stores one
row per checkpoint with `channel_values` **inline** as a single msgpack blob:

```sql
CREATE TABLE checkpoints (
    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
    type, checkpoint BLOB, metadata BLOB, PRIMARY KEY (...)
);
```

LangGraph writes one checkpoint per super-step. HPCA's graph alternates
`orchestrator → execute_tool → orchestrator`, which measures at ~2.6
super-steps per tool call. So the cost of a turn is not the size of what it
appends — it is the size of the whole conversation, once per step. Storage is
quadratic in tool rounds.

Measured on a real thread in a developer's `checkpoints.db`:

| | |
|---|---|
| checkpoint rows | 115 |
| bytes of blobs stored | 7.73 MB |
| the conversation those blobs describe | 150 KB |
| amplification | **51×** — 98.1% of the file is superseded copies |

The growth curve is the whole argument: the last sixty rows are each 120–150 KB
and each is a full copy of the same conversation.

Composition of that 150 KB state, which is where the constant factor lives:

| channel | size | note |
|---|---|---|
| `messages` (88) | 67 KB | tool results verbatim |
| `thinking` (44) | **67 KB** | never sent to the model, copied every step |
| `calls` (43) | 24 KB | full unfolded `arguments`, duplicating `messages` |
| `plan`, `pending_tool`, `tool_rounds` | <1 KB | |

This has a second-order cost that is worse than the first. `dbcache.sync()`
rebuilds `checkpoints.db` with `VACUUM INTO` **onto NFS** — the temp file is a
sibling of the destination (`dbcache.py:253`) — every 60 s for as long as the
fingerprint keeps changing, which during an active session is every tick. A
1.5 GB checkpoint database therefore means a multi-minute, full-bandwidth NFS
write once a minute, on the same node whose home directory the UI's own
synchronous file IO (`proc_logs`, `scripts/`, chatlogs) has to reach. And while
`VACUUM INTO` holds its read transaction the live checkpointer keeps
committing, so the WAL cannot be checkpointed away and grows — which guarantees
the next tick copies again. Size feeds back into latency.

### 1.1 Two key facts this exploits

**Nothing in HPCA ever reads a historical checkpoint.** Every read is
`aget_state(config)` with no `checkpoint_id`: `graph.py:671,713,757,783,793,863,906`,
`scheduler.py:1024`, `service.py:858`. There is no `alist`, no
`get_state_history`, no time travel in any command. Even the chat rewind is not
one — `TRUNCATE_TO` (`graph.py:63`) goes through `aupdate_state` and writes a
*new* checkpoint. The 152 superseded rows in the file above were never readable
by any feature. Keeping them is not a conservative choice; it is dead weight
nothing was ever going to lift.

**The three fat channels are append-only by construction.** `_append`
(`graph.py:78`) and `_append_messages` (`graph.py:84`) only ever return
`left + right`; `messages`, `thinking` and `calls` all use one of them
(`graph.py:147,151,162`). So the difference between consecutive checkpoints is
*the tail*, and a checkpoint does not need to store a list at all — it needs to
store a **length**.

The single dict-shaped exception is the truncate sentinel, which shrinks a list
rather than extending it. §3.3 handles it, and handles it as the general case
rather than as a special one.

## 2. What this does not change

- The state shape. `AgentState` (`graph.py:146`) keeps the same channels with
  the same reducers. This is a storage change under an unchanged graph.
- Compaction. `/compact` appends a summary and sets `compacted.upto`; the
  stored history is never rewritten (`history.py:265`), which is exactly the
  shape an append log wants.
- `fold_old_payloads`. It builds a view for the model and touches nothing
  stored.
- Durability. A checkpoint is still written per super-step, synchronously with
  respect to the step that follows it. See §7.1 for the lever that was
  considered and rejected here.

## 3. Design

### 3.1 Schema

```sql
-- Every item ever appended to a list channel, stored exactly once.
CREATE TABLE channel_items (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    idx INTEGER NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, idx)
);

-- The channels that are not append-only (plan, pending_tool, tool_rounds,
-- compacted): stored whole, but versioned, so a checkpoint that did not touch
-- one shares the row rather than copying it.
CREATE TABLE channel_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- The checkpoint itself: a manifest, ~1.5 KB whatever the conversation weighs.
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    meta_type TEXT, -- the serde's type tag for `meta`
    meta BLOB,      -- v, ts, id, channel_versions, versions_seen, updated_channels
    manifest TEXT,  -- {"messages": {"log": 88, "chain": "…"}, "plan": {"blob": "…"}}
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Unchanged, and still keyed to a checkpoint_id.
CREATE TABLE writes (...);
```

Three details the design sketch got wrong and the implementation settled.
`version` is **TEXT**, not INTEGER: the versions are LangGraph's strings (§6),
and INTEGER affinity would silently coerce `"000…05.6100862733136080"` to a
real and drop the hash suffix. `meta` needs a `meta_type` beside it because
the serializer's unit is a `(type, bytes)` pair. And `manifest` is TEXT holding
JSON — its contents are entirely this saver's own (ints, strings, hex), so
there is nothing there the serde is needed for, and being able to read it with
`sqlite3` at a prompt is worth something.

Splitting channel values out of the checkpoint row is not novel — upstream's
`BasePostgresSaver` already keeps them in a separate blob table keyed by
`(thread, ns, channel, version)`, and sqlite is the odd saver out for storing
them inline. Per-channel blobs alone would not be enough here, though:
`messages`, `thinking` and `calls` change on nearly every step, so each would
get a fresh full copy anyway. `channel_items` is the part that goes beyond
upstream and the part that removes the quadratic.

### 3.2 Write path (`aput`)

For each channel named in `checkpoint["updated_channels"]`:

- **Append-only, prefix verified (§3.3)** — insert only `value[stored_len:]`
  into `channel_items`, one row per item, then record `{"log": len(value)}` in
  the manifest.
- **Anything else** — one `channel_blobs` row at the channel's new version,
  recorded as `{"blob": version}`.

Then one small `checkpoints` row.

What makes an unchanged `plan` free is *not*, as first designed, carrying the
previous manifest entry forward for channels absent from `updated_channels`.
There is nothing to carry forward: `create_checkpoint`
(`langgraph/pregel/_checkpoint.py`) puts **every** channel that has a version
into `channel_values`, on every checkpoint, so the saver sees the whole state
each time. The dedup is the blob key instead — `(thread, ns, channel,
version)` names one value for all time, because LangGraph bumps a channel's
version exactly when it is written, so a put whose `plan` did not change
resolves to the row an earlier put already wrote and writes nothing. That is
strictly better than reading `updated_channels`, which is `None` on some paths
and would have had to be treated as "everything changed" when it was.

Which channels are treated as logs is a property of the *saver*, declared once
(`LOG_CHANNELS = ("messages", "thinking", "calls")`), not inferred from the
graph. A saver that guessed from reducer identity would be reading the graph's
internals to decide how to store its data, and would silently change behaviour
the day someone edits a reducer. Declared, plus verified per put, plus a
fallback that is always correct — that is the whole safety argument.

### 3.3 Prefix verification, and the fallback

The append-only claim must be checked on every put, in O(Δ), never assumed.

Each `(thread_id, ns, channel)` carries a rolling chain in memory:
`h_0 = H(b"")`, `h_i = H(h_{i-1} ‖ serialized(item_i))`, cached alongside the
length that was last written. Extending it for a put costs only the new items,
so verification is proportional to what was appended and not to what is already
stored. The manifest records `(len, chain)` so a restart can re-derive the same
value by reading the log.

A put is a valid append iff `len(new) >= stored_len` and the prefix
`new[:stored_len]` is what was written. When it is not — **rewrite the whole
log from index 0 and carry on.** That is the entire error
policy, and it is what makes the design safe to land:

| situation | what happens |
|---|---|
| ordinary step | tail insert |
| `TRUNCATE_TO` rewind (`graph.py:63`) | length shrinks → snapshot, then tails resume |
| core restart mid-thread | chain not in memory → one snapshot, then tails resume |
| a future non-append reducer | snapshot, forever, correctly |
| corrupted or partially written log | snapshot |
| a log channel whose value stopped being a list | `channel_blobs`, forever, correctly |
| anything at all raising inside the check | snapshot |

The failure mode of a bad prefix check is a larger row, never a wrong restore.
That is a claim about the *shape* of the code, not about how many cases were
thought of: `_plan_channel` builds the snapshot and only `_verified_append`
can narrow it, so an unforeseen case is a bigger row by construction. The last
two rows of that table are what that buys — neither was in the design.

There is one snapshot mechanism, not two. The design said "write the channel
as a `channel_blobs` snapshot" and then, a paragraph later, that a snapshot
re-seeds the log from index 0; doing both would mean the same list stored in
two shapes and two ways to read it back. Only the re-seed is implemented: a
list-valued log channel *always* ends up as `{"log": n, "chain": …}`, with the
items rewritten from index 0 with `INSERT OR REPLACE` and rows above `n`
deleted. `channel_blobs` is where a log channel goes only when its value is
not a list at all — which is also how a state that dropped `thinking`, or a
reducer that stopped producing lists, is handled without the saver caring.

The chain's cost, honestly: extending it for a put costs only the new items,
but *checking* the prefix is O(prefix) unless something cheaper decides it
first. The something cheaper is object identity — the reducers return
`left + right`, so an ordinary step hands the saver the very objects it
already stored, and the check is a walk of pointers. The hash chain is the
fallback for when identity does not hold (items rebuilt by a deserialization,
a fork, a copying reducer), and re-serializes the prefix to decide. The
assumption identity rests on, stated so it can be broken deliberately rather
than by accident: **an item already appended to a log channel is never mutated
in place.** Which is the one property this design deliberately
gives up, stated plainly: **the log is the current conversation, not its
history.** A rewind overwrites the items it rolled past, and a checkpoint older
than the rewind can no longer be restored. Nothing reads those checkpoints
(§1.1), and "the database reflects the current state" is the requirement this
spec was written to satisfy — but it is a property, and it must be written down
here rather than discovered by whoever first wants time travel.

### 3.4 Read path (`aget_tuple`)

Read the manifest row; then one indexed range scan per log channel
(`WHERE thread_id=? AND channel=? AND idx < ? ORDER BY idx`) and one row per
blob channel. Reassemble `channel_values` and hand back a `CheckpointTuple`
that is byte-identical in meaning to what the inline saver would have returned.

More queries than one blob read, and it does not matter: this runs on session
open and on the first step of a turn, not per step, and the bytes it reads are
the same bytes today's blob contains. The per-step cost that this spec exists
to remove is on the write side.

### 3.5 Retention

With a ~1 KB manifest per step, keeping every checkpoint row costs ~3 MB for a
3000-step session, so the retention policy is no longer forced by size. Keep
the last **`KEEP_CHECKPOINTS = 32`** per thread anyway and delete older rows and
their `writes` together: superseded rows are unreachable through any HPCA code
path (§1.1), they are what pins `channel_blobs` versions alive, and a bounded
count is what makes the file's size a function of the conversation rather than
of how long it has been open. Thirty-two is comfortably more than the two
LangGraph needs to resume an interrupt, and small enough to stay negligible.

`channel_blobs` rows not referenced by any surviving manifest are deleted in
the same statement. `channel_items` is never pruned by age: it *is* the current
conversation.

The database is created with `PRAGMA auto_vacuum=INCREMENTAL` so those
deletions can return pages. The design also called for `PRAGMA
incremental_vacuum` from the sync worker; that was not done and `dbcache` is
unchanged, because `dbcache.sync()` already rebuilds `checkpoints.db` with
`VACUUM INTO` — a rebuild reclaims everything an incremental vacuum would, and
adding a second reclaimer on the same file would only be a second thing to get
wrong.

## 4. What this touches elsewhere

- **`dbcache`** — the win that matters for latency. A checkpoints.db of a few
  MB turns the 60 s `VACUUM INTO` onto NFS (`dbcache.py:253`) into a non-event,
  and the WAL stops growing under a long-held read transaction.
  `COMPACT_DB_NAMES` can keep `checkpoints.db` in it; there is simply far less
  to compact.
- **Interrupts and `writes`** — unchanged. `aput_writes` keeps its own table
  keyed to `checkpoint_id`; those rows are small and now die with their
  checkpoint under §3.5.
- **`fork_thread` (`graph.py:845`)** — re-appends the source's messages into a
  fresh thread, so the new thread writes a fresh log. One full copy at fork
  time, exactly as today.
- **`adelete_thread` (`service.py:1249`)** — deletes from four tables instead
  of two.
- **Compaction (`graph.py:724`)** — `aupdate_state` on the `compacted` channel
  only, which is a blob channel. One small row.

## 5. Sequencing

Three independent levers. Land them in this order; each is useful alone.

What actually happened: C was built on its own. A is subsumed — retention
(§3.5) is part of the saver, so the 1.5 GB file and the sync storm are fixed by
the same change. B is still open and still worth doing; it is now a ~40% cut of
a database that is already two orders of magnitude smaller, so it has stopped
being urgent.

| | change | file size | write per step | risk |
|---|---|---|---|---|
| **A** | prune superseded checkpoints (§3.5) behind today's saver | O(1) | unchanged | low |
| **B** | shrink what is in the state (§5.1) | ~40% | ~40% | low |
| **C** | this spec | O(N) | O(Δ) | medium |

A is a thin `BaseCheckpointSaver` wrapper around `AsyncSqliteSaver` that after
each `aput` deletes rows older than the last `KEEP_CHECKPOINTS` for that thread.
It is hours of work, it needs no schema, and it is what stops a 1.5 GB file and
the sync storm today. It does not reduce the per-step write, which is why it is
not the whole answer.

### 5.1 Lever B, because it makes everything else cheaper

`thinking` is 67 KB of a 150 KB state — as large as `messages` — and it is
explicitly firewalled from the model (`graph.py:148`), already persisted to the
transcript and the chatlog. It is in the checkpoint only so `TurnResult` can
hand it back to the scheduler. `calls` stores the full unfolded `arguments`
(`graph.py:401`) that `messages` already holds whole.

Taking both out of the checkpointed state is a ~40% cut with no interface risk
and no dependency on C. It is a separate change with its own spec-sized
question (where does the transcript then read them from), and it is named here
only so the sequencing is deliberate.

## 6. Cost, stated honestly

Implementing `BaseCheckpointSaver` means owning `aget_tuple`, `alist`, `aput`,
`aput_writes` and `adelete_thread`. The sync variants can stay unimplemented —
the base class raises `NotImplementedError` for them by default and HPCA only
ever drives the graph asynchronously — but that is a fact about langgraph
1.2.9, not a promise.

That is the real cost: **the checkpoint contract stops being upstream's problem
and becomes ours.** The `Checkpoint` shape (`v` is 4 as of langgraph 1.2.9, not
the 1 this document first said; the saver stores whatever `v` it is handed and
does not read it), the meaning of `updated_channels`, `get_next_version`, the
resume protocol between `aput` and `aput_writes` — all
of it is internal to langgraph and free to move between releases. Today a
langgraph upgrade is a version bump. After this it is a version bump plus a
read of their changelog.

`get_next_version` is **not** the base class's integer default, which this
document assumed. It is the string scheme `AsyncSqliteSaver` used, copied
verbatim (`f"{n:032}.{random():016}"`), and the reason is the migration: a
database written by the old saver holds string versions in every
`channel_versions` and `versions_seen`, LangGraph compares versions with `<`,
and a thread holding both shapes raises `TypeError`. Keeping the string scheme
makes §10 a pure re-shaping of storage with nothing rewritten. The base class
raises `NotImplementedError` when handed a string, so this override is not
optional.

The failure mode is "the agent forgets a conversation", so the pin has to be
deliberate: an exact-version dependency, and §9's suite run against every bump.

## 7. Alternatives considered

### 7.1 `durability="exit"` — rejected

LangGraph's `durability` defaults to `"async"`; `"exit"` writes one checkpoint
per `ainvoke` instead of per super-step (`pregel/_loop.py:1023,1324`), and
interrupts still persist because an interrupt *is* an exit. On the measured
thread that is 110 rows → ~3, a ~40× cut for one keyword.

Not taken, and the current default stays. A core crash mid-turn would lose the
whole turn's record, and HPCA turns submit Slurm jobs. Losing "I submitted job
12345" while the job keeps running on the cluster is a worse bug than a large
database — the database wastes bytes, that wastes an allocation and lies to the
user about what exists. The point of this spec is to keep per-step durability
and stop paying for it.

### 7.2 Per-channel blobs only (the Postgres shape) — insufficient

Dedups the channels that did not change, which here is `plan`, `pending_tool`
and `compacted` — under 1 KB. `messages`, `thinking` and `calls` change on
nearly every step and would each get a full copy. Necessary (it is §3.1's
`channel_blobs`), not sufficient.

### 7.3 Content-addressed items with a hash manifest — rejected

Store each item under `H(item)` and let the manifest list hashes. Robust to any
mutation, makes forks free, and preserves time travel across a rewind. But the
manifest is then O(N) per checkpoint — 32 bytes × 3000 items × 3000 steps is
288 MB, quadratic again with a smaller constant — unless the manifest is itself
stored as a delta against its parent, at which point the design has grown a
second delta mechanism to serve a feature (§1.1) that does not exist.

### 7.4 Postgres — rejected

Upstream's own recommendation for write-heavy checkpointing, and it solves this
properly. It also means a server process on a compute node, in a deployment
model whose whole premise is tmux and a `$TMPDIR`. Out of scope.

## 8. Measurement

The change is justified by a number taken before and after, on the same thread.

```sql
-- amplification: what is stored, over what it describes
SELECT SUM(LENGTH(checkpoint)) FROM checkpoints WHERE thread_id = ?;
SELECT LENGTH(checkpoint) FROM checkpoints WHERE thread_id = ?
  ORDER BY checkpoint_id DESC LIMIT 1;
```

Baseline, measured on the thread in §1: 7.73 MB stored for 150 KB of
conversation, 51×.

Measured after the fact, both savers driven through the *real* graph over the
same scripted conversation — twelve turns, three tool calls each, tool results
of ~1.8 KB, reasoning on every decision — and compared on final state as well
as on bytes. The final states agree exactly, once the per-write `datetime.now`
stamp `_append_messages` adds is set aside.

| turns | conversation | inline: rows / stored | log: rows / stored | smaller |
|---|---|---|---|---|
| 6 | 62 KB | 54 / 1.72 MB | 32 / 0.11 MB | 15.5× |
| 12 | 123 KB | 108 / 6.81 MB | 32 / 0.17 MB | 39.3× |
| 24 | 246 KB | 216 / 27.15 MB | 32 / 0.30 MB | 90.9× |
| 48 | 491 KB | 432 / 108.47 MB | 32 / 0.55 MB | 197.3× |

At twelve turns the file on disk goes 7.1 MB → 0.34 MB (20.7×); at forty-eight,
109.9 MB → 0.81 MB (135×). The ratio climbing with the session is the whole
claim and not a bonus: the baseline is quadratic in steps and this is linear in
the conversation, so any single factor quoted here is a fact about the length
that was measured. What the 12-turn run costs, broken out: 125 KB of items
(the conversation itself, once), 47 KB of manifests over 32 surviving rows —
1.47 KB a row, and it is `versions_seen`, a dict per node per channel — and
1.2 KB of blobs.

Reproduced with a throwaway script driving `hpca.agent.graph`; it was not
committed, on the grounds that it measures a thing that is now a test.

Second measurement, and the one the user actually feels: with
`HPCA_LOOPLAG=checkpoint-log` set (`ui/boot.py:553`), the spike list in
`<app_dir>/looplag.log` should lose its once-a-minute band. Those spikes are
the sync, and they are labelled with `activity_label()`, so a stall during
`idle` is the sync and a stall during `running run_bash` is not this spec's
problem (it is `builtin_tools.py:666`, which reads a whole stdout file on the
event loop to keep 4000 characters of it).

## 9. Testing

The suite is the pin against langgraph's contract, so it tests the contract and
not the implementation:

- **Round trip.** Every channel, every type in `AgentState`, put then
  `aget_tuple`, compared against what `InMemorySaver` returns for the same
  sequence. This is the oracle for all of the below.
- **Resume across an interrupt.** Put, `aput_writes`, restart the saver, resume
  with `Command(resume=...)`, assert the turn completes — the one path that
  reads `writes` and `parent_checkpoint_id`.
- **Rewind.** `TRUNCATE_TO` then further appends; assert the restored state,
  and assert the snapshot fallback fired rather than a silent tail insert.
- **Restart mid-thread.** Drop the in-memory chain between puts; assert one
  snapshot then tails.
- **Fork and compaction.** `fork_thread` and `apply_compaction` round-trip.
- **Retention.** After `KEEP_CHECKPOINTS + 10` puts, exactly
  `KEEP_CHECKPOINTS` rows survive, their `writes` and `channel_blobs` survive
  with them, orphans are gone, and the latest state is unchanged.
- **Fuzz against the oracle.** N random puts drawn from {append, truncate,
  blob-channel update, restart}, comparing `aget_tuple` to `InMemorySaver` at
  every step.
- **Migration.** An old-format file with several threads migrates to the same
  latest state per thread, the old table is gone, the surviving checkpoint
  keeps its `writes`, one unreadable thread does not cost the others, running
  it twice is a no-op, and a file the migration never touched still starts.

All hermetic; none of this needs a backend. `tests/test_checkpointer.py`, 49
tests, about a second.

## 10. Migration

`migrate_inline_format(path)` detects the old format — a `checkpoints` table
carrying a `checkpoint BLOB` column and no `manifest` — and migrates in place:
for each thread, read the latest row, write it as an initial log plus manifest,
drop the old table, `VACUUM`. One pass, and it collapses the existing 160 MB /
1.5 GB on the spot, which is a better first impression than a new empty file
beside a large old one. The migrated row's `parent_checkpoint_id` is cleared:
the parent did not survive, and a head that points at nothing is what a
thread's first checkpoint looks like anyway.

Migration runs from `Core.start`, before the graph is built, on
`asyncio.to_thread` — on a 1.5 GB file it is a minute of sqlite, and a silent
minute of blocked frames at startup reads as a hang. What it has to say rides
the existing `notices` list that `_open_db_cache` returns, which `_events_out`
puts on the wire after the handshake; there is no separate mechanism for it.

It never raises. Three things were added over the design because "the app must
still start" is not something an error path can be trusted to do by accident:

- The rename to `checkpoints_inline` **commits on its own**, before anything is
  read, so a failure anywhere after it leaves a file the next start resumes
  from rather than a half-converted one.
- A thread that cannot be read — a serializer that raises, *or* a corrupt blob
  that decodes to something that is not a checkpoint — is logged and skipped.
  Every other thread still migrates.
- The saver's own `setup()` is the last line of defence: opening a file still
  in the inline shape (a migration that failed, or was never called at all),
  it moves the old table aside itself and creates the new schema. Nothing is
  destroyed, the app starts, and the next run migrates.

`VACUUM` is attempted last and separately: it is the step most likely to run
out of room, and failing it leaves a correct database that is merely large.

## 11. Files

- `src/hpca/checkpointer.py` — new: `CheckpointLogSaver(BaseCheckpointSaver)`,
  `LOG_CHANNELS`, `KEEP_CHECKPOINTS`, the chain helpers, `migrate_inline_format`
- `src/hpca/ui/boot.py` — `AsyncSqliteSaver.from_conn_string` → the new saver;
  migration notice
- `src/hpca/core/service.py` — unchanged; `adelete_thread` deletes from four
  tables inside the saver, and the call site never knew how many there were
- `src/hpca/dbcache.py` — unchanged; `COMPACT_DB_NAMES` keeps `checkpoints.db`
- `tests/test_checkpointer.py` — new (§9)

No existing test needed changing. `test_graph.py`'s
`test_sqlite_checkpointer_survives_graph_rebuild` still drives
`AsyncSqliteSaver` directly and still passes — it is a claim about langgraph,
not about which saver the app opens; the equivalent claim about this one is in
`tests/test_checkpointer.py`.
