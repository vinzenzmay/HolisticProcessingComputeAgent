# Spec: node-local sqlite working copies ("DB local cache")

**Status:** design agreed (2026-07-27), implemented on `fix/db-local-cache`.

**One-line purpose:** run the app's sqlite databases on node-local storage
(`/tmp`, or `$TMPDIR` under Slurm) instead of the NFS home directory, and sync
them back to home periodically and on exit — so every sqlite call costs
microseconds instead of NFS round-trips.

---

## 1. Context — why

On a cluster node `$HOME` is NFS. Every sqlite call there is network round-trips
plus a remote fsync: hundreds of milliseconds each, seconds under lock
contention. HPCA hits sqlite constantly:

| Writer | Path | Frequency |
|---|---|---|
| LangGraph checkpointer | `checkpoints.db` | every graph step of every turn |
| tool handlers (jobs, registry, symbols, episodic, RAG) | `hpca.db`, `rag.db` | several per tool call |
| poll timers (processes, jobs), sessions rebuild | `hpca.db` | every 2s / 30s |

Version 0.7.2 (`91fe544`) moved the **poll timers and the sessions rebuild** off
the Textual event loop onto a dedicated DB worker thread (`hpca.db.DbIO`). That
fixed the UI freeze caused by *those* callers, but it did not make sqlite any
faster, and it did not cover the agent side: the graph's tool handlers are
`async def` and run sync sqlite inline on the event loop, and the checkpointer
writes on every step. So the TUI still lags while the agent works.

`DbIO` treats the symptom (who waits). This spec treats the cause (how long the
wait is). The two are complementary and both stay.

**Key fact this exploits:** a Slurm job's `/tmp` is on the node executing the
job, so it is local disk, not NFS.

## 2. Design

### 2.1 What moves

All three sqlite files, so there is one uniform mechanism:

- `hpca.db` — jobs, sessions, path registry, processes, symbols, messages
- `checkpoints.db` — LangGraph conversation checkpoints (the heaviest writer)
- `rag.db` — sqlite-vec vector store. *(Since replaced by one index per
  profile, `rag/<profile>.db`, taken into the cache when a profile's index is
  first opened rather than at start — `DbCache.adopt`.)*

Everything else in the app dir (`settings.json`, `profiles/`, `skills/`,
`chatlogs/`, `scripts/`, `proc_logs/`, `job_logs/`, `trash/`) stays in home and
is untouched. Those are not on the hot path.

### 2.2 Location

`local_root()`, first that applies:

1. `$HPCA_LOCAL_DIR` — explicit override (also what tests use)
2. `settings.database.local_dir` — configured override
3. `tempfile.gettempdir()` — honours `$TMPDIR`, which Slurm sets per job

The working dir is `<local_root>/hpca-<uid>-<digest>`, where `digest` is the
first 12 hex chars of `sha256(str(app_dir().resolve()))`. Keying on the app dir
means two different `$HPCA_HOME`s (different tests, a scratch profile) never
share a working dir, and the same home always maps back to the same one — which
is what makes crash recovery possible. The uid keeps users apart in a shared
`/tmp`, and the dir is `chmod 0700` on every acquire (not just at creation: it
may be inherited from a crashed run) — `/tmp` is shared on a cluster node and
these databases hold whole conversations.

### 2.3 Copying

Every copy in both directions is one of two mechanisms, never `shutil.copy`:
the **sqlite online-backup API** (`sqlite3.Connection.backup`), or a
**rebuild** (`VACUUM INTO`) for the databases in `COMPACT_DB_NAMES`. Both are
transactionally consistent under a live writer and both leave a standalone
destination; they differ in what they do with free pages (§2.4).

The backup API is the default. Reasons:

- it is page-level and transactionally consistent even while another connection
  is writing the source (it restarts if the source changes mid-copy), so the
  live checkpointer and RagStore connections need not be quiesced;
- it deals with WAL correctly — the destination is a single self-contained file,
  no `-wal`/`-shm` sidecars need copying;
- it works on `rag.db`'s `vec0` virtual tables from a plain connection with the
  sqlite-vec extension *not* loaded (verified empirically), because it copies
  pages, not rows.

After each backup the destination gets `PRAGMA wal_checkpoint(TRUNCATE)` so the
file left in home is standalone. A rebuild needs no such step — its output is
standalone already.

The copy lands in a sibling temp file (`<name>.backup-tmp`) that is renamed
over the destination only once complete. Backing up straight into the
destination proved able to destroy it: an interrupted write on the network
filesystem left a home copy with exactly its first page zeroed (BIH cluster,
2026-08-04), and the backup API then refused that file in *both* directions
("file is not a database") — one torn file blocked seed, recovery and every
later sync-back at once. The rename is atomic, so the destination is only ever
its old self or the finished copy.

### 2.4 Compaction

sqlite never shrinks a database on its own. `auto_vacuum` is `NONE`, so the
pages a delete releases go on the free list and the file keeps its size
forever — and the backup API reproduces that free list page for page, so the
bloat is copied across NFS at every seed, every sync and every exit.

`checkpoints.db` is where this bites. LangGraph writes a full state snapshot
per super-step, so an ordinary session churns thousands of pages, and deleting
a session frees every page it ever wrote. Measured on a real app dir after a
handful of sessions: **159.8 MB of file holding 8.7 MB of live checkpoints —
36,767 of 39,013 pages free.** Deleting a session *does* remove its rows
(`AgentService._delete_session` drops the session row, the episodic messages,
the watches and the checkpointer thread); what it cannot do is give the space
back.

So the databases in `COMPACT_DB_NAMES` — `hpca.db` and `checkpoints.db` — are
**rebuilt rather than duplicated**, in every direction: seed, recovery and
sync-back. `VACUUM INTO` writes only the live pages, which makes the copy both
smaller and cheaper to make; on seed it is the bigger win, because the pages
never read are pages never fetched over NFS.

`rag.db` is deliberately excluded, and *not* because a rebuild would fail —
VACUUM copies a virtual table's shadow tables and carries its schema row across
without instantiating the module, so `vec0` survives it untouched (verified
empirically). It is excluded because it has nothing to reclaim: an embedding
index grows, it does not churn, and it measured **zero** free pages beside
`checkpoints.db`'s 36,767. Rebuilding it would re-create every index over a
hundred thousand chunks, every sync, to save nothing.

A rebuild that fails anyway falls back to the page copy — logged, and
remembered per source path (`_COMPACT_UNSUPPORTED`) so a whole failed rebuild
is not wasted on every tick. `COMPACT_DB_NAMES` is the default, not a promise.

### 2.5 Skipping a sync that would change nothing

A periodic sync copies only the databases that have been *written to* since
their last successful sync. Before this, every tick pushed all three whole
files to home whether or not anything had touched them — and `rag.db` alone is
hundreds of megabytes that only change when something is indexed.

"Written to" is `fingerprint(path)`: the size and `st_mtime_ns` of the database
**and of its `-wal` sidecar**, both, because a WAL-mode commit lands in the
sidecar and can leave the main file untouched for a long time. Two stats on
node-local storage against a whole-file copy over NFS. A missing file is a
value like any other, so a WAL appearing or being checkpointed away both read
as a change.

The fingerprint is read *before* the copy and stored after: opening and closing
a connection to a WAL database can checkpoint it and remove the sidecar, which
would move the stamp under us. Stale in that direction costs one needless copy
on the next tick, which is the harmless direction.

Two rules keep the skip safe:

- **`release()` passes `force=True`.** A skipped periodic sync is only ever
  safe because the local copy still exists — a hard kill in the window leaves
  the working dir behind and the next start recovers it. The final sync has no
  such window: the working dir is about to be deleted, so it never skips.
- **A name with no recorded fingerprint is always copied.** So the first sync
  of a run copies regardless, which is what lets a run that never happens to
  write to a database still replace a bloated home copy with a compacted one.

A failed copy clears the name's stamp, so the next tick retries it. The lease
heartbeat is refreshed on every tick regardless of what was skipped. `sync()`
still returns "did everything get home", and a database that was already there
counts as arrived.

**Corrupt-file quarantine:** a file under a database's name that is non-empty
but does not start with sqlite's 16-byte magic is moved aside to
`<name>.corrupt-<timestamp>` rather than overwritten or trusted — the torn
bytes stay recoverable offline (`.recover` can salvage rows from the intact
later pages). This fires in two places: a corrupt sync/recovery *destination*
is quarantined just before the fresh copy is renamed in; a corrupt seed
*source* is quarantined so the app starts that database afresh instead of
carrying the blockage through the whole run. Each quarantine logs an ERROR to
`dbcache.log` and appends a one-line notice to `DbCache.warnings`, which the
app drains into toasts (`drain_warnings()`) at startup and after each sync
tick. An empty file is *not* condemned — sqlite treats it as a database with
no tables yet.

### 2.4 Lifecycle

**Startup** (`DbCache.acquire()`):

1. Take the lease (§2.5). Failure ⇒ inactive, run directly on home as before.
2. **Recover:** if the working dir already holds databases, a previous run died
   without syncing. Copy those local→home *first*, before anything overwrites
   them.
3. **Seed:** for each database that exists in home, copy home→local.
4. Report active. `path_for(name)` now returns the local path.

**Running:** a timer syncs local→home every `sync_interval_s` (default 60,
0 disables), on a worker thread, skipped while a previous sync is still in
flight, and per database skipped again if nothing has written to it (§2.5).
The lease heartbeat is refreshed by the same tick either way.

**Exit** (`DbCache.release()`): final sync — `force=True`, so it never skips a
database — delete the working dir, drop the
lease. Deleting is deliberate — after a *clean* exit home is authoritative, so
leaving the copies behind would only make the next start do pointless recovery
work. A crash skips this, which is exactly when recovery is wanted.

The working dir is deleted **only if the final sync got everything home**
(`sync()` returns that). If it did not, the working dir holds the only copy of
what is missing, so it is kept and the next start recovers it — a failed exit
is treated as a crash.

`sync()` and `release()` are serialised by a reentrant lock: both run on worker
threads, and shutdown can begin while a periodic sync is still copying.

**Saying so.** The final sync is the one the user waits on, and it has no face:
the TUI has stopped painting and the shell prompt is not back yet, so a copy
that takes NFS-minutes is indistinguishable from a hang. Textual dispatches
`Unmount` with the alt screen still up, which is why the app stops application
mode itself before releasing — anything printed before that is thrown away with
the alt screen. On the terminal the user is left with:

    please WAIT a moment while the chat log databases are being copied ...
    chat log databases copied.

Nothing is printed when local mode was never active: the databases are already
home and `release()` returns at once.

Leaving application mode also gives the terminal its signal handling back, so
from that point Ctrl+C — the reflex this message exists to head off — really
would kill the copy. The first press is answered rather than obeyed:

    still copying — press Ctrl+C again to abort (the next start then finishes the copy)

A second press restores the default handler and aborts, so a hung mount is
never a trap. Aborting costs nothing permanent: the working dir and lease
survive it, which is precisely the crash case recovery is built for.

### 2.5 Concurrency — the lease

Whole-file sync-back is last-writer-wins over the **entire** database. Two
instances each holding a private copy would silently erase each other's
sessions. (Sharing one file on NFS, as today, does not have this problem —
sqlite's own locking handles concurrent writers.) So local mode is exclusive:

`<app_dir>/db.lease` holds JSON `{host, pid, started_at, heartbeat, local_dir}`.

- **acquire:** a *live* lease ⇒ do not take local mode; fall back to direct-home
  mode (today's exact behaviour, correct under concurrency) and tell the user
  why via a toast.
- **live** means: same host and the pid still exists (`os.kill(pid, 0)`;
  `PermissionError` counts as alive), or a different host whose heartbeat is
  younger than `stale_after_s` (default 180 = 3 sync intervals).
- **release:** the lease is removed only if it is still ours.

A lease that a crash left behind on this host goes stale as soon as the pid is
gone; from another node it takes `stale_after_s`.

### 2.6 Failure policy

Every failure degrades to direct-home mode or to "skip this sync", never to a
crash and never to data loss:

- working dir not creatable / not writable ⇒ inactive, toast the reason;
- `local_dir` resolving to the app dir itself ⇒ inactive (nothing to gain, and
  copying a file onto itself must not be attempted);
- a sync failing ⇒ logged to `<app_dir>/dbcache.log`, next tick tries again.
  `sync()` never raises: one unreadable database costs only its own copy, and
  an unwritable lease costs only the heartbeat. The log has a file handler and
  `propagate = False` — a TUI owns the terminal, and logging's last-resort
  handler writes `WARNING`+ to stderr, which would shred the display. The app
  additionally toasts once when sync-back *starts* failing (and again if it
  recovers and fails anew) — a home copy quietly falling behind for hours
  proved too easy to miss in the log alone;
- a corrupt home copy ⇒ quarantined to `<name>.corrupt-<timestamp>`, replaced
  by the local copy (or started afresh if there is none), toasted (§2.3);
- `settings.database.local_cache = false` ⇒ inactive, no lease, no copies.

## 3. Configuration

```json
"database": {
  "local_cache": true,
  "local_dir": null,
  "sync_interval_s": 60
}
```

## 4. Durability

A clean exit loses nothing. A hard kill (Slurm walltime, node failure, SIGKILL)
loses at most `sync_interval_s` of database writes — and only if the node's
`/tmp` is also gone, since a surviving working dir is recovered on the next
start.

This is a real change in durability: before, a committed write was on NFS
immediately. It is the price of the speedup and the reason the sync is periodic
rather than exit-only.

## 5. Files

- `src/hpca/dbcache.py` — new: `local_root`, `local_dir_for`, `copy_database`,
  `Lease`, `DbCache`
- `src/hpca/config.py` — new `DatabaseSettings` section
- `src/hpca/tui/app.py` — `on_mount` wiring, sync timer, `on_unmount` release,
  `_announce_final_db_sync` / `_write_to_terminal` / `_sync_interrupt_guard`,
  `_file_logger` (generalised out of `_autoconnect_logger`)
- `src/hpca/db.py` — `checkpoints_db_path()` removed (its only caller now asks
  the cache); its rationale moved to `DB_NAMES`
- `tests/test_dbcache.py`, `tests/test_tui_dbcache.py` — new
