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
- `rag.db` — sqlite-vec vector store

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

Every copy in both directions uses the **sqlite online-backup API**
(`sqlite3.Connection.backup`), never `shutil.copy`. Reasons:

- it is page-level and transactionally consistent even while another connection
  is writing the source (it restarts if the source changes mid-copy), so the
  live checkpointer and RagStore connections need not be quiesced;
- it deals with WAL correctly — the destination is a single self-contained file,
  no `-wal`/`-shm` sidecars need copying;
- it works on `rag.db`'s `vec0` virtual tables from a plain connection with the
  sqlite-vec extension *not* loaded (verified empirically), because it copies
  pages, not rows.

After each backup the destination gets `PRAGMA wal_checkpoint(TRUNCATE)` so the
file left in home is standalone.

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
flight. The lease heartbeat is refreshed by the same tick.

**Exit** (`DbCache.release()`): final sync, delete the working dir, drop the
lease. Deleting is deliberate — after a *clean* exit home is authoritative, so
leaving the copies behind would only make the next start do pointless recovery
work. A crash skips this, which is exactly when recovery is wanted.

The working dir is deleted **only if the final sync got everything home**
(`sync()` returns that). If it did not, the working dir holds the only copy of
what is missing, so it is kept and the next start recovers it — a failed exit
is treated as a crash.

`sync()` and `release()` are serialised by a reentrant lock: both run on worker
threads, and shutdown can begin while a periodic sync is still copying.

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
  handler writes `WARNING`+ to stderr, which would shred the display;
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
  `_file_logger` (generalised out of `_autoconnect_logger`)
- `src/hpca/db.py` — `checkpoints_db_path()` removed (its only caller now asks
  the cache); its rationale moved to `DB_NAMES`
- `tests/test_dbcache.py`, `tests/test_tui_dbcache.py` — new
