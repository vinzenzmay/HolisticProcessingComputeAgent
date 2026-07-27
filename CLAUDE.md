# HPCA — agent notes

Terminal AI agent (Textual TUI) for HPC Slurm clusters. Design doc: [project.md](project.md).

## Running tests

- `pixi run -e dev test` — the default suite: hermetic unit tests, parallel via
  pytest-xdist (`-n auto`; ~48s on a 12-core box). Use this for routine verification.
- `pixi run -e dev test-serial` — same tests, serial; use when debugging
  (readable output, `-x`/`--pdb` behave normally).
- `pixi run -e dev test-live` — the live-backend integration tests
  (`-m integration`). These call a REAL LLM (default `http://localhost:20001/v1`,
  usually an SSH tunnel to a cluster vLLM; override with `$HPCA_TEST_LLM_URL`).
  If the endpoint is key-locked, set `HPCA_TEST_LLM_KEY=<key>` — otherwise the
  probe gets a 401 and every live test silently skips. They run real generations
  (~40s when the backend is fast, minutes when it is loaded). **Never fold them
  into the default run** — they are deselected via `addopts = "-m 'not
  integration'"` in pyproject.toml, and must only be run separately and
  deliberately.

A dev machine often has live tunnels on ports 20000 (embeddings) / 20001 (LLM), so
"something answers on localhost" is normal and does not mean tests should use it.

## Where the databases are

`hpca.db`, `checkpoints.db` and `rag.db` are *kept* in the app dir, but while
the app runs they are opened from a node-local working dir (`$TMPDIR`, else
`/tmp`) and synced back every 60s and on exit — `$HOME` is NFS on a cluster
node, where each sqlite call costs network round-trips. See `hpca.dbcache` and
specs-db-local-cache.md; `settings.database.local_cache` turns it off.

Consequences when debugging: mid-run, the app dir's copies are stale by up to
one sync interval; `<app_dir>/dbcache.log` records recovery and sync failures;
and a leftover `db.lease` plus a surviving working dir is what a crashed run
looks like, recovered on the next start.

## Dev dependencies

Add dev deps by editing `[project.optional-dependencies].dev` in pyproject.toml,
then `pixi install -e dev`. Do NOT use `pixi add --pypi --feature dev`: it writes a
`[dependency-groups]` table that silently replaces the optional-dependencies list.
