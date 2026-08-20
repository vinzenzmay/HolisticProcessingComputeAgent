# HPCA — agent notes

Terminal AI agent (Textual TUI) for HPC Slurm clusters. Design doc: [project.md](project.md).

## Running tests

- `pixi run -e dev test` — the default suite: hermetic unit tests, parallel via
  pytest-xdist (`-n auto`; 48s–2min on a 12-core box, the upper end when the box
  is otherwise busy). Use this for routine verification.
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

## Assessing file-tool changes

Any change touching edit_file/read_file/create_file, the middleware retry
loop, or editing guidance gets measured, not eyeballed: run the live eval in
`evals/edit_eval.py` as baseline-ref-vs-working-tree — recipe and the
v0.17.0→v0.18.0 reference numbers in specs-edit-eval.md. Like test-live it
costs real generations; run it deliberately, never as part of a default suite.

Tiers: `core` and `hard` are the standing regression set. `paths` is the
long-context one (~95k of session in front of a ~200-character path, over
create/edit/delete) — it costs ~15 min a side because each run re-prefills, so
reach for it when a change touches how a file is *named*, and see
specs-path-registry.md for what it settled. `shift` measured the key-vs-path
friction of an interface that no longer exists; it is history, not a check.

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
