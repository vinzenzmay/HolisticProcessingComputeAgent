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

## Colours

There are no colour constants. Every colour is a setting
(`config.PaletteSettings`, under `display.palette`) resolved by `hpca.ui.theme`,
and drawing code reads `theme.chrome` / `theme.ok` / `theme.warn` /
`theme.danger` / `theme.agent` / `theme.user` / `theme.faint` **at paint time**.
That indirection is load-bearing: `from hpca.ui.ansi import CYAN` bound the
string at import, so a palette that changes while the app runs could not have
been constants. Same rule for any table of styles — hold the *role name* and
resolve it in the function (`state.MODE_ROLES`, `toasts.ROLES`), never a dict of
finished escape sequences at module level.

A colour is an xterm index (`"215"`) or a hex triple (`"#ffaf5f"`); the settings
model refuses anything else, and `theme` falls back per-role for whatever still
reaches it. Hot reload is free — the palette rides `DisplaySettings`, which the
core already restates on save (`core.service._apply_settings` →
`DisplayChanged` → `RowUI.set_display`). There is no file watcher.

The focus flash (`display.focus_flash_seconds`, 0 turns it off) is the only
background colour this UI draws; the `REVERSE` cursor row opts out of it, since
reverse would turn the tint into the text colour.

## Where the app dir is

`config.app_dir()` is the one answer, and it is computed rather than constant:
`$HPCA_HOME` if set (what the whole suite and `evals/smoke_pty.py` run behind),
else `~/work/.HolisticProcessingComputeAgent` when `~/work` is a directory, else
`~/.HolisticProcessingComputeAgent` — with an app dir that *already exists*
beating the rule, so a `~/work` appearing does not strand a user's databases.
Then one hop: that directory's settings.json may name an `app_dir`, which wins.

Consequences for tests: the rule reads the real filesystem, so a test that does
not set `$HPCA_HOME` and does not fake `$HOME` will resolve against the machine
it runs on. `Settings.app_dir` is a modelled field on purpose — `save()` writes
the whole model back, so an unmodelled key would be erased by the next save
from the config editor.

## Where the databases are

`hpca.db`, `checkpoints.db` and `rag.db` are *kept* in the app dir, but while
the app runs they are opened from a node-local working dir (`$TMPDIR`, else
`/tmp`) and synced back every 60s and on exit — `$HOME` is NFS on a cluster
node, where each sqlite call costs network round-trips. See `hpca.dbcache` and
specs-db-local-cache.md; `settings.database.local_cache` turns it off.

Copies are not naive. `hpca.db` and `checkpoints.db` are *rebuilt* on every
copy (`VACUUM INTO`) rather than page-copied, because sqlite never shrinks a
file and checkpoint churn had left one 94% free pages — 160 MB of file around
9 MB of live rows, all of it crossing NFS four times a run. `rag.db` stays on
the backup API: no free pages to reclaim, and a rebuild would re-index it. And
a periodic sync skips any database nothing has written to since it last went
home; the final sync never skips. See §2.4–2.5 of the spec.

Consequences when debugging: mid-run, the app dir's copies are stale by up to
one sync interval — or longer for a database nothing is writing to, which is
skipped entirely until it changes; `<app_dir>/dbcache.log` records recovery and
sync failures; and a leftover `db.lease` plus a surviving working dir is what a
crashed run looks like, recovered on the next start.

## Dev dependencies

Add dev deps by editing `[project.optional-dependencies].dev` in pyproject.toml,
then `pixi install -e dev`. Do NOT use `pixi add --pypi --feature dev`: it writes a
`[dependency-groups]` table that silently replaces the optional-dependencies list.
