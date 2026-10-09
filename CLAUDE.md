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
`evals/edit_eval.py` as baseline-ref-vs-working-tree — recipe in
specs/specs-edit-eval.md. Compare a baseline you ran yourself, not a number quoted
in that document: the task checks were tightened on 2026-08-24 (§10), and
every figure recorded before that is an upper bound measured with checks that
scored some corrupted files as successes. Like test-live it
costs real generations; run it deliberately, never as part of a default suite.

Every run first checks its own task checks — against a correct outcome and
against every one-line corruption of it — and aborts rather than producing a
number it cannot stand behind; `--self-check` alone does just that, over all
four tiers, in about a second, and is the thing to run after touching a task.

Tiers: `core` and `hard` are the standing regression set. `paths` is the
long-context one (~95k of session in front of a ~200-character path, over
create/edit/delete) — it costs ~15 min a side because each run re-prefills, so
reach for it when a change touches how a file is *named*, and see
specs/specs-path-registry.md for what it settled. `shift` measured the key-vs-path
friction of an interface that no longer exists; it is history, not a check.

## Two front-ends, one runtime

`hpca` draws a TUI; `hpca -p "..."` runs one turn and prints the reply. Both
build the same `Core`, which lives in **`hpca.core.boot`** — not `hpca.ui.boot`,
which is now only the wiring between that class and a screen. So a change to
how the runtime starts or stops belongs in `core/boot.py`, and
`tests/test_core_headless.py` will fail it if it reaches for a front-end.

`hpca/headless.py` is the second front-end and its docstring is the reasoning:
what a run answers when there is no human (gated call → yes, triage offer → no,
compaction → yes), why the session pins its backend by label, and why the exit
code rather than the message is the interface. The approval default is the one
that looks wrong and is not — refusing an `edit_file` does not stop the edit,
it moves it into `run_bash` and `sed -i`, losing the trash backup; that is
measured, not assumed (§3.0 of project.md).

## The tool registry changes at runtime

User tools (`<app_dir>/tools/*.py`, `hpca.user_tools`, project.md §5.1) are
loaded into the same `ToolRegistry` the graph decides from, and
`reload_user_tools` swaps them in place. So a tool can disappear between a
decision and its execution: the graph answers that call with a tool error
rather than raising. Startup loads only file versions a reload approved
(`tools/.approved.json`); a test that wants a tool live at startup has to
record that first. A new built-in whose name a user's tool already has wins,
and that user file stops loading. Only `build_service`'s default registry gets
user tools; a test that passes its own `tools=` gets none.

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
specs/specs-db-local-cache.md; `settings.database.local_cache` turns it off.

Copies are not naive. `hpca.db` and `checkpoints.db` are *rebuilt* on every
copy (`VACUUM INTO`) rather than page-copied, because sqlite never shrinks a
file and checkpoint churn had left one 94% free pages — 160 MB of file around
9 MB of live rows, all of it crossing NFS four times a run. `rag.db` stays on
the backup API: no free pages to reclaim, and a rebuild would re-index it. And
a periodic sync skips any database nothing has written to since it last went
home; the final sync never skips. See §2.4–2.5 of the spec.

`checkpoints.db` is not LangGraph's saver. `hpca.checkpointer.CheckpointLogSaver`
stores `messages`/`thinking`/`calls` as an append log (`channel_items`, one row
per item, written once) and the checkpoint as a manifest naming how much of each
log is live — so a step's write is what it appended, not the whole conversation.
It keeps the last 32 checkpoints per thread and nothing older, which is a
property, not an oversight: the log is the *current* conversation, so a rewind
overwrites what it rolled past and checkpoints before it cannot be restored.
Nothing reads a historical checkpoint (every read is `aget_state` with no
`checkpoint_id`). An old inline-format file is converted on first start —
`migrate_inline_format`, off the loop, keeping the latest state per thread, with
a notice on the wire. See specs/specs-checkpoint-log.md.

Consequences when debugging: mid-run, the app dir's copies are stale by up to
one sync interval — or longer for a database nothing is writing to, which is
skipped entirely until it changes; `<app_dir>/dbcache.log` records recovery and
sync failures; and a leftover `db.lease` plus a surviving working dir is what a
crashed run looks like, recovered on the next start.

## Dev dependencies

Add dev deps by editing `[project.optional-dependencies].dev` in pyproject.toml,
then `pixi install -e dev`. Do NOT use `pixi add --pypi --feature dev`: it writes a
`[dependency-groups]` table that silently replaces the optional-dependencies list.
