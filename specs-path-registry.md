# Spec: removing the path registry

**Status:** implemented (v0.24.0); measured against the live backend.

**One-line purpose:** replace the `{key → absolute path}` registry with plain
path arguments, and settle by measurement — not by taste — whether the one
argument the registry had left survives contact with a long context.

---

## 1. What the registry was

A per-`(profile, session)` table in sqlite mapping a short key to an absolute
path. Tools took keys: `read_file(registry_key, subpath)`,
`create_file(dir_key, name, content_lines)`, `edit_file(registry_key, subpath,
…)`, `move_file(source_key, subpath, dest_dir_key, new_name)`. `register_path`
minted a key; `list_paths` listed them; paths a tool discovered (a job's stdout,
a file it created) were auto-registered and announced to the model by key.

It had been softened once already (v0.19, specs-edit-eval.md §6): every
key-taking argument also accepted a literal absolute path, auto-registering it
on the way through. That removed the mandatory register-then-call round trip.
What it did not remove was the *vocabulary*.

## 2. Why it went

**Its original premise had already failed measurement.** The registry existed
because "a small model mis-copies a 90-character cluster path". Asked to read a
161-character one, the 27B reproduced it byte-for-byte 20 times out of 20
(specs-edit-eval.md §5). Constrained decoding cannot help there either — inside
a JSON string every character is legal — so nothing about the interface was
protecting anything.

**What kept it alive was durable naming.** A key resolves from sqlite whatever
is in the context window. A 200-character path that has scrolled out of a
compacted conversation is simply gone, and the agent has to ask or re-discover
it. This is a real argument, and it is the one this spec had to answer.

**What it cost was constant.** `dir_key`, `subpath`, `source_key`,
`dest_dir_key`, `registry_key` — five argument names, none of which appear in
any agent corpus the model was post-trained on, all of them positioned where the
model wants to put a `path`. The shift tier (specs-edit-eval.md §7) had already
measured what that costs on the cases where a key is wrong or missing.

**The failure that started this** was a `create_file` with `dir_key: "."` in a
fresh session, answered with:

```
[tool error] create_file: UnknownKeyError: Unknown registry key '.'.
Available keys: (none)
```

Three things wrong at once. The session's registry was genuinely empty, so the
"available keys" list — the part meant to make the error actionable — was
empty too. `.` is unambiguous and could simply have been resolved. And the whole
thing escaped as an unhandled exception rather than as a `NOT created: …`
sentence, which is the shape specs-edit-eval.md §1 asks every refusal to have.
A bounce into the retry loop, for a call the tool could have carried out.

## 3. What replaced it

`hpca.paths` — three functions, no state, no table:

* `resolve_path(value, workdir)` — `~` expands; a relative path is anchored at
  the session's working directory; `..` folds **lexically**, so a file that does
  not exist yet still resolves (every write tool resolves its target before
  creating it). Symlinks are deliberately *not* followed: rewriting the user's
  own path into a different one makes a result they cannot recognise.
* `display_path` — a result always prints the absolute form, never the relative
  one the model sent, so what a result claims is checkable against the disk.
* `contains` — resolved, because the question it answers is about symlinks.

The tools follow the shell rather than the old argument pairs:

| before | after |
|---|---|
| `read_file(registry_key, subpath, …)` | `read_file(path, …)` |
| `create_file(dir_key, name, content_lines)` | `create_file(path, content_lines)` |
| `edit_file(registry_key, subpath, old, new)` | `edit_file(path, old, new)` |
| `delete_file(registry_key, subpath)` | `delete_file(path)` |
| `move_file(source_key, subpath, dest_dir_key, new_name)` | `move_file(source_path, dest_path)` |
| `copy_file(…same…)` | `copy_file(source_path, dest_path)` |
| `list_paths()` | `list_scripts()` |
| `register_path(key, path)` | *(gone)* |

`move_file`/`copy_file` take `mv`/`cp` semantics: an existing directory takes the
file under its own name, anything else *is* the target path. A rename therefore
needs no `new_name` argument — the old interface had no way to express one
without it.

**One handle was never a path and stays: a script's name.** It is the file's own
name in `ctx.scripts_dir`, so `create_script(name=…)` writes `<name>.<suffix>`
there, `{name}` in a run_bash line expands to it, and `start_background_script` /
`submit_job` take it. All of that resolves by looking in the directory
(`script_path`), so naming survived without a table.

Also gone: the `path_registry` table, the fork's alias copying (the paths a
copied history mentions mean the same thing in the fork), and the sweep in
`SessionStore.delete`.

### What the model is told now

`PATH_WORKFLOW_GUIDANCE` shrank to the two things a path-taking interface still
has to say — where a relative path is anchored, and that a kept script is named
rather than pathed. The old text's advice to "register_path the paths it
printed and use them by key" is gone from `DISCOVERY_GUIDANCE`.

## 4. How it was measured

`evals/edit_eval.py --tier paths`, nine tasks over create/edit/delete, in three
halves that ask different questions:

* **`_long_ctx`** — a ~200-character cluster path in the ask itself, with ~95k
  of prior session behind it. A copying test: at that distance, does the path
  still arrive in the tool call intact?
* **`_recall`** — the same path established **once**, ~95k tokens back, by an
  earlier assistant turn; the ask refers to it only by description ("the
  cluster-annotation results directory you told me about earlier"). This is the
  durable-naming case, put where a key is supposed to win. On the registry side
  that establishing turn also announces the key, the way an auto-registration
  does — so each branch meets the case with its own production affordance.
* **`_recall_mid`** — the same, with the establishing turn buried in the
  *middle* of the padding. Position is not incidental here: a model holds the
  start and the end of a long window far better than its middle, so `_recall`
  alone tests the kindest position there is. This is both the harder case and
  the realistic one — a path mentioned somewhere in the middle of a long
  session. Kept alongside rather than replacing `_recall`, because a gap
  between the two would itself be the finding: it would say the failure is
  retrieval and not the interface.

The delete check has two halves on purpose: the right file gone **and** the
neighbour untouched. A near-miss path that deletes `annotation.tsv` instead of
`annotation_backup.tsv` is exactly the failure this tier is looking for, and
"the backup is gone" alone would score it as a pass.

Both sides run the same harness file (`HAS_REGISTRY` is the single flag the
branch-dependent bits key on), each supplying its own `PATH_WORKFLOW_GUIDANCE`
and `SCRIPT_GUIDANCE` — that difference *is* what is under test.

The context size is measured, not assumed: `prompt_tokens` is recorded per run.
These per-cluster tables tokenize at ~2.2 chars a token where prose runs ~4, so
300k chars overran the 112k-context server outright, 200k came in at 91,108, and
the 207k the tier uses lands at **94,931** (95,304 on the registry side, whose
guidance text is a little longer). That matters because a context
overrun is a 400, which scores as a zero indistinguishable from the model
getting the path wrong.

### Recipe

```sh
export HPCA_TEST_LLM_KEY=<key>          # else the probe 401s and everything skips

# plumbing, no backend (expect 1.0):
pixi run -e dev python evals/edit_eval.py --tier paths --dry-run

# treatment = working tree:
pixi run -e dev python evals/edit_eval.py \
    --tier paths --repeats 3 --label no-registry --out /tmp/treatment.json

# baseline = a worktree of the last ref that still had the registry:
git worktree add /tmp/hpca-baseline <ref>
cp evals/edit_eval.py /tmp/hpca-baseline/evals/
pixi run -e dev python /tmp/hpca-baseline/evals/edit_eval.py \
    --tier paths --repeats 3 --label registry --out /tmp/baseline.json
```

The baseline run does **not** need its own pixi environment: the harness puts
its own repo's `src/` on `sys.path` ahead of the editable install, and
`has_registry` in the summary confirms which side actually ran.

Run the two sequentially. They share one backend, and queueing on one would show
up as latency on the other.

## 5. Results

Qwen3.8-27B-FP8 via a local vLLM (112k window), 3 repeats, run 2026-08-20.
Baseline = `main` @ 83a0c90 (registry), treatment = this branch. Each tier ran
both sides sequentially against the same backend.

**`paths` — 9 tasks × 3, at ~95k of context** (baseline peak 95,304 prompt
tokens; treatment 94,935):

| | registry | no registry |
|---|---|---|
| success | **100%** (27/27) | **100%** (27/27) |
| tool calls / run | 1.41 | 1.33 |
| tool errors / run | 0.00 | 0.00 |
| failed edits / run | 0.00 | 0.00 |

Every task passed on both sides — including all three placements of the path:
in the ask, established 95k tokens earlier at the head of the window, and buried
in the middle of it. Not one run put the file in the wrong place, and the delete
check (right file gone, neighbour untouched) held in all 27.

**`core` — 12 tasks × 3** (the standing regression tier):

| | registry | no registry |
|---|---|---|
| success | 100% (36/36) | 100% (36/36) |
| tool calls / run | 2.00 | 1.92 |
| failed edits / run | 0.06 | 0.00 |

**`hard` — 7 tasks × 3:**

| | registry | no registry |
|---|---|---|
| success | 90.5% (19/21) | **100%** (21/21) |
| tool calls / run | 3.57 | 3.76 |
| failed edits / run | 0.00 | 0.05 |

The whole difference is `long_specs_write`, 1/3 on the registry side against 3/3
without it. Read that as suggestive and not settled: that task is a 12-call
skeleton-then-fill write and has been the tier's marginal one before
(specs-edit-eval.md §8), and three repeats cannot separate a real effect from
its variance.

### The detour, in the model's own words

The interesting evidence is not in the totals. A kept baseline transcript
(`--keep`, `edit_long_ctx`) shows what the extra calls were:

```
register_path  {"key": "annotation_tsv", "path": "/tmp/…/annotation.tsv"}
read_file      {"registry_key": "annotation_tsv", "subpath": "", …}
edit_file      {"registry_key": "annotation_tsv", "subpath": "", …}
```

Handed a path, the model spent a round trip minting a key for it — the tax the
v0.19 guidance rewrite was supposed to have ended, still being paid about one
run in three under the *current* guidance. And in the runs where it did not, it
did this instead:

```
read_file      {"registry_key": "/tmp/…/annotation.tsv", "subpath": "", …}
```

— the argument named for a key, filled with a path. That is the interface being
worked around rather than used, in both directions, in the same task.

## 6. What the numbers settle

**Durable naming does not show up.** It was the registry's last argument and the
reason this had to be measured rather than argued: a key resolves from sqlite
whatever is in the window, a long path scrolls away. At ~95k of context, with a
~200-character cluster path, across create/edit/delete and all three positions
in the window, the 27B reproduced the path exactly every time. Whatever the
registry was insuring against, this backend at this context length does not do
it. The premise had already failed once at 161 characters in a short window
(specs-edit-eval.md §5); it now fails at 95k too.

**The cost was real and is gone.** Fewer calls per run on both the paths tier
(1.33 vs 1.41) and core (1.92 vs 2.00), no `[tool error]` results on either side,
and the `register_path` detour cannot happen because the tool no longer exists.

**No regression anywhere.** core is 100% both sides; hard is 100% treatment
against 90.5% baseline.

### What this does not say

* **It is not a claim about every backend.** A smaller or differently trained
  model may well lose a 200-character path at 95k, and the tier is the
  instrument to check that when the backend changes.
* **It is not a claim about compaction.** These runs carry the whole
  conversation. HPCA folds *tool-call payloads* by age
  (`hpca.agent.history`), which is not the same as dropping the turn a path was
  mentioned in, and a session compacted hard enough to lose that turn is a case
  this tier does not reach. The honest fallback there is the same one a human
  has: look it up again.
* **Three repeats is three repeats.** The success rates are separated by wide
  margins or not at all; the per-run call counts differ by ~5%, which is inside
  what this many samples can resolve.

