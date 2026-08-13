# Spec: live edit-tooling eval (`evals/edit_eval.py`)

**Status:** implemented and measured (2026-08-07, v0.18.0).

**One-line purpose:** measure — against the real backend, through the real
middleware decision loop and real tool handlers — how well the live model
drives `edit_file` / `read_file` / `create_file`, so that changes to the file
tools are judged by measured effect on the model's behavior, not by taste.

---

## 1. Context — why

v0.18.0 redesigned the file tools after earendil-works/pi, under one
philosophy: **absorb model imperfection inside the tool instead of bouncing
errors back into the retry loop** — the retry loop is where the 27B backend
gets confused. The eval existed first as the instrument to prove that
redesign, comparing v0.17.0 (baseline) to v0.18.0 (treatment) on the live
Qwen3.6-27B:

| Tier | Metric | v0.17.0 | v0.18.0 |
|---|---|---|---|
| core (12 tasks × 3) | success rate | 97.2% | 100% |
| core | failed edit calls / run | 0.111 | 0.028 |
| hard (3 tasks × 3) | success rate | 33.3% | 100% |
| hard | tool calls / task | 4.1 | 2.3 |

What actually failed at baseline was structural, not model stupidity: the
middle of a 700-line file was unreachable (head/tail truncation, no offset),
and every edit of a CRLF file silently rewrote it to LF.

**Negative result worth keeping:** the smart-quote transcription trap
(`smart_quote_line`, hard tier) never fired — Qwen3.6 transcribes typographic
quotes, en-dashes and NBSP faithfully, and the task passed 3/3 on the *old*
exact-match code too. The fuzzy ladder's unicode level is therefore insurance
against other/smaller backends, not a measured need of this one; if it ever
gets in the way, it can go — and the trap task itself is perhaps not needed
at all. Keep that in mind before adding more unicode-normalization machinery.

## 2. How to run a future assessment

The harness is the standing instrument for any change that touches the file
tools, the middleware retry loop, prompts/guidance around editing, or a
backend/model swap. The short form to ask the agent for:
*"run the edit eval, baseline `<ref>` vs this branch"*.

The agent (or you) then does:

```sh
# 0. prerequisites: live backend answering on port 20001 (SSH tunnel),
#    and the key exported — otherwise the script exits 2 on the 401.
export HPCA_TEST_LLM_KEY=cubi

# 1. plumbing sanity, no backend needed (expect success_rate 1.0):
pixi run -e dev python evals/edit_eval.py --dry-run --tier all

# 2. treatment = the working tree:
pixi run -e dev python evals/edit_eval.py \
    --tier all --repeats 3 --label treatment --out /tmp/treatment.json

# 3. baseline = any git ref, via a worktree (the script self-bootstraps
#    sys.path to <its repo>/src, so a copied script measures the OLD code):
git worktree add /tmp/hpca-baseline <ref>
mkdir -p /tmp/hpca-baseline/evals
cp evals/edit_eval.py /tmp/hpca-baseline/evals/
(cd /tmp/hpca-baseline && pixi run -e dev python evals/edit_eval.py \
    --tier all --repeats 3 --label baseline --out /tmp/baseline.json)
git worktree remove --force /tmp/hpca-baseline

# 4. compare the summary blocks of the two JSONs.
```

Reading the numbers: `success_rate` is the headline; `mean_failed_edits`
(rejected "NOT edited" calls, i.e. retry-loop entries) and `mean_tool_calls`
are the friction metrics — a change can hold success constant and still be a
win by cutting both. Run baseline and treatment **sequentially**, not in
parallel: they share the backend, and contention skews wall time and can skew
quality under load. Three repeats is signal enough for the big effects seen
so far; raise `--repeats` for small ones.

Extending: add tasks to `build_tasks()` (core: everyday shapes) or
`build_hard_tasks()` (shapes the tooling has structurally mishandled) in
`evals/edit_eval.py`, each with a `fake_calls` script so `--dry-run` keeps
proving the plumbing. Keep tasks representative of real cluster files — the
hard tier earns its name from realism (long generated configs, Windows-edited
ini), not from puzzle construction. A task's `files` are written and
registered; its `missing` keys are registered and deliberately *not* created,
which is the only way to set up a key that resolves to nothing.

## 3. The long-write problem (v0.19.0)

A real session (the cut_locus grilling, 2026-08-06) surfaced the next
failure class: a 12-decision specs.md is 250+ lines, more than one
create_file call fits under the 4096-token decision cap, so the write
truncated and the turn died with nothing on disk. Fixes, in order of load
they carry:

- **Truncated-call salvage** (`middleware._salvage_truncated_call`): a
  cut-off create_file/edit_file is not retried wholesale — the complete
  prefix of its final line-array runs, with a `TBD` marker line planted
  where the text stops and a repair note in the result. Arrays-last argument
  ordering is what makes the prefix parseable. Degenerate loop tails
  (`'  ,' × 200`, the other way generations hit the cap) are trimmed, not
  written.
- **Placeholder tracking** (`file_tools._tbd_note`): every create/edit
  result counts remaining `TBD` lines and names the next one — the tool side
  of the skeleton-then-fill protocol, and what keeps a 27B from declaring
  "done" two placeholders early.
- **Skeleton-then-fill guidance**, kept deliberately small: one sentence in
  SCRIPT_GUIDANCE and the create_file description; the reactive
  TRUNCATION_FEEDBACK (costs nothing until it fires) teaches the same shape.

Measured on `long_specs_write` (hard tier; transcription of ten agreed
decisions into a ten-section specs.md, 6 reps): v0.18.1 lost the ENTIRE
write to double-truncation in 3 of 6 runs (dead turn, no file); v0.19.0
produced a substantial specs.md in 6 of 6, fully complete in 3, the rest one
section short or with an honest TBD marker left. The residual incompleteness
is long-horizon instruction compliance of the 27B — one interactive
"section X is missing" turn in practice — not tool mechanics; do not try to
prompt it away with more standing text.

## 4. Keys that point at nothing (v0.19.2)

`register_path` used to refuse a path that did not exist, on the reasoning
that a missing path is a typo. It is equally often the file the agent is about
to write, and refusing left no key to name it with in the call that would have
created it. It now registers either way and says which case it is; the two
tools that meet such a key follow:

- `read_file` names the absence instead of letting `read_text` raise a bare
  `FileNotFoundError` into the loop.
- `create_file` treats a `dir_key` that is not there yet as a directory to
  make, not a call to refuse — it already made intermediate parents, so the
  guard was rejecting work the tool does anyway.

Two hard-tier tasks cover it: `write_into_unmade_dir` (the output directory is
registered but never created) and `read_key_pointing_at_nothing` (a file key
resolving to nothing; the model must notice and write the file).

Measured v0.19.1 (baseline) vs working tree, 3 reps, Qwen3.6-27B via a local
ollama (`qwen27_32k`, num_ctx 32000, temp 0.25):

| Task | Metric | baseline | treatment |
|---|---|---|---|
| write_into_unmade_dir | success | 33.3% | 100% |
| write_into_unmade_dir | failed edits / run | 2.0 | 0.0 |
| write_into_unmade_dir | tool calls / run | 5.67 | 1.0 |
| write_into_unmade_dir | completion tokens | 6485 | 356 |
| read_key_pointing_at_nothing | success | 100% | 100% |
| read_key_pointing_at_nothing | tool calls / run | 2.67 | 2.33 |

The first is the whole point: refused at the door, the 27B spent its entire
8-decision budget twice over and still left nothing on disk. The second is a
**null result worth keeping** — a bare `FileNotFoundError` was already enough
for this backend to recover from, so `read_file`'s sentence is a clarity fix,
not a measured win; the call-count drop is inside the noise at 3 reps.

Core tier over the same pair (12 tasks × 2) as a regression check: success
100% treatment vs 95.8% baseline, failed edits 0.042 vs 0.167 — no regression.
Read that delta as noise: the one baseline failure was a degenerate 214s
generation on `simple_replace`, a path neither change touches.

Backend note for whoever runs this next: the local ollama died mid-run once
(`Server disconnected`, then connection refused for the remaining reps). It
fails loudly in the per-run `error` field — discard those cells and re-run
them rather than reading the summary, which happily averages a dead run in.

## 5. What it deliberately is not

- Not part of the default or live pytest suites — it costs real generations
  and minutes of wall time; it runs only when someone asks for an assessment.
- Not HITL-gated — handlers are called directly; it measures the model+tool
  contract, not the approval UX.
- Not a benchmark of the model — the fixed reference is the task set; the
  thing under test is HPCA's tooling.
