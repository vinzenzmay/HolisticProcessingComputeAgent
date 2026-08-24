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

#    every run does this first and aborts on failure; alone, it checks the
#    task checks themselves over all four tiers in about a second (§10):
pixi run -e dev python evals/edit_eval.py --self-check

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

## 5. The distribution-shift tier (`--tier shift`)

HPCA's file tools take a *registry key* where every agent-trained model expects
a *path*. The `shift` tier measures what that costs, in three tasks that are
all winnable on the old code — the friction shows up as extra calls and
`[tool error]` results, not as a rigged 0% baseline:

- `repoint_stale_key` — the key is registered at a misspelled directory and the
  user's message carries the real path. Old code: `register_path` on an
  existing key raises `RegistryError` ("pick a different key"), so the model
  must invent a second key for the same directory.
- `edit_by_literal_path` / `create_by_literal_path` — the target is named only
  by its literal absolute path and is registered under no key. Old code: a
  `register_path` ceremony before the call that does the work.

Harness API these needed, for whoever adds tasks next:

- `Task.templated: bool` — opt-in `.format(workspace=...)` of the prompt, and
  `{workspace}` substitution inside `fake_calls` arguments. Opt-in because
  several prompts carry literal braces in shell snippets, and a formatting
  crash would silently zero a task's success rate.
- `Task.system_note: str` — replaces the task-facing half of the system prompt.
  Core and hard keep `_DEFAULT_SYSTEM_NOTE` **byte-identical** (it forbids
  `register_path`); the shift tier uses `_SHIFT_SYSTEM_NOTE`, which allows it.
  Never edit the default — a character moves both older tiers' baselines.
- `Task.unregistered: dict[str, str | bytes]` — files written into the
  workspace and deliberately registered under no key (parents made). The only
  way to set up a file reachable solely by its literal path.
- `register_path` is now in the eval's tool subset for every tier.
- Metric `tool_errors` (summary: `mean_tool_errors`) — results starting with
  `[tool error]`, any tool. `failed_edits` only ever counted
  edit_file/create_file, so refusals from the path layer were invisible.

A `fake_calls` script must drive the route that works on **both** sides of a
comparison (register a fresh key, then act), so `--dry-run` stays green when
the harness is copied into a baseline worktree.

The history shape is a hook, for the same reason: `hpca.agent.history.
tool_exchange(tool_name, arguments, result)` is imported with a fallback that
reproduces the bare `[tool result]` user turn verbatim. A baseline checkout
without that module measures the old conversation shape; the treatment tree
measures the new one.

## 6. Moving the interface toward the model (v0.20.0)

Four changes, each measured as its own rung against v0.19.2 on the cluster
vLLM (`Qwen3.6-27B-AWQ`, :20001, constrained decoding ON), 3 reps unless a
number says otherwise. Read this for the method as much as the numbers — most
of what went wrong was measurement, not code.

| change | measured effect |
|---|---|
| the model sees its own tool call (`agent/history.py`) | core failed edits 0.167 → 0.028, calls 2.222 → 2.028; nearly all of it on `python_indent` (4.67 → 2.33 calls, 2.0 → 0.33 failed) |
| `register_path` repoints a dead key | **null at every rung** — the model never attempts a repoint, it invents a fresh key |
| file tools take a literal path where they take a key | nothing on its own |
| rewriting the guidance that change makes false | with it: `edit_by_literal_path` 3.33 → 2.00 calls, core 97% → 100%, `delete_lines` 72% → 88% (n=25) |

The last two are ONE change. With the capability in and the text still saying
"tools take registry KEYS, never literal paths", the model kept registering
first and nothing moved. The win arrives only when the standing text stops
contradicting the tool — and the same thing recurred one layer down, where
SCRIPT_GUIDANCE still said "create_file with the directory's *key*" (11 of 20
first decisions still registered; 12 of 20 went straight to the tool after).

Keep change 2 on correctness grounds — a mistyped key is unusable for the rest
of a session and there is no unregister tool — but do not claim a measured
win for it.

**A wide prompt cut was tried and rejected.** Halving the whole standing
guidance (−47% of system prompt + tool listing) looked excellent at 3 reps —
core 100%, fewest calls of any variant — and was wrong. At n=25 it took
`delete_lines` from 72% to **52%**, including runs where the model answered
instead of acting at all. Cutting text that had become *false* is worth a call
per task; cutting text that was merely *long* (fail-fast scripts, no-heredoc,
skeleton-then-fill) removed facts a 27B cannot guess. The two are
indistinguishable if you count tokens and opposite in kind. Final state keeps
the guidance at full length with the path block and one `create_file` phrase
rewritten.

**Three reps cannot measure a prompt change.** Every 3-rep signal here that
was not a removed round-trip failed to replicate: a "core collapse" that was a
backend stall (`LLMError` at exactly the 180s client timeout — check each
run's `error` field before believing any drop), a "regression" that was the
harness bug below, a 33%-vs-100% task gap that was 10/10 both sides at n=10.
Tool-mechanics changes — a call the model no longer has to make — are
deterministic enough for 3. Prompt changes shift how often a stochastic model
*chooses* to act and need n≥20, cheaply had by sampling only the FIRST
decision on one task instead of running whole tasks: build the task's
messages, call `decide` N times, count which tool it picks. That probe settled
every prompt question in this section; whole-task runs at n=3 got two of them
backwards.

**The harness bug worth remembering.** `_system_prompt` assembled
RESPOND_VS_TOOL + SCRIPT guidance and omitted PATH_WORKFLOW_GUIDANCE — the one
block that says what a file tool's key argument accepts. The shift tier was
judging the key-or-path change with the guidance about paths deleted, and
produced a confident, entirely artefactual regression: the model answered
instead of acting, twice reporting a file it had never written. Two commits
were written to repair that phantom; both measured worse than the untouched
text and were reverted. **An eval that omits part of the production prompt
does not measure a weaker system, it measures a different one.** When a block
is added to `orchestrator_system_prompt`, add it here too.

Also fixed while running this: the reachability probe allowed 5s, while a
tunnelled cluster LLM answers `/models` in 3-8s when idle or loaded, so a
whole stage died at the door reporting "unreachable" against a healthy
backend. Now 30s, `HPCA_TEST_LLM_PROBE_TIMEOUT` overrides.

## 7. Envelope vs native tool calling (v0.21.0)

`--tool-protocol {envelope,native}` runs the same task set over either
protocol (§4.3 in project.md), so the hand-rolled decision object can be
compared against the backend's own tool-calling channel on identical code.
`tool_protocol` is recorded in the summary block, because two result files
that differ only in protocol are otherwise indistinguishable.

Two things have to move together with the protocol or the comparison measures
something else:

- **The respond guidance.** `RESPOND_VS_TOOL_GUIDANCE` names
  `{"action": "respond", ...}`, which under native is a format the model
  cannot emit. Left in, the core tier ran 12/17 before the run was stopped,
  and *every* failure had one shape: a single `read_file`, then prose claiming
  the edit was made, with the file untouched. The envelope's two-branch
  listing keeps "call a tool" visibly available; once that listing moves into
  the `tools` array, the prose has to say it instead
  (`RESPOND_VS_TOOL_GUIDANCE_NATIVE`).
- **The history shape.** A native run must build its exchanges with the call
  id, or the model is shown envelope-shaped history while calling natively.
  The harness threads `decision.call_id` through, falling back for a baseline
  checkout that has neither.

### 7.1 The example the schema could not give

The first native run failed `simple_replace` in a way no metric named:
1 tool call, 2 decisions, 0 failed edits, 0 tool errors, file untouched. It
looked like the model narrating instead of acting. Replaying the kept
transcript's last decision showed the opposite — it *did* call `edit_file`,
with `old_lines` as a bare string where the schema wants a list of them:

```
"old_lines": "echo \"starting run\""      ← sent
"old_lines": ["echo \"starting run\""]    ← required
```

Validation rejected it and the model answered in prose on the retry rather
than correcting the shape, so the turn ended with a `DirectResponse` and the
counters saw a task that simply stopped. 6 of 6 generations did this.

The envelope protocol never had the problem because `format_instruction`
shows a *filled* example per tool (`_example_args`), while the `tools` array
carried only the JSON schema — and `"type": "array"` is not enough for a 27B.
This is the same finding `_example_args` was written for. Putting that same
example into each tool's `description` took `simple_replace` from 0/6 to 6/6.

**Reading for the future: when a native-protocol task fails with no error and
fewer calls than it should have, replay the transcript's last decision before
believing the shape the metrics suggest.** A rejected call and a refusal to
act are indistinguishable from the outside; they are not the same bug.

### 7.2 Results (2026-08-13, Qwen3.6-27B-AWQ, same code both sides)

core tier, 3 repeats, n=36 each:

| | envelope | native |
|---|---|---|
| success | 0.944 | **0.972** |
| failed edits / run | 0.056 | **0.028** |
| tool calls / run | 2.028 | **1.944** |
| completion tokens | **3292** | 4423 |
| wall | 337.6s | **198.7s** |

shift tier, 3 repeats, n=9 each: both **1.0**; calls/run 1.889 envelope vs
1.667 native; 853 vs 1043 completion tokens.

One failure against two at n=36 does not separate the two protocols, and
should not be read as native winning. What the run does show is that they
fail *differently*, which matters more than the rate:

- The envelope's failure was a constrained-decoding loop: 223.9s of one run,
  truncated at `max_tokens` mid-`edit_file`, the turn lost. That is two
  thirds of its entire wall time in one run, and it is a tail the native
  channel does not have — there is no grammar to loop. Excluding it, the
  envelope is *faster* per run (3.25s vs 5.5s) and cheaper in tokens.
- `middle_of_large_file` failed once on each side. That one is the task, not
  the protocol.

So: envelope stays the default because it runs on any OpenAI-compatible
backend, and native is a setting worth turning on where the server supports
it — a little more token spend for fewer calls and no grammar-loop tail.

### 7.3 Re-measured on Qwen3.8-27B-FP8 (2026-08-17)

The cluster moved to `Qwen3.8-27B-FP8` on vLLM with `--enable-auto-tool-choice
--tool-call-parser qwen3_coder --reasoning-parser qwen3`, which is what
prompted re-running §7.2. Two rounds, core tier, 3 repeats, **n=36 per
protocol per round**, run **sequentially and in opposite orders** (round 1
envelope→native, round 2 native→envelope) so that drift in cluster load
cannot be mistaken for a protocol effect.

| | envelope | native |
|---|---|---|
| round 1 success | **1.0** | 0.972 |
| round 2 success | **1.0** | **1.0** |
| pooled success (n=72) | **1.0** (0 failures) | 0.986 (1 failure) |
| failed edits / run | 0.069 | **0.056** |
| tool calls / run | 2.028 | **2.000** |
| completion tokens | **6858** | 8541 (+24.5%) |
| wall | **450.8s** | 542.2s (+20.3%) |

**The §7.1 shape bug is gone.** Not one native run mis-shaped an argument —
`old_lines` arrived as a list of strings every time. Some of that credit
belongs to the parser rather than the model: `qwen3_coder` coerces each
`<parameter=…>` block against the tool's JSON schema, so an argument declared
`"type": "array"` is *built* as one.

**Native is nevertheless not better here.** It never won a round, and the two
differences that repeat in both orderings are the ones against it: ~24% more
completion tokens and ~20% more wall time, for the same work. The likely cause
is encoding — `qwen3_coder`'s XML-ish call is more verbose than the envelope's
compact JSON, and the gap is ~11 completion tokens per decision, which is the
right order of magnitude for it. A server running the `hermes` parser would be
worth re-measuring before generalising this; it is a statement about *this*
deployment, not about native tool calling.

The envelope's own §7.2 tail — a constrained-decoding loop costing 224s — did
not recur in either round.

**Decision: envelope stays the default**, now on measured grounds rather than
only portability. Native is correct, tested and one setting away
(`llm.tool_protocol`, or the per-entry override on a backend in the catalog).

### 7.4 What a cut-off call looks like on each channel

The finding that matters more than the table, because it is a capability
difference rather than a few percent. Probed directly (2026-08-17):

```
max_tokens=120   finish_reason=tool_calls   completion_tokens=120   args={dir_key, name}
max_tokens=300   finish_reason=tool_calls   completion_tokens=300   args={dir_key, name}
max_tokens=4096  finish_reason=tool_calls   completion_tokens=3716  args={dir_key, name, content_lines×200}
```

A `create_file` cut off inside `content_lines` does **not** report
`finish_reason: "length"` and does not return the fragment. `qwen3_coder`
drops the argument it was part-way through and hands back a syntactically
perfect call of the arguments it did finish, with `content` null. The only
trace is `completion_tokens == max_tokens`.

Consequences, all reproduced live:

- The envelope's salvage (§ commit 35a0ffd) **cannot** apply on this channel.
  `_salvage_truncated_call` rebuilds a write from its prefix; here there is no
  prefix to rebuild from.
- Before the fix, the truncated call surfaced as an ordinary validation error
  (`content_lines: Field required`). The model re-sent the same 200-line
  write, truncated again, and the turn died on `DecisionError` — verified by
  replaying one request with the guard disabled.
- `middleware._hit_token_cap` now reinterprets *an argument validation that
  failed having spent the entire token budget* as truncation, and feeds the
  skeleton-then-fill message instead. Same request, same cap, after the fix:
  a valid `create_file` of 31 lines (cap 400) and 13 lines (cap 1200) — the
  model writes something that fits rather than losing the turn.
- It is deliberately checked only *after* validation fails, so a complete call
  that happens to end at the cap is still kept (§`test_a_complete_call_at_the
  _cap_is_kept`).
- This is why **no tool may declare an optional array argument**. Scanned at
  the time of writing: none does. If one did, a dropped array would validate
  as its default and the call would execute with the content silently missing
  — the same failure, but acted on instead of caught.

> **Reading this after v0.24.0:** the interface described below is gone. Every
> file tool took a *registry key* then; they take a path now, and `register_path`
> / `list_paths` / `dir_key` / `subpath` no longer exist. The measurements stand
> as the record of what was true when they were taken — including the ones that
> argued the registry was worth keeping — and specs-path-registry.md is where
> the removal and its head-to-head live. The `shift` tier in particular measures
> a friction that no longer has anything to rub against.

## 8. Folding the model's own record (v0.23.3)

A session on 2026-08-19 produced fifteen corrupted files. The model wrote a
24-row annotation TSV, was asked for a second file like it, and sent back the
placeholder that `hpca.agent.history` had put in place of its own payload —
so the placeholder went to disk and the file collapsed to four lines. Folding
*that* left four lines again, so every rewrite confirmed the loss, and the
model concluded its own writer was truncating and spent a dozen rounds
bisecting a bug that did not exist.

New hard-tier task `second_file_after_first`: read a registered CSV, write a
TSV from it, then write a second filtered TSV in the same layout. The source
is a file rather than text in the prompt, because re-deriving from the prompt
is the escape route the real session did not have. `check` requires both files
and no placeholder in either — the row counts alone would pass an abandoned
file, and the marker alone would pass a file of the right length with the
marker in the middle.

Three variants measured against v0.23.2, 10 reps each, Qwen3.8-27B, envelope:

| Variant | success | failed edits/run | tool calls/task |
|---|---|---|---|
| v0.23.2 baseline | 0.8 | 0.2 | 5.6 |
| fold at write time + refuse it back | 0.9 | 1.8 | 5.5 |
| …with the marker no longer quoted in the refusal | 1.0 | 0.0 | 4.0 |
| fold by age (shipped) | **1.0** | **0.0** | **3.4** |

Two findings worth keeping, both invisible to the unit tests:

**A refusal that quotes the offending line feeds the loop it is refusing.**
The first treatment echoed the placeholder back inside its "NOT created"
message, which returned it to the context; the model composed its next call
out of the refusal it had just read and was refused in the same words —
seventeen times in one run, until the decision budget died. Removing the quote
and keeping only the line number took `mean_failed_edits` from 1.8 to 0.0.
This is the retry-loop cost §1 is about, arriving through the one door nobody
watches: the error message itself.

**The skeleton-then-fill protocol was paying for the write-time fold all
along.** `long_specs_write` (§3) writes a long specs.md and then edits it
section by section, so it is the task most exposed to a change in what the
model can see of its own writes — the reason it was checked at all. Measured
separately, 3 reps:

| | success | failed edits/run | tool calls |
|---|---|---|---|
| v0.23.2 baseline | 0/3 | 4.0 | 18.0 |
| fold by age | 2/3 | 0.0 | 11.7 |

The baseline never finishes it. Its `old_lines` stop matching because the
record of what it wrote was folded away the moment it wrote it, so each edit
is a guess at text it can no longer see — 4 rejected edits per run, 18 calls,
no file. This is the same defect as the corrupted-TSV cascade seen from the
other side: there the model copied the placeholder forward, here it cannot
reconstruct what the placeholder replaced. §3 read the residue as "long-horizon
instruction compliance of the 27B"; part of it was the tooling after all.

The remaining 1-in-3 is the documented incompleteness — a section short, or an
honest TBD left — and still not tool mechanics.

**Guarding a failure is worth less than not having it.** Refusing the write
prevented every corrupted file, but cost a livelock in 1 run of 10. Deferring
the fold so recent calls keep their payloads removed the failure at its source
and came out ahead of the baseline on friction as well — the guard stayed, but
as a backstop for sessions checkpointed before the change, not as the
correctness mechanism.

## 9. What it deliberately is not

- Not part of the default or live pytest suites — it costs real generations
  and minutes of wall time; it runs only when someone asks for an assessment.
- Not HITL-gated — handlers are called directly; it measures the model+tool
  contract, not the approval UX.
- Not a benchmark of the model — the fixed reference is the task set; the
  thing under test is HPCA's tooling.

### 6.1 Was the registry's premise ever true?

The registry exists because "models mis-copy long paths". Measured directly
(2026-08-13, same backend, n=20 first decisions): asked to read a
161-character cluster path — `/data/cephfs-1/work/groups/cubi/projects/`
`2026-01-15_scRNAseq_pilot/results/alignment/sample_A12_rep2/outs/`
`filtered_feature_bc_matrix/barcodes_S12_L002_R1_001.tsv`, with repeated
near-identical segments, a date, and sample ids that recur with suffixes —
**20/20 byte-exact**. Constrained decoding gives no help there: inside a JSON
string every character is legal, so the grammar cannot correct a typo. This
model simply copies paths well.

That does not make the registry pointless, it means it is justified by
something else: a key resolves from sqlite whatever is in the window, while a
long path that has scrolled out of a compacted conversation is gone. Keep the
transcription argument out of the rationale; it is not what the numbers show.

Exposure is unchanged by the key-or-path change, because the model transcribes
a path exactly once either way — in v0.19 into `register_path`, now into the
first tool call, whose result hands back the auto-registered key that the
model then uses (verified in a kept transcript). What was lost is one warning:
`create_file` with a literal directory path skips the registration step that
used to say "nothing exists there yet", and it makes missing parents, so a
typo lands the file somewhere plausible and reports success. Hence the caution
in that result — the tool knows it is writing where nothing was.


## 10. Checking the checks (2026-08-24)

A `success_rate` is worth exactly what its checks are worth, and several of
these were worth less than they looked. `simple_replace` — change one line of
a five-line script — was graded like this:

```python
check=lambda ws: 'echo "starting run v2"' in (ws / "runner.sh").read_text()
```

Replaying a transcript by hand (the §7.1 habit, applied to something else)
turned up a run that had written the new line over `set -euo pipefail`: one
line destroyed, the line it was told to change still sitting there untouched,
and `success: true`. **Presence of the wanted text is not absence of the wrong
outcome**, and eight of the thirty-one tasks across the four tiers were graded
on presence alone. The failure mode is quiet by construction — a check that is
too generous does not error, it just returns a number that is too high, which
then goes into a document and gets believed.

### What a task declares now

A `Task` may carry an `expected`: one correct outcome, as whole files, keyed
by workspace-relative name (`None` for a file that has to be gone). It does
two jobs.

It is the **default check** — whole-file equality, which is the only check a
run that also destroyed a neighbouring line cannot satisfy. A `str` compares
through universal newlines (so `crlf_file` grades every character except the
line endings, which are `crlf_preserved`'s business); `bytes` compares byte
for byte. The single tolerance is a missing final newline, a text-file wart no
task here grades.

And it is the **fixture `--self-check` mutates**. Declare it even where the
check itself has to stay tolerant, because tolerance is exactly what needs the
guard. Four checks are still hand-written, each for a reason the whole-file
comparison would get wrong:

| task | why it stays tolerant |
|---|---|
| `trailing_space_trap` | the planted trailing space is the trap, not the grade — whether the new line carries one is the model's business, so the comparison is per line and rstripped |
| `smart_quote_line` | the typography is what the *match* has to absorb, not what the write has to reproduce; the title line is graded on its words, the two lines around it exactly |
| `second_file_after_first` | two files, a header, and a row count — and an elision placeholder is the thing it exists to catch |
| the `create_*` / write tasks | "exactly these two lines" means the content, not blank-line taste (`_wrote_lines`) |

Checks are also wrapped so an exception reads as a failure rather than an
error: most of them read a file the task was supposed to write, and a check
that cannot read it did not see a success.

### The guard

`evals/edit_eval.py --self-check` runs every task's check against a correct
outcome and against every one-line corruption of it, over all four tiers, in
about a second. It runs **before every eval too**, dry or live, and aborts the
run — an eval whose checks cannot tell a correct file from a wrecked one is
worse than no eval, because the number still looks like a measurement.

Two mutation families, because they catch different tolerances:

- **a line dropped** — stands in for a neighbour overwritten, a block deleted,
  an edit landing one line up. This is what caught `simple_replace`.
- **a line added** — content written twice, a placeholder copied back as text,
  a rewrite that kept the old line and appended the new one. This is what a
  `lines[:3] == wanted` check waves through, and one of those was in the
  `paths` tier.

Plus the file missing entirely, and — for a file the task was told to delete —
still being there.

The guard was checked in both directions: with the old presence-only checks
restored in a copy, `--self-check` fails and names `simple_replace` and the
three `create_*` path tasks; against the tightened ones it passes, and
`--dry-run --tier all` still scores 1.0 over 28 runs, so nothing was tightened
past what the tooling actually produces.

### What this costs the numbers already in this document

Every `success_rate` recorded before 2026-08-24 was measured with at least
some presence-only checks and is therefore an **upper bound**, not comparable
with anything measured after. That includes the v0.17.0→v0.18.0 reference
figures, the envelope-vs-native tables in §7, and the arm comparison on
`exp/edit-arms`. The direction of the error is known — old numbers are too
high, never too low — and the effects those runs measured were large enough
that the ranking is unlikely to move; but a future comparison has to be
baseline-vs-treatment on *this* harness, not against a figure quoted here.

### What is still unguarded

`long_specs_write` declares no expected outcome: it asks for 150+ lines of
plan prose, where "one line dropped" is not corruption. Its check stays
structural (every section present, every agreed decision present, no `TBD`),
now also refusing a document carrying an elision placeholder. `--self-check`
names it on every run rather than passing over it silently.

One pre-existing breakage, unrelated: `--dry-run --tier shift` scores 0.0 on
this branch, because each shift route scripts a `register_path` call and there
is no path registry any more. That tier is history rather than a check (see
CLAUDE.md), and its scripts were not repaired.
