# edit_eval — how well the live model drives edit_file / read_file / create_file

This is the standing instrument for assessing any change to the file tools,
the editing guidance, or the backend — method, measured v0.17.0→v0.18.0
results, and the quick "baseline ref vs working tree" recipe live in
[specs-edit-eval.md](../specs-edit-eval.md).

`edit_eval.py` runs file-editing tasks in two tiers — `--tier core` (default;
12 everyday tasks: one-line replacement, mid-file edits in a ~300-line file,
CRLF, unicode context, append idiom, multi-line block, duplicated-block
disambiguation, nested indentation, line deletion, create_file, config value,
planted trailing-space trap), `--tier hard` (shapes the old tooling
structurally mishandled: an edit deep in a 700-line file, byte-preserved CRLF,
a typographic-quote line), or `--tier all` — against the live LLM
through HPCA's real middleware decision loop and real tool handlers (real
ToolContext, PathRegistry, TrashManager) in a throwaway temp workspace per
run. HITL gating is bypassed: handlers are called directly.

Per task × repeat it records: success (predicate on the final file content),
tool calls, failed/rejected edit calls ("NOT edited" / validation failures),
decisions, completion tokens, wall time — written to a JSON file, with an
aggregate summary printed at the end.

## Prerequisites

- A live OpenAI-compatible backend (vLLM). Default `http://localhost:20001/v1`,
  usually an SSH tunnel to the cluster; override with `HPCA_TEST_LLM_URL`.
- **`HPCA_TEST_LLM_KEY` must be exported** if the endpoint is key-locked —
  otherwise the probe gets a 401 and the script exits with code 2.
- Optional: pin the model with `HPCA_TEST_LLM_MODEL` (otherwise discovered
  from `/v1/models`).

## Sanity check (no backend needed)

```sh
pixi run -e dev python evals/edit_eval.py --dry-run
```

Runs every task once against a scripted fake model that emits a correct call
per task — proves the plumbing (registry, context, handlers, predicates)
end-to-end. Expect success_rate 1.0.

## Treatment run (this branch)

```sh
export HPCA_TEST_LLM_KEY=...   # if the endpoint is key-locked
pixi run -e dev python evals/edit_eval.py \
    --out evals/results_treatment.json --repeats 3 --label treatment
```

## Baseline run (main)

Make a worktree of main and run the *same eval script* against main's code by
copying the script in and pointing `PYTHONPATH` at the worktree's `src`
(edit_eval.py prepends `<script dir>/../src` to `sys.path` itself, so copying
it into the worktree is the simplest way to get the worktree's hpca):

```sh
git worktree add /tmp/hpca-baseline main
mkdir -p /tmp/hpca-baseline/evals
cp evals/edit_eval.py /tmp/hpca-baseline/evals/
export HPCA_TEST_LLM_KEY=...
# same environment (the dev env has all runtime deps); PYTHONPATH belt-and-braces:
PYTHONPATH=/tmp/hpca-baseline/src pixi run -e dev \
    python /tmp/hpca-baseline/evals/edit_eval.py \
    --out evals/results_baseline.json --repeats 3 --label baseline
git worktree remove /tmp/hpca-baseline
```

Then compare the two JSON files' `summary` blocks (success_rate,
mean_failed_edits are the headline numbers).

## Options

- `--tier core|hard|all` — task tier (default core).
- `--repeats N` — repeats per task (default 3; dry-run always runs 1).
- `--tasks N` — only the first N tasks (quick smoke).
- `--only SUBSTR` — only tasks whose name contains SUBSTR.
- `--keep` — keep each run's workdir and dump `transcript.json` into it
  (diagnosis: read what the model actually saw and wrote).
- `--label name` — recorded in the output JSON.
- Exit codes: 0 ok; 2 backend unreachable or 401.

## A negative result to remember

The `smart_quote_line` hard task was built to catch a model transcribing
typographic quotes/dashes as ASCII — and it never fired: Qwen3.6-27B copies
them faithfully, and the task passed 3/3 even on the pre-fuzzy v0.17.0 code.
The trap (and the fuzzy ladder's unicode level it was meant to justify) is
insurance for other backends, perhaps not needed at all; don't grow that
machinery without a measured failure first.

---

# skill_draft_eval — how well the live model drafts a skill from one line

The instrument for `/skill-creator <what it should do>` (§5.1): ten requests a
user would plausibly type, run through the real
`hpca.agent.skill_drafter.propose_skill` against a live backend. Two of the ten
carry a synthetic conversation, because "write a skill based on our
conversation" is the case the feature exists for; one plants a job id and a
one-off path in the request to check they do not end up in the skill.

Each draft is graded on the five things the form actually needs — **invocable**
(kebab-case name that survives as a slash command), **described** (a one-line
description that is not the name again), **procedural** (a body with real
steps, not a sentence), **on topic** (a concrete term the request implies shows
up), **generalised** (today's ids and paths did not leak into name or body).

```sh
export HPCA_TEST_LLM_KEY=...   # if the endpoint is key-locked
pixi run -e dev python evals/skill_draft_eval.py --dry-run     # plumbing only
pixi run -e dev python evals/skill_draft_eval.py --repeats 2 --out results.json
pixi run -e dev python evals/skill_draft_eval.py --show        # print each draft
```

Options: `--repeats N` (default 1), `--only SUBSTR`, `--label`, `--out`,
`--show`, `--dry-run`. Exit codes as above: 0 ok, 2 backend unreachable or 401.

## Measured, v0.21.0

Qwen3.6-27B-AWQ, 10 tasks × 2 repeats: **20/20 on all five checks**, mean 9.2
body lines, mean 7.3s per draft (total 146s). One draft per command is the
whole cost, so the form appears in under ten seconds.

## Two prompt findings from building it

Both were failures the eval caught, fixed in `skill_drafter.SYSTEM_PROMPT`:

- **"one line" is not a length.** Descriptions came back at 130-140 characters
  and were being truncated mid-sentence into the form; asking for *at most 25
  words* and raising the cleaner's cap to the schema's own 200 fixed it.
- **Forbidding specifics in general does nothing; forbidding them where the
  model is writing does.** "Today's job ids do not belong in the skill", placed
  as a closing rule, still left `run17` in the body of 2 of 3 drafts as an
  "e.g."; moving the constraint into the *body* bullet ("write a placeholder —
  `<job-id>` — never the actual id from the request, not even as an example")
  took it to 3/3, without the model dropping the real tools the conversation
  had established (`papermill`, `--mem`, `cProfile`, `line_profiler` all
  survive, parameterised).

---

# smoke_pty — does the real binary come up in a real terminal?

Everything else in `tests/` drives the UI through a harness that never opens a
pty: `Screen` writes into a buffer, keys arrive as method calls, and the
terminal is assumed. The first run on a cluster node is where that assumption
gets tested for the first time. This opens a real pty, runs the real console
script under it, feeds it a quit, and prints what it painted.

It answers the one question the suite cannot: what does a fresh install do
when **nothing answers**? Two scenarios, two failures wearing the same face:

```sh
pixi run -e dev python evals/smoke_pty.py bare 14   # empty $HPCA_HOME
pixi run -e dev python evals/smoke_pty.py down 14   # settings name a dead port
```

`bare` is a UI question — does it tell the user, and offer the fix? `down` is
a networking one: a TCP connect to a port with nothing behind it is the
startup path most likely to *hang* rather than fail, and a hang here is a UI
that never draws. Both must reach a painted frame, an open manage-LLMs
screen, and exit 0. Exit code is 0 when clean, 1 on a traceback or a non-zero
child; the throwaway `$HPCA_HOME` is kept only on failure, so there is
something to look at when there is.

No backend needed — the point is that there isn't one. It does sweep the
network for endpoints, so what the discovered list contains depends on the
box, and on a dev machine with tunnels up it will not be empty.

## Measured, v0.25.0 (before the Textual deletion)

Both scenarios: full layout painted, the no-backend warning opened by itself,
the sweep finished (`6 endpoint(s) found` on a box with local ollama and two
tunnels), the exit sync printed its wait message, exit 0, no traceback.
Frame times in the status line ran 0.08–0.53 ms.

One thing to know before reading the output as a bug: with a `settings.toml`
naming a backend, the *configured* panel still says "no backends configured".
That panel lists the **catalog**, which is a different list from
`settings.llm` — correct, and confusing exactly once.
