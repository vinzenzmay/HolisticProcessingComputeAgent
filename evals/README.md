# edit_eval — how well the live model drives edit_file / read_file / create_file

This is the standing instrument for assessing any change to the file tools,
the editing guidance, or the backend — method, measured v0.17.0→v0.18.0
results, and the quick "baseline ref vs working tree" recipe live in
[specs-edit-eval.md](../specs-edit-eval.md).

`edit_eval.py` runs file-editing tasks in two tiers — `--tier core` (default;
12 everyday tasks: one-line replacement, mid-file edits in a ~300-line file,
CRLF, unicode context, append idiom, multi-line block, duplicated-block
disambiguation, Python indentation, line deletion, create_file, config value,
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
- `--label name` — recorded in the output JSON.
- Exit codes: 0 ok; 2 backend unreachable or 401.

## A negative result to remember

The `smart_quote_line` hard task was built to catch a model transcribing
typographic quotes/dashes as ASCII — and it never fired: Qwen3.6-27B copies
them faithfully, and the task passed 3/3 even on the pre-fuzzy v0.17.0 code.
The trap (and the fuzzy ladder's unicode level it was meant to justify) is
insurance for other backends, perhaps not needed at all; don't grow that
machinery without a measured failure first.
