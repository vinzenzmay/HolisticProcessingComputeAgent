#!/usr/bin/env python
"""skill_draft_eval — how well the live model drafts a skill from a one-line
request (`/skill-creator <what it should do>`).

Ten requests a user would plausibly type, run through the REAL
``hpca.agent.skill_drafter.propose_skill`` against a live backend. Per draft
it checks what the form actually needs:

- **invocable**   the name survives as a slash command: kebab-case, non-empty,
                  short enough for the field.
- **described**   a one-line description that is not just the name again.
- **procedural**  a body with real substance — several lines or steps, not a
                  sentence of throat-clearing.
- **on topic**    at least one of the concrete terms the request implies
                  (``sbatch``, ``papermill``, …) shows up in the draft. A
                  fluent draft about the wrong thing is a failed draft.
- **generalised** today's specifics (a literal job id, a one-off path) did not
                  become part of the name.

Two of the ten carry a synthetic conversation, because "write a skill based on
our conversation" is the case the feature was asked for: the check there is
that the draft picked up what was actually said.

Usage
-----
    export HPCA_TEST_LLM_KEY=...          # if the endpoint is key-locked
    pixi run -e dev python evals/skill_draft_eval.py --out evals/results.json

    pixi run -e dev python evals/skill_draft_eval.py --dry-run   # no backend

Exit codes: 0 ran, 2 backend unreachable or key-locked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Run against this checkout's hpca without installing it — and, when copied
# into a worktree of another ref, against that ref's (see edit_eval.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from hpca.agent.skill_drafter import propose_skill  # noqa: E402
from hpca.config import LLMSettings  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402

LIVE_URL = os.environ.get("HPCA_TEST_LLM_URL", "http://localhost:20001/v1")
LIVE_KEY = os.environ.get("HPCA_TEST_LLM_KEY")
PROBE_TIMEOUT_S = float(os.environ.get("HPCA_TEST_LLM_PROBE_TIMEOUT", "30"))

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MIN_BODY_CHARS = 120
MIN_BODY_LINES = 3


# --------------------------------------------------------------------- tasks


@dataclass
class Task:
    """One `/skill-creator <request>` a user might type."""

    name: str
    request: str
    #: concrete terms the draft must show it understood — any one counts.
    expects: tuple[str, ...]
    #: the open conversation, when the request leans on it.
    messages: list[dict] = field(default_factory=list)
    #: session-specific noise that must not survive into the draft.
    forbidden_in_name: tuple[str, ...] = ()


def build_tasks() -> list[Task]:
    return [
        Task(
            name="jupyter_monitor",
            request=(
                "write a skill that starts and monitors jupyter notebook "
                "executions"
            ),
            expects=("notebook", "jupyter", "papermill", "nbconvert", "kernel"),
        ),
        Task(
            name="from_conversation",
            request="write a skill based on our conversation",
            expects=("papermill", "notebook", "mem", "oom"),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "my notebook runs keep dying halfway through on the "
                        "cluster"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "That is the OOM killer. Run them headless with "
                        "papermill under sbatch instead of a live kernel: "
                        "`papermill in.ipynb out.ipynb -p n 100`, submitted "
                        "with --mem=32G. Then watch the job with `squeue -j "
                        "<id>` and read out.ipynb for the failing cell."
                    ),
                },
                {"role": "user", "content": "that worked, --mem=32G was the fix"},
            ],
        ),
        Task(
            name="submit_job",
            request=(
                "a skill for submitting a batch job to slurm and watching it "
                "until it finishes"
            ),
            expects=("sbatch", "squeue", "slurm", "sacct", "job"),
        ),
        Task(
            name="failed_job_triage",
            request="skill: work out why a slurm job failed",
            expects=("sacct", "squeue", "slurm-", "exit", "log", "seff"),
        ),
        Task(
            name="conda_env",
            request=(
                "make a skill that sets up a reproducible conda environment "
                "for a project"
            ),
            expects=("conda", "environment.yml", "mamba", "env"),
        ),
        Task(
            name="rsync_transfer",
            request=(
                "a skill to move a large dataset between my laptop and the "
                "cluster scratch filesystem"
            ),
            expects=("rsync", "scp", "scratch", "transfer"),
        ),
        Task(
            name="gpu_check",
            request="skill for checking which gpus are free before I submit",
            expects=("gpu", "sinfo", "nvidia-smi", "gres", "squeue"),
        ),
        Task(
            name="quota_cleanup",
            request=(
                "write a skill for when I hit my disk quota — find what is "
                "big and clean it up safely"
            ),
            expects=("quota", "du ", "df ", "disk", "space"),
        ),
        Task(
            name="conversation_profiling",
            request=(
                "turn what we just did into a skill I can reuse next time"
            ),
            expects=("profil", "cprofile", "bottleneck", "line_profiler", "hot"),
            messages=[
                {
                    "role": "user",
                    "content": "my simulation is slow and I do not know why",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Profile before you optimise: `python -m cProfile -o "
                        "run.prof sim.py`, then sort by cumulative time with "
                        "pstats. If one line dominates, narrow it down with "
                        "line_profiler's @profile decorator rather than "
                        "guessing."
                    ),
                },
                {
                    "role": "user",
                    "content": "cProfile showed it was all in the neighbour loop",
                },
            ],
        ),
        Task(
            name="specific_case_generalises",
            request=(
                "a skill from this: I had to rerun job 4471902 in "
                "/scratch/mayv/run17 after it timed out, with a longer "
                "walltime and a checkpoint restart"
            ),
            expects=("walltime", "time", "checkpoint", "restart", "requeue"),
            # The name must describe the class of work, not this incident.
            forbidden_in_name=("4471902", "run17", "mayv"),
        ),
    ]


# -------------------------------------------------------------------- checks


def grade(task: Task, draft) -> dict:
    """Which of the five checks this draft passed, and why not."""
    name, description, body = draft.name, draft.description, draft.body
    haystack = f"{name} {description} {body}".lower()
    lines = [line for line in body.splitlines() if line.strip()]

    checks = {
        "invocable": bool(NAME_RE.match(name)) and len(name) <= 40,
        "described": bool(description)
        and "\n" not in description
        and description.strip("- ").lower() != name.replace("-", " "),
        "procedural": len(body) >= MIN_BODY_CHARS and len(lines) >= MIN_BODY_LINES,
        "on_topic": any(term.lower() in haystack for term in task.expects),
        # Neither the handle nor the procedure may be about this one incident:
        # a skill that names today's job id is dead the moment it is saved.
        "generalised": not any(
            bad.lower() in f"{name}\n{body}".lower()
            for bad in task.forbidden_in_name
        ),
    }
    return checks


# ------------------------------------------------------------------ backends


def discover_backend() -> str:
    """The model the endpoint serves, or a diagnosis and exit 2."""
    headers = {"Authorization": f"Bearer {LIVE_KEY}"} if LIVE_KEY else {}
    override = os.environ.get("HPCA_TEST_LLM_MODEL")
    try:
        response = httpx.get(
            f"{LIVE_URL}/models", timeout=PROBE_TIMEOUT_S, headers=headers
        )
    except Exception as e:
        print(f"No backend at {LIVE_URL}: {e}")
        print("Set HPCA_TEST_LLM_URL, or use --dry-run.")
        sys.exit(2)
    if response.status_code == 401:
        print(f"{LIVE_URL} rejected the request (401). Export HPCA_TEST_LLM_KEY.")
        sys.exit(2)
    if response.status_code != 200:
        print(f"{LIVE_URL} answered {response.status_code}.")
        sys.exit(2)
    if override:
        return override
    data = response.json().get("data", [])
    if not data:
        print(f"{LIVE_URL} serves no models.")
        sys.exit(2)
    return data[0]["id"]


def make_live_llm(model: str) -> LLMClient:
    settings = LLMSettings(
        base_url=LIVE_URL,
        model=model,
        api_key=LIVE_KEY,
        request_timeout_s=180,
        # propose_skill forces thinking off per call; this only sets the default.
        enable_thinking=False,
    )
    print(f"Backend: {LIVE_URL}  model: {model}")
    return LLMClient(settings)


class FakeLLM:
    """--dry-run: proves the plumbing (tasks, grading, reporting) with no
    backend. Answers every task with a draft built from its own request."""

    def __init__(self, tasks: list[Task]) -> None:
        self._by_request = {t.request: t for t in tasks}

    async def chat(self, messages, **kwargs):
        request = messages[-1]["content"]
        task = next(
            (t for r, t in self._by_request.items() if r in request), None
        )
        term = task.expects[0] if task else "thing"
        content = json.dumps(
            {
                "name": f"handle {term}".strip(),
                "description": f"When the user needs to {term} on the cluster",
                "body": (
                    f"1. Check the state of the {term} first.\n"
                    f"2. Run the {term} step, and read the output.\n"
                    "3. If it fails, report the error rather than retrying blind."
                ),
            }
        )

        class _Resp:
            def __init__(self, content):
                self.content = content
                self.usage = {"completion_tokens": 0}

        return _Resp(content)

    async def close(self):
        return None


# ---------------------------------------------------------------------- runs


async def run_task(task: Task, llm, repeat: int) -> dict:
    started = time.monotonic()
    record = {
        "task": task.name,
        "repeat": repeat,
        "request": task.request,
        "with_conversation": bool(task.messages),
    }
    try:
        draft = await propose_skill(llm, task.request, messages=task.messages)
    except Exception as e:  # DraftError is expected; a timeout is data too
        record.update(
            success=False,
            error=f"{type(e).__name__}: {e}",
            checks={},
            wall_s=round(time.monotonic() - started, 1),
        )
        return record
    checks = grade(task, draft)
    record.update(
        success=all(checks.values()),
        error="",
        checks=checks,
        failed=[k for k, ok in checks.items() if not ok],
        name=draft.name,
        description=draft.description,
        body=draft.body,
        body_lines=len([line for line in draft.body.splitlines() if line.strip()]),
        wall_s=round(time.monotonic() - started, 1),
    )
    return record


def summarise(runs: list[dict], label: str, model: str) -> dict:
    graded = [r for r in runs if r["checks"]]
    checks = ("invocable", "described", "procedural", "on_topic", "generalised")
    # Empty when every draft errored out: there is nothing to rate, and a
    # zero rate would read as "the model got them all wrong".
    per_check = {
        check: round(sum(1 for r in graded if r["checks"][check]) / len(graded), 3)
        for check in checks
    } if graded else {}
    return {
        "label": label,
        "model": model,
        "n_runs": len(runs),
        "success_rate": round(sum(r["success"] for r in runs) / len(runs), 3),
        "per_check_rate": per_check,
        "mean_body_lines": round(
            sum(r.get("body_lines", 0) for r in graded) / max(len(graded), 1), 1
        ),
        "mean_wall_s": round(sum(r["wall_s"] for r in runs) / len(runs), 1),
        "total_wall_s": round(sum(r["wall_s"] for r in runs), 1),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the full JSON record here")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", help="run only tasks whose name contains this")
    parser.add_argument("--label", default="treatment")
    parser.add_argument("--dry-run", action="store_true", help="no backend")
    parser.add_argument("--show", action="store_true", help="print each draft")
    args = parser.parse_args()

    tasks = build_tasks()
    if args.only:
        tasks = [t for t in tasks if args.only in t.name]
    if not tasks:
        print("No tasks matched.")
        return 2

    if args.dry_run:
        model, llm = "fake", FakeLLM(tasks)
    else:
        model = discover_backend()
        llm = make_live_llm(model)

    print(f"{len(tasks)} tasks x {args.repeats} repeat(s)\n")
    runs = []
    try:
        for repeat in range(args.repeats):
            for task in tasks:
                record = await run_task(task, llm, repeat)
                runs.append(record)
                mark = "ok " if record["success"] else "FAIL"
                detail = record["error"] or (
                    f"“{record['name']}” — {record['body_lines']} lines"
                    + (
                        f"  [{', '.join(record['failed'])}]"
                        if record["failed"]
                        else ""
                    )
                )
                print(f"[{mark}] {task.name:<28} {detail}  {record['wall_s']}s")
                if args.show and not record["error"]:
                    print(f"       {record['description']}")
                    for line in record["body"].splitlines():
                        print(f"       | {line}")
    finally:
        await llm.close()

    summary = summarise(runs, args.label, model)
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
