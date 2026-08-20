"""Synthetic content, so the UI can be looked at before it is wired to data.

Deliberately over-long: the question this exists to answer is what the UI feels
like once a turn has produced hundreds of steps, and a six-entry conversation
answers nothing. Nothing here opens a database or talks to a backend.
"""

from __future__ import annotations

from hpca.ui.ansi import BLUE, GREEN, RED, YELLOW
from hpca.ui.app import RowUI, SessionState
from hpca.ui.pane import Item

TASKS = [
    "annotate the cohort BAMs with sniffles",
    "why did the snakemake run stall at merge_vcf",
    "write the methods section for the SV paper",
    "check GPU utilisation on the last training job",
    "rebuild the reference index on scratch",
    "compare coverage between the two batches",
]

LONG_ASK = (
    "Use the reference on scratch rather than the one in my home directory, "
    "keep the intermediate BAMs so I can check them afterwards, and if the "
    "merge step stalls again do not retry it silently — stop and tell me which "
    "shards were still open, because last time two of them wrote to the same "
    "temp path and I only found out from the log three hours later."
)

SNIPPETS = [
    "/scratch/proj/cohort/run3/annotation.tsv (412 lines)",
    "wrote 38 lines, backup kept in trash",
    "exit 0 in 2.4s",
    "no match — falling back to a wider search",
    "/home/mayv_c/projects/sv-paper/methods.md",
    "12 files, 3.2 GB",
]

REPLIES = [
    "The merge rule stalled because two shards wrote to the same temp path.",
    "Coverage is 31x in batch A and 28x in batch B; the gap is one flowcell.",
    "I have written the methods section — it is 42 lines, have a look.",
    "The job is still queued behind a reservation; nothing is wrong with it.",
]

SETTINGS_JSON = """{
  "llm": {
    "base_url": "http://localhost:20001/v1",
    "model": "qwen3-27b-fp8",
    "context_window": 112000,
    "temperature": 0.2
  },
  "database": {
    "local_cache": true,
    "sync_interval_s": 60
  },
  "logging": {
    "enabled": true,
    "dir": ""
  },
  "agent": {
    "mode": "agent",
    "max_steps": 40
  }
}
"""

LEARNINGS = {
    "hpc": (
        "The cluster's scratch is /scratch/proj, and $HOME is NFS — never write\n"
        "large intermediates to $HOME.\n"
        "Slurm partitions: gpu (a100), cpu-long, cpu-short.\n"
        "Prefers sniffles over cuteSV for long-read SV calling.\n"
    ),
    "default": "(nothing learned yet)\n",
    "writing": "Writes in British spelling. Dislikes bullet lists in prose.\n",
}


def sample_chat(count: int, seed: int = 0, task: str = "") -> list[Item]:
    """One session's conversation. ``seed`` shifts the content so that two
    sessions never look alike, and ``task`` is what the user keeps asking
    about — a session is about one thing, and that is what makes a switch
    visible at a glance."""
    tools = ["read_file", "edit_file", "create_file", "run_bash", "list_dir"]
    task = task or TASKS[seed % len(TASKS)]
    items: list[Item] = []
    for n in range(count):
        i = n + seed * 7
        slot = n % 3
        if slot == 0:
            # Every fourth one is long, so that the rewind's preview has
            # something to truncate and the wrapping is exercised by running
            # the thing rather than only by the headless tests.
            said = task if n % 12 else f"{task}. {LONG_ASK}"
            items.append(
                Item(head=f"you   {said}", accent=BLUE, kind="user", text=said)
            )
        elif slot == 1:
            steps = 3 + (i * 5) % 18
            names = [tools[(i + k) % len(tools)] for k in range(steps)]
            items.append(
                Item(
                    head=f"      {steps} steps · " + " → ".join(names[:3]) + " …",
                    body=[
                        f"{name:<14}{SNIPPETS[(i + k) % len(SNIPPETS)]}"
                        for k, name in enumerate(names)
                    ],
                )
            )
        else:
            reply = REPLIES[i % len(REPLIES)]
            items.append(
                Item(
                    head=f"hpca  {reply}",
                    body=[
                        reply,
                        "The detail is in the log at "
                        "/scratch/proj/cohort/run3/logs/merge_vcf.log, and the "
                        "two shards are listed at the bottom of it.",
                    ],
                    accent=YELLOW,
                )
            )
    return items


def sample_watchers(count: int, seed: int = 0) -> list[Item]:
    states = [("RUNNING", GREEN), ("PENDING", ""), ("COMPLETED", ""), ("FAILED", RED)]
    out = []
    for n in range(count):
        i = n + seed * 3
        state, accent = states[i % len(states)]
        out.append(
            Item(
                head=(
                    f"{'job ' + str(4821000 + i):<16}{state:<11}"
                    f"last write {3 + i * 11}s ago"
                ),
                body=[
                    f"/scratch/proj/cohort/run{seed}/logs/step{n}.log",
                    "[12:41:07] merging shard 3 of 8",
                    "[12:41:44] merging shard 4 of 8",
                ],
                accent=accent,
            )
        )
    return out


def sample_profiles() -> list[Item]:
    return [
        Item(
            head=f"{'hpc':<18}12 memories · 4 skills · 3 sessions",
            body=["created 2026-04-02", "used by the open session"],
            accent=GREEN,
        ),
        Item(
            head=f"{'default':<18}0 memories · 0 skills · 1 session",
            body=["created 2026-01-11", "the fallback; cannot be deleted"],
        ),
        Item(
            head=f"{'writing':<18}5 memories · 1 skill · 0 sessions",
            body=["created 2026-06-20", "copied from hpc"],
        ),
        Item(head="(new profile)"),
    ]


def sample_llms() -> tuple[list[Item], list[Item]]:
    discovered = [
        Item(
            head="● 10.12.4.31:20001    qwen3-27b-fp8      112k ctx",
            body=["found in the shared endpoints manifest", "node gpu014, 42s ago"],
            accent=GREEN,
        ),
        Item(
            head="● localhost:20001     qwen3-27b-fp8      112k ctx",
            body=["found by localhost scan (ssh tunnel)"],
            accent=GREEN,
        ),
        Item(
            head="○ 10.12.4.55:20001    llama-3.3-70b       128k ctx",
            body=["in the manifest, did not answer the probe"],
        ),
    ]
    # The whole width, which is the point of stacking these rather than
    # putting them in a column: url, model and context all fit on one line.
    configured = [
        Item(
            head="★ ● cluster-qwen   http://10.12.4.31:20001/v1   qwen3-27b-fp8   112k",
            body=["the active default", "answered the last probe 42s ago"],
            accent=GREEN,
        ),
        Item(
            head="  ● tunnel-qwen    http://localhost:20001/v1     qwen3-27b-fp8   112k",
            body=["the same server through an ssh tunnel"],
            accent=GREEN,
        ),
        Item(
            head="  ○ big-llama      http://10.12.4.55:20001/v1    llama-3.3-70b   128k",
            body=["not answering; last seen 3d ago"],
        ),
    ]
    return discovered, configured


def build(chat: int = 400, sessions: int = 14, watchers: int = 5) -> RowUI:
    states = [
        SessionState(
            title=TASKS[i % len(TASKS)],
            profile=("hpc", "writing", "default")[i % 3],
            model="qwen3-27b-fp8" if i % 2 == 0 else "llama-3.3-70b",
            started=f"{2 + i * 7}m ago",
            # Varied on purpose: a session with a handful of entries next to
            # one with hundreds is what shows the chat row taking up the slack.
            size=max(6, chat // (1 + i % 4)),
            # And some with no watches at all, which is the case the layout
            # charges nothing for.
            watch_count=watchers if i == 0 else (i * 3) % 4,
            index=i,
        )
        for i in range(sessions)
    ]
    return RowUI(
        states,
        learnings=dict(LEARNINGS),
        settings_json=SETTINGS_JSON,
        llms=sample_llms(),
    )
