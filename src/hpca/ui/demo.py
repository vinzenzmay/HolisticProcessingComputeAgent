"""Synthetic content, and a core that serves it over the real protocol.

    pixi run -e dev python -m hpca.ui.run --demo

Deliberately over-long: the question this exists to answer is what the UI feels
like once a turn has produced hundreds of steps, and a six-entry conversation
answers nothing. Nothing here opens a database or talks to a backend.

What changed at M2 is *how* the content reaches the screen. The UI used to
manufacture its own chat — `SessionState` imported this module to make one up
— which meant the seam between the renderer and the data was a fiction and the
demo could not tell you whether the real one worked. Now `DemoCore` answers
`protocol.Command`s with `protocol.Event`s, exactly as `AgentService` will
(M3), and the demo goes through `ui.client` and `ui.state` like anything else.
Running `--demo` is therefore a test of the seam: if the client stopped
handling `chat.reset`, the demo would be empty.

The loopback is synchronous — commands are handed to `DemoCore.handle` in place
rather than put on a queue — so `--demo` still needs no event loop. The wire
shapes are identical either way; `tests/test_ui_client.py` drives the same
client over a real `InProcessConnection`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from hpca import protocol
from hpca.ui.app import RowUI
from hpca.ui.client import UIClient
from hpca.ui.state import ProfileInfo, SkillInfo

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

PROFILES = ["hpc", "writing", "default"]
# The real three (`hpca.agent.modes.MODES`), so that the mode bar's colours
# and its hints are the ones a user will actually see.
MODES = ["auto", "manual", "full-auto"]

# What a conversation is called before anything has been said in it — the
# store's placeholder, and what a session made from `(new session)` shows.
UNTITLED = "(untitled)"
# Two backends, so switching session moves the model line (§4.3 item 20) and
# not only the title.
MODELS = ["qwen3-27b-fp8", "llama-3.3-70b", ""]

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

ARCHIVE_TEXT = (
    "## [rag] aged out 2026-03-01\n"
    "The old queue names before the January reorganisation: gpu-a100, gpu-v100.\n"
)

SKILL_TEXT = (
    "---\nname: merge-vcfs\ndescription: how this lab merges shard VCFs\n---\n\n"
    "1. bcftools merge the shards in the order the manifest lists them.\n"
    "2. Never write the intermediate to $HOME.\n"
)

# What HPCA ships, as the wide `skill.list` scope reports them: callable from
# the "/" menu on a fresh install, and not any one profile's to edit or
# remove. The names are the real package's (`hpca/data/skills`).
SHIPPED_SKILLS = (
    ("plan", "break a piece of work into steps before starting it"),
    ("grillme", "interrogate a plan for what it has not thought about"),
)

# How often each command has been run, as `command.counts` reports it. Held by
# the core rather than by the UI, which is the whole point of that event: the
# table is the core's and rule 2 of §4.2 keeps a front-end out of it.
COMMAND_COUNTS: dict[str, int] = {"compact": 5, "memorize": 2, "conclude": 1}

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

TOOLS = ["read_file", "edit_file", "create_file", "run_bash", "list_dir"]

# The gated call one demo session is parked on (§4.3 item 21). Built the way
# the graph builds one: an execution gate in manual mode, with the script the
# user is actually deciding about, the tool's schema blurb it must NOT show, a
# plumbing argument it must drop, and the script's own lines among the
# arguments so the prompt can be seen not repeating them above the block.
DEMO_DECISION = {
    "tool": "run_bash",
    "kind": "execution",
    "description": (
        "Run a registered script as a tracked background process for work "
        "that outlives this turn; returns a process id to poll."
    ),
    "arguments": {
        "key": "merge_vcf",
        "content_lines": ["set -euo pipefail", "bcftools merge -o merged.vcf"],
        "timeout_s": 600,
    },
    "script": (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "workdir=/scratch/proj/cohort/run3\n"
        "rm -f $workdir/tmp/shard_*.partial\n"
        "bcftools merge -Oz -o $workdir/merged.vcf.gz $workdir/shards/*.vcf.gz"
    ),
}
WATCH_STATES = [
    ("RUNNING", "watch watch-live"),
    ("PENDING", "watch watch-idle"),
    ("COMPLETED", "watch watch-done"),
    ("FAILED", "watch watch-dead"),
]


def sample_entries(count: int, seed: int = 0, task: str = "") -> list[protocol.Entry]:
    """One session's conversation, as the core would send it.

    ``seed`` shifts the content so that two sessions never look alike, and
    ``task`` is what the user keeps asking about — a session is about one
    thing, and that is what makes a switch visible at a glance.

    ``seq`` numbers every row from 1, which is what a `chat.reset` promises;
    ``index`` is set on the user's own messages only, and to the entry's
    position, because that is what a real thread's message index comes to when
    every message is one entry.
    """
    task = task or TASKS[seed % len(TASKS)]
    entries: list[protocol.Entry] = []
    for n in range(count):
        i = n + seed * 7
        slot = n % 3
        if slot == 0:
            # Every fourth one is long, so that the rewind's preview has
            # something to truncate and the wrapping is exercised by running
            # the thing rather than only by the headless tests.
            said = task if n % 12 else f"{task}. {LONG_ASK}"
            entries.append(
                protocol.Entry(kind="user", text=said, seq=n + 1, index=n)
            )
        elif slot == 1:
            steps = 3 + (i * 5) % 18
            names = [TOOLS[(i + k) % len(TOOLS)] for k in range(steps)]
            entries.append(
                protocol.Entry(
                    kind="thinking",
                    text="",
                    seq=n + 1,
                    steps=steps,
                    parts=[
                        protocol.Part(
                            kind="call",
                            text="",
                            tool=name,
                            result=SNIPPETS[(i + k) % len(SNIPPETS)],
                            done=True,
                        )
                        for k, name in enumerate(names)
                    ],
                )
            )
        else:
            reply = REPLIES[i % len(REPLIES)]
            entries.append(
                protocol.Entry(
                    kind="assistant",
                    seq=n + 1,
                    text=(
                        f"{reply}\nThe detail is in the log at "
                        "/scratch/proj/cohort/run3/logs/merge_vcf.log, and the "
                        "two shards are listed at the bottom of it."
                    ),
                )
            )
    return entries


def sample_watches(count: int, seed: int = 0) -> list[protocol.PanelRow]:
    """The right column for one session, as `panel.update` carries it."""
    rows = []
    for n in range(count):
        i = n + seed * 3
        state, classes = WATCH_STATES[i % len(WATCH_STATES)]
        rows.append(
            protocol.PanelRow(
                key=f"w{4821000 + i}",
                title=f"job {4821000 + i}",
                text=(
                    f"{state:<11}last write {3 + i * 11}s ago\n"
                    f"/scratch/proj/cohort/run{seed}/logs/step{n}.log\n"
                    "[12:41:07] merging shard 3 of 8\n"
                    "[12:41:44] merging shard 4 of 8"
                ),
                classes=classes,
                kind=protocol.PANEL_WATCH,
                ref=str(4821000 + i),
            )
        )
    return rows


def sample_profiles() -> list[ProfileInfo]:
    """The profiles screen's rows, exactly as `profile.rows` carries them.

    Counts and flags and no bodies: each of the two editable files is fetched
    when its editor opens (`profile.get`, `skill.get`), and `DemoCore` answers
    those from `LEARNINGS`, `ARCHIVE_TEXT` and `SKILL_TEXT` below.
    """
    return [
        ProfileInfo(
            name="hpc",
            memories=12,
            sessions=3,
            working=True,
            skills=[
                SkillInfo("merge-vcfs", "how this lab merges shard VCFs"),
                SkillInfo("submit-gpu", "the partition and the flags that work"),
            ],
        ),
        ProfileInfo(name="default", memories=0, sessions=1, default=True),
        ProfileInfo(name="writing", memories=5, copied_from="hpc"),
    ]


def sample_catalog() -> list[protocol.LLMEntry]:
    """The catalog, as `llm.catalog` carries it — labels and marks, no keys.

    One entry deliberately unprobed and one deliberately down, because the
    three-state `reachable` is the thing the manage-LLMs panel has to draw
    correctly and two of the three states are easy to conflate.
    """
    return [
        protocol.LLMEntry(
            label="qwen3-27b-fp8 @ 10.12.4.31:20001",
            model="qwen3-27b-fp8",
            base_url="http://10.12.4.31:20001/v1",
            max_model_len=112000,
            active=True,
            reachable=True,
        ),
        protocol.LLMEntry(
            label="qwen3-27b-fp8 @ localhost:20001",
            model="qwen3-27b-fp8",
            base_url="http://localhost:20001/v1",
            max_model_len=112000,
            reachable=None,
        ),
        protocol.LLMEntry(
            label="llama-3.3-70b",
            model="llama-3.3-70b",
            base_url="http://10.12.4.55:20001/v1",
            max_model_len=128000,
            needs_key=True,
            reachable=False,
        ),
    ]


# What the demo's first scan turns up, one row per frame. Neither shares a
# base_url with `sample_catalog`, because a discovered row for something
# already configured is the duplicate the screen dedups away — worth having a
# demo that shows two rows rather than one that silently shows none.
DISCOVERED = [
    protocol.LLMEntry(
        label="mistral-small @ localhost:20003",
        model="mistral-small-3.1",
        base_url="http://localhost:20003/v1",
        max_model_len=32000,
        reachable=True,
        discovered=True,
    ),
    protocol.LLMEntry(
        label="(api key required) @ 10.12.4.90:20001",
        model="(api key required)",
        base_url="http://10.12.4.90:20001/v1",
        needs_key=True,
        reachable=True,
        discovered=True,
    ),
]


class DemoCore:
    """A core made of lists: it answers commands with the events a real one
    would, and holds the transcripts so that a fork or a rollback means
    something.

    Every command the M2 UI can send has an answer here, and nothing else does
    — a command this does not know is answered with a warning `notify`, the
    same channel `AgentService` uses for one it cannot carry out.
    """

    def __init__(self, *, chat: int = 400, sessions: int = 14, watchers: int = 5):
        self.emit: Callable[[protocol.Event], None] = lambda event: None
        self.rows: list[protocol.SessionRow] = [
            protocol.SessionRow(
                session_id=f"9f3c{i:04x}",
                title=TASKS[i % len(TASKS)],
                profile=PROFILES[i % len(PROFILES)],
                mode=MODES[i % len(MODES)],
                model=MODELS[i % len(MODELS)],
            )
            for i in range(sessions)
        ]
        # Varied on purpose: a session with a handful of entries next to one
        # with hundreds is what shows the chat row taking up the slack. And
        # some with no watches at all, which is the case the layout charges
        # nothing for.
        self._sizes = {
            row.session_id: max(6, chat // (1 + i % 4))
            for i, row in enumerate(self.rows)
        }
        self._watch_counts = {
            row.session_id: watchers if i == 0 else (i * 3) % 4
            for i, row in enumerate(self.rows)
        }
        self._entries: dict[str, list[protocol.Entry]] = {}
        self._watches: dict[str, list[protocol.PanelRow]] = {}
        # What `profile.list` and `skill.list` answer. Held by the core rather
        # than handed to the UI, which is the whole shape M8 restored: every
        # list and every body the screens draw comes down the wire.
        self.profiles = sample_profiles()
        self._forks = 0
        self._made = 0
        # One session is left mid-turn, because a turn in flight is the thing
        # M4b builds and a demo that only ever shows finished conversations
        # cannot show it: opening this one starts the spinner, hands it a tool
        # that never answers, and waits to be interrupted.
        self._busy = self.rows[0].session_id if self.rows else ""
        # And one left parked on an approval, for the same reason: the inline
        # prompt is what M5 builds. The fifth row rather than the second, and
        # deliberately: it is a session in manual mode, which is the gate the
        # payload above is, and it is far enough down the list that the "!"
        # has to be noticed in the sidebar rather than being what opens first.
        self._parked = self.rows[4].session_id if len(self.rows) > 4 else ""
        self._live: dict[str, protocol.Entry] = {}
        # What the manage-LLMs screen has done to the catalog: the scans it
        # has run, what they turned up, and what `r` removed.
        self._scans = 0
        self._found: list[protocol.LLMEntry] = []
        self._removed: set[str] = set()

    # ------------------------------------------------------------- content

    def _index(self, session_id: str) -> int:
        return next(
            (i for i, r in enumerate(self.rows) if r.session_id == session_id), 0
        )

    def entries(self, session_id: str) -> list[protocol.Entry]:
        """Built on first open, as a real core reads a thread on switch:
        fourteen sessions of four hundred steps should not all exist because
        one of them is open."""
        if session_id not in self._entries:
            index = self._index(session_id)
            self._entries[session_id] = sample_entries(
                self._sizes.get(session_id, 0),
                index,
                self.rows[index].title,
            )
        return self._entries[session_id]

    def watches(self, session_id: str) -> list[protocol.PanelRow]:
        if session_id not in self._watches:
            index = self._index(session_id)
            self._watches[session_id] = sample_watches(
                self._watch_counts.get(session_id, 0), index
            )
        return self._watches[session_id]

    # ------------------------------------------------------------ the wire

    def start(self) -> None:
        """The first frame on connect, as `coreproc` would deliver it."""
        self.emit(protocol.Hello(profile="hpc", settings_digest="demo"))

    def handle(self, cmd: protocol.Command) -> None:
        handler = getattr(self, f"_do_{type(cmd).__name__}", None)
        if handler is None:
            self.emit(
                protocol.Notify(
                    severity="warning",
                    text=f"the demo core has no answer for {cmd.TYPE}",
                )
            )
            return
        handler(cmd)

    def _do_SessionList(self, cmd: protocol.SessionList) -> None:
        self.emit(protocol.SessionRows(rows=list(self.rows)))
        if self._parked:
            # Re-emitted on subscribe, as a real core does with a decision it
            # is holding (§4.4): a turn parked before the front-end existed
            # must still be answerable once one arrives.
            self.emit(
                protocol.DecisionRequested(
                    session_id=self._parked, payload=dict(DEMO_DECISION)
                )
            )

    def _do_SessionOpen(self, cmd: protocol.SessionOpen) -> None:
        entries = self.entries(cmd.session_id)
        self.emit(
            protocol.ChatReset(session_id=cmd.session_id, entries=list(entries))
        )
        self._estimate(cmd.session_id)
        self._panel(cmd.session_id)
        if cmd.session_id == self._busy:
            self._start_live_turn(cmd.session_id)

    def _start_live_turn(self, session_id: str) -> None:
        """A turn caught mid-tool: the spinner, and the steps so far.

        Three events, in the order a real core sends them — `turn.started`
        says there is something to stop, `chat.append` puts the working on
        screen *while* it happens, and `turn.activity` names the tool and
        stamps when the turn began, twelve seconds ago so the clock has
        something to count.
        """
        entries = self.entries(session_id)
        live = protocol.Entry(
            kind="thinking",
            text="",
            seq=len(entries) + 1,
            steps=3,
            parts=[
                protocol.Part(
                    kind="call",
                    text="",
                    tool="read_file",
                    target="/scratch/proj/cohort/run3/logs/merge_vcf.log",
                    result="412 lines\n[12:41:07] merging shard 3 of 8",
                    done=True,
                ),
                protocol.Part(
                    kind="reasoning",
                    text=(
                        "Two shards wrote to the same temp path; check which "
                        "of them is still open before retrying the merge."
                    ),
                    done=True,
                ),
                protocol.Part(
                    kind="call",
                    text="",
                    tool="run_bash",
                    target="lsof /scratch/proj/cohort/run3/tmp",
                    done=False,  # the live row: a call with no result yet
                ),
            ],
        )
        entries.append(live)
        self._live[session_id] = live
        self.emit(protocol.TurnStarted(session_id=session_id))
        self.emit(protocol.ChatAppend(session_id=session_id, entry=live))
        self.emit(
            protocol.TurnActivity(
                session_id=session_id,
                activity="running run_bash",
                started_at=(
                    datetime.now(timezone.utc) - timedelta(seconds=12)
                ).isoformat(),
            )
        )

    def _finish_live_turn(self, session_id: str, result: str) -> None:
        """Fill the live row's waiting call *in that row* (`chat.update`)."""
        live = self._live.pop(session_id, None)
        if live is None:
            return
        if session_id == self._busy:
            self._busy = ""
        live.parts[-1].result = result
        live.parts[-1].done = True
        self.emit(protocol.ChatUpdate(session_id=session_id, entry=live))

    def _do_SessionFocus(self, cmd: protocol.SessionFocus) -> None:
        if cmd.session_id:
            self._panel(cmd.session_id)

    def _estimate(self, session_id: str) -> None:
        """Roughly what a conversation that long would cost, capped so a demo
        of four hundred entries does not report 300% of the window."""
        used = min(90 * len(self.entries(session_id)), 108_000)
        self.emit(
            protocol.ContextEstimate(
                session_id=session_id, used=used, window=112_000
            )
        )

    def _panel(self, session_id: str) -> None:
        self.emit(
            protocol.PanelUpdate(
                profile="hpc",
                session_id=session_id,
                rows=list(self.watches(session_id)),
            )
        )

    def _do_TurnSubmit(self, cmd: protocol.TurnSubmit) -> None:
        entries = self.entries(cmd.session_id)
        if cmd.session_id in (self._busy, self._parked):
            # One turn per session: what arrives while that one runs is
            # queued, and the queued row is what the user gets back.
            waiting = protocol.Entry(
                kind="queued", text=cmd.text, seq=len(entries) + 1
            )
            entries.append(waiting)
            self.emit(
                protocol.ChatAppend(session_id=cmd.session_id, entry=waiting)
            )
            return
        self._finish_live_turn(cmd.session_id, "no such process")
        self.emit(protocol.TurnStarted(session_id=cmd.session_id))
        said = protocol.Entry(
            kind="user",
            text=cmd.text,
            seq=len(entries) + 1,
            index=len(entries),
        )
        entries.append(said)
        self.emit(protocol.ChatAppend(session_id=cmd.session_id, entry=said))
        reply = protocol.Entry(
            kind="assistant",
            seq=len(entries) + 1,
            text=(
                "There is no backend behind the demo, so this is where a turn "
                "would start."
            ),
        )
        entries.append(reply)
        self.emit(protocol.ChatAppend(session_id=cmd.session_id, entry=reply))
        self.emit(
            protocol.TurnFinished(session_id=cmd.session_id, reply=reply.text)
        )

    def _do_TurnInterrupt(self, cmd: protocol.TurnInterrupt) -> None:
        self._finish_live_turn(cmd.session_id, "interrupted")
        self.emit(protocol.TurnFinished(session_id=cmd.session_id))

    def _do_TurnUnqueue(self, cmd: protocol.TurnUnqueue) -> None:
        """Take the queued row back, by the name the core gave it.

        A `seq` that is no longer queued is refused with a warning rather than
        an event — it has already started, and stopping that is a different
        question (`protocol.TurnUnqueue`).
        """
        entries = self.entries(cmd.session_id)
        found = next(
            (e for e in entries if e.seq == cmd.seq and e.kind == "queued"),
            None,
        )
        if found is None:
            self.emit(
                protocol.Notify(
                    severity="warning",
                    text="Too late — that message is already running.",
                )
            )
            return
        entries.remove(found)
        self.emit(
            protocol.TurnUnqueued(
                session_id=cmd.session_id, seq=found.seq, text=found.text
            )
        )

    def _do_DecisionResolve(self, cmd: protocol.DecisionResolve) -> None:
        """The answer to the parked approval: the prompt goes and the turn
        goes on, with a refusal's reason recorded where the user can read it
        back — which is where the model reads it too."""
        if cmd.session_id == self._parked:
            self._parked = ""
        self.emit(protocol.DecisionCleared(session_id=cmd.session_id))
        entries = self.entries(cmd.session_id)
        said = "ran the script" if cmd.approved else "skipped the script"
        if cmd.reason:
            said += f" \u2014 \u201c{cmd.reason}\u201d"
        note = protocol.Entry(kind="event", text=said, seq=len(entries) + 1)
        entries.append(note)
        self.emit(protocol.ChatAppend(session_id=cmd.session_id, entry=note))
        self.emit(protocol.TurnFinished(session_id=cmd.session_id))

    def _do_ConfirmResolve(self, cmd: protocol.ConfirmResolve) -> None:
        yes = "yes" if cmd.confirmed else "no"
        self.emit(protocol.Notify(text=f"confirmation {cmd.id}: {yes}"))

    def _do_ModeSet(self, cmd: protocol.ModeSet) -> None:
        """Persist the mode and say so, which is the sidebar's copy of it."""
        self.rows[self._index(cmd.session_id)].mode = cmd.mode
        self.emit(protocol.SessionRows(rows=list(self.rows)))

    def _do_SessionNew(self, cmd: protocol.SessionNew) -> None:
        """A conversation with nothing in it, under the profile that was
        picked — `session.created` first, because the UI has to open it."""
        self._made += 1
        row = protocol.SessionRow(
            session_id=f"new{self._made:04x}",
            title=UNTITLED,
            profile=cmd.profile or "hpc",
            mode=MODES[0],
            model=cmd.backend or "",
        )
        self.rows.insert(0, row)
        self._sizes[row.session_id] = 0
        self._watch_counts[row.session_id] = 0
        self._entries[row.session_id] = []
        self.emit(protocol.SessionCreated(row=row))
        self.emit(protocol.SessionRows(rows=list(self.rows)))

    def _catalog(self) -> list[protocol.LLMEntry]:
        """What the catalog is *now*: the configured rows a remove has not
        taken away, plus whatever the running scan has turned up so far."""
        return [
            x for x in sample_catalog() if x.label not in self._removed
        ] + list(self._found)

    def _do_LLMList(self, cmd: protocol.LLMList) -> None:
        """The catalog, and — when probes were asked for — the catalog again.

        Twice on purpose: that is the shape a real core answers in
        (`protocol.LLMCatalog.probed`), and a demo that only ever sent the
        settled frame would never show the mark a client has to draw for "not
        asked yet".
        """
        entries = self._catalog()
        self.emit(
            protocol.LLMCatalog(
                entries=[e.model_copy(update={"reachable": None}) for e in entries]
            )
        )
        if cmd.probe:
            self.emit(protocol.LLMCatalog(entries=entries, probed=True))

    def _do_BackendScan(self, cmd: protocol.BackendScan) -> None:
        """The scan, in the shape a real one answers in (`protocol.BackendScan`).

        Every hit restates the catalog, so the discovered panel fills a row at
        a time and the closing `backend.scanned` carries the verdict. The
        demo's loopback is synchronous, so the frames arrive in one breath
        rather than over a minute — the *order* is what the screen is being
        shown, and it is the order that decides whether it can fill
        incrementally at all.

        A second scan finds nothing on purpose, which is the other half of
        what this screen has to draw: the off-cluster case, where the verdict
        is a tunnel recipe in a window rather than rows in a panel.
        """
        self._scans += 1
        # Cleared first and said so, as the core does: a rescan must take a
        # row for an endpoint that has gone away off the screen when the sweep
        # *starts*, not when it ends.
        self._found = []
        self.emit(protocol.LLMCatalog(entries=self._catalog()))
        if self._scans > 1:
            self.emit(
                protocol.BackendScanned(
                    help=(
                        "No LLM endpoints found.\n\n"
                        "You are probably not on the cluster. Open a tunnel:\n\n"
                        "    ssh -N -L 20001:node042:20001 login.example.org\n"
                        "    ssh -N -L 20000:node042:20000 login.example.org\n\n"
                        "Then press s here to scan again."
                    )
                )
            )
            return
        for found in DISCOVERED:
            self._found.append(found)
            self.emit(protocol.LLMCatalog(entries=self._catalog()))
        self.emit(
            protocol.BackendScanned(found=len(self._found), cluster=1)
        )

    def _do_BackendProbe(self, cmd: protocol.BackendProbe) -> None:
        """`backend.probed`, in all three of its shapes.

        Which one depends on the URL, so that the form's three branches — a
        refused key, an endpoint that serves one model, an endpoint that
        serves several — can all be reached from the demo rather than only
        from a test.
        """
        if "locked" in cmd.base_url and not cmd.api_key:
            self.emit(
                protocol.BackendProbed(base_url=cmd.base_url, needs_key=True)
            )
            return
        if "nothing" in cmd.base_url:
            self.emit(protocol.BackendProbed(base_url=cmd.base_url))
            return
        models = ["qwen3-27b-fp8", "llama-3.3-70b"] if "many" in cmd.base_url else [
            "qwen3-27b-fp8"
        ]
        self.emit(
            protocol.BackendProbed(
                base_url=cmd.base_url,
                models=[
                    protocol.LLMEntry(
                        label=name,
                        model=name,
                        base_url=cmd.base_url,
                        max_model_len=112000,
                        reachable=True,
                        discovered=True,
                    )
                    for name in models
                ],
            )
        )

    def _do_BackendRemove(self, cmd: protocol.BackendRemove) -> None:
        self._removed.add(cmd.label)
        self.emit(protocol.Notify(text=f"Removed {cmd.label}"))
        self.emit(protocol.LLMCatalog(entries=self._catalog(), probed=True))

    def _do_ProfileList(self, cmd: protocol.ProfileList) -> None:
        """`profile.rows`: counts and flags, which is all the event carries.

        Answerable now that the bodies come down the wire too — before M8
        closed the stopgap the client filled them off disk, and answering here
        would have replaced the demo's profiles with rows whose editors had
        nothing to open.
        """
        self.emit(
            protocol.ProfileRows(
                rows=[
                    protocol.ProfileRow(
                        name=x.name,
                        memories=x.memories,
                        copied_from=x.copied_from,
                        is_default=x.default,
                        working=x.working,
                    )
                    for x in self.profiles
                ]
            )
        )

    def _do_ThinkingSet(self, cmd: protocol.ThinkingSet) -> None:
        self.rows[self._index(cmd.session_id)].thinking = cmd.effort
        self.emit(protocol.SessionRows(rows=list(self.rows)))
        self.emit(protocol.Notify(text=f"Thinking effort: {cmd.effort}"))

    def _do_BackendSet(self, cmd: protocol.BackendSet) -> None:
        model = str(cmd.backend.get("model", "")) or "the new backend"
        if cmd.session_id is None:
            self.emit(protocol.Notify(text=f"Default backend: {model}"))
            return
        self.rows[self._index(cmd.session_id)].model = model
        self.emit(protocol.SessionRows(rows=list(self.rows)))
        self.emit(protocol.Notify(text=f"This session now uses {model}"))

    def _do_ProfileSet(self, cmd: protocol.ProfileSet) -> None:
        self.emit(protocol.Notify(text=f"Working profile: {cmd.name}"))

    def _do_ProfileSave(self, cmd: protocol.ProfileSave) -> None:
        self.emit(protocol.Notify(text=f"Saved {cmd.kind} for {cmd.name}."))

    def _do_ProfileCreate(self, cmd: protocol.ProfileCreate) -> None:
        self.emit(protocol.Notify(text=f"Created profile {cmd.name}."))

    def _do_ProfileDuplicate(self, cmd: protocol.ProfileDuplicate) -> None:
        self.emit(
            protocol.Notify(text=f"Copied {cmd.source or 'hpc'} to {cmd.name}.")
        )

    def _do_ProfileDelete(self, cmd: protocol.ProfileDelete) -> None:
        self.emit(protocol.Notify(text=f"Deleted profile {cmd.name}."))

    def _do_SkillSave(self, cmd: protocol.SkillSave) -> None:
        self.emit(protocol.Notify(text=f"Saved skill {cmd.name}."))

    def _do_SkillDelete(self, cmd: protocol.SkillDelete) -> None:
        self.emit(protocol.Notify(text=f"Removed skill {cmd.name}."))

    # ------------------------------------------------------- the read paths

    def _do_ProfileGet(self, cmd: protocol.ProfileGet) -> None:
        """`profile.get`: one editable body, verbatim.

        The demo's answer to the rule that closed M8's stopgap — the UI reads
        no files, so every editor's text comes down the wire, here included.
        """
        if cmd.kind == "archive":
            self.emit(
                protocol.ProfileBody(
                    name=cmd.name, kind=cmd.kind, text=ARCHIVE_TEXT
                )
            )
            return
        self.emit(
            protocol.ProfileBody(
                name=cmd.name,
                kind=cmd.kind,
                text=LEARNINGS.get(cmd.name, ""),
                error="" if cmd.name in LEARNINGS else "no such profile here",
            )
        )

    def _do_SkillList(self, cmd: protocol.SkillList) -> None:
        """Both scopes, because the real core answers both: a menu asks what
        is callable and an editor asks what it may overwrite. The shipped
        skills are in the wide answer only — which is what makes `/plan` work
        here the way it works on a fresh install."""
        info = next((x for x in self.profiles if x.name == cmd.profile), None)
        own = [
            protocol.SkillRow(
                name=x.name, description=x.description, level="profile"
            )
            for x in (info.skills if info else [])
        ]
        shipped = [
            protocol.SkillRow(
                name=name, description=description, level="builtin"
            )
            for name, description in SHIPPED_SKILLS
        ]
        self.emit(
            protocol.SkillRows(
                profile=cmd.profile,
                scope=cmd.scope,
                skills=(own + shipped) if cmd.scope == "visible" else own,
            )
        )

    def _do_SkillDraft(self, cmd: protocol.SkillDraft) -> None:
        """There is no model behind the demo, so the "draft" is the request
        itself, shaped like one — enough for the form to open pre-filled."""
        words = [w for w in cmd.request.split() if w.isalnum()][:3]
        self.emit(
            protocol.SkillDrafted(
                profile=cmd.profile,
                request=cmd.request,
                name="-".join(words).lower() or "new-skill",
                description=f"when the task is: {cmd.request}",
                body=f"1. {cmd.request}\n2. check it worked\n",
            )
        )

    def _do_CommandList(self, cmd: protocol.CommandList) -> None:
        self.emit(protocol.CommandCounts(counts=dict(COMMAND_COUNTS)))

    def _do_SkillGet(self, cmd: protocol.SkillGet) -> None:
        self.emit(
            protocol.SkillBody(
                profile=cmd.profile, name=cmd.name, text=SKILL_TEXT
            )
        )

    def _do_SettingsGet(self, cmd: protocol.SettingsGet) -> None:
        self.emit(protocol.SettingsBody(text=SETTINGS_JSON))

    def _do_SettingsSave(self, cmd: protocol.SettingsSave) -> None:
        """Restated on save, which is what the real core does and why: what
        lands on disk is the validated model written back out, not the bytes
        that were sent."""
        self.emit(protocol.SettingsBody(text=cmd.text))
        self.emit(protocol.Notify(text="Settings saved."))

    def _do_MemoryResolve(self, cmd: protocol.MemoryResolve) -> None:
        kept = sum(1 for x in cmd.approved if x)
        self.emit(
            protocol.Notify(text=f"{kept} of {len(cmd.approved)} memories kept.")
        )

    def _do_CommandRun(self, cmd: protocol.CommandRun) -> None:
        # Counted the way the real core counts, so the menu's frequency sort
        # is visible in the demo rather than only in the tests.
        COMMAND_COUNTS[cmd.name] = COMMAND_COUNTS.get(cmd.name, 0) + 1
        self.emit(protocol.CommandCounts(counts=dict(COMMAND_COUNTS)))
        self.emit(protocol.Notify(text=f"the demo core does not run /{cmd.name}"))

    def _do_SessionRename(self, cmd: protocol.SessionRename) -> None:
        if not cmd.title.strip():
            self.emit(
                protocol.Notify(
                    severity="warning", text="A session needs a name."
                )
            )
            return
        self.rows[self._index(cmd.session_id)].title = cmd.title
        self.emit(protocol.SessionRows(rows=list(self.rows)))

    def _do_SessionRetitle(self, cmd: protocol.SessionRetitle) -> None:
        """There is no model behind the demo, so the "title" is the first
        thing that was said — which is what the real core falls back to."""
        entries = self.entries(cmd.session_id)
        said = next((e.text for e in entries if e.kind == "user"), "")
        if not said:
            self.emit(
                protocol.Notify(
                    severity="warning", text="Nothing to summarize yet."
                )
            )
            return
        title = " ".join(said.split()[:6])
        self.rows[self._index(cmd.session_id)].title = title
        self.emit(protocol.SessionRows(rows=list(self.rows)))
        self.emit(protocol.Notify(text=f"Renamed to \u201c{title}\u201d"))

    def _do_SessionDelete(self, cmd: protocol.SessionDelete) -> None:
        gone = [r for r in self.rows if r.session_id == cmd.session_id]
        self.rows = [r for r in self.rows if r.session_id != cmd.session_id]
        self._entries.pop(cmd.session_id, None)
        self._watches.pop(cmd.session_id, None)
        self._live.pop(cmd.session_id, None)
        # A turn and a parked approval go with the conversation, the way
        # `TurnScheduler.forget_session` drops them in the real core.
        if self._busy == cmd.session_id:
            self._busy = ""
        if self._parked == cmd.session_id:
            self._parked = ""
        self.emit(protocol.SessionRows(rows=list(self.rows)))
        for row in gone:
            self.emit(protocol.Notify(text=f"Deleted \u201c{row.title}\u201d"))

    def _do_SessionFork(self, cmd: protocol.SessionFork) -> None:
        source = self.rows[self._index(cmd.session_id)]
        self._forks += 1
        row = protocol.SessionRow(
            session_id=f"fork{self._forks:04x}",
            title=f"{source.title} (fork)",
            profile=source.profile,
            mode=source.mode,
        )
        cut = self._cut(cmd.session_id, cmd.index)
        self.rows.insert(0, row)
        self._sizes[row.session_id] = cut
        self._watch_counts[row.session_id] = 0
        self._entries[row.session_id] = [
            entry.model_copy() for entry in self.entries(cmd.session_id)[:cut]
        ]
        self.emit(protocol.SessionCreated(row=row))
        self.emit(protocol.SessionRows(rows=list(self.rows)))

    def _do_SessionRollback(self, cmd: protocol.SessionRollback) -> None:
        entries = self.entries(cmd.session_id)
        cut = self._cut(cmd.session_id, cmd.index)
        dropped = len(entries) - cut
        del entries[cut:]
        self.emit(
            protocol.ChatReset(session_id=cmd.session_id, entries=list(entries))
        )
        self._estimate(cmd.session_id)
        self.emit(
            protocol.Notify(
                text=(
                    "rolled back to just before that message "
                    f"({dropped} entries gone)"
                )
            )
        )

    def _cut(self, session_id: str, index: int) -> int:
        """Where a rewind aimed at message ``index`` falls in the entry list.

        The conversion `_Rewind` says the core owns: the UI names a message it
        was shown, and only this side can turn that into a length.
        """
        entries = self.entries(session_id)
        return next(
            (i for i, entry in enumerate(entries) if entry.index == index),
            len(entries),
        )

    def _do_WatchPeek(self, cmd: protocol.WatchPeek) -> None:
        for rows in self._watches.values():
            for row in rows:
                if row.ref == str(cmd.watch_id):
                    self.emit(
                        protocol.WatchPeeked(
                            watch_id=cmd.watch_id,
                            title=row.title,
                            text=row.text.split("\n")[-1],
                        )
                    )
                    return
        self.emit(
            protocol.Notify(severity="warning", text="no such watch any more")
        )

    def _do_WatchDrop(self, cmd: protocol.WatchDrop) -> None:
        for session_id, rows in self._watches.items():
            kept = [row for row in rows if row.ref != str(cmd.watch_id)]
            if len(kept) != len(rows):
                self._watches[session_id] = kept
                self._panel(session_id)
                return


def build(chat: int = 400, sessions: int = 14, watchers: int = 5) -> RowUI:
    """A UI with a demo core behind it, already handshaken.

    The client is reachable as ``ui.send``'s closure and the core as the
    client's sink; returning the UI keeps all three alive, which is all the
    ownership this needs while the loopback is synchronous.
    """
    ui = RowUI()
    core = DemoCore(chat=chat, sessions=sessions, watchers=watchers)
    client = UIClient(ui, send=core.handle)
    core.emit = client.apply
    core.start()
    return ui
