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

from hpca import protocol
from hpca.ui.ansi import GREEN
from hpca.ui.app import RowUI
from hpca.ui.client import UIClient
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

PROFILES = ["hpc", "writing", "default"]
MODES = ["agent", "agent", "plan"]

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

TOOLS = ["read_file", "edit_file", "create_file", "run_bash", "list_dir"]
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
        self._forks = 0

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

    def _do_SessionOpen(self, cmd: protocol.SessionOpen) -> None:
        entries = self.entries(cmd.session_id)
        self.emit(
            protocol.ChatReset(session_id=cmd.session_id, entries=list(entries))
        )
        self._estimate(cmd.session_id)
        self._panel(cmd.session_id)

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
        self.emit(protocol.TurnFinished(session_id=cmd.session_id))

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
    ui = RowUI(
        learnings=dict(LEARNINGS),
        settings_json=SETTINGS_JSON,
        llms=sample_llms(),
        profiles=sample_profiles(),
    )
    core = DemoCore(chat=chat, sessions=sessions, watchers=watchers)
    client = UIClient(ui, send=core.handle)
    core.emit = client.apply
    core.start()
    return ui
