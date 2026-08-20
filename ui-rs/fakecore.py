"""A core that speaks the protocol and runs no agent — for testing a front-end.

The point of this file is that it imports the *real* `hpca.protocol` and
`hpca.transport` rather than reimplementing them. That is what makes it a test
of interoperability instead of a test of my own idea of the wire: every command
the Rust client sends is parsed by `protocol.parse`, which is `extra="forbid"`
on every model, so a field this front-end got wrong, renamed or omitted fails
here loudly rather than being quietly tolerated.

It is not a stand-in for `hpca --serve`, which does not exist yet (wave 3 of
specs-core-process.md). It is the harness that lets a second front-end be
developed before it does.

    python fakecore.py /tmp/core.sock [--script]

``--script`` plays a short scripted turn — a submitted message, reasoning, a
tool call that lands, an approval request — so a front-end can be driven
through its states without a backend.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from hpca.protocol import (
    ChatAppend,
    ChatReset,
    ConfirmRequested,
    ContextEstimate,
    DecisionCleared,
    DecisionRequested,
    Entry,
    Hello,
    Notify,
    PANEL_WATCH,
    PanelRow,
    Part,
    ProtocolError,
    SessionRow,
    SessionRows,
    TurnActivity,
    TurnFinished,
    TurnStarted,
    TurnUsage,
    parse,
)
from hpca.transport import Connection, serve_unix

SESSIONS = [
    SessionRow(
        session_id="s1",
        title="align the reads",
        profile="default",
        mode="auto",
        flags=[],
    ),
    SessionRow(
        session_id="s2",
        title="why did 4417 die",
        profile="default",
        mode="manual",
        flags=[],
    ),
]

WATCHES = [
    PanelRow(
        key="w1",
        title="job 4417",
        text="RUNNING  12:04 elapsed\nnode-023  32 cpus",
        kind=PANEL_WATCH,
        ref="1",
    ),
    PanelRow(
        key="w2",
        title="align.err",
        text="quiet for 3m",
        kind=PANEL_WATCH,
        ref="2",
    ),
]

TRANSCRIPT = [
    Entry(kind="user", text="did the alignment run finish cleanly?", index=0),
    Entry(
        kind="thinking",
        text="",
        steps=1,
        reasoning_chars=1840,
        parts=[
            Part(kind="reasoning", text="Check the error log first, then squeue."),
            Part(
                kind="call",
                tool="run_bash",
                target="align.err",
                text="tail -n 40 /scratch/run/align.err",
                result="slurmstepd: error: JOB 4417 CANCELLED DUE TO TIME LIMIT",
                done=True,
            ),
        ],
    ),
    Entry(
        kind="assistant",
        text=(
            "No — it hit the wall clock. The step was cancelled at the time "
            "limit rather than failing, so the outputs written up to that "
            "point are intact and the run can resume from the last checkpoint."
        ),
        index=1,
    ),
    Entry(kind="event", text="job 4418 finished (exit 0)"),
]


class FakeCore:
    def __init__(self, conn: Connection, *, script: bool) -> None:
        self.conn = conn
        self.script = script
        self.open_session: str | None = None
        self.seen: list[str] = []

    async def run(self) -> None:
        await self.conn.send(Hello(profile="default", settings_digest="fake"))
        await self.conn.send(SessionRows(rows=SESSIONS))
        await self.conn.send(PanelUpdate_rows())
        async for env in self.conn:
            try:
                # The real parser, with the real extra="forbid": a command the
                # front-end shaped wrongly dies here, which is the whole point.
                cmd = parse(env)
            except ProtocolError as e:
                print(f"REJECTED {env.type!r}: {e}", flush=True)
                await self.conn.send(
                    Notify(severity="error", text=f"bad command: {e}")
                )
                continue
            self.seen.append(env.type)
            print(f"ok {env.type}", flush=True)
            await self.dispatch(env.type, cmd)

    async def dispatch(self, kind: str, cmd) -> None:
        if kind == "session.list":
            await self.conn.send(SessionRows(rows=SESSIONS))
        elif kind == "session.open":
            self.open_session = cmd.session_id
            await self.conn.send(
                ChatReset(session_id=cmd.session_id, entries=TRANSCRIPT)
            )
            await self.conn.send(
                ContextEstimate(session_id=cmd.session_id, used=24_600, window=32_768)
            )
        elif kind == "turn.submit":
            await self.play_turn(cmd.session_id, cmd.text)
        elif kind == "turn.interrupt":
            await self.conn.send(
                TurnActivity(session_id=cmd.session_id, activity="")
            )
            await self.conn.send(
                Notify(severity="warning", text="turn interrupted")
            )
        elif kind == "decision.resolve":
            await self.conn.send(DecisionCleared(session_id=cmd.session_id))
            verdict = "approved" if cmd.approved else f"refused ({cmd.reason or '—'})"
            await self.conn.send(Notify(text=f"call {verdict}"))
        elif kind == "command.run":
            await self.conn.send(
                Notify(text=f"/{cmd.name} {cmd.args}".strip() + " — acknowledged")
            )
        elif kind == "mode.set":
            await self.conn.send(Notify(text=f"mode is now {cmd.mode}"))
        elif kind == "watch.drop":
            await self.conn.send(Notify(text=f"dropped watch {cmd.watch_id}"))

    async def play_turn(self, session_id: str, text: str) -> None:
        await self.conn.send(
            ChatAppend(
                session_id=session_id,
                entry=Entry(kind="user", text=text, index=len(TRANSCRIPT)),
            )
        )
        await self.conn.send(TurnStarted(session_id=session_id))
        for activity, pause in (("thinking", 0.4), ("run_bash", 0.6)):
            await self.conn.send(
                TurnActivity(session_id=session_id, activity=activity)
            )
            await asyncio.sleep(pause if self.script else 0.05)
        if self.script:
            await self.conn.send(
                DecisionRequested(
                    session_id=session_id,
                    payload={
                        "question": "run this on the login node?",
                        "command": "sbatch --time=8:00:00 align.sh",
                    },
                )
            )
            await asyncio.sleep(0.8)
            await self.conn.send(DecisionCleared(session_id=session_id))
        await self.conn.send(
            ChatAppend(
                session_id=session_id,
                entry=Entry(
                    kind="thinking",
                    text="",
                    steps=1,
                    reasoning_chars=920,
                    parts=[
                        Part(
                            kind="call",
                            tool="run_bash",
                            target="squeue",
                            text="squeue -u $USER",
                            result="4419 align  RUNNING  0:12",
                            done=True,
                        )
                    ],
                ),
            )
        )
        await self.conn.send(
            ChatAppend(
                session_id=session_id,
                entry=Entry(
                    kind="assistant",
                    text="Resubmitted as 4419 with an eight-hour limit.",
                    index=len(TRANSCRIPT) + 1,
                ),
            )
        )
        await self.conn.send(
            TurnUsage(
                session_id=session_id, prompt_tokens=26_800, max_model_len=32_768
            )
        )
        await self.conn.send(TurnActivity(session_id=session_id, activity=""))
        await self.conn.send(TurnFinished(session_id=session_id, reply="ok"))
        await self.conn.send(
            ConfirmRequested(id="c1", question="remember this log signature?")
        )


def PanelUpdate_rows():
    from hpca.protocol import PanelUpdate

    return PanelUpdate(profile="default", session_id=None, rows=WATCHES)


async def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    path = Path(sys.argv[1])
    script = "--script" in sys.argv
    if path.exists():
        path.unlink()

    async def handler(conn: Connection) -> None:
        await FakeCore(conn, script=script).run()

    server = await serve_unix(path, handler)
    print(f"listening on {path}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        server.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
