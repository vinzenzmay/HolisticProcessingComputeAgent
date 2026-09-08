"""`hpca -p`: the front-end with no screen, driven against a scripted core.

The runtime is not built here. `hpca.headless.run` opens databases and an
`AgentService`; `_drive` — the half that pins a backend, submits the turn and
answers what nobody is there to answer — takes only a wire, which is why it is
a separate function. So this file is hermetic and fast, and what it asserts is
the whole of what this front-end decides:

* the four answers a run gives without a human (§ the module docstring), and
  that the default for a *gated* call is yes, with the reason the trail in
  `hpca.headless` records;
* that the session is pinned to a named backend rather than to whatever is
  active when the turn happens to start;
* that the tool record comes from the chat parts and not from `turn.activity`,
  which double-counts every approved call because `interrupt()` re-runs its
  node on resume;
* that stdout carries the reply and nothing else, and that the exit code says
  which of the four things happened.

The transport is real (`InProcessConnection.pair()`), as everywhere else in
this suite: the edges — a closed peer, a dropped frame — behave here as they
do in a run.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from hpca import protocol as P
from hpca.__main__ import build_parser
from hpca.config import LLMBackend, Settings
from hpca.headless import (
    EXIT_NO_BACKEND,
    EXIT_OK,
    EXIT_TIMEOUT,
    EXIT_TURN_FAILED,
    EXIT_USAGE,
    Outcome,
    _drive,
    _resolve_label,
    _Session,
)
from hpca.transport import InProcessConnection

SESSION = "s-1"
CATALOG = [
    P.LLMEntry(label="qwen", model="qwen", base_url="http://x/v1", active=True),
    P.LLMEntry(label="other", model="other", base_url="http://y/v1"),
]


def args_for(*argv: str):
    """Real argv through the real parser, so the CLI is under test too."""
    return build_parser().parse_args(["-p", "do the thing", *argv])


def settings_for() -> Settings:
    settings = Settings()
    settings.backends = [
        LLMBackend(model="qwen", base_url="http://x/v1"),
        LLMBackend(model="other", base_url="http://y/v1"),
    ]
    settings.llm.base_url = "http://x/v1"
    settings.llm.model = "qwen"
    return settings


class Core:
    """The core, scripted: it answers the handful of commands this front-end
    sends, and keeps everything it heard so an assertion can be made about it.

    `reply_to` is the whole script — a mapping from a command type to what the
    core says back — so a test states only its own difference from the happy
    path.
    """

    def __init__(self, conn) -> None:
        self.conn = conn
        self.commands: list = []
        self.answer = True  # whether session.new is answered at all

    async def listen(self) -> None:
        async for env in self.conn:
            message = P.parse(env)
            self.commands.append(message)
            if isinstance(message, P.LLMList):
                await self.conn.send(P.LLMCatalog(entries=CATALOG, probed=True))
            elif isinstance(message, P.SessionNew) and self.answer:
                await self.conn.send(
                    P.SessionCreated(
                        row=P.SessionRow(
                            session_id=SESSION, title="t", profile="default"
                        )
                    )
                )

    def took(self, kind: type) -> list:
        return [c for c in self.commands if isinstance(c, kind)]

    def one(self, kind: type):
        found = self.took(kind)
        assert found, f"no {kind.__name__}: {[type(c).__name__ for c in self.commands]}"
        return found[-1]

    def sent(self, kind: type) -> bool:
        return bool(self.took(kind))


@asynccontextmanager
async def driven(*argv: str, settings=None):
    """A headless run's command half, against a scripted core."""
    ours, theirs = InProcessConnection.pair()
    args = args_for(*argv)
    session = _Session(ours, approve=not args.refuse_gated, quiet=True)
    core = Core(theirs)
    no_backend = asyncio.Event()
    tasks = [
        asyncio.ensure_future(session.pump()),
        asyncio.ensure_future(core.listen()),
    ]
    drive = asyncio.ensure_future(
        _drive(P, session, args, settings or settings_for(), no_backend)
    )
    try:
        yield session, core, drive, no_backend
    finally:
        drive.cancel()
        await ours.close()
        await theirs.close()
        for task in [*tasks, drive]:
            task.cancel()


async def settle(times: int = 6) -> None:
    """Let the two ends finish talking. The wire is queues, so a turn of the
    loop is all it takes; several, because a round trip is two."""
    for _ in range(times):
        await asyncio.sleep(0)


async def turn_ends(core: Core, drive, event) -> None:
    await settle()
    await core.conn.send(event)
    await asyncio.wait_for(drive, 5)


# --------------------------------------------------------------- the pinning


class TestPinning:
    async def test_the_session_names_the_backend_it_wants(self):
        async with driven() as (session, core, drive, _):
            await settle()
            assert core.one(P.SessionNew).backend == "qwen", (
                "a session that pins nothing is one auto-connect can move"
            )

    async def test_an_explicit_label_wins(self):
        async with driven("--backend", "other") as (session, core, drive, _):
            await settle()
            assert core.one(P.SessionNew).backend == "other"

    async def test_a_label_nobody_has_stops_the_run(self):
        async with driven("--backend", "nope") as (session, core, drive, _):
            await asyncio.wait_for(drive, 5)
            assert session.outcome.exit_code == EXIT_USAGE
            assert not core.sent(P.SessionNew), "a typo must not create a session"

    async def test_resuming_makes_no_session(self):
        async with driven("--session", "old-1") as (session, core, drive, _):
            await settle()
            assert not core.sent(P.SessionNew)
            assert core.one(P.TurnSubmit).session_id == "old-1"

    def test_an_unconfigured_run_lets_the_core_pick(self):
        # Empty string, not None: None is the error return, and the two must
        # not be confused by the caller.
        assert _resolve_label([], settings_for(), None) == ""


# ------------------------------------------------- the answers nobody gives


class TestTheAnswersNobodyGives:
    async def test_a_gated_call_is_approved_by_default(self):
        async with driven() as (session, core, drive, _):
            await settle()
            await core.conn.send(
                P.DecisionRequested(session_id=SESSION, payload={"tool": "edit_file"})
            )
            await settle()
            answer = core.one(P.DecisionResolve)
            assert answer.approved
            assert answer.reason == "", "an approval explains nothing"

    async def test_refuse_gated_refuses_and_says_not_to_reroute(self):
        async with driven("--refuse-gated") as (session, core, drive, _):
            await settle()
            await core.conn.send(
                P.DecisionRequested(session_id=SESSION, payload={"tool": "edit_file"})
            )
            await settle()
            answer = core.one(P.DecisionResolve)
            assert not answer.approved
            # The measured failure this sentence exists for: refused once, the
            # model did the same edit with `sed -i` through run_bash.
            assert "another way" in answer.reason
            assert session.outcome.refusals

    async def test_a_triage_offer_is_declined(self):
        async with driven() as (session, core, drive, _):
            await settle()
            await core.conn.send(
                P.ConfirmRequested(id="c1", session_id=SESSION, question="learn this?")
            )
            await settle()
            answer = core.one(P.ConfirmResolve)
            assert answer.id == "c1" and not answer.confirmed

    async def test_a_compaction_is_accepted(self):
        async with driven() as (session, core, drive, _):
            await settle()
            await core.conn.send(
                P.CompactProposed(session_id=SESSION, summary="…", folded=3)
            )
            await settle()
            assert core.one(P.CompactResolve).action == "accept"


# ------------------------------------------------------------ what it reports


class TestWhatItReports:
    async def test_the_tools_come_from_the_parts_not_the_activity(self):
        """One approved edit is one edit, however many times it is announced.

        `interrupt()` re-runs its node on resume, so an approved `edit_file`
        reports `running edit_file` twice; the parts of the row report it once.
        """
        async with driven() as (session, core, drive, _):
            await settle()
            for activity in ("running edit_file", "working", "running edit_file"):
                await core.conn.send(
                    P.TurnActivity(session_id=SESSION, activity=activity)
                )
            await core.conn.send(
                P.ChatUpdate(
                    session_id=SESSION,
                    entry=P.Entry(
                        kind="thinking",
                        text="",
                        seq=2,
                        parts=[
                            P.Part(kind="call", text="", tool="edit_file",
                                   target="calc.py", done=True),
                        ],
                    ),
                )
            )
            await turn_ends(
                core, drive, P.TurnFinished(session_id=SESSION, reply="done")
            )
            assert session.outcome.tools == [
                {"tool": "edit_file", "target": "calc.py", "failed": False}
            ]

    async def test_a_revised_row_replaces_rather_than_repeats(self):
        async with driven() as (session, core, drive, _):
            await settle()
            for done in (False, True):
                await core.conn.send(
                    P.ChatUpdate(
                        session_id=SESSION,
                        entry=P.Entry(
                            kind="thinking",
                            text="",
                            seq=2,
                            parts=[
                                P.Part(kind="call", text="", tool="run_bash", done=done)
                            ],
                        ),
                    )
                )
            await turn_ends(
                core, drive, P.TurnFinished(session_id=SESSION, reply="done")
            )
            assert len(session.outcome.tools) == 1

    async def test_a_warning_is_kept_and_information_is_not(self):
        async with driven() as (session, core, drive, _):
            await settle()
            await core.conn.send(P.Notify(severity="warning", text="no window"))
            await core.conn.send(P.Notify(severity="information", text="hello"))
            await turn_ends(
                core, drive, P.TurnFinished(session_id=SESSION, reply="done")
            )
            assert session.outcome.warnings == ["no window"]


# --------------------------------------------------------------- exit codes


class TestExitCodes:
    async def test_a_finished_turn_is_zero(self):
        async with driven() as (session, core, drive, _):
            await turn_ends(
                core, drive, P.TurnFinished(session_id=SESSION, reply="hi")
            )
            assert session.outcome.exit_code == EXIT_OK
            assert session.outcome.reply == "hi"

    async def test_a_failed_turn_is_four(self):
        async with driven() as (session, core, drive, _):
            await turn_ends(
                core, drive, P.TurnFailed(session_id=SESSION, error="boom")
            )
            assert session.outcome.exit_code == EXIT_TURN_FAILED
            assert session.outcome.error == "boom"

    async def test_a_failed_turn_with_nothing_answering_is_two(self):
        """The distinction the caller most needs: a broken task, or no LLM."""
        async with driven() as (session, core, drive, no_backend):
            no_backend.set()
            await turn_ends(
                core, drive, P.TurnFailed(session_id=SESSION, error="connect failed")
            )
            assert session.outcome.exit_code == EXIT_NO_BACKEND

    async def test_no_answer_at_all_is_three(self):
        async with driven("--timeout", "0.05") as (session, core, drive, _):
            await asyncio.wait_for(drive, 5)
            assert session.outcome.exit_code == EXIT_TIMEOUT
            assert "0.05" in (session.outcome.error or "")

    async def test_nothing_is_submitted_without_a_session(self):
        """A core that never answers `session.new` gets no turn anyway.

        The wait itself is the module's 60s and is not shortened here — what
        this holds is the order: nothing is sent into a session that does not
        exist, so a hung create cannot become a turn against the wrong id.
        """
        async with driven() as (session, core, drive, _):
            core.answer = False
            await settle()
            assert not core.sent(P.TurnSubmit)
            assert not core.sent(P.ModeSet)


# ------------------------------------------------------- the stdout contract


def returning(monkeypatch, outcome: Outcome):
    """`main` with the run replaced by its result.

    `main` is the output contract and nothing else — what it does with an
    `Outcome` — so the runtime under it is the part to take away. The coroutine
    is closed rather than dropped, or the test passes with a warning attached.
    """
    from hpca import headless

    def instead(coro):
        coro.close()
        return outcome

    monkeypatch.setattr(headless.asyncio, "run", instead)
    return headless


class TestStdout:
    """One thing on stdout, so `$(hpca -p ...)` is the answer."""

    def test_the_reply_alone(self, capsys, monkeypatch):
        headless = returning(
            monkeypatch,
            Outcome(session_id="s", reply="the answer", exit_code=EXIT_OK),
        )
        code = headless.main(args_for())
        captured = capsys.readouterr()
        assert code == EXIT_OK
        assert captured.out == "the answer\n"

    def test_an_error_stays_off_stdout(self, capsys, monkeypatch):
        headless = returning(
            monkeypatch, Outcome(error="boom", exit_code=EXIT_TURN_FAILED)
        )
        code = headless.main(args_for())
        captured = capsys.readouterr()
        assert code == EXIT_TURN_FAILED
        assert captured.out == "", "a caller capturing stdout must get nothing"
        assert "boom" in captured.err

    def test_json_carries_what_the_reply_cannot(self, capsys, monkeypatch):
        headless = returning(
            monkeypatch,
            Outcome(
                session_id="s-9",
                reply="done",
                tools=[{"tool": "edit_file", "target": "a.py", "failed": False}],
            ),
        )
        headless.main(args_for("--json"))
        body = json.loads(capsys.readouterr().out)
        assert body["session_id"] == "s-9", "a caller continues the session by id"
        assert body["tools"][0]["tool"] == "edit_file"

    def test_the_prompt_is_required(self, capsys):
        from hpca import headless

        args = build_parser().parse_args([])
        assert headless.main(args) == EXIT_USAGE
        assert capsys.readouterr().out == ""
