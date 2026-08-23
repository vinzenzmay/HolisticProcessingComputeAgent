"""The timed sweeps, with nothing rendering them (specs-core-process §5, §7).

These are the same behaviours ``test_tui_jobs``, ``test_tui_watches`` and
``test_tui_processes`` cover through a running Textual app; here they are
asserted on what the core *says* instead — the events it emits and the work it
hands the scheduler. Which is the point of the extraction: none of this needs
a screen to be true, and none of these tests needs a cluster, an LLM or a
wall clock to run.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from os import utime
from pathlib import Path

import pytest

from hpca.agent.explainer import ProposedSignature
from hpca.config import Settings
from hpca.core.deps import CoreDeps
from hpca.core.pollers import Pollers
from hpca.db import connect, init_db
from hpca.jobs import JobStore
from hpca.llm import ChatResponse
from hpca.protocol import PANEL_WATCH, Notify, PanelUpdate
from hpca.runner import ProcessChange, ProcessRunner
from hpca.sessions import SessionStore
from hpca.slurm import SlurmClient
from hpca.triage import Candidate, LogFinding
from hpca.watches import KIND_JOB, KIND_LOG, WatchStore


# One sacct row, in the field order SACCT_FIELDS asks for.
def sacct(job_id: str, state: str) -> str:
    return f"{job_id}|{state}|0:0|00:01:00||25G|5-00:00:00\n"


class FakeRun:
    """Stands in for the subprocess half of SlurmClient. Raises when its
    script runs out, which is how a test says "nothing more may be asked"."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    async def __call__(self, argv):
        self.calls.append(argv)
        if not self.responses:
            raise AssertionError(f"unexpected slurm call: {argv}")
        return self.responses.pop(0)


class Recorder:
    """CoreDeps' two outward channels, collected rather than delivered.

    The whole test surface: a poller that used to notify a widget and append
    to the app's work queue now emits and calls back, so asserting on two
    lists is asserting on everything it does.
    """

    def __init__(self) -> None:
        self.events: list = []
        self.submitted: list[tuple[str, str]] = []

    def emit(self, event) -> None:
        self.events.append(event)

    def submit(self, session_id: str, text: str) -> None:
        self.submitted.append((session_id, text))

    def of(self, model) -> list:
        return [e for e in self.events if isinstance(e, model)]

    @property
    def toasts(self) -> list[str]:
        return [e.text for e in self.of(Notify)]

    @property
    def rows(self) -> list:
        """The rows of the last panel update, or [] if there was none."""
        updates = self.of(PanelUpdate)
        return list(updates[-1].rows) if updates else []

    @property
    def keys(self) -> list[str]:
        return [row.key for row in self.rows]


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "hpca.db")
    init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def deps(conn, tmp_path, recorder):
    async def run_db(fn):
        # The real DbIO runs this on its own thread; a poller written against
        # `await deps.db(fn)` cannot tell, which is why the seam is worth
        # having (see hpca.core.deps).
        return fn(conn)

    return CoreDeps(
        settings=Settings(),
        app_dir=tmp_path,
        db=run_db,
        emit=recorder.emit,
        conn=conn,
    )


@pytest.fixture
def session(conn, deps):
    """One open session — what `session.focus` would have named."""
    created = SessionStore(conn).create(profile="default")
    deps.focused_session_id = created.session_id
    return created


def pollers(deps, recorder, **kwargs) -> Pollers:
    return Pollers(deps, submit_event=recorder.submit, **kwargs)


def add_job(conn, session, job_id="27744534", script_key="pipeline"):
    return JobStore(conn).add(
        job_id=job_id,
        kind="sbatch",
        session_id=session.session_id,
        profile="default",
        script_key=script_key,
        stdout_path=f"/logs/{job_id}.out",
        stderr_path=f"/logs/{job_id}.err",
    )


def watched_job(conn, session, job_id="27744534"):
    """One box whose rendering does not tick.

    A log box counts up from its last write, so two repaints a few
    milliseconds apart can still differ if a second boundary falls between
    them. A job box is its state and nothing else, which is what makes it the
    right instrument for asserting a column was or was not re-sent.
    """
    return WatchStore(conn).add(
        kind=KIND_JOB,
        target=job_id,
        profile="default",
        session_id=session.session_id,
    )


def written(path, text="progress\n", *, age_s=0.0):
    """A log file whose last write was ``age_s`` seconds ago."""
    path.write_text(text)
    if age_s:
        when = datetime.now().timestamp() - age_s
        utime(path, (when, when))
    return path


class TestJobPoll:
    async def test_a_terminal_state_notifies_and_reaches_the_agent(
        self, conn, deps, recorder, session
    ):
        add_job(conn, session)
        deps.slurm = SlurmClient(run=FakeRun([(0, sacct("27744534", "COMPLETED"), "")]))
        await pollers(deps, recorder).poll_jobs()

        assert any("SUBMITTED → COMPLETED" in t for t in recorder.toasts)
        assert len(recorder.submitted) == 1
        sid, text = recorder.submitted[0]
        assert sid == session.session_id
        assert "[job completed]" in text
        assert "27744534" in text
        assert "triage_job" in text  # and what to do about it

    async def test_a_job_merely_starting_is_news_for_the_reader_only(
        self, conn, deps, recorder, session
    ):
        """RUNNING is worth a toast and not a turn: there is nothing for the
        agent to do about it, and a turn costs a real model call."""
        add_job(conn, session)
        deps.slurm = SlurmClient(run=FakeRun([(0, sacct("27744534", "RUNNING"), "")]))
        await pollers(deps, recorder).poll_jobs()

        assert any("→ RUNNING" in t for t in recorder.toasts)
        assert recorder.submitted == []

    async def test_a_failing_sacct_call_degrades_to_a_notify(
        self, conn, deps, recorder, session
    ):
        add_job(conn, session)
        deps.slurm = SlurmClient(run=FakeRun([(1, "", "sacct: connection timed out")]))
        await pollers(deps, recorder).poll_jobs()  # must not raise

        assert any("Job polling failed" in t for t in recorder.toasts)
        assert recorder.of(PanelUpdate) == []

    async def test_a_terminal_job_stops_costing_a_call(
        self, conn, deps, recorder, session
    ):
        """JobStore.active() is what the sweep asks about, so a COMPLETED job
        is not in the next one — FakeRun raises if it were."""
        add_job(conn, session)
        run = FakeRun([(0, sacct("27744534", "COMPLETED"), "")])
        deps.slurm = SlurmClient(run=run)
        poller = pollers(deps, recorder)
        await poller.poll_jobs()
        await poller.poll_jobs()
        assert len(run.calls) == 1


class TestPanelOrder:
    async def test_the_column_is_watches_and_nothing_else(
        self, conn, deps, recorder, session, tmp_path
    ):
        """A submitted job and a finished subprocess used to get rows of their
        own under a "── this session ──" heading. Both are in the chat log a
        column to the left; the panel is for what was asked to be watched."""
        WatchStore(conn).add(
            kind=KIND_LOG, target=str(written(tmp_path / "a.log")), profile="default",
            session_id=session.session_id,
        )
        add_job(conn, session)
        conn.execute(
            "INSERT INTO processes (pid, session_id, name, cmd, state, "
            "stdout_path, stderr_path, started_at) "
            "VALUES (4242, ?, 'align', 'bash /tmp/x.sh', 'finished', "
            "'/tmp/o', '/tmp/e', '2026-07-19T10:00:00+00:00')",
            (session.session_id,),
        )
        conn.commit()
        await pollers(deps, recorder).refresh_panel()

        assert recorder.keys == ["w1"]
        assert [row.kind for row in recorder.rows] == [PANEL_WATCH]

    async def test_boxes_come_out_in_the_order_the_store_holds_them(
        self, conn, deps, recorder, session, tmp_path
    ):
        """The user arranges the column with alt+↑/alt+↓ (WatchStore.move), so
        a poll has to paint the store's order and never impose its own."""
        store = WatchStore(conn)
        for name in ("a", "b", "c"):
            store.add(
                kind=KIND_LOG, target=str(written(tmp_path / f"{name}.log")),
                profile="default", session_id=session.session_id,
            )
        moved = store.list(session_id=session.session_id)[2]
        store.move(moved.id, -1)
        await pollers(deps, recorder).refresh_panel()

        assert recorder.keys == ["w1", "w3", "w2"]

    async def test_rows_carry_the_id_a_keypress_acts_on(
        self, conn, deps, recorder, session, tmp_path
    ):
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(written(tmp_path / "a.log")),
            label="sniffles", profile="default", session_id=session.session_id,
        )
        await pollers(deps, recorder).refresh_panel()

        row = recorder.rows[0]
        assert row.kind == PANEL_WATCH
        assert row.ref == str(watch.id)
        assert row.title == "sniffles"

    async def test_a_session_with_no_watches_gets_an_empty_column(
        self, conn, deps, recorder, session
    ):
        """It used to fill with the session's run history instead, which is
        why the boxes were never the first thing read."""
        add_job(conn, session)
        await pollers(deps, recorder).refresh_panel()
        assert recorder.keys == []

    async def test_a_watch_belongs_to_the_session_that_made_it(
        self, conn, deps, recorder, session, tmp_path
    ):
        """The column shows the focused session's own watches, so one
        registered in another session stays out of it."""
        WatchStore(conn).add(
            kind=KIND_LOG, target=str(written(tmp_path / "a.log")),
            profile="default", session_id="some-other-session",
        )
        await pollers(deps, recorder).refresh_panel()
        assert recorder.keys == []

    async def test_with_no_session_focused_there_are_no_watches_to_show(
        self, conn, deps, recorder, session, tmp_path
    ):
        """A watch describes a conversation, so with none open there is
        nothing of anyone's the column could honestly be showing."""
        WatchStore(conn).add(
            kind=KIND_LOG, target=str(written(tmp_path / "a.log")),
            profile="default", session_id=session.session_id,
        )
        deps.focused_session_id = None
        await pollers(deps, recorder).refresh_panel()
        assert recorder.keys == []


class TestPanelPush:
    async def test_an_unchanged_column_is_not_resent(
        self, conn, deps, recorder, session
    ):
        watched_job(conn, session)
        poller = pollers(deps, recorder)
        await poller.refresh_panel()
        await poller.refresh_panel()
        assert len(recorder.of(PanelUpdate)) == 1

    async def test_force_resends_it_for_a_client_with_a_blank_screen(
        self, conn, deps, recorder, session
    ):
        watched_job(conn, session)
        poller = pollers(deps, recorder)
        await poller.refresh_panel()
        await poller.refresh_panel(force=True)
        assert len(recorder.of(PanelUpdate)) == 2

    async def test_a_ticking_watch_box_keeps_being_sent(
        self, conn, deps, recorder, session, tmp_path
    ):
        """"last write 4s ago" counts up on its own, so those rows differ
        every pass even when nothing in the database moved.

        Both clocks are pinned. Reading ``changed_at`` off the real clock
        while the two ``now`` values stayed fixed made this pass only when
        the suite ran before noon UTC: any later and both ``now`` values sit
        *behind* the write, so both ages clamp to zero and the rows match.
        """
        wrote_at = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(tmp_path / "a.log"), profile="default"
        )
        WatchStore(conn).update(
            watch.id,
            state="writing",
            changed_at=wrote_at.isoformat(),
        )
        poller = pollers(deps, recorder)
        first = poller.panel_rows(
            WatchStore(conn).list(profile="default"),
            now=wrote_at + timedelta(seconds=4),
        )
        later = poller.panel_rows(
            WatchStore(conn).list(profile="default"),
            now=wrote_at + timedelta(minutes=5),
        )
        assert first != later

    async def test_the_panel_names_the_session_it_describes(
        self, conn, deps, recorder, session
    ):
        watched_job(conn, session)
        await pollers(deps, recorder).refresh_panel()
        update = recorder.of(PanelUpdate)[0]
        assert update.session_id == session.session_id
        assert update.profile == "default"


class TestWatchedLogs:
    async def test_the_first_poll_of_a_new_watch_is_not_news(
        self, conn, deps, recorder, session, tmp_path
    ):
        """It moves the watch off "" into a state, which is not a change of
        state — and an hours-old log the user just asked about is not a
        surprise to them."""
        WatchStore(conn).add(
            kind=KIND_LOG,
            target=str(written(tmp_path / "a.log", age_s=7200)),
            profile="default",
        )
        await pollers(deps, recorder).poll_watched_logs()
        assert recorder.toasts == []

    async def test_a_log_going_quiet_notifies_about_nothing(
        self, conn, deps, recorder, session, tmp_path
    ):
        """A log not being written to for a while is not an event. The box
        already says when the last write was, and that number climbs on its
        own; a toast for it interrupts to report that nothing happened."""
        log = written(tmp_path / "sniffles.log")
        WatchStore(conn).add(
            kind=KIND_LOG, target=str(log), label="sniffles",
            profile="default", session_id=session.session_id,
        )
        poller = pollers(deps, recorder)
        await poller.poll_watched_logs()

        written(log, age_s=3600)  # nothing new for an hour
        recorder.toasts.clear()
        await poller.poll_watched_logs()
        assert recorder.toasts == []

        await poller.poll_watched_logs()  # and still nothing, however long
        assert recorder.toasts == []

    async def test_a_vanished_log_says_so(
        self, conn, deps, recorder, session, tmp_path
    ):
        log = written(tmp_path / "gone.log")
        WatchStore(conn).add(
            kind=KIND_LOG, target=str(log), label="job out", profile="default"
        )
        poller = pollers(deps, recorder)
        await poller.poll_watched_logs()
        log.unlink()
        await poller.poll_watched_logs()
        assert any("the file is gone" in t for t in recorder.toasts)

    async def test_a_change_repaints_the_column(
        self, conn, deps, recorder, session, tmp_path
    ):
        WatchStore(conn).add(
            kind=KIND_LOG, target=str(written(tmp_path / "a.log")), profile="default",
            session_id=session.session_id,
        )
        await pollers(deps, recorder).poll_watched_logs()
        assert recorder.keys == ["w1"]


class TestWatchedJobs:
    async def test_a_state_change_notifies_and_repaints(
        self, conn, deps, recorder, session
    ):
        watch = WatchStore(conn).add(
            kind=KIND_JOB, target="27744534", label="sniffles",
            profile="default", session_id=session.session_id,
        )
        WatchStore(conn).update(watch.id, state="PENDING")
        squeue = "27744534|RUNNING|node042|00:10:00|1-00:00:00|smk|\n"
        deps.slurm = SlurmClient(run=FakeRun([(0, squeue, "")]))
        await pollers(deps, recorder).poll_watched_jobs()

        assert any("sniffles: PENDING → RUNNING" in t for t in recorder.toasts)
        assert recorder.keys == ["w1"]

    async def test_a_settled_job_stops_costing_a_call(
        self, conn, deps, recorder, session
    ):
        watch = WatchStore(conn).add(
            kind=KIND_JOB, target="27744534", profile="default"
        )
        WatchStore(conn).update(watch.id, state="COMPLETED")
        run = FakeRun()  # raises if anything is asked
        deps.slurm = SlurmClient(run=run)
        await pollers(deps, recorder).poll_watched_jobs()
        assert run.calls == []

    async def test_a_failing_squeue_call_degrades_to_a_notify(
        self, conn, deps, recorder, session
    ):
        watch = WatchStore(conn).add(
            kind=KIND_JOB, target="27744534", profile="default"
        )
        WatchStore(conn).update(watch.id, state="PENDING")
        deps.slurm = SlurmClient(run=FakeRun([(1, "", "slurm_load_jobs error")]))
        await pollers(deps, recorder).poll_watched_jobs()  # must not raise

        assert any("Job watch failed" in t for t in recorder.toasts)


class TestFinishedProcesses:
    async def test_a_finished_background_script_reaches_the_agent(
        self, conn, deps, recorder, session, tmp_path
    ):
        runner = ProcessRunner(
            conn, session_id=session.session_id, log_dir=tmp_path / "logs"
        )
        record = await runner.start(
            ["bash", "-c", "echo '187 reads extracted'"],
            name="extract",
            background=True,
        )
        await runner.wait(record.pid)
        await pollers(deps, recorder).watch_processes()

        assert len(recorder.submitted) == 1
        sid, text = recorder.submitted[0]
        assert sid == session.session_id
        assert "[process finished]" in text
        assert "187 reads extracted" in text

    async def test_a_failure_carries_the_triage_text(
        self, conn, deps, recorder, session, tmp_path
    ):
        runner = ProcessRunner(
            conn, session_id=session.session_id, log_dir=tmp_path / "logs"
        )
        record = await runner.start(
            ["bash", "-c", "echo 'mamba: command not found' >&2; exit 127"],
            name="sniffles_run",
            background=True,
        )
        await runner.wait(record.pid)
        await pollers(deps, recorder).watch_processes()

        _, text = recorder.submitted[0]
        assert "[process failed]" in text
        assert "mamba: command not found" in text  # the cause, inline
        assert "fix the script" in text

    async def test_a_completion_is_delivered_exactly_once(
        self, conn, deps, recorder, session, tmp_path
    ):
        runner = ProcessRunner(
            conn, session_id=session.session_id, log_dir=tmp_path / "logs"
        )
        record = await runner.start(["true"], name="quick", background=True)
        await runner.wait(record.pid)
        poller = pollers(deps, recorder)
        await poller.watch_processes()
        await poller.watch_processes()
        assert len(recorder.submitted) == 1

    async def test_it_leaves_the_repaint_to_the_panel_sweep(
        self, conn, deps, recorder, session, tmp_path
    ):
        runner = ProcessRunner(
            conn, session_id=session.session_id, log_dir=tmp_path / "logs"
        )
        record = await runner.start(["true"], name="quick", background=True)
        await runner.wait(record.pid)
        await pollers(deps, recorder).watch_processes()
        assert recorder.of(PanelUpdate) == []


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return ChatResponse(content=self.reply)


def tier_two() -> LogFinding:
    """What the keyword scan leaves behind when no signature matched: lines
    that might be the cause, and no way to say which."""
    return LogFinding(
        tier=2,
        title="Possible error lines",
        matched_line="Error: contig chr1 not in reference",
        excerpt="reading BAM\nError: contig chr1 not in reference\n",
        candidates=[
            Candidate(
                line_no=12,
                line="Error: contig chr1 not in reference",
                score=3.0,
                excerpt="reading BAM\nError: contig chr1 not in reference\n",
            ),
            Candidate(
                line_no=3, line="warning: 4 reads skipped", score=1.0, excerpt=""
            ),
        ],
    )


def explanation(**over) -> str:
    body = {
        "why": "the reference lacks that contig",
        "cause_line": "Error: contig chr1 not in reference",
        "suggested_fix": "point it at the matching reference",
        "proposed_signature": None,
    }
    body.update(over)
    return json.dumps(body)


class TestTierThree:
    """The one branch that spends a model call: the scan found candidates but
    cannot say which one stopped the run."""

    async def test_a_conclusive_explanation_replaces_the_candidate_list(
        self, deps, recorder
    ):
        llm = FakeLLM(explanation())
        poller = pollers(deps, recorder, llm=lambda: llm)
        text = await poller._explain_candidates(
            _change(), tier_two(), "[process failed] head\n\nfallback body"
        )
        assert "the reference lacks that contig" in text
        assert "[process failed] head" in text  # the head is kept
        assert "fallback body" not in text

    async def test_an_inconclusive_one_is_not_dressed_up(self, deps, recorder):
        llm = FakeLLM(explanation(cause_line=""))
        poller = pollers(deps, recorder, llm=lambda: llm)
        text = await poller._explain_candidates(
            _change(), tier_two(), "the fallback"
        )
        assert text == "the fallback"

    async def test_no_backend_means_the_fallback_text_unchanged(
        self, deps, recorder
    ):
        poller = pollers(deps, recorder)
        text = await poller._explain_candidates(
            _change(), tier_two(), "the fallback"
        )
        assert text == "the fallback"

    async def test_an_explainer_failure_degrades_to_a_notify(
        self, deps, recorder
    ):
        class Broken:
            async def chat(self, messages, **kwargs):
                raise RuntimeError("backend down")

        poller = pollers(deps, recorder, llm=lambda: Broken())
        text = await poller._explain_candidates(
            _change(), tier_two(), "the fallback"
        )
        assert text == "the fallback"
        assert any("Log explainer failed" in t for t in recorder.toasts)

    async def test_a_proposed_signature_is_offered_not_written(
        self, deps, recorder
    ):
        """The library is the user's; one model call is thin evidence."""
        asked: list[tuple[str, str]] = []
        llm = FakeLLM(
            explanation(
                proposed_signature={
                    "id": "missing_contig",
                    "title": "Contig not in reference",
                    "patterns": ["contig .* not in reference"],
                    "hint": "check the reference",
                }
            )
        )
        poller = pollers(
            deps, recorder, llm=lambda: llm,
            confirm=lambda session_id, question, on_yes: asked.append(
                (session_id, question)
            ),
        )
        await poller._explain_candidates(_change(), tier_two(), "head\n\nbody")
        assert len(asked) == 1
        session_id, question = asked[0]
        assert "missing_contig" in question
        # The two things whoever answers needs and a poll is the only one
        # holding: which conversation this came out of, and which job failed.
        assert session_id == "s1"
        assert "sniffles_run" in question

    async def test_with_nothing_to_ask_with_the_proposal_is_dropped(
        self, deps, recorder
    ):
        llm = FakeLLM(
            explanation(
                proposed_signature={
                    "id": "missing_contig", "title": "t",
                    "patterns": ["nope"], "hint": "",
                }
            )
        )
        poller = pollers(deps, recorder, llm=lambda: llm)
        text = await poller._explain_candidates(
            _change(), tier_two(), "head\n\nbody"
        )
        assert "the reference lacks that contig" in text  # still diagnosed

    async def test_a_yes_writes_the_signature_and_says_where(
        self, deps, recorder, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        accepted: list = []
        poller = pollers(
            deps, recorder,
            confirm=lambda session_id, question, on_yes: accepted.append(on_yes),
        )
        poller._offer_signature(
            _change(),
            _proposed(id="missing_contig", patterns=["contig .* not in reference"]),
        )
        await accepted[0]()
        assert any("Saved error signature" in t for t in recorder.toasts)
        assert "missing_contig" in (tmp_path / "error_signatures.yaml").read_text()

    async def test_a_regex_that_does_not_compile_is_not_even_offered(
        self, deps, recorder, tmp_path, monkeypatch
    ):
        """A pattern the model got wrong must not take the timer down with
        it, and there is nothing to ask the user about."""
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        accepted: list = []
        poller = pollers(
            deps, recorder,
            confirm=lambda session_id, question, on_yes: accepted.append(on_yes),
        )
        poller._offer_signature(
            _change(), _proposed(id="broken", patterns=["(unclosed"])
        )
        assert accepted == []
        assert any("Ignored a proposed error signature" in t for t in recorder.toasts)
        assert not (tmp_path / "error_signatures.yaml").exists()


def _change():
    return ProcessChange(
        pid=1, session_id="s1", name="sniffles_run", state="failed",
        exit_code=1, stdout_path=Path("/tmp/o"), stderr_path=Path("/tmp/e"),
    )


def _proposed(**over):
    body = {"id": "x", "title": "a title", "patterns": ["boom"], "hint": "h"}
    body.update(over)
    return ProposedSignature(**body)


class TestQuietWhenThereIsNothingToSay:
    """Every sweep runs on a timer whether or not anything happened, so the
    empty case is the common one — and a core that chatters on an idle
    machine costs the socket, the renderer and the user's attention."""

    async def test_nothing_pending_emits_nothing_at_all(
        self, conn, deps, recorder
    ):
        run = FakeRun()  # raises if the cluster is asked anything
        deps.slurm = SlurmClient(run=run)
        poller = pollers(deps, recorder)
        await poller.refresh_panel()
        await poller.poll_jobs()
        await poller.poll_watched_logs()
        await poller.poll_watched_jobs()
        await poller.watch_processes()

        assert recorder.events == []
        assert recorder.submitted == []
        assert run.calls == []

    async def test_an_open_but_idle_session_is_just_as_quiet(
        self, conn, deps, recorder, session
    ):
        poller = pollers(deps, recorder)
        await poller.refresh_panel()
        await poller.watch_processes()
        assert recorder.events == []


class TestSchedule:
    """The cadence is the core's, the loop is not."""

    def test_without_slurm_the_cluster_sweeps_are_absent(self, deps, recorder):
        names = [fn.__name__ for _, fn in pollers(deps, recorder).timers()]
        assert "poll_jobs" not in names
        assert "poll_watched_jobs" not in names
        assert "refresh_panel" in names
        assert "watch_processes" in names

    def test_the_sacct_cadence_has_a_floor(self, deps, recorder):
        deps.slurm = SlurmClient(run=FakeRun())
        deps.settings.cluster.job_poll_seconds = 1
        intervals = {
            fn.__name__: seconds
            for seconds, fn in pollers(deps, recorder).timers()
        }
        assert intervals["poll_jobs"] == 5.0
