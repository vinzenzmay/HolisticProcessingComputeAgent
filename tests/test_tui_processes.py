"""Tests for the right column: process list, inspect, kill (§3.3)."""

import json

import pytest
from pathlib import Path

from textual.widgets import ListView, Static

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.inspect_screen import InspectScreen


def is_title_request(json_schema):
    """The app names a session by asking the model (§3 sessions column); that
    call is not one of the queued decisions."""
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})

class FakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def start_session_with_process(app, pilot, argv, name):
    """Open a session via chat, then start a tracked process in its runner."""
    await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "hello"
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    record = await app._tool_ctx.runner.start(argv, name=name)
    await app.refresh_processes()
    await pilot.pause()
    return record


class TestProcessList:
    async def test_process_appears_in_right_column(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["sleep", "5"], "sleeper"
            )
            processes_list = app.query_one("#processes-list", ListView)
            assert len(processes_list) == 1
            assert getattr(processes_list.children[0], "data_record").pid == record.pid
            await app._tool_ctx.runner.kill(record.pid)

    async def test_finished_process_shows_state(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["echo", "hi"], "quickie"
            )
            await app._tool_ctx.runner.wait(record.pid)
            await app.refresh_processes()
            await pilot.pause()
            items = app.query_one("#processes-list", ListView).children
            assert getattr(items[0], "data_record").state == "finished"


class TestInspect:
    async def test_enter_opens_inspect_with_logs(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["bash", "-c", "echo needle-out; echo needle-err >&2"],
                "loggy",
            )
            await app._tool_ctx.runner.wait(record.pid)
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, InspectScreen)
            body = app.screen.body_text()
            assert "needle-out" in body
            assert "needle-err" in body
            await pilot.press("escape")
            assert not isinstance(app.screen, InspectScreen)


class TestKill:
    async def test_k_confirms_then_kills(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["sleep", "60"], "victim"
            )
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("k")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await app._tool_ctx.runner.wait(record.pid)
            await pilot.pause()
            assert app._tool_ctx.runner.get(record.pid).state == "killed"

    async def test_kill_denied_leaves_process_running(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(
                app, pilot, ["sleep", "60"], "survivor"
            )
            processes_list = app.query_one("#processes-list", ListView)
            processes_list.focus()
            processes_list.index = 0
            await pilot.press("k")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert app._tool_ctx.runner.get(record.pid).state == "running"
            await app._tool_ctx.runner.kill(record.pid)


# ------------------------------------------------- persisted process history

from datetime import datetime, timedelta, timezone  # noqa: E402

from hpca.tui.app import format_started  # noqa: E402
from hpca.tui.inspect_screen import format_process  # noqa: E402


def process_labels(app):
    return [
        str(item.query_one(Static).render())
        for item in app.query_one("#processes-list", ListView).children
    ]


def panel_rows(app):
    """(name, has_record) for each row, read from the attached data."""
    rows = []
    for item in app.query_one("#processes-list", ListView).children:
        record = getattr(item, "data_record", None)
        rows.append(record.name if record is not None else None)
    return rows


class TestHistoryPersists:
    """The reported bug: the panel emptied whenever the session was reopened,
    because it read the live runner and the TUI builds a new one per turn."""

    async def test_survives_leaving_and_reopening_the_session(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await start_session_with_process(app, pilot, ["true"], "align")
            session = app.active_session
            assert "align" in panel_rows(app)

            await app.close_session()
            await app.refresh_processes()
            await pilot.pause()
            assert panel_rows(app) == []  # nothing open, nothing to show

            await app.open_session(session)
            await app.refresh_processes()
            await pilot.pause()
            assert "align" in panel_rows(app)

    async def test_survives_a_restart(self, hpca_home):
        """A fresh app over the same database still has the history."""
        app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await start_session_with_process(app, pilot, ["true"], "align")
            session_id = app.active_session.session_id

        app2 = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app2.run_test(size=(120, 40)) as pilot:
            session = app2.session_store.get(session_id)
            await app2.open_session(session)
            await app2.refresh_processes()
            await pilot.pause()
            assert "align" in panel_rows(app2)

    async def test_a_processes_history_is_its_own_sessions(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await start_session_with_process(app, pilot, ["true"], "first-session")
            await app.start_new_session()
            await app.refresh_processes()
            await pilot.pause()
            assert "first-session" not in panel_rows(app)

    async def test_older_rows_are_summarised_not_silently_dropped(self, hpca_home):
        from hpca.tui.app import PROCESS_HISTORY_LIMIT

        app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await start_session_with_process(app, pilot, ["true"], "p0")
            session_id = app.active_session.session_id
            for i in range(PROCESS_HISTORY_LIMIT + 4):
                app._conn.execute(
                    "INSERT INTO processes (pid, session_id, name, cmd, state, "
                    "stdout_path, stderr_path, started_at) "
                    "VALUES (?, ?, ?, ?, 'finished', '/tmp/o', '/tmp/e', ?)",
                    (9000 + i, session_id, f"p{i}", "bash /tmp/x.sh",
                     f"2026-07-19T10:{i:02d}:00+00:00"),
                )
            app._conn.commit()
            app._panel_keys = None
            await app.refresh_processes()
            await pilot.pause()
            labels = [
                lbl for lbl in process_labels(app) if "older" in lbl
            ]
            assert labels, "a truncated list must say so"


class TestStartTime:
    def test_today_shows_only_the_clock(self):
        now = datetime.now(timezone.utc)
        assert format_started(now.isoformat()).strip() == now.astimezone().strftime(
            "%H:%M"
        )

    def test_another_day_shows_the_date(self):
        earlier = datetime.now(timezone.utc) - timedelta(days=3)
        text = format_started(earlier.isoformat())
        assert text == earlier.astimezone().strftime("%m-%d %H:%M")

    def test_utc_is_converted_to_local(self):
        """Stored UTC, read next to a wall clock."""
        stamp = "2026-07-19T10:00:00+00:00"
        expected = datetime.fromisoformat(stamp).astimezone().strftime("%H:%M")
        assert expected in format_started(stamp)

    def test_missing_or_broken_stamps_do_not_crash(self):
        assert format_started("") == "  --  "
        assert format_started("not a date") == "  --  "

    async def test_the_panel_shows_the_start_time(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app.run_test(size=(120, 40)) as pilot:
            record = await start_session_with_process(app, pilot, ["true"], "align")
            expected = format_started(record.started_at).strip()
            assert any(expected in label for label in process_labels(app))


class TestScriptInspection:
    """Output without the script behind it is half the story."""

    async def test_inspect_shows_the_script_that_ran(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app.run_test(size=(120, 40)) as pilot:
            script = hpca_home / "scripts" / "align.sh"
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("#!/bin/bash\nsamtools view -b in.bam\n")
            await start_session_with_process(
                app, pilot, ["bash", str(script)], "align"
            )
            app.query_one("#processes-list", ListView).index = 0
            app.inspect_selected_process()
            await pilot.pause()
            body = app.screen.body_text()
            assert "── script" in body
            assert "samtools view -b in.bam" in body
            assert "── stdout tail" in body  # the output is still there

    def test_a_process_without_a_script_shows_no_script_section(self):
        from hpca.runner import ProcessRecord

        record = ProcessRecord(
            pid=1, name="ls", cmd="ls -la", state="finished",
            stdout_path=Path("/tmp/o"), stderr_path=Path("/tmp/e"),
            started_at="2026-07-19T10:00:00+00:00",
        )
        assert "── script" not in format_process(record)

    def test_a_long_script_is_capped(self, tmp_path):
        from hpca.runner import ProcessRecord
        from hpca.tui.inspect_screen import SCRIPT_LINES

        script = tmp_path / "big.sh"
        script.write_text("\n".join(f"echo {i}" for i in range(SCRIPT_LINES + 50)))
        record = ProcessRecord(
            pid=1, name="big", cmd=f"bash {script}", state="finished",
            stdout_path=Path("/tmp/o"), stderr_path=Path("/tmp/e"),
            started_at="2026-07-19T10:00:00+00:00",
        )
        body = format_process(record)
        assert "50 more lines" in body

    def test_a_deleted_script_says_so_rather_than_failing(self, tmp_path):
        from hpca.runner import ProcessRecord

        record = ProcessRecord(
            pid=1, name="gone", cmd=f"bash {tmp_path / 'missing.sh'}",
            state="finished", stdout_path=Path("/tmp/o"),
            stderr_path=Path("/tmp/e"), started_at="2026-07-19T10:00:00+00:00",
        )
        assert "── script" not in format_process(record)  # nothing to point at


class TestKillAcrossTurns:
    async def test_killing_a_process_the_current_runner_does_not_own(
        self, hpca_home
    ):
        """The panel now shows earlier turns' processes, and the runner that
        started them is gone — killing must not raise."""
        app = HpcaApp(llm=FakeLLM([respond_json("ok")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await start_session_with_process(app, pilot, ["sleep", "60"], "long")
            record = app._tool_ctx.runner.list()[0]
            # a later turn replaces the runner; the old one is forgotten
            app._tool_ctx = app._make_tool_ctx(app.active_session, None)
            assert not app._tool_ctx.runner.owns(record.pid)
            await app._kill_process_and_refresh(record.pid)
            await pilot.pause()
            row = app._conn.execute(
                "SELECT state FROM processes WHERE pid = ?", (record.pid,)
            ).fetchone()
            assert row["state"] == "killed"
