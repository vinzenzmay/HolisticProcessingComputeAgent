"""The right column as a monitor: watch boxes, their hotkeys, their repaint.

The panel used to show only what hpca itself started, which is both a small
slice of what runs on a cluster and a slice already in the chat log. These
tests are about the other half — the log the user asked to be shown, and the
three keys that make it useful: Enter to peek, D to drop, arrows to move
without the repaint stealing the cursor.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import ListView, Static

from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, HpcaApp
from hpca.watches import KIND_JOB, KIND_LOG, WatchStore


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        return ChatResponse(content=self._outputs.pop(0) if self._outputs else "")

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def written(path, text="progress\n", *, age_s=0.0):
    """A log file whose last write was ``age_s`` seconds ago."""
    path.write_text(text)
    if age_s:
        when = datetime.now().timestamp() - age_s
        os.utime(path, (when, when))
    return path


async def open_session(app, pilot):
    await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = "hello"
    await pilot.press("enter")
    await app.workers.wait_for_complete()


async def add_log_watch(app, pilot, path, label=""):
    # Registered against the session that is actually open, as watch_log does:
    # a watch belongs to its session now, so a hardcoded id would register it
    # somewhere the panel is right not to show it.
    watch = app.watch_store.add(
        kind=KIND_LOG,
        target=str(path),
        label=label,
        profile=app._panel_profile(),
        session_id=app._panel_session() or "",
    )
    await app.poll_watched_logs()
    await app.refresh_processes()
    await pilot.pause()
    return watch


def panel(app):
    return app.query_one("#processes-list", ListView)


def box_texts(app):
    """What every row of the column actually renders."""
    return [
        str(item.query_one(Static).render())
        for item in panel(app).children
    ]


def watch_rows(app):
    return [
        item for item in panel(app).children
        if getattr(item, "data_watch", None) is not None
    ]


def toasts(app):
    return [n.message for n in app._notifications]


class TestTheBox:
    async def test_a_watched_log_gets_a_box_of_its_own(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            log = written(tmp_path / "sniffles.log")
            await add_log_watch(app, pilot, log, "sniffles")
            assert len(watch_rows(app)) == 1
            text = box_texts(app)[0]
            assert "last write" in text
            # No alive/dead word: the mtime cannot support that claim.
            assert "writing" not in text and "idle" not in text

    async def test_the_box_says_how_long_ago_the_log_was_written(
        self, hpca_home, tmp_path
    ):
        """The whole point, and the whole claim: when it was last written."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(
                app, pilot, written(tmp_path / "old.log", age_s=3700)
            )
            text = box_texts(app)[0]
            assert "last write 1h01m ago" in text
            assert "idle" not in text

    async def test_the_boxes_come_before_the_run_history(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            record = await app._tool_ctx.runner.start(["true"], name="align")
            await add_log_watch(app, pilot, written(tmp_path / "a.log"))
            keys = [getattr(i, "data_key", "") for i in panel(app).children]
            assert keys[0].startswith("w")
            assert f"p{record.pid}" in keys
            assert "h:session" in keys  # and the two halves are labelled

    async def test_a_watch_stays_in_the_session_it_was_made_in(
        self, hpca_home, tmp_path
    ):
        """A watch belongs to its conversation, not to the machine.

        It was the other way round, and in practice that meant every session
        showed every other session's boxes — sessions on one profile are the
        normal case — so the column stopped describing what was being read.
        """
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "kept")
            first = app.active_session
            assert len(watch_rows(app)) == 1

            await app.start_new_session()
            await app.refresh_processes()
            await pilot.pause()
            assert watch_rows(app) == []

            # ...and it is still there on the way back.
            await app.open_session(first)
            await app.refresh_processes()
            await pilot.pause()
            assert len(watch_rows(app)) == 1

    async def test_with_no_session_open_the_column_shows_no_watches(
        self, hpca_home, tmp_path
    ):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "kept")
            await app.close_session()
            await app.refresh_processes()
            await pilot.pause()
            assert watch_rows(app) == []

    async def test_it_survives_a_restart(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "kept")

        app2 = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app2.run_test(size=(120, 40)) as pilot:
            # Reopened rather than merely restarted: the watch belongs to its
            # session, so finding it again means going back to that session.
            await app2.open_session(app2.session_store.list_all()[0])
            await app2.refresh_processes()
            await pilot.pause()
            assert len(watch_rows(app2)) == 1


class TestPeek:
    async def test_enter_flashes_the_tail_of_the_log(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            log = written(
                tmp_path / "sniffles.log",
                "reading BAM\n[ERROR] contig not found\n",
            )
            await add_log_watch(app, pilot, log, "sniffles")
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert any("contig not found" in t for t in toasts(app))

    async def test_the_peek_is_a_toast_not_a_screen_to_dismiss(
        self, hpca_home, tmp_path
    ):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"))
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is app.screen_stack[0]

    async def test_peeking_a_job_box_shows_its_state(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            watch = app.watch_store.add(
                kind=KIND_JOB, target="27744534", label="snakemake",
                profile=app._panel_profile(),
                session_id=app._panel_session() or "",
            )
            app.watch_store.update(
                watch.id, state="RUNNING", head="01:02:03", detail="node042"
            )
            await app.refresh_processes()
            await pilot.pause()
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any("RUNNING" in t and "node042" in t for t in toasts(app))

    async def test_a_job_hpca_submitted_peeks_its_output_too(
        self, hpca_home, tmp_path
    ):
        """The user's case: an sbatch job whose stdout is the snakemake log."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            out = written(tmp_path / "job.out", "rule sniffles: 12 of 30 done\n")
            app.job_store.add(
                job_id="27744534", kind="sbatch",
                session_id=app.active_session.session_id, profile="default",
                script_key="pipeline", stdout_path=str(out),
                stderr_path=str(tmp_path / "job.err"),
            )
            watch = app.watch_store.add(
                kind=KIND_JOB, target="27744534", label="pipeline",
                profile=app._panel_profile(),
                session_id=app._panel_session() or "",
            )
            app.watch_store.update(watch.id, state="RUNNING")
            await app.refresh_processes()
            await pilot.pause()
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any("12 of 30 done" in t for t in toasts(app))


class TestDrop:
    async def test_d_removes_the_watch_and_its_box(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "done-with")
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("d")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert watch_rows(app) == []
            assert app.watch_store.list(profile=app._panel_profile()) == []

    async def test_d_leaves_the_log_file_alone(self, hpca_home, tmp_path):
        """Dropping a box stops the watching, it does not delete anything."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            log = written(tmp_path / "a.log")
            await add_log_watch(app, pilot, log)
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("d")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert log.exists()

    async def test_d_on_a_process_row_does_nothing(self, hpca_home):
        """Only watches are droppable; a process row keeps its own k."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await app._tool_ctx.runner.start(["true"], name="align")
            await app.refresh_processes()
            await pilot.pause()
            panel(app).focus()
            panel(app).index = 0
            await pilot.press("d")
            await pilot.pause()
            assert len(panel(app).children) == 1


class TestRepaint:
    async def test_the_cursor_stays_put_while_the_clock_ticks(
        self, hpca_home, tmp_path
    ):
        """A box counts up every repaint. Rebuilding the list for that used to
        drop the highlight back to the top twice a second, which makes the
        column impossible to arrow through."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            for name in ("a.log", "b.log", "c.log"):
                await add_log_watch(app, pilot, written(tmp_path / name))
            panel(app).focus()
            panel(app).index = 2
            first = box_texts(app)[2]
            await app.refresh_processes()
            await pilot.pause()
            assert panel(app).index == 2
            assert box_texts(app)[2] == first  # same row, still ours

    async def test_the_text_still_refreshes_in_place(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            log = written(tmp_path / "a.log")
            watch = await add_log_watch(app, pilot, log)
            app.watch_store.update(
                watch.id,
                state="idle",
                head="9.9 MB",
                changed_at=(
                    datetime.now(timezone.utc) - timedelta(minutes=5)
                ).isoformat(),
            )
            await app.refresh_processes()
            await pilot.pause()
            assert "9.9 MB" in box_texts(app)[0]
            assert "last write 5m ago" in box_texts(app)[0]

    async def test_a_new_watch_appears_without_losing_the_selection(
        self, hpca_home, tmp_path
    ):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            kept = await add_log_watch(app, pilot, written(tmp_path / "a.log"))
            panel(app).focus()
            panel(app).index = 0
            await add_log_watch(app, pilot, written(tmp_path / "b.log"))
            assert len(watch_rows(app)) == 2
            assert getattr(panel(app).highlighted_child, "data_watch").id == kept.id


class TestQuietLogNotice:
    async def test_a_log_going_quiet_raises_nothing(self, hpca_home, tmp_path):
        """It used to toast "no new output". That is not an event.

        The box already says when the last write was and that number climbs
        on its own; interrupting the user to report that nothing happened
        trains them to ignore the toasts that matter.
        """
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            log = written(tmp_path / "sniffles.log")
            await add_log_watch(app, pilot, log, "sniffles")
            written(log, age_s=3600)  # nothing new for an hour
            await app.poll_watched_logs()
            await pilot.pause()
            assert not any("no new output" in t for t in toasts(app))

    async def test_a_log_that_vanishes_still_raises_a_toast(
        self, hpca_home, tmp_path
    ):
        """The one thing a log poll observes rather than infers."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            log = written(tmp_path / "sniffles.log")
            await add_log_watch(app, pilot, log, "sniffles")
            log.unlink()
            await app.poll_watched_logs()
            await pilot.pause()
            assert any("sniffles" in t and "gone" in t for t in toasts(app))

    async def test_the_first_poll_of_a_fresh_watch_is_not_news(
        self, hpca_home, tmp_path
    ):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            store = WatchStore(app._conn)
            store.add(
                kind=KIND_LOG,
                target=str(written(tmp_path / "a.log", age_s=7200)),
                profile=app._panel_profile(),
            )
            await app.poll_watched_logs()
            await pilot.pause()
            assert not any("no new output" in t for t in toasts(app))
