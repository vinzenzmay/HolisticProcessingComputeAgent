"""The watchers column: the boxes, their hotkeys, their order, their repaint.

The panel used to show what hpca itself started — a small slice of what runs
on a cluster, and a slice already in the chat log — with the watch boxes above
it. The history is gone and the boxes are the whole column now. These tests
cover what makes it useful: Enter to peek, D to drop, alt+↑/alt+↓ to arrange,
and arrows that keep working while the clock in every box ticks.
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
    await app.refresh_watchers()
    await pilot.pause()
    return watch


def panel(app):
    return app.query_one("#watchers-list", ListView)


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

    async def test_the_boxes_are_the_whole_column(self, hpca_home, tmp_path):
        """A subprocess hpca ran used to get a row under a heading. The run is
        in the chat log already, and the history crowded out the boxes the user
        asked for on any terminal short enough to matter."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            record = await app._tool_ctx.runner.start(["true"], name="align")
            await app._tool_ctx.runner.wait(record.pid)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"))
            keys = [getattr(i, "data_key", "") for i in panel(app).children]
            assert keys == [f"w{app.watch_store.list()[0].id}"]

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
            await app.refresh_watchers()
            await pilot.pause()
            assert watch_rows(app) == []

            # ...and it is still there on the way back.
            await app.open_session(first)
            await app.refresh_watchers()
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
            await app.refresh_watchers()
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
            await app2.refresh_watchers()
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
            await app.refresh_watchers()
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
            await app.refresh_watchers()
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

    async def test_d_on_an_empty_column_does_nothing(self, hpca_home):
        """Nothing highlighted means nothing to unwatch — and check_action
        keeps the key out of the footer rather than offering a no-op."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            panel(app).focus()
            await pilot.press("d")
            await pilot.pause()
            assert len(panel(app).children) == 0


class TestReorder:
    """alt+↑/alt+↓ carry a box past its neighbour.

    Registration order is not importance order: three settled boxes can sit
    above the one log the user is actually waiting on, and on a short terminal
    that one is off the bottom of the column.
    """

    async def three_boxes(self, app, pilot, tmp_path):
        for name in ("a.log", "b.log", "c.log"):
            await add_log_watch(app, pilot, written(tmp_path / name), name)
        panel(app).focus()
        return self.titles(app)

    def titles(self, app):
        return [
            item.query_one(Static).border_title for item in panel(app).children
        ]

    async def press_move(self, app, pilot, key):
        await pilot.press(key)
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def test_alt_up_swaps_with_the_box_above(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await self.three_boxes(app, pilot, tmp_path)
            panel(app).index = 2
            await self.press_move(app, pilot, "alt+up")
            assert self.titles(app) == ["a.log", "c.log", "b.log"]

    async def test_alt_down_swaps_with_the_box_below(self, hpca_home, tmp_path):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await self.three_boxes(app, pilot, tmp_path)
            panel(app).index = 0
            await self.press_move(app, pilot, "alt+down")
            assert self.titles(app) == ["b.log", "a.log", "c.log"]

    async def test_the_cursor_follows_the_box_it_moved(self, hpca_home, tmp_path):
        """Otherwise a held-down alt+↑ swaps the same pair back and forth
        instead of walking one box to the top."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await self.three_boxes(app, pilot, tmp_path)
            panel(app).index = 2
            await self.press_move(app, pilot, "alt+up")
            assert panel(app).index == 1
            await self.press_move(app, pilot, "alt+up")
            assert self.titles(app) == ["c.log", "a.log", "b.log"]
            assert panel(app).index == 0

    async def test_the_ends_are_silent(self, hpca_home, tmp_path):
        """Arriving at the top is the normal end of holding the key down, not
        something to warn about."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await self.three_boxes(app, pilot, tmp_path)
            panel(app).index = 0
            await self.press_move(app, pilot, "alt+up")
            assert self.titles(app) == ["a.log", "b.log", "c.log"]
            assert toasts(app) == []

    async def test_the_arrangement_outlives_the_repaint(self, hpca_home, tmp_path):
        """The column is rebuilt from the store every two seconds, so an order
        the poll does not know about would be undone almost at once."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await self.three_boxes(app, pilot, tmp_path)
            panel(app).index = 2
            await self.press_move(app, pilot, "alt+up")
            await app.poll_watched_logs()
            await app.refresh_watchers()
            await pilot.pause()
            assert self.titles(app) == ["a.log", "c.log", "b.log"]

    async def test_shift_arrows_do_the_same_in_a_multiplexer(
        self, hpca_home, tmp_path
    ):
        """zellij and tmux bind alt+arrows for pane navigation, so the keypress
        may never reach hpca at all — see project.md's reserved list."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await self.three_boxes(app, pilot, tmp_path)
            panel(app).index = 2
            await self.press_move(app, pilot, "shift+up")
            assert self.titles(app) == ["a.log", "c.log", "b.log"]

    async def test_an_empty_column_has_nothing_to_move(self, hpca_home):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            panel(app).focus()
            await self.press_move(app, pilot, "alt+up")
            assert len(panel(app).children) == 0


class TestTheColumnAlwaysHasACursor:
    """A column with boxes in it must have one of them highlighted.

    ``check_action`` answers about the highlighted row, so with nothing
    highlighted the footer loses peek, unwatch and both moves and the column
    reads as inert. The reported symptom was exactly that: arrowing between
    the three columns sometimes landed on watchers with the wrong hotkeys.
    """

    def hotkeys(self, app):
        """What the footer offers for the highlighted row, via check_action —
        the same question the footer asks."""
        watchers = panel(app)
        return {
            action
            for action in ("peek_watch", "drop_watch", "move_watch")
            if watchers.check_action(action, ())
        }

    ALL = {"peek_watch", "drop_watch", "move_watch"}

    async def test_the_first_paint_lands_on_the_top_box(
        self, hpca_home, tmp_path
    ):
        """Nothing was highlighted before, so there was no row to restore and
        the cursor was left nowhere at all."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "a.log")
            assert panel(app).index == 0
            assert self.hotkeys(app) == self.ALL

    async def test_arriving_by_arrow_key_finds_the_hotkeys(
        self, hpca_home, tmp_path
    ):
        """←/→ between the columns is how the user gets here."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "a.log")
            app._focus_column("chat")
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            assert app.focused_column_id == "watchers"
            assert self.hotkeys(app) == self.ALL

    async def test_switching_to_a_session_with_other_boxes_keeps_a_cursor(
        self, hpca_home, tmp_path
    ):
        """The remembered row belongs to the session being left, so it is never
        among the new keys — which used to leave the column unhighlighted."""
        app = HpcaApp(llm=FakeLLM([respond_json("a"), respond_json("b")]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "a.log")
            panel(app).focus()
            panel(app).index = 0

            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "b.log"), "b.log")
            highlighted = panel(app).highlighted_child
            assert getattr(highlighted, "data_watch").title == "b.log"
            assert self.hotkeys(app) == self.ALL

    async def test_dropping_a_box_holds_the_position_not_the_row(
        self, hpca_home, tmp_path
    ):
        """The row is gone by definition, so the cursor keeps its place in the
        column: unwatching the middle of three leaves it on what is now the
        middle, rather than jumping back to the top."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            for name in ("a.log", "b.log", "c.log"):
                await add_log_watch(app, pilot, written(tmp_path / name), name)
            panel(app).focus()
            panel(app).index = 1
            await pilot.press("d")
            await app.workers.wait_for_complete()
            await pilot.pause()
            highlighted = panel(app).highlighted_child
            assert getattr(highlighted, "data_watch").title == "c.log"
            assert self.hotkeys(app) == self.ALL

    async def test_dropping_the_last_box_clamps_instead_of_overrunning(
        self, hpca_home, tmp_path
    ):
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            for name in ("a.log", "b.log"):
                await add_log_watch(app, pilot, written(tmp_path / name), name)
            panel(app).focus()
            panel(app).index = 1
            await pilot.press("d")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert panel(app).index == 0
            assert getattr(panel(app).highlighted_child, "data_watch").title == "a.log"

    async def test_an_empty_column_offers_nothing(self, hpca_home, tmp_path):
        """The other half of the same rule: once the last box is gone there is
        genuinely nothing to act on, and the footer has to say so."""
        app = HpcaApp(llm=FakeLLM([respond_json()]))
        async with app.run_test(size=(120, 40)) as pilot:
            await open_session(app, pilot)
            await add_log_watch(app, pilot, written(tmp_path / "a.log"), "a.log")
            panel(app).focus()
            assert self.hotkeys(app) == self.ALL
            await pilot.press("d")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert self.hotkeys(app) == set()


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
            await app.refresh_watchers()
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
            await app.refresh_watchers()
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
