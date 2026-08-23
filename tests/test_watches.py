"""Tests for hpca.watches: the store, the rendering, and the two pollers.

The squeue samples mirror ``squeue -h -o '%i|%T|%N|%M|%L|%j|%r'`` on the target
site, including the "(null)"/"N/A" fields Slurm writes where a column does not
apply to a job's state.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from hpca.db import connect, init_db
from hpca.slurm import JobDetail, JobStatus, parse_squeue_details
from hpca.watches import (
    JOB_GONE,
    KIND_JOB,
    KIND_LOG,
    LOG_GONE,
    LOG_PRESENT,
    WatchStore,
    apply_job_details,
    format_age,
    format_size,
    is_settled,
    job_fields,
    log_fields,
    peek,
    poll_log_watches,
    watch_class,
    watch_lines,
)

NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    return WatchStore(conn)


def touch(path, *, age_s=0.0):
    """A file whose last write was ``age_s`` seconds before NOW."""
    path.write_text("x")
    when = (NOW - timedelta(seconds=age_s)).timestamp()
    os.utime(path, (when, when))
    return path


class TestStore:
    def test_registering_returns_a_row_with_an_id(self, store):
        watch = store.add(kind=KIND_LOG, target="/data/run.log", profile="p")
        assert watch.id
        assert watch.kind == KIND_LOG
        assert [w.id for w in store.list(profile="p")] == [watch.id]

    def test_the_same_target_twice_is_one_box(self, store):
        """Re-registering is how a box gets renamed, not how it gets doubled."""
        first = store.add(kind=KIND_LOG, target="/data/run.log", profile="p")
        again = store.add(
            kind=KIND_LOG, target="/data/run.log", label="sniffles", profile="p"
        )
        assert again.id == first.id
        assert again.label == "sniffles"
        assert len(store.list(profile="p")) == 1

    def test_a_log_and_a_job_may_share_a_target_string(self, store):
        store.add(kind=KIND_LOG, target="123", profile="p")
        store.add(kind=KIND_JOB, target="123", profile="p")
        assert len(store.list(profile="p")) == 2

    def test_watches_are_scoped_to_the_profile(self, store):
        store.add(kind=KIND_LOG, target="/a.log", profile="alpha")
        store.add(kind=KIND_LOG, target="/b.log", profile="beta")
        assert [w.target for w in store.list(profile="alpha")] == ["/a.log"]

    def test_order_is_insertion_not_freshness(self, store):
        """The cursor lives in this list; rows that reorder under it every
        poll cannot be navigated."""
        for name in ("a", "b", "c"):
            store.add(kind=KIND_LOG, target=f"/{name}.log", profile="p")
        store.update(store.list(profile="p")[0].id, state=LOG_PRESENT)
        assert [w.target for w in store.list(profile="p")] == [
            "/a.log",
            "/b.log",
            "/c.log",
        ]

    def test_remove(self, store):
        watch = store.add(kind=KIND_LOG, target="/a.log", profile="p")
        assert store.remove(watch.id) is True
        assert store.list(profile="p") == []
        assert store.remove(watch.id) is False

    def test_a_quiet_poll_does_not_reset_the_change_clock(self, store):
        """"last write 4m ago" has to keep counting up when nothing happens."""
        watch = store.add(kind=KIND_LOG, target="/a.log", profile="p")
        store.update(
            watch.id, state=LOG_PRESENT, changed_at="2026-07-30T11:00:00+00:00"
        )
        store.update(watch.id, state=LOG_PRESENT)
        assert store.get(watch.id).changed_at == "2026-07-30T11:00:00+00:00"

    def test_unknown_kinds_are_refused(self, store):
        with pytest.raises(ValueError):
            store.add(kind="pipeline", target="x", profile="p")


class TestReordering:
    """alt+↑/alt+↓ in the watchers column, at the store level.

    The point of the feature is that the boxes the user is actually watching
    can be put at the top, so what is asserted here is the resulting order and
    not the numbers behind it.
    """

    def three(self, store, session_id="s1"):
        for name in ("a", "b", "c"):
            store.add(
                kind=KIND_LOG,
                target=f"/{name}.log",
                profile="p",
                session_id=session_id,
            )
        return store.list(session_id=session_id)

    def order(self, store, session_id="s1"):
        return [w.target for w in store.list(session_id=session_id)]

    def test_moving_up_trades_places_with_the_box_above(self, store):
        _, b, _ = self.three(store)
        assert store.move(b.id, -1) is True
        assert self.order(store) == ["/b.log", "/a.log", "/c.log"]

    def test_moving_down_trades_places_with_the_box_below(self, store):
        _, b, _ = self.three(store)
        assert store.move(b.id, +1) is True
        assert self.order(store) == ["/a.log", "/c.log", "/b.log"]

    def test_the_new_order_is_what_the_next_read_sees(self, store):
        """It is the panel's own source, so the arrangement has to survive the
        repaint two seconds later — and the restart after that."""
        a, _, c = self.three(store)
        store.move(c.id, -1)
        store.move(c.id, -1)
        store.update(a.id, state=LOG_PRESENT)  # a poll must not undo it
        assert self.order(store) == ["/c.log", "/a.log", "/b.log"]

    def test_the_arrangement_survives_a_restart(self, store, tmp_path):
        """What the column is *for*: an order put right yesterday has to still
        be right tomorrow, which is the whole reason it is a database column
        and not something the panel remembers."""
        _, _, c = self.three(store)
        store.move(c.id, -1)
        store._conn.close()

        reopened = connect(tmp_path / "hpca.db")
        init_db(reopened)  # every start runs the migrations; none may undo it
        assert self.order(WatchStore(reopened)) == ["/a.log", "/c.log", "/b.log"]
        reopened.close()

    def test_a_column_of_one_has_nowhere_to_go(self, store):
        alone = store.add(
            kind=KIND_LOG, target="/only.log", profile="p", session_id="s1"
        )
        assert store.move(alone.id, -1) is False
        assert store.move(alone.id, +1) is False
        assert self.order(store) == ["/only.log"]

    def test_a_dropped_box_leaves_the_rest_in_order(self, store):
        """Removal punches a hole in the numbers and nothing else: what is
        left keeps the arrangement, and the next move closes the gaps."""
        a, b, c = self.three(store)
        store.move(c.id, -1)  # a, c, b
        store.remove(a.id)
        assert self.order(store) == ["/c.log", "/b.log"]
        assert store.move(b.id, -1) is True
        assert self.order(store) == ["/b.log", "/c.log"]

    def test_the_top_box_cannot_go_further_up(self, store):
        a, _, _ = self.three(store)
        assert store.move(a.id, -1) is False
        assert self.order(store) == ["/a.log", "/b.log", "/c.log"]

    def test_the_bottom_box_cannot_go_further_down(self, store):
        _, _, c = self.three(store)
        assert store.move(c.id, +1) is False
        assert self.order(store) == ["/a.log", "/b.log", "/c.log"]

    def test_moving_a_watch_that_is_gone_is_not_an_error(self, store):
        """The panel repaints on a timer, so the box under the cursor can be
        dropped between the keypress and the write."""
        a, _, _ = self.three(store)
        store.remove(a.id)
        assert store.move(a.id, -1) is False

    def test_a_move_leaves_another_session_alone(self, store):
        """Each session has its own column; one cannot reshuffle another's."""
        self.three(store, session_id="s1")
        self.three(store, session_id="s2")
        store.move(store.list(session_id="s1")[2].id, -1)
        assert self.order(store, "s1") == ["/a.log", "/c.log", "/b.log"]
        assert self.order(store, "s2") == ["/a.log", "/b.log", "/c.log"]

    def test_a_new_watch_lands_at_the_bottom_of_a_reordered_column(self, store):
        """Renumbering a column reuses low numbers, so a new box has to be
        given one past everything — otherwise it appears in the middle."""
        _, _, c = self.three(store)
        store.move(c.id, -1)
        store.move(c.id, -1)
        store.add(kind=KIND_LOG, target="/d.log", profile="p", session_id="s1")
        assert self.order(store) == ["/c.log", "/a.log", "/b.log", "/d.log"]

    def test_a_watch_from_before_the_column_existed_keeps_its_place(self, store):
        """Rows written by an older hpca all carry position 0. Ordering has to
        fall back to the id they were sorted by, or every one of them would
        pile up above the boxes the user has arranged."""
        self.three(store)
        store._conn.execute("UPDATE watches SET position = 0")
        store._conn.commit()
        assert self.order(store) == ["/a.log", "/b.log", "/c.log"]
        moved = store.list(session_id="s1")[2]
        assert store.move(moved.id, -1) is True
        assert self.order(store) == ["/a.log", "/c.log", "/b.log"]


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0s"), (4.7, "4s"), (59, "59s"), (60, "1m"), (3599, "59m"),
         (3600, "1h00m"), (7860, "2h11m"), (86400, "1d00h"), (270000, "3d03h")],
    )
    def test_age_is_short_and_never_overstates(self, seconds, expected):
        assert format_age(seconds) == expected

    def test_age_of_a_clock_skewed_future_stamp_is_zero_not_negative(self):
        assert format_age(-30) == "0s"

    @pytest.mark.parametrize(
        "size,expected",
        [(0, "0 B"), (900, "900 B"), (2048, "2.0 KB"), (4_404_019, "4.2 MB"),
         (50 * 1024**2, "50 MB")],
    )
    def test_sizes(self, size, expected):
        assert format_size(size) == expected


class TestLogPolling:
    def test_a_log_that_exists_reports_its_size_and_last_write(self, tmp_path):
        state, head, changed = log_fields(touch(tmp_path / "s.log", age_s=5), now=NOW)
        assert state == LOG_PRESENT
        assert head == "1 B"
        assert changed.startswith("2026-07-30T11:59:55")

    def test_an_untouched_log_is_still_only_present(self, tmp_path):
        """Age does not become a state. A job can be alive and not writing —
        buffered output, a long compute phase — so the mtime cannot be read as
        running-or-dead, and the box does not pretend otherwise."""
        state, _, _ = log_fields(touch(tmp_path / "s.log", age_s=86_400), now=NOW)
        assert state == LOG_PRESENT

    def test_a_missing_log_says_so_rather_than_raising(self, tmp_path):
        assert log_fields(tmp_path / "never.log", now=NOW)[0] == LOG_GONE

    def test_poll_writes_the_state_back_and_reports_the_change(self, store, tmp_path):
        log = touch(tmp_path / "s.log", age_s=5)
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        changes = poll_log_watches(store, [watch], now=NOW)
        assert [c.new_state for c in changes] == [LOG_PRESENT]
        assert store.get(watch.id).state == LOG_PRESENT

    def test_a_second_poll_with_nothing_new_reports_no_change(self, store, tmp_path):
        log = touch(tmp_path / "s.log", age_s=5)
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        again = store.list(profile="p")
        assert poll_log_watches(store, again, now=NOW) == []

    def test_a_log_going_quiet_is_not_a_change_at_all(self, store, tmp_path):
        """It used to be: the box flipped writing → idle and toasted about it.

        A log not being written to for a while is not an event. The box says
        when the last write was and that number keeps climbing on its own; a
        toast for it interrupts the user to tell them nothing happened.
        """
        log = touch(tmp_path / "s.log", age_s=5)
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        later = NOW + timedelta(days=1)
        assert poll_log_watches(store, store.list(profile="p"), now=later) == []

    def test_a_log_that_vanishes_is_still_a_change(self, store, tmp_path):
        """The one thing a log poll can actually observe."""
        log = touch(tmp_path / "s.log", age_s=5)
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        log.unlink()
        changes = poll_log_watches(store, store.list(profile="p"), now=NOW)
        assert [(c.old_state, c.new_state) for c in changes] == [
            (LOG_PRESENT, LOG_GONE)
        ]

    def test_polling_ignores_job_watches(self, store):
        watch = store.add(kind=KIND_JOB, target="42", profile="p")
        assert poll_log_watches(store, [watch], now=NOW) == []
        assert store.get(watch.id).state == ""


SQUEUE_MIXED = """\
27744534|RUNNING|node042|1-04:42:02|3-19:17:58|align_pipeline|None
27744999|PENDING|(null)|0:00|5-00:00:00|sniffles_call|Resources
"""


class TestSqueueParsing:
    def test_a_running_and_a_pending_job(self):
        details = parse_squeue_details(SQUEUE_MIXED)
        assert set(details) == {"27744534", "27744999"}
        running = details["27744534"]
        assert (running.state, running.nodes, running.elapsed) == (
            "RUNNING", "node042", "1-04:42:02"
        )
        assert running.reason == ""  # "None" is not a reason

    def test_placeholder_fields_are_blanked_not_shown(self):
        pending = parse_squeue_details(SQUEUE_MIXED)["27744999"]
        assert pending.nodes == ""
        assert pending.reason == "Resources"

    def test_a_short_row_still_yields_a_job(self):
        details = parse_squeue_details("999|RUNNING|node1")
        assert details["999"].state == "RUNNING"
        assert details["999"].elapsed == ""

    def test_blank_output_is_no_jobs(self):
        assert parse_squeue_details("\n \n") == {}


class TestJobPolling:
    def test_a_running_job_shows_where_and_how_long(self, store):
        detail = parse_squeue_details(SQUEUE_MIXED)["27744534"]
        state, head, line = job_fields(detail)
        assert state == "RUNNING"
        assert head == "1-04:42:02"
        assert "node042" in line and "3-19:17:58 left" in line

    def test_a_pending_job_shows_why_it_is_waiting(self, store):
        state, head, line = job_fields(parse_squeue_details(SQUEUE_MIXED)["27744999"])
        assert state == "PENDING"
        assert "Resources" in line

    def test_a_job_that_left_the_queue_gets_its_final_state_from_sacct(self, store):
        watch = store.add(kind=KIND_JOB, target="27744534", profile="p")
        store.update(watch.id, state="RUNNING")
        finished = {
            "27744534": JobStatus(
                job_id="27744534", state="COMPLETED", raw_state="COMPLETED",
                exit_code=0, elapsed_s=3661, max_rss_bytes=1024**3,
            )
        }
        changes = apply_job_details(
            store, store.list(profile="p"), {}, finished, now=NOW
        )
        assert [(c.old_state, c.new_state) for c in changes] == [
            ("RUNNING", "COMPLETED")
        ]
        row = store.get(watch.id)
        assert row.head == "01:01:01"
        assert "exit 0" in row.detail

    def test_a_job_neither_squeue_nor_sacct_knows_is_marked_gone(self, store):
        watch = store.add(kind=KIND_JOB, target="1", profile="p")
        apply_job_details(store, store.list(profile="p"), {}, {}, now=NOW)
        assert store.get(watch.id).state == JOB_GONE

    def test_state_changes_stamp_the_clock_and_quiet_polls_do_not(self, store):
        watch = store.add(kind=KIND_JOB, target="7", profile="p")
        detail = {"7": JobDetail(job_id="7", state="RUNNING", elapsed="00:01:00")}
        apply_job_details(store, store.list(profile="p"), detail, {}, now=NOW)
        stamped = store.get(watch.id).changed_at
        later = NOW + timedelta(minutes=5)
        apply_job_details(store, store.list(profile="p"), detail, {}, now=later)
        assert store.get(watch.id).changed_at == stamped

    def test_a_finished_job_stops_costing_an_squeue_call(self, store):
        watch = store.add(kind=KIND_JOB, target="7", profile="p")
        store.update(watch.id, state="COMPLETED")
        assert is_settled(store.get(watch.id))

    def test_a_missing_log_keeps_being_polled(self, store):
        """The job has not created it yet — the commonest reason to watch one."""
        watch = store.add(kind=KIND_LOG, target="/not/yet.log", profile="p")
        store.update(watch.id, state=LOG_GONE)
        assert not is_settled(store.get(watch.id))


class TestBoxContents:
    def test_a_log_box_says_its_size_and_how_long_ago_it_was_written(
        self, store, tmp_path
    ):
        log = touch(tmp_path / "sniffles.log", age_s=12)
        watch = store.add(
            kind=KIND_LOG, target=str(log), label="sniffles", profile="p"
        )
        poll_log_watches(store, [watch], now=NOW)
        lines = watch_lines(store.get(watch.id), now=NOW)
        # No state word: "writing"/"idle" were a claim about the job that the
        # file's mtime cannot support. The last write is the whole message.
        assert lines[0] == "○ 1 B"
        assert lines[1] == "last write 12s ago"

    def test_a_stale_log_box_reads_the_same_but_for_the_clock(
        self, store, tmp_path
    ):
        log = touch(tmp_path / "sniffles.log", age_s=90_000)
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        lines = watch_lines(store.get(watch.id), now=NOW)
        assert lines[0] == "○ 1 B"
        assert lines[1].startswith("last write 1d")

    def test_a_box_is_two_lines_whatever_it_holds(self, store):
        watch = store.add(kind=KIND_JOB, target="7", profile="p")
        assert len(watch_lines(store.get(watch.id))) == 2

    def test_an_unpolled_box_admits_it(self, store):
        watch = store.add(kind=KIND_LOG, target="/a.log", profile="p")
        assert watch_lines(store.get(watch.id)) == [
            "○ not polled yet",
            "not polled yet",
        ]

    def test_a_missing_file_reads_as_missing_not_as_stale(self, store, tmp_path):
        watch = store.add(kind=KIND_LOG, target=str(tmp_path / "no.log"), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        assert watch_lines(store.get(watch.id))[1] == "no such file"

    def test_the_label_falls_back_to_the_file_name(self, store):
        watch = store.add(kind=KIND_LOG, target="/data/deep/run.log", profile="p")
        assert watch.title == "run.log"

    def test_a_log_is_never_coloured_as_live(self, store, tmp_path):
        """Green would be the same unsupportable claim in another form."""
        log = touch(tmp_path / "a.log")
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        assert watch_class(store.get(watch.id)) == "watch-idle"
        poll_log_watches(
            store, store.list(profile="p"), now=NOW + timedelta(hours=1)
        )
        assert watch_class(store.get(watch.id)) == "watch-idle"

    def test_a_vanished_log_is_coloured_dead(self, store, tmp_path):
        log = touch(tmp_path / "a.log")
        watch = store.add(kind=KIND_LOG, target=str(log), profile="p")
        poll_log_watches(store, [watch], now=NOW)
        log.unlink()
        poll_log_watches(store, store.list(profile="p"), now=NOW)
        assert watch_class(store.get(watch.id)) == "watch-dead"


class TestPeek:
    def test_the_tail_is_what_comes_back(self, tmp_path):
        log = tmp_path / "a.log"
        log.write_text("\n".join(f"line {i}" for i in range(500)))
        text = peek(log, chars=40)
        assert text.endswith("line 499")
        assert text.startswith("…")  # never pretends to be the whole file

    def test_a_short_log_is_shown_whole(self, tmp_path):
        log = tmp_path / "a.log"
        log.write_text("Done.\n")
        assert peek(log) == "Done."

    def test_a_huge_log_is_not_read_into_memory(self, tmp_path):
        """These are job logs; a gigabyte of progress bars is normal."""
        log = tmp_path / "big.log"
        with log.open("w") as handle:
            handle.write("x" * 5_000_000)
            handle.write("\nFINISHED")
        assert peek(log).endswith("FINISHED")

    def test_an_unreadable_log_explains_itself(self, tmp_path):
        assert "could not read" in peek(tmp_path / "missing.log")

    def test_an_empty_log_says_empty(self, tmp_path):
        log = tmp_path / "a.log"
        log.write_text("")
        assert peek(log) == "(empty)"


class TestForgetSession:
    """Watches are session-scoped, so a deleted session's must go with it —
    otherwise they are invisible forever and still polled forever."""

    def test_it_drops_only_that_sessions_watches(self, store):
        keep = store.add(kind=KIND_LOG, target="/a.log", session_id="s1")
        drop = store.add(kind=KIND_LOG, target="/b.log", session_id="s2")
        assert store.forget_session("s2") == 1
        assert [w.id for w in store.list()] == [keep.id]
        assert store.get(drop.id) is None

    def test_a_session_with_no_watches_is_not_an_error(self, store):
        assert store.forget_session("never-watched") == 0
