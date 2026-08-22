"""Tests for hpca.sessions: session rows backing the left column (§5.4)."""

import pytest

from hpca.db import connect, init_db
from hpca.sessions import SessionStore


@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield SessionStore(conn)
    conn.close()


class TestSessionStore:
    def test_create_returns_unique_ids(self, store):
        a = store.create(profile="default")
        b = store.create(profile="default")
        assert a.session_id != b.session_id

    def test_create_default_title_is_untitled(self, store):
        session = store.create(profile="default")
        assert session.title == "untitled"

    def test_list_newest_first(self, store):
        a = store.create(profile="default", title="first")
        b = store.create(profile="default", title="second")
        listed = store.list(profile="default")
        assert [s.session_id for s in listed] == [b.session_id, a.session_id]

    def test_list_filters_by_profile(self, store):
        store.create(profile="p1", title="one")
        store.create(profile="p2", title="two")
        assert [s.title for s in store.list(profile="p1")] == ["one"]

    def test_rename(self, store):
        session = store.create(profile="default")
        store.rename(session.session_id, "BAM subsetting")
        assert store.list(profile="default")[0].title == "BAM subsetting"

    def test_get(self, store):
        session = store.create(profile="default", title="x")
        fetched = store.get(session.session_id)
        assert fetched.title == "x"
        assert store.get("nonexistent") is None

    def test_checkpoint_ref_equals_session_id(self, store):
        # thread_id for the LangGraph checkpointer
        session = store.create(profile="default")
        assert session.checkpoint_ref == session.session_id


class TestLastActive:
    """When something last happened here — the sidebar's right-hand column.

    Stored on the session rather than read back off the thread, because the
    sidebar draws every row at once and answering this from the transcript
    would mean opening every conversation to paint a list.
    """

    def test_a_new_session_is_active_now(self, store):
        # Being made is the first thing that happens in a conversation, so a
        # session with no turns yet reads as new rather than as never having
        # happened at all.
        session = store.create(profile="default")
        assert session.last_active == session.created_at

    def test_and_it_survives_the_round_trip(self, store):
        session = store.create(profile="default")
        assert store.get(session.session_id).last_active == session.last_active

    def test_touching_moves_it(self, store):
        session = store.create(profile="default")
        store.touch(session.session_id, "2030-01-01T00:00:00+00:00")
        assert store.get(session.session_id).last_active == (
            "2030-01-01T00:00:00+00:00"
        )

    def test_and_a_bare_touch_is_now(self, store):
        session = store.create(profile="default")
        store.touch(session.session_id, "2000-01-01T00:00:00+00:00")
        store.touch(session.session_id)
        assert store.get(session.session_id).last_active > session.created_at

    def test_touching_one_leaves_the_others_alone(self, store):
        a = store.create(profile="default")
        b = store.create(profile="default")
        store.touch(a.session_id, "2030-01-01T00:00:00+00:00")
        assert store.get(b.session_id).last_active == b.last_active


class TestDelete:
    def test_delete_removes_only_that_session(self, store):
        keep = store.create(profile="default", title="keep me")
        drop = store.create(profile="default", title="drop me")
        store.delete(drop.session_id)
        assert [s.session_id for s in store.list(profile="default")] == [
            keep.session_id
        ]
        assert store.get(drop.session_id) is None
        assert store.get(keep.session_id) is not None

    def test_deleting_an_unknown_session_is_quiet(self, store):
        store.delete("no-such-session")  # must not raise

    def test_jobs_outlive_the_session_they_were_submitted_from(self, store):
        from hpca.jobs import JobStore

        session = store.create(profile="default")
        jobs = JobStore(store._conn)
        jobs.add(
            job_id="27744534",
            kind="sbatch",
            session_id=session.session_id,
            profile="default",
            script_key="align",
            stdout_path="/tmp/out",
            stderr_path="/tmp/err",
        )
        store.delete(session.session_id)
        # the cluster job runs on; its record must not vanish with the chat
        assert jobs.get("27744534") is not None


class TestReordering:
    """alt+↑/alt+↓ in the sidebar, at the store level.

    The same gesture the watch column has, on the list the user spends the
    day in: newest-first is a good default and a bad permanent arrangement,
    because the conversation somebody works in all week sinks under every
    throwaway one. What is asserted here is the resulting order and not the
    numbers behind it.
    """

    def three(self, store):
        for title in ("first", "second", "third"):
            store.create(profile="default", title=title)
        # Newest first, so the list reads back the other way round.
        return store.list_all()

    def order(self, store):
        return [s.title for s in store.list_all()]

    def test_the_sidebar_starts_newest_first(self, store):
        self.three(store)
        assert self.order(store) == ["third", "second", "first"]

    def test_moving_up_trades_places_with_the_row_above(self, store):
        _, middle, _ = self.three(store)
        assert store.move(middle.session_id, -1) is True
        assert self.order(store) == ["second", "third", "first"]

    def test_moving_down_trades_places_with_the_row_below(self, store):
        _, middle, _ = self.three(store)
        assert store.move(middle.session_id, +1) is True
        assert self.order(store) == ["third", "first", "second"]

    def test_the_arrangement_survives_a_restart(self, store, tmp_path):
        """The reason this is a column and not a front-end's memory: an order
        arranged today has to be the order tomorrow opens with."""
        _, _, oldest = self.three(store)
        store.move(oldest.session_id, -1)
        store.move(oldest.session_id, -1)
        store._conn.close()

        reopened = connect(tmp_path / "hpca.db")
        init_db(reopened)  # the migrations run on every start; none may undo it
        assert self.order(SessionStore(reopened)) == ["first", "third", "second"]
        reopened.close()

    def test_the_top_row_cannot_go_further_up(self, store):
        newest, _, _ = self.three(store)
        assert store.move(newest.session_id, -1) is False
        assert self.order(store) == ["third", "second", "first"]

    def test_the_bottom_row_cannot_go_further_down(self, store):
        _, _, oldest = self.three(store)
        assert store.move(oldest.session_id, +1) is False
        assert self.order(store) == ["third", "second", "first"]

    def test_a_sidebar_of_one_has_nowhere_to_go(self, store):
        alone = store.create(profile="default", title="only")
        assert store.move(alone.session_id, -1) is False
        assert store.move(alone.session_id, +1) is False
        assert self.order(store) == ["only"]

    def test_moving_a_session_that_is_gone_is_not_an_error(self, store):
        """The sidebar can lose a row between the keypress and the write — a
        deletion from another front-end, or the one the user just pressed."""
        _, _, oldest = self.three(store)
        store.delete(oldest.session_id)
        assert store.move(oldest.session_id, -1) is False
        assert self.order(store) == ["third", "second"]

    def test_a_deleted_row_leaves_the_rest_arranged(self, store):
        """Deleting punches a hole in the numbers and nothing else: the order
        of what is left is untouched, and the next move closes the gaps."""
        newest, middle, oldest = self.three(store)
        store.move(oldest.session_id, -1)  # third, first, second
        store.delete(newest.session_id)
        assert self.order(store) == ["first", "second"]
        assert store.move(middle.session_id, -1) is True
        assert self.order(store) == ["second", "first"]

    def test_a_new_session_lands_at_the_top_of_a_rearranged_sidebar(self, store):
        """Where newest-first always put it, and where the front-end that
        opens it expects to find it — the opposite end from a new watch box,
        because the two lists are read in opposite directions."""
        _, _, oldest = self.three(store)
        store.move(oldest.session_id, -1)
        store.create(profile="default", title="fourth")
        assert self.order(store) == ["fourth", "third", "first", "second"]

    def test_a_profile_listing_shows_the_arrangement_too(self, store):
        """One sidebar, one order: `list` is `list_all` with rows hidden, so
        it must not fall back to an order of its own."""
        store.create(profile="p1", title="mine")
        store.create(profile="p2", title="theirs")
        store.create(profile="p1", title="mine too")
        arranged = store.list_all()[0]  # "mine too", at the top
        store.move(arranged.session_id, +1)
        store.move(arranged.session_id, +1)
        assert [s.title for s in store.list(profile="p1")] == ["mine", "mine too"]

    def test_a_move_reaches_across_profiles(self, store):
        """The sidebar draws every profile's sessions interleaved, so a swap
        that skipped the differently-profiled row between two rows would land
        somewhere the user did not aim."""
        store.create(profile="p1", title="mine")
        store.create(profile="p2", title="theirs")
        bottom = store.list_all()[1].session_id  # "mine", under "theirs"
        assert store.move(bottom, -1) is True
        assert self.order(store) == ["mine", "theirs"]

    def test_a_session_from_before_the_column_existed_keeps_its_place(self, store):
        """Rows written by an older hpca all carry position 0. The ordering
        has to fall back to newest-first, or every one of them would pile up
        above the rows the user has arranged."""
        self.three(store)
        store._conn.execute("UPDATE sessions SET position = 0")
        store._conn.commit()
        assert self.order(store) == ["third", "second", "first"]
        assert store.move(store.list_all()[2].session_id, -1) is True
        assert self.order(store) == ["third", "first", "second"]
