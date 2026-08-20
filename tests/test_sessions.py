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
