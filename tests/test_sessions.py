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
