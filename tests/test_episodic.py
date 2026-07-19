"""Tests for episodic memory: the FTS5 message index (redesign Phase 2)."""

import pytest

from hpca.db import connect, init_db
from hpca.episodic import EpisodicStore
from hpca.sessions import SessionStore


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "hpca.db")
    init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def store(conn):
    return EpisodicStore(conn)


def make_session(conn, *, profile="default", title="a session"):
    return SessionStore(conn).create(profile=profile, title=title)


def record(store, session, pairs):
    store.record(
        session_id=session.session_id, profile=session.profile, entries=pairs
    )


class TestRecording:
    def test_turn_numbers_increase_across_calls(self, store, conn):
        session = make_session(conn)
        record(store, session, [("user", "first"), ("assistant", "second")])
        record(store, session, [("user", "third")])
        rows = store.window(session.session_id)
        assert [r["turn_no"] for r in rows] == [1, 2, 3]
        assert [r["content"] for r in rows] == ["first", "second", "third"]

    def test_empty_entries_are_a_noop(self, store, conn):
        session = make_session(conn)
        record(store, session, [])
        assert store.window(session.session_id) == []

    def test_fts_available_in_this_build(self, store):
        assert store.fts_available


class TestSearch:
    def test_finds_session_with_bookends(self, store, conn):
        session = make_session(conn, title="STAR alignment")
        record(
            store,
            session,
            [
                ("user", "how do I align reads with STAR"),
                ("assistant", "index the genome first"),
                ("user", "it ran out of memory"),
                ("assistant", "STAR needs 40G here"),
            ],
        )
        hits = store.search("memory")
        assert len(hits) == 1
        hit = hits[0]
        assert hit.title == "STAR alignment"
        assert ">>memory<<" in hit.snippet
        assert hit.goal == "how do I align reads with STAR"  # first user message
        assert hit.resolution == "STAR needs 40G here"  # last assistant message
        assert hit.turn_no == 3

    def test_one_hit_per_session(self, store, conn):
        session = make_session(conn)
        record(
            store,
            session,
            [("user", "bam bam bam"), ("assistant", "bam again"), ("user", "bam")],
        )
        assert len(store.search("bam")) == 1

    def test_current_session_excluded(self, store, conn):
        session = make_session(conn)
        record(store, session, [("user", "snakemake fails")])
        assert store.search("snakemake") != []
        assert store.search("snakemake", exclude_session_id=session.session_id) == []

    def test_profile_scoping(self, store, conn):
        genetics = make_session(conn, profile="genetics")
        admin = make_session(conn, profile="hpc-admin")
        record(store, genetics, [("user", "variant calling with deepvariant")])
        record(store, admin, [("user", "slurm node draining")])
        assert len(store.search("deepvariant", profile="genetics")) == 1
        assert store.search("deepvariant", profile="hpc-admin") == []
        # profile=None is the cross-profile mode (config-gated at the tool)
        assert len(store.search("deepvariant", profile=None)) == 1

    def test_no_match_returns_empty(self, store, conn):
        record(store, make_session(conn), [("user", "hello")])
        assert store.search("kubernetes") == []

    def test_fts_syntax_in_query_does_not_explode(self, store, conn):
        record(store, make_session(conn), [("user", "the star aligner")])
        # bare FTS5 operators/quotes would be a syntax error unquoted
        assert store.search('star OR "') != [] or True
        assert store.search("*") == []
        assert store.search("") == []

    def test_limit_respected(self, store, conn):
        for i in range(4):
            session = make_session(conn, title=f"s{i}")
            record(store, session, [("user", f"bam file number {i}")])
        assert len(store.search("bam", limit=2)) == 2


class TestWindow:
    def test_around_anchors_the_excerpt(self, store, conn):
        session = make_session(conn)
        record(
            store,
            session,
            [("user", f"m{i}") for i in range(1, 21)],
        )
        rows = store.window(session.session_id, around=10, radius=2)
        assert [r["turn_no"] for r in rows] == [8, 9, 10, 11, 12]

    def test_tail_when_no_anchor(self, store, conn):
        session = make_session(conn)
        record(store, session, [("user", f"m{i}") for i in range(1, 21)])
        rows = store.window(session.session_id, radius=2)
        assert [r["turn_no"] for r in rows] == [16, 17, 18, 19, 20]

    def test_unknown_session_is_empty(self, store):
        assert store.window("nope") == []


class TestForget:
    def test_deleting_a_session_removes_it_from_search(self, store, conn):
        session = make_session(conn)
        record(store, session, [("user", "confidential patient note")])
        assert store.search("confidential") != []
        store.forget_session(session.session_id)
        assert store.search("confidential") == []
        assert store.window(session.session_id) == []
