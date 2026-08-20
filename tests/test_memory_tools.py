"""Tests for the session_search tool (redesign Phase 2)."""

import pytest

from hpca.agent.context import ToolContext
from hpca.agent.memory_tools import (
    SessionSearchParams,
    add_memory_tools,
    session_search,
)
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.episodic import EpisodicStore
from hpca.runner import ProcessRunner
from hpca.sessions import SessionStore


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "hpca.db")
    init_db(connection)
    yield connection
    connection.close()


@pytest.fixture
def env(conn, tmp_path):
    store = EpisodicStore(conn)
    sessions = SessionStore(conn)
    settings = Settings()

    def context(profile="default", session_id="current", episodic=store):
        return ToolContext(
            workdir=tmp_path,
            runner=ProcessRunner(conn, session_id=session_id, log_dir=tmp_path),
            settings=settings,
            scripts_dir=tmp_path,
            session_id=session_id,
            profile=profile,
            episodic=episodic,
        )

    return store, sessions, settings, context


async def search(ctx, **kwargs):
    return await session_search(SessionSearchParams(**kwargs), ctx)


class TestDiscovery:
    async def test_reports_goal_match_and_resolution(self, env):
        store, sessions, _, context = env
        session = sessions.create(profile="default", title="STAR alignment")
        store.record(
            session_id=session.session_id,
            profile="default",
            entries=[
                ("user", "how do I align reads with STAR"),
                ("assistant", "index the genome first"),
                ("user", "it ran out of memory"),
                ("assistant", "STAR needs 40G on this cluster"),
            ],
        )
        out = await search(context(), query="memory")
        assert "STAR alignment" in out
        assert "goal: how do I align reads with STAR" in out
        assert "resolution: STAR needs 40G on this cluster" in out
        assert ">>memory<<" in out

    async def test_no_match_says_so(self, env):
        _, _, _, context = env
        assert "No past-session matches" in await search(context(), query="kubernetes")

    async def test_empty_input_asks_for_one(self, env):
        _, _, _, context = env
        assert "Give a query" in await search(context(), query="   ")

    async def test_result_is_char_bounded(self, env):
        store, sessions, _, context = env
        for i in range(5):
            session = sessions.create(profile="default", title=f"session {i}")
            store.record(
                session_id=session.session_id,
                profile="default",
                entries=[("user", "bam " + "x" * 3000), ("assistant", "y" * 3000)],
            )
        out = await search(context(), query="bam")
        assert len(out) <= 2600


class TestProfileScoping:
    async def test_other_profiles_hidden_by_default(self, env):
        store, sessions, _, context = env
        other = sessions.create(profile="hpc-admin", title="node draining")
        store.record(
            session_id=other.session_id,
            profile="hpc-admin",
            entries=[("user", "how do I drain a slurm node")],
        )
        out = await search(context(profile="genetics"), query="drain")
        assert "No past-session matches" in out

    async def test_cross_profile_search_when_enabled(self, env):
        store, sessions, settings, context = env
        settings.memory.cross_profile_search = True
        other = sessions.create(profile="hpc-admin", title="node draining")
        store.record(
            session_id=other.session_id,
            profile="hpc-admin",
            entries=[("user", "how do I drain a slurm node")],
        )
        out = await search(context(profile="genetics"), query="drain")
        assert "node draining" in out
        assert "profile hpc-admin" in out  # provenance shown when crossing


class TestReadMode:
    async def test_reads_around_a_turn(self, env):
        store, sessions, _, context = env
        session = sessions.create(profile="default", title="long one")
        store.record(
            session_id=session.session_id,
            profile="default",
            entries=[("user", f"message {i}") for i in range(1, 21)],
        )
        out = await search(context(), session_id=session.session_id, around=10)
        assert "[10] user: message 10" in out
        assert "message 3" not in out

    async def test_tail_without_anchor(self, env):
        store, sessions, _, context = env
        session = sessions.create(profile="default", title="long one")
        store.record(
            session_id=session.session_id,
            profile="default",
            entries=[("user", f"message {i}") for i in range(1, 21)],
        )
        out = await search(context(), session_id=session.session_id)
        assert "message 20" in out
        assert "session tail" in out

    async def test_unknown_session(self, env):
        _, _, _, context = env
        assert "No messages found" in await search(context(), session_id="nope")


class TestUnavailable:
    async def test_no_store_degrades_gracefully(self, env):
        _, _, _, context = env
        out = await search(context(episodic=None), query="anything")
        assert "unavailable" in out


class TestRegistration:
    def test_tool_registered(self):
        registry = add_memory_tools(ToolRegistry())
        assert "session_search" in registry.names()
