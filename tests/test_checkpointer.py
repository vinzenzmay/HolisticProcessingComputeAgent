"""`hpca.checkpointer` — the log-structured saver, against the one oracle.

The suite is the pin against langgraph's checkpoint contract, so it tests the
contract and not the implementation: nearly everything below drives the same
sequence through `InMemorySaver` and through `CheckpointLogSaver` and asserts
the two agree. What is asserted *of* the implementation is only the thing the
oracle cannot see — whether a put appended a tail or rewrote the log, and
whether retention left the rows it promised.

`specs/specs-checkpoint-log.md` §9 is the list this covers.
"""

from __future__ import annotations

import json
import operator
import random
import sqlite3
from typing import Annotated, TypedDict

import aiosqlite
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from hpca.checkpointer import (
    KEEP_CHECKPOINTS,
    LOG_CHANNELS,
    CheckpointLogSaver,
    migrate_inline_format,
)

# --------------------------------------------------------------------- helpers


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "checkpoints.db")


@pytest.fixture
async def saver(db):
    async with CheckpointLogSaver.from_conn_string(db) as saver:
        yield saver


def config_for(thread_id: str, checkpoint_id: str | None = None, ns: str = ""):
    configurable = {"thread_id": thread_id, "checkpoint_ns": ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


class Thread:
    """A hand-driven thread: builds checkpoints the way the pregel loop does,
    and puts each one into every saver it was given so they can be compared.

    Deliberately not a real graph. The graph exercises one path through the
    contract; this needs to reach the ones it does not — a channel that
    shrinks, a state with a channel `LOG_CHANNELS` has never heard of, a
    `LOG_CHANNELS` name that is not in the state at all.
    """

    def __init__(self, *savers, thread_id="t1", ns=""):
        self.savers = savers
        self.thread_id = thread_id
        self.ns = ns
        self.values: dict = {}
        self.versions: dict = {}
        self.parent: str | None = None
        self.step = 0

    async def put(self, updates: dict, *, metadata=None):
        """Apply `updates` to the state and checkpoint the result."""
        primary = self.savers[0]
        new_versions = {}
        for channel, value in updates.items():
            self.values[channel] = value
            self.versions[channel] = primary.get_next_version(
                self.versions.get(channel), None
            )
            new_versions[channel] = self.versions[channel]
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = dict(self.values)
        checkpoint["channel_versions"] = dict(self.versions)
        checkpoint["updated_channels"] = sorted(updates)
        meta = metadata or {"source": "loop", "step": self.step, "parents": {}}
        self.step += 1
        config = config_for(self.thread_id, self.parent, self.ns)
        for one in self.savers:
            await one.aput(config, checkpoint, meta, new_versions)
        self.parent = checkpoint["id"]
        return checkpoint["id"]

    async def state(self, saver):
        tup = await saver.aget_tuple(config_for(self.thread_id, ns=self.ns))
        return None if tup is None else tup.checkpoint["channel_values"]


async def rows(db, table, **where):
    conn = await aiosqlite.connect(db)
    try:
        clause = " AND ".join(f"{k}=?" for k in where)
        query = f"SELECT COUNT(*) FROM {table}"
        if clause:
            query += f" WHERE {clause}"
        async with conn.execute(query, tuple(where.values())) as cur:
            return (await cur.fetchone())[0]
    finally:
        await conn.close()


class AppendState(TypedDict, total=False):
    """An append-only list channel, which is the shape `AgentState` has.

    At module scope because `from __future__ import annotations` turns the
    annotation into a string that LangGraph resolves against the defining
    module's globals — a schema declared inside a test function is not found.
    """

    messages: Annotated[list, operator.add]


def message(text, role="user"):
    return {"role": role, "content": text}


# ------------------------------------------------------------------ round trip


class TestRoundTrip:
    """Every channel of `AgentState`, put and read back, against the oracle."""

    async def test_every_channel_and_type_survives(self, saver):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put(
            {
                "messages": [message("hello"), message("hi", "assistant")],
                "thinking": [{"after": 1, "text": "thought about it"}],
                "calls": [{"after": 1, "tool": "run_bash", "arguments": {"a": 1}}],
                "plan": [{"text": "do the thing", "done": False}],
                "pending_tool": None,
                "tool_rounds": 3,
                "compacted": {"upto": 1, "summary": message("a summary", "system")},
            }
        )
        assert await thread.state(saver) == await thread.state(oracle)

    async def test_a_thread_nothing_wrote_reads_as_nothing(self, saver):
        assert await saver.aget_tuple(config_for("never-used")) is None

    async def test_the_config_and_parent_come_back(self, saver):
        thread = Thread(saver)
        first = await thread.put({"messages": [message("one")]})
        second = await thread.put({"messages": [message("one"), message("two")]})
        tup = await saver.aget_tuple(config_for("t1"))
        assert tup.config["configurable"]["checkpoint_id"] == second
        assert tup.parent_config["configurable"]["checkpoint_id"] == first

    async def test_metadata_round_trips(self, saver):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put(
            {"messages": [message("x")]},
            metadata={"source": "update", "step": 7, "parents": {}},
        )
        mine = await saver.aget_tuple(config_for("t1"))
        theirs = await oracle.aget_tuple(config_for("t1"))
        assert mine.metadata == theirs.metadata

    async def test_an_explicit_checkpoint_id_reads_that_checkpoint(self, saver):
        thread = Thread(saver)
        first = await thread.put({"messages": [message("one")]})
        await thread.put({"messages": [message("one"), message("two")]})
        tup = await saver.aget_tuple(config_for("t1", first))
        assert tup.checkpoint["channel_values"]["messages"] == [message("one")]

    async def test_threads_and_namespaces_do_not_bleed(self, saver):
        a = Thread(saver, thread_id="a")
        b = Thread(saver, thread_id="b")
        sub = Thread(saver, thread_id="a", ns="child")
        await a.put({"messages": [message("in a")]})
        await b.put({"messages": [message("in b")]})
        await sub.put({"messages": [message("in the subgraph")]})
        assert (await a.state(saver))["messages"] == [message("in a")]
        assert (await b.state(saver))["messages"] == [message("in b")]
        assert (await sub.state(saver))["messages"] == [message("in the subgraph")]


class TestChannelsItDoesNotKnowAbout:
    """§ non-negotiable 2: a `LOG_CHANNELS` name missing from the state, and a
    state channel `LOG_CHANNELS` never heard of, are both ordinary."""

    async def test_a_log_channel_absent_from_the_state_is_not_an_error(self, saver):
        # What a branch that removed `thinking` from AgentState looks like.
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put({"messages": [message("one")], "calls": []})
        await thread.put({"messages": [message("one"), message("two")], "calls": []})
        assert await thread.state(saver) == await thread.state(oracle)
        assert "thinking" not in await thread.state(saver)

    async def test_a_channel_nobody_declared_a_log_is_stored_whole(self, saver, db):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put({"notes": ["a", "b"], "messages": [message("one")]})
        assert await thread.state(saver) == await thread.state(oracle)
        assert await rows(db, "channel_items", channel="notes") == 0
        assert await rows(db, "channel_blobs", channel="notes") == 1

    async def test_a_log_channel_that_stopped_being_a_list_is_stored_whole(
        self, saver, db
    ):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put({"messages": {"not": "a list"}})
        assert await thread.state(saver) == await thread.state(oracle)
        assert await rows(db, "channel_items", channel="messages") == 0
        assert await rows(db, "channel_blobs", channel="messages") == 1

    async def test_and_it_can_go_back_to_being_a_list(self, saver):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put({"messages": {"not": "a list"}})
        await thread.put({"messages": [message("one")]})
        await thread.put({"messages": [message("one"), message("two")]})
        assert await thread.state(saver) == await thread.state(oracle)


# ------------------------------------------------------------------- the tails


class TestTheLogIsALog:
    async def test_an_ordinary_step_writes_only_the_tail(self, saver, db):
        thread = Thread(saver)
        await thread.put({"messages": [message("one")]})
        first = await rows(db, "channel_items", channel="messages")
        history = [message("one"), message("two"), message("three")]
        await thread.put({"messages": history})
        assert first == 1
        assert await rows(db, "channel_items", channel="messages") == 3

    async def test_the_conversation_is_stored_once_not_once_per_step(self, saver, db):
        """The whole point: N appends leave N rows, not N(N+1)/2."""
        thread = Thread(saver)
        history: list = []
        for n in range(40):
            history = history + [message(f"turn {n}" * 20)]
            await thread.put({"messages": list(history)})
        assert await rows(db, "channel_items", channel="messages") == 40

    async def test_a_step_that_appends_nothing_writes_no_items(self, saver, db):
        thread = Thread(saver)
        await thread.put({"messages": [message("one")]})
        await thread.put({"tool_rounds": 1})
        await thread.put({"tool_rounds": 2})
        assert await rows(db, "channel_items", channel="messages") == 1

    async def test_an_unchanged_blob_channel_shares_its_row(self, saver, db):
        thread = Thread(saver)
        plan = [{"text": "step", "done": False}]
        await thread.put({"plan": plan, "messages": []})
        for n in range(5):
            await thread.put({"messages": [message(f"m{n}")] * (n + 1)})
        assert await rows(db, "channel_blobs", channel="plan") == 1


class TestSnapshotIsTheFallback:
    """Every failure of the prefix check must widen to a full rewrite."""

    async def test_a_rewind_snapshots_rather_than_tailing(self, saver, db):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        history = [message(f"m{n}") for n in range(5)]
        await thread.put({"messages": list(history)})
        # `TRUNCATE_TO` shrinks the list; the reducer has already applied it by
        # the time a checkpoint reaches the saver, so this is what it sees.
        await thread.put({"messages": history[:2]})
        assert await rows(db, "channel_items", channel="messages") == 2
        assert await thread.state(saver) == await thread.state(oracle)

    async def test_and_tails_resume_after_it(self, saver, db):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        history = [message(f"m{n}") for n in range(5)]
        await thread.put({"messages": list(history)})
        after = history[:2] + [message("a different m2")]
        await thread.put({"messages": list(after)})
        await thread.put({"messages": after + [message("m3 again")]})
        assert await rows(db, "channel_items", channel="messages") == 4
        assert await thread.state(saver) == await thread.state(oracle)

    async def test_a_rewritten_prefix_is_caught_by_the_chain(self, saver, db):
        """Same length, different content, and the identity check defeated —
        the case only the hash chain can see."""
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put({"messages": [message("one"), message("two")]})
        # Fresh objects, so identity cannot decide it, and a changed prefix.
        rewritten = [message("ONE"), message("two"), message("three")]
        saver._logs[("t1", "", "messages")].items = []
        await thread.put({"messages": rewritten})
        assert await thread.state(saver) == await thread.state(oracle)
        assert (await thread.state(saver))["messages"][0] == message("ONE")

    async def test_an_unchanged_prefix_of_fresh_objects_still_tails(self, saver, db):
        """The chain's other half: identity fails, content matches, so this is
        an append and must be stored as one."""
        thread = Thread(saver)
        await thread.put({"messages": [message("one"), message("two")]})
        saver._logs[("t1", "", "messages")].items = []
        await thread.put(
            {"messages": [message("one"), message("two"), message("three")]}
        )
        assert await rows(db, "channel_items", channel="messages") == 3

    async def test_a_serializer_that_raises_does_not_lose_the_channel(self, saver):
        """The fallback has to cover the unforeseen too, not only the cases
        that were thought of."""
        thread = Thread(saver)
        await thread.put({"messages": [message("one")]})
        original = saver.serde.dumps_typed
        calls = {"n": 0}

        def flaky(value):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("no")
            return original(value)

        saver.serde.dumps_typed = flaky
        try:
            await thread.put({"messages": [message("one"), message("two")]})
        finally:
            saver.serde.dumps_typed = original
        assert (await thread.state(saver))["messages"] == [
            message("one"),
            message("two"),
        ]

    async def test_restart_mid_thread_snapshots_once_then_tails(self, db):
        """The cold-cache case: a new process knows nothing about the log, so
        it rewrites it whole, and only then goes back to tailing."""
        oracle = InMemorySaver()
        thread = Thread(oracle)  # the oracle keeps the sequence across savers
        history = [message(f"m{n}") for n in range(4)]

        async with CheckpointLogSaver.from_conn_string(db) as first:
            thread.savers = (first, oracle)
            await thread.put({"messages": list(history)})
            assert first._logs[("t1", "", "messages")].length == 4

        async with CheckpointLogSaver.from_conn_string(db) as second:
            thread.savers = (second, oracle)
            assert not second._logs
            history = history + [message("m4")]
            await thread.put({"messages": list(history)})
            # The whole log was rewritten, so all five rows carry this put.
            assert second._logs[("t1", "", "messages")].length == 5
            history = history + [message("m5")]
            await thread.put({"messages": list(history)})
            assert await rows(db, "channel_items", channel="messages") == 6
            assert await thread.state(second) == await thread.state(oracle)


# ------------------------------------------------------------------- interrupt


class TestWritesAndResume:
    async def test_pending_writes_come_back_with_the_checkpoint(self, saver):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        checkpoint_id = await thread.put({"messages": [message("one")]})
        config = config_for("t1", checkpoint_id)
        for one in (saver, oracle):
            await one.aput_writes(config, [("messages", [message("two")])], "task-1")
        mine = await saver.aget_tuple(config_for("t1"))
        theirs = await oracle.aget_tuple(config_for("t1"))
        assert mine.pending_writes == theirs.pending_writes
        assert mine.pending_writes[0][1] == "messages"

    async def test_an_interrupt_resumes_through_a_real_graph(self, tmp_path):
        """The one path that reads `writes` and `parent_checkpoint_id`: park a
        graph at `interrupt()`, close the saver, reopen it, resume."""
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt

        db = str(tmp_path / "interrupt.db")

        def build(saver):
            builder = StateGraph(dict)

            def ask(state):
                answer = interrupt("how many?")
                return {"log": state.get("log", []) + [answer]}

            builder.add_node("ask", ask)
            builder.add_edge(START, "ask")
            builder.add_edge("ask", END)
            return builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "resume-me"}}
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            await build(saver).ainvoke({"log": ["before"]}, config)

        async with CheckpointLogSaver.from_conn_string(db) as saver:
            graph = build(saver)
            state = await graph.aget_state(config)
            assert state.tasks and state.tasks[0].interrupts
            result = await graph.ainvoke(Command(resume="forty two"), config)
            assert result["log"] == ["before", "forty two"]

    async def test_a_thread_survives_the_graph_being_rebuilt(self, tmp_path):
        """`test_graph.py`'s sqlite claim, made of this saver."""
        from langgraph.graph import END, START, StateGraph

        db = str(tmp_path / "rebuild.db")

        def build(saver, reply):
            builder = StateGraph(AppendState)
            builder.add_node("say", lambda state: {"messages": [reply]})
            builder.add_edge(START, "say")
            builder.add_edge("say", END)
            return builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "s1"}}
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            await build(saver, "one").ainvoke({"messages": []}, config)
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            out = await build(saver, "two").ainvoke({}, config)
        assert out["messages"] == ["one", "two"]


# ------------------------------------------------------- fork and compaction


class TestForkAndCompaction:
    async def test_fork_thread_and_apply_compaction_round_trip(self, saver, tools_llm):
        from hpca.agent.graph import apply_compaction, build_graph, fork_thread
        from hpca.agent.graph import run_turn

        llm, tools = tools_llm
        graph = build_graph(llm=llm, tools=tools, checkpointer=saver)
        await run_turn(graph, session_id="s1", user_text="first")
        await run_turn(graph, session_id="s1", user_text="second")

        applied = await apply_compaction(
            graph, session_id="s1", upto=2, summary=message("a summary", "system")
        )
        assert applied
        state = (await graph.aget_state(config_for("s1"))).values
        assert state["compacted"]["upto"] == 2

        kept = await fork_thread(
            graph, source_session_id="s1", target_session_id="s2", keep=2
        )
        forked = (await graph.aget_state(config_for("s2"))).values
        assert forked["messages"] == kept
        assert len(forked["messages"]) == 2
        # The fork is a fresh log, not a share of the source's.
        assert (await graph.aget_state(config_for("s1"))).values["messages"] != kept


# ------------------------------------------------------------------ retention


class TestRetention:
    async def test_only_the_last_thirty_two_manifests_survive(self, saver, db):
        thread = Thread(saver)
        history: list = []
        for n in range(KEEP_CHECKPOINTS + 10):
            history = history + [message(f"m{n}")]
            await thread.put({"messages": list(history)})
        assert await rows(db, "checkpoints", thread_id="t1") == KEEP_CHECKPOINTS
        # And the conversation is untouched by the pruning.
        assert len((await thread.state(saver))["messages"]) == KEEP_CHECKPOINTS + 10
        assert await rows(db, "channel_items") == KEEP_CHECKPOINTS + 10

    async def test_the_writes_of_a_pruned_checkpoint_go_with_it(self, saver, db):
        thread = Thread(saver)
        first = await thread.put({"messages": [message("m0")]})
        await saver.aput_writes(
            config_for("t1", first), [("messages", [message("x")])], "task-1"
        )
        assert await rows(db, "writes", checkpoint_id=first) == 1
        history = [message("m0")]
        for n in range(KEEP_CHECKPOINTS + 5):
            history = history + [message(f"m{n + 1}")]
            await thread.put({"messages": list(history)})
        assert await rows(db, "writes", checkpoint_id=first) == 0

    async def test_a_live_blob_survives_and_an_orphan_does_not(self, saver, db):
        thread = Thread(saver)
        await thread.put({"plan": [{"text": "first plan", "done": False}]})
        for n in range(KEEP_CHECKPOINTS + 10):
            await thread.put({"tool_rounds": n})
        # `plan` was written once, long ago, and nothing has touched it since:
        # every surviving manifest still names that version, so it stays.
        assert await rows(db, "channel_blobs", channel="plan") == 1
        # `tool_rounds` changed every step; only the versions the surviving
        # manifests name are still there.
        assert await rows(db, "channel_blobs", channel="tool_rounds") == (
            KEEP_CHECKPOINTS
        )
        state = await thread.state(saver)
        assert state["plan"] == [{"text": "first plan", "done": False}]
        assert state["tool_rounds"] == KEEP_CHECKPOINTS + 9

    async def test_retention_is_per_thread(self, saver, db):
        for thread_id in ("a", "b"):
            thread = Thread(saver, thread_id=thread_id)
            for n in range(KEEP_CHECKPOINTS + 5):
                await thread.put({"tool_rounds": n})
        assert await rows(db, "checkpoints", thread_id="a") == KEEP_CHECKPOINTS
        assert await rows(db, "checkpoints", thread_id="b") == KEEP_CHECKPOINTS


class TestDeleteThread:
    async def test_it_takes_all_four_tables_with_it(self, saver, db):
        a = Thread(saver, thread_id="a")
        b = Thread(saver, thread_id="b")
        for thread in (a, b):
            checkpoint_id = await thread.put(
                {"messages": [message("one")], "plan": [{"text": "p", "done": False}]}
            )
            await saver.aput_writes(
                config_for(thread.thread_id, checkpoint_id),
                [("messages", [message("x")])],
                "task-1",
            )
        await saver.adelete_thread("a")
        for table in ("checkpoints", "writes", "channel_items", "channel_blobs"):
            assert await rows(db, table, thread_id="a") == 0
            assert await rows(db, table, thread_id="b") > 0
        assert await saver.aget_tuple(config_for("a")) is None
        assert await b.state(saver) is not None

    async def test_and_a_deleted_thread_can_be_used_again(self, saver):
        oracle = InMemorySaver()
        thread = Thread(saver, oracle)
        await thread.put({"messages": [message("one"), message("two")]})
        await saver.adelete_thread("t1")
        await oracle.adelete_thread("t1")
        fresh = Thread(saver, oracle)
        await fresh.put({"messages": [message("brand new")]})
        assert await fresh.state(saver) == await fresh.state(oracle)


# ----------------------------------------------------------------------- alist


class TestList:
    async def test_it_walks_the_thread_newest_first(self, saver):
        thread = Thread(saver)
        ids = []
        history: list = []
        for n in range(3):
            history = history + [message(f"m{n}")]
            ids.append(await thread.put({"messages": list(history)}))
        listed = [t async for t in saver.alist(config_for("t1"))]
        assert [t.config["configurable"]["checkpoint_id"] for t in listed] == ids[::-1]
        assert listed[0].checkpoint["channel_values"]["messages"] == history

    async def test_limit_and_filter(self, saver):
        thread = Thread(saver)
        await thread.put({"tool_rounds": 0}, metadata={"source": "input", "step": -1})
        await thread.put({"tool_rounds": 1}, metadata={"source": "loop", "step": 0})
        limited = [t async for t in saver.alist(config_for("t1"), limit=1)]
        assert len(limited) == 1
        filtered = [
            t async for t in saver.alist(config_for("t1"), filter={"source": "input"})
        ]
        assert len(filtered) == 1
        assert filtered[0].metadata["source"] == "input"


# ------------------------------------------------------------------------ fuzz


class TestFuzzAgainstTheOracle:
    @pytest.mark.parametrize("seed", range(8))
    async def test_random_puts_agree_with_in_memory_at_every_step(self, seed, db):
        """N random puts drawn from {append, truncate, blob update, restart},
        compared against `InMemorySaver` after each one."""
        rng = random.Random(seed)
        oracle = InMemorySaver()
        thread = Thread(oracle)
        saver = None
        try:
            async with CheckpointLogSaver.from_conn_string(db) as first:
                saver = first
                thread.savers = (saver, oracle)
                for step in range(60):
                    choice = rng.random()
                    if choice < 0.55:  # append to a log channel
                        channel = rng.choice(LOG_CHANNELS)
                        current = list(thread.values.get(channel) or [])
                        current.append(
                            {"n": step, "text": "x" * rng.randint(1, 40)}
                        )
                        await thread.put({channel: current})
                    elif choice < 0.7:  # truncate one
                        channel = rng.choice(LOG_CHANNELS)
                        current = list(thread.values.get(channel) or [])
                        if current:
                            keep = rng.randrange(len(current))
                            await thread.put({channel: current[:keep]})
                    elif choice < 0.9:  # a blob channel
                        await thread.put(
                            {
                                rng.choice(("plan", "tool_rounds", "compacted")): (
                                    {"step": step} if rng.random() < 0.5 else None
                                )
                            }
                        )
                    else:  # a restart: the saver forgets everything in memory
                        saver._logs.clear()
                        saver._blobs.clear()
                    assert await thread.state(saver) == await thread.state(oracle), (
                        f"diverged at step {step}"
                    )
        finally:
            if saver is not None:
                saver._logs.clear()


# ------------------------------------------------------------------- migration


def write_inline_db(path, threads: dict[str, list[dict]]):
    """A `checkpoints.db` in the shape `AsyncSqliteSaver` leaves behind."""
    serde = JsonPlusSerializer()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );
        CREATE TABLE writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            value BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        );
        """
    )
    for thread_id, states in threads.items():
        parent = None
        for step, values in enumerate(states):
            # Real uuid6 ids, not readable ones: the saver orders checkpoints
            # lexicographically, and an id that does not sort like a uuid6
            # would make a migrated row outrank everything written after it.
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = values
            checkpoint["channel_versions"] = {
                k: f"{step + 1:032}.{0.5:016}" for k in values
            }
            type_, blob = serde.dumps_typed(checkpoint)
            conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    "",
                    checkpoint["id"],
                    parent,
                    type_,
                    blob,
                    json.dumps({"source": "loop", "step": step}).encode(),
                ),
            )
            conn.execute(
                "INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", checkpoint["id"], "task", 0, "messages", *serde
                 .dumps_typed([message("pending")])),
            )
            parent = checkpoint["id"]
    conn.commit()
    conn.close()


class TestMigration:
    def history(self, n):
        return [
            {
                "messages": [message(f"m{i}") for i in range(step + 1)],
                "thinking": [{"after": step, "text": "t"}],
                "plan": [{"text": "p", "done": False}],
                "tool_rounds": step,
            }
            for step in range(n)
        ]

    async def test_the_latest_state_of_every_thread_survives(self, db):
        write_inline_db(db, {"a": self.history(5), "b": self.history(3)})
        result = migrate_inline_format(db)
        assert result.ran and result.error is None and result.threads == 2
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            a = (await saver.aget_tuple(config_for("a"))).checkpoint
            b = (await saver.aget_tuple(config_for("b"))).checkpoint
        assert len(a["channel_values"]["messages"]) == 5
        assert len(b["channel_values"]["messages"]) == 3
        assert a["channel_values"]["plan"] == [{"text": "p", "done": False}]

    async def test_the_old_table_is_gone_and_the_file_shrinks(self, db):
        write_inline_db(db, {"a": self.history(60)})
        before = migrate_inline_format(db)
        conn = sqlite3.connect(db)
        try:
            tables = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        finally:
            conn.close()
        assert "checkpoints_inline" not in tables
        assert before.bytes_after < before.bytes_before

    async def test_the_surviving_checkpoint_keeps_its_writes(self, db):
        write_inline_db(db, {"a": self.history(4)})
        migrate_inline_format(db)
        assert await rows(db, "writes", thread_id="a") == 1
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            tup = await saver.aget_tuple(config_for("a"))
        assert tup.pending_writes and tup.pending_writes[0][1] == "messages"

    async def test_the_thread_carries_on_from_where_it_was(self, db):
        write_inline_db(db, {"a": self.history(4)})
        migrate_inline_format(db)
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            values = (await saver.aget_tuple(config_for("a"))).checkpoint
            thread = Thread(saver, thread_id="a")
            thread.values = dict(values["channel_values"])
            thread.versions = dict(values["channel_versions"])
            await thread.put(
                {"messages": thread.values["messages"] + [message("after")]}
            )
            after = await thread.state(saver)
        assert len(after["messages"]) == 5
        assert after["messages"][-1] == message("after")

    async def test_running_it_twice_is_a_no_op(self, db):
        write_inline_db(db, {"a": self.history(3)})
        migrate_inline_format(db)
        second = migrate_inline_format(db)
        assert not second.ran and second.error is None
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            assert await saver.aget_tuple(config_for("a")) is not None

    async def test_a_fresh_file_needs_nothing(self, db):
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            await saver.setup()
        result = migrate_inline_format(db)
        assert not result.ran and result.error is None

    async def test_a_missing_file_is_not_an_error(self, tmp_path):
        result = migrate_inline_format(tmp_path / "nothing.db")
        assert not result.ran and result.error is None

    async def test_an_unreadable_thread_is_skipped_not_fatal(self, db):
        write_inline_db(db, {"a": self.history(2), "b": self.history(2)})
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE checkpoints SET checkpoint = ? WHERE thread_id = 'b'",
            (b"not msgpack at all",),
        )
        conn.commit()
        conn.close()
        result = migrate_inline_format(db)
        assert result.error is None and result.threads == 1
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            assert await saver.aget_tuple(config_for("a")) is not None
            assert await saver.aget_tuple(config_for("b")) is None

    async def test_a_migration_that_never_ran_does_not_stop_the_app(self, db):
        """The last line of defence: the saver opens a file still in the old
        shape, moves it aside, and starts."""
        write_inline_db(db, {"a": self.history(3)})
        async with CheckpointLogSaver.from_conn_string(db) as saver:
            thread = Thread(saver, thread_id="fresh")
            await thread.put({"messages": [message("a new conversation")]})
            assert (await thread.state(saver))["messages"] == [
                message("a new conversation")
            ]
        conn = sqlite3.connect(db)
        try:
            tables = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        finally:
            conn.close()
        # Moved aside, not destroyed — the next start migrates it.
        assert "checkpoints_inline" in tables
        result = migrate_inline_format(db)
        assert result.ran and result.threads == 1


@pytest.fixture
def tools_llm():
    """A graph's worth of doubles: no tools, and a model that only ever
    answers. Enough to drive `run_turn`, which is all this needs."""
    from hpca.agent.tools import ToolRegistry
    from hpca.llm import ChatResponse

    class FakeLLM:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, *, json_schema=None, **kwargs):
            self.calls.append({"messages": list(messages)})
            return ChatResponse(
                content=json.dumps({"action": "respond", "response": "sure"})
            )

        async def supports_constrained_decoding(self):
            return True

    return FakeLLM(), ToolRegistry()
