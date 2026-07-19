"""Tests for hpca.agent.graph: the checkpointed orchestrator loop (§4.1, §4.2)."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from hpca.agent.graph import MAX_TOOL_ROUNDS, build_graph, run_turn
from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse


class EchoParams(BaseModel):
    text: str = Field(description="Text to echo")


class DeleteParams(BaseModel):
    target: str = Field(description="What to delete")


async def echo_handler(args, ctx):
    return f"echo: {args.text}"


async def delete_handler(args, ctx):
    return f"deleted {args.target}"


@pytest.fixture
def tools():
    registry = ToolRegistry()
    registry.register(
        Tool(name="echo", description="Echo text", params=EchoParams, handler=echo_handler)
    )
    registry.register(
        Tool(
            name="delete",
            description="Delete something",
            params=DeleteParams,
            handler=delete_handler,
            destructive=True,
        )
    )
    return registry


class FakeLLM:
    def __init__(self, outputs, reasoning=None):
        self._outputs = list(outputs)
        self._reasoning = list(reasoning or [])
        self.calls: list[dict] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append({"messages": list(messages), "json_schema": json_schema})
        return ChatResponse(
            content=self._outputs.pop(0),
            reasoning=self._reasoning.pop(0) if self._reasoning else None,
        )

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


def make_graph(llm, tools):
    return build_graph(llm=llm, tools=tools, checkpointer=InMemorySaver())


class TestDirectResponse:
    async def test_assistant_message_appended(self, tools):
        llm = FakeLLM([respond_json("hello there")])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="hi")
        assert result.reply == "hello there"
        assert result.interrupt is None
        assert result.messages[-1] == {"role": "assistant", "content": "hello there"}

    async def test_system_prompt_first(self, tools):
        llm = FakeLLM([respond_json()])
        graph = make_graph(llm, tools)
        await run_turn(graph, session_id="s1", user_text="hi")
        assert llm.calls[0]["messages"][0]["role"] == "system"


class TestToolLoop:
    async def test_tool_result_fed_back_then_final_answer(self, tools):
        llm = FakeLLM([tool_json("echo", text="hi"), respond_json("all done")])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="echo hi")
        assert result.reply == "all done"
        # tool result became part of the conversation for the second call
        second_call = llm.calls[1]["messages"]
        assert any(
            m["role"] == "user" and "echo: hi" in m["content"] for m in second_call
        )

    async def test_tool_rounds_capped_then_summarises(self, tools):
        # MAX tool calls, then one tools-stripped call that answers from what
        # the results already show — the model does not just say "I stopped".
        llm = FakeLLM(
            [tool_json("echo", text="x")] * MAX_TOOL_ROUNDS
            + [respond_json("From the results so far: it is GRCh38.")]
        )
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="loop forever")
        assert result.interrupt is None
        assert "GRCh38" in result.reply  # answered, not a canned stop
        # MAX tool decisions + 1 summary decision
        assert len(llm.calls) == MAX_TOOL_ROUNDS + 1

    async def test_the_summary_call_has_no_tools(self, tools):
        llm = FakeLLM(
            [tool_json("echo", text="x")] * MAX_TOOL_ROUNDS
            + [respond_json("done summarising")]
        )
        graph = make_graph(llm, tools)
        await run_turn(graph, session_id="s1", user_text="go")
        # the last decision was made with the budget note and no tool listing
        last = llm.calls[-1]["messages"]
        combined = " ".join(m["content"] for m in last)
        assert "tool budget" in combined
        assert "To call a tool" not in combined  # tool listing withdrawn

    async def test_budget_is_configurable(self, tools):
        llm = FakeLLM([tool_json("echo", text="x")] * 3 + [respond_json("stopped")])
        graph = build_graph(
            llm=llm, tools=tools, checkpointer=InMemorySaver(), max_tool_rounds=3
        )
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.interrupt is None
        assert len(llm.calls) == 3 + 1  # 3 tool rounds, then the summary

    async def test_summary_falling_over_still_stops_cleanly(self, tools):
        # if even the summary call cannot produce a valid decision, the turn
        # ends with a plain message rather than looping or raising
        llm = FakeLLM([tool_json("echo", text="x")] * MAX_TOOL_ROUNDS + ["garbage"] * 4)
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.interrupt is None
        assert "tool call" in result.reply.lower()

    async def test_handler_exception_surfaces_as_tool_error(self, tools):
        async def boom(args, ctx):
            raise RuntimeError("disk on fire")

        tools.register(
            Tool(name="boom", description="explodes", params=EchoParams, handler=boom)
        )
        llm = FakeLLM([tool_json("boom", text="x"), respond_json("I saw an error")])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.reply == "I saw an error"
        second_call = llm.calls[1]["messages"]
        assert any("disk on fire" in m["content"] for m in second_call)


class TestDestructiveGate:
    async def test_destructive_tool_interrupts_with_details(self, tools):
        llm = FakeLLM([tool_json("delete", target="results/")])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="delete results")
        assert result.interrupt is not None
        assert result.interrupt["tool"] == "delete"
        assert result.interrupt["arguments"] == {"target": "results/"}

    async def test_approve_executes_tool(self, tools):
        llm = FakeLLM([tool_json("delete", target="results/"), respond_json("gone")])
        graph = make_graph(llm, tools)
        first = await run_turn(graph, session_id="s1", user_text="delete results")
        assert first.interrupt is not None
        result = await run_turn(
            graph, session_id="s1", resume=Command(resume={"approved": True})
        )
        assert result.reply == "gone"
        second_call = llm.calls[1]["messages"]
        assert any("deleted results/" in m["content"] for m in second_call)

    async def test_reject_skips_execution(self, tools):
        llm = FakeLLM([tool_json("delete", target="results/"), respond_json("ok")])
        graph = make_graph(llm, tools)
        await run_turn(graph, session_id="s1", user_text="delete results")
        result = await run_turn(
            graph, session_id="s1", resume=Command(resume={"approved": False})
        )
        assert result.reply == "ok"
        second_call = llm.calls[1]["messages"]
        assert any("denied" in m["content"].lower() for m in second_call)
        assert not any("deleted results/" in m["content"] for m in second_call)

    async def test_conditionally_destructive_tool(self, tools):
        class FlagParams(BaseModel):
            danger: bool = False

        async def flag_handler(args, ctx):
            return "flagged"

        tools.register(
            Tool(
                name="flag",
                description="conditionally destructive",
                params=FlagParams,
                handler=flag_handler,
                is_destructive_call=lambda args, ctx: args.danger,
            )
        )
        llm = FakeLLM(
            [tool_json("flag", danger=False), respond_json("safe done"),
             tool_json("flag", danger=True)]
        )
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="safe")
        assert result.interrupt is None and result.reply == "safe done"
        result = await run_turn(graph, session_id="s2", user_text="dangerous")
        assert result.interrupt is not None
        assert result.interrupt["tool"] == "flag"

    async def test_non_destructive_tool_does_not_interrupt(self, tools):
        llm = FakeLLM([tool_json("echo", text="x"), respond_json("done")])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="echo")
        assert result.interrupt is None


class TestPersistence:
    async def test_second_turn_sees_first_turn(self, tools):
        llm = FakeLLM([respond_json("first answer"), respond_json("second answer")])
        graph = make_graph(llm, tools)
        await run_turn(graph, session_id="s1", user_text="first question")
        await run_turn(graph, session_id="s1", user_text="second question")
        second_call = llm.calls[1]["messages"]
        contents = [m["content"] for m in second_call]
        assert "first question" in contents
        assert "first answer" in contents
        assert "second question" in contents

    async def test_threads_isolated(self, tools):
        llm = FakeLLM([respond_json("a"), respond_json("b")])
        graph = make_graph(llm, tools)
        await run_turn(graph, session_id="s1", user_text="in thread one")
        await run_turn(graph, session_id="s2", user_text="in thread two")
        second_call = llm.calls[1]["messages"]
        assert not any("thread one" in m["content"] for m in second_call)

    async def test_sqlite_checkpointer_survives_graph_rebuild(self, tools, tmp_path):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db = str(tmp_path / "checkpoints.db")
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            llm = FakeLLM([respond_json("answer one")])
            graph = build_graph(llm=llm, tools=tools, checkpointer=saver)
            await run_turn(graph, session_id="s1", user_text="remember me")

        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            llm2 = FakeLLM([respond_json("answer two")])
            graph2 = build_graph(llm=llm2, tools=tools, checkpointer=saver)
            await run_turn(graph2, session_id="s1", user_text="what did I say?")
            contents = [m["content"] for m in llm2.calls[0]["messages"]]
            assert "remember me" in contents


class TestDecisionFailure:
    async def test_exhausted_retries_surface_to_user(self, tools):
        llm = FakeLLM(["garbage"] * 10)
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="hi")
        assert result.interrupt is None
        assert "valid" in result.reply.lower() or "fail" in result.reply.lower()


# --------------------------------------------------------- integration tests

from hpca.config import LLMSettings  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402

from tests.live_backend import LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@integration
class TestGraphLive:
    async def test_full_tool_loop_with_live_model(self, tools):
        llm = LLMClient(
            LLMSettings(
                base_url=LIVE_URL,
                model=LIVE_MODEL,
                request_timeout_s=120,
                # these test routing, not reasoning; thinking is ~15x slower
                enable_thinking=False,
            )
        )
        graph = make_graph(llm, tools)
        result = await run_turn(
            graph,
            session_id="live-1",
            user_text="Use the echo tool on the text 'BAM123', then tell me what it returned.",
        )
        assert result.interrupt is None
        assert result.reply is not None
        # the tool must actually have run
        assert any(
            m["role"] == "user" and "echo: BAM123" in m["content"]
            for m in result.messages
        ), result.messages
        await llm.close()

    async def test_live_destructive_gate_roundtrip(self, tools):
        llm = LLMClient(
            LLMSettings(
                base_url=LIVE_URL,
                model=LIVE_MODEL,
                request_timeout_s=120,
                # these test routing, not reasoning; thinking is ~15x slower
                enable_thinking=False,
            )
        )
        graph = make_graph(llm, tools)
        first = await run_turn(
            graph,
            session_id="live-2",
            user_text="Use the delete tool to delete 'old_logs'.",
        )
        assert first.interrupt is not None, first.reply
        assert first.interrupt["tool"] == "delete"
        result = await run_turn(
            graph, session_id="live-2", resume=Command(resume={"approved": True})
        )
        assert result.interrupt is None
        assert any("deleted old_logs" in m["content"] for m in result.messages)
        await llm.close()


class TestThinkingState:
    async def test_reasoning_is_kept_out_of_the_messages(self, tools):
        llm = FakeLLM([respond_json("42")], reasoning=["Let me think about this."])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="t1", user_text="how many?")
        assert all(
            "Let me think" not in m["content"] for m in result.messages
        ), "reasoning must never enter the conversation fed back to the model"
        assert result.thinking == [{"after": 1, "reasoning": "Let me think about this."}]

    async def test_reasoning_anchors_to_the_message_it_produced(self, tools):
        llm = FakeLLM(
            [tool_json("echo", text="hi"), respond_json("done")],
            reasoning=["I should echo first.", "Now I can answer."],
        )
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="t2", user_text="echo hi")
        # 0: user, 1: tool result (from the first decision), 2: the answer
        assert result.thinking == [
            {"after": 1, "reasoning": "I should echo first."},
            {"after": 2, "reasoning": "Now I can answer."},
        ]
        assert result.messages[1]["content"].startswith("[tool result]")
        assert result.messages[2]["role"] == "assistant"

    async def test_no_reasoning_no_entries(self, tools):
        llm = FakeLLM([respond_json("hi")])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="t3", user_text="hello")
        assert result.thinking == []

    async def test_blank_reasoning_ignored(self, tools):
        llm = FakeLLM([respond_json("hi")], reasoning=["   \n "])
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="t4", user_text="hello")
        assert result.thinking == []

    async def test_thinking_accumulates_across_turns(self, tools):
        llm = FakeLLM([respond_json("a"), respond_json("b")], reasoning=["one", "two"])
        graph = make_graph(llm, tools)
        await run_turn(graph, session_id="t5", user_text="first")
        result = await run_turn(graph, session_id="t5", user_text="second")
        assert result.thinking == [
            {"after": 1, "reasoning": "one"},
            {"after": 3, "reasoning": "two"},
        ]


class TestNewThisTurn:
    async def test_first_new_marks_the_turns_own_messages(self, tools):
        llm = FakeLLM([respond_json("a"), respond_json("b")])
        graph = make_graph(llm, tools)
        first = await run_turn(graph, session_id="n1", user_text="one")
        assert first.first_new == 0  # the whole session is new
        second = await run_turn(graph, session_id="n1", user_text="two")
        assert second.first_new == 2
        assert [m["content"] for m in second.messages[second.first_new:]] == ["two", "b"]

    async def test_interrupt_and_resume_split_the_turn_without_overlap(self, tools):
        llm = FakeLLM([tool_json("delete", target="x"), respond_json("gone")])
        graph = make_graph(llm, tools)
        first = await run_turn(graph, session_id="n2", user_text="delete x")
        assert first.interrupt is not None
        assert first.first_new == 0
        resumed = await run_turn(
            graph, session_id="n2", resume=Command(resume={"approved": True})
        )
        # the resume logs only what it added: the tool result and the answer
        new = [m["content"] for m in resumed.messages[resumed.first_new:]]
        assert len(new) == 2
        assert new[0].startswith("[tool result] delete")
        assert new[1] == "gone"


class TestCompaction:
    """Redesign Phase 6: fold the old history before it overflows the window.

    The stored history is never rewritten — only the view sent to the model —
    so the transcript keeps everything the user can scroll back to.
    """

    def long_history(self, count=40, chars=200):
        return [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"m{i} " + "x" * chars,
            }
            for i in range(count)
        ]

    async def prime(self, graph, session_id, messages):
        await graph.aupdate_state(
            {"configurable": {"thread_id": session_id}}, {"messages": messages}
        )

    async def test_off_without_a_known_window(self, tools):
        llm = FakeLLM([respond_json("ok")])
        graph = build_graph(llm=llm, tools=tools, checkpointer=InMemorySaver())
        await self.prime(graph, "s1", self.long_history())
        await run_turn(graph, session_id="s1", user_text="and now?")
        # every message still went to the model
        assert len(llm.calls[0]["messages"]) > 40

    async def test_folds_the_old_history_when_the_window_is_small(self, tools):
        llm = FakeLLM(["a summary of the earlier work", respond_json("ok")])
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            max_model_len=lambda: 2000,
        )
        await self.prime(graph, "s1", self.long_history())
        result = await run_turn(graph, session_id="s1", user_text="and now?")
        assert result.reply == "ok"
        summarize_call, decide_call = llm.calls[0], llm.calls[1]
        assert summarize_call["json_schema"] is None  # the summarizer is free-form
        # the decision saw a folded view: far fewer messages, summary first
        sent = decide_call["messages"]
        assert len(sent) < 25
        assert any(
            "a summary of the earlier work" in str(m["content"]) for m in sent
        )
        # the recent tail survived verbatim
        assert any("and now?" == str(m["content"]) for m in sent)

    async def test_stored_history_is_not_rewritten(self, tools):
        llm = FakeLLM(["a summary", respond_json("ok")])
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            max_model_len=lambda: 2000,
        )
        await self.prime(graph, "s1", self.long_history())
        result = await run_turn(graph, session_id="s1", user_text="and now?")
        # the transcript keeps the whole session, summary or not
        assert len(result.messages) > 40
        assert any("m0 " in str(m["content"]) for m in result.messages)

    async def test_evicted_messages_are_offered_for_extraction(self, tools):
        seen = []

        async def on_evict(messages):
            seen.append(messages)

        llm = FakeLLM(["a summary", respond_json("ok")])
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            max_model_len=lambda: 2000,
            on_evict=on_evict,
        )
        await self.prime(graph, "s1", self.long_history())
        await run_turn(graph, session_id="s1", user_text="and now?")
        assert seen and len(seen[0]) > 10
        assert "m0 " in str(seen[0][0]["content"])  # the oldest, before it goes

    async def test_a_failing_summarizer_does_not_break_the_turn(self, tools):
        class Failing(FakeLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if json_schema is None:  # the summarize call
                    raise RuntimeError("backend down")
                return await super().chat(messages, json_schema=json_schema, **kwargs)

        llm = Failing([respond_json("ok")])
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            max_model_len=lambda: 2000,
        )
        await self.prime(graph, "s1", self.long_history())
        result = await run_turn(graph, session_id="s1", user_text="and now?")
        assert result.reply == "ok"  # oversized beats not running at all

    async def test_compaction_persists_across_turns(self, tools):
        llm = FakeLLM(["a summary", respond_json("first"), respond_json("second")])
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            max_model_len=lambda: 2000,
        )
        await self.prime(graph, "s1", self.long_history())
        await run_turn(graph, session_id="s1", user_text="first question")
        await run_turn(graph, session_id="s1", user_text="second question")
        # only one summarize call: the second turn reused the stored fold
        free_form = [c for c in llm.calls if c["json_schema"] is None]
        assert len(free_form) == 1

    async def test_budget_exhaustion_path_compacts_too(self, tools):
        """The tools-withdrawn call is the one that rescues the turn's
        findings, so it must not be the one that overflows."""
        llm = FakeLLM(
            [tool_json("echo", text="x")] * 2
            + ["a summary of the earlier work", respond_json("here is what I found")]
        )
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            max_tool_rounds=2,
            max_model_len=lambda: 2000,
        )
        await self.prime(graph, "s1", self.long_history())
        result = await run_turn(graph, session_id="s1", user_text="dig into this")
        assert result.reply == "here is what I found"
        final = llm.calls[-1]["messages"]
        assert len(final) < 25  # folded, not the whole history
        assert any("a summary of the earlier work" in str(m["content"]) for m in final)
