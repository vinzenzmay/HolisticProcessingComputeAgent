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
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append({"messages": list(messages), "json_schema": json_schema})
        return ChatResponse(content=self._outputs.pop(0))

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

    async def test_tool_rounds_capped(self, tools):
        llm = FakeLLM([tool_json("echo", text="x")] * (MAX_TOOL_ROUNDS + 5))
        graph = make_graph(llm, tools)
        result = await run_turn(graph, session_id="s1", user_text="loop forever")
        assert result.interrupt is None
        assert "tool" in result.reply.lower()  # explains the cap was hit
        assert len(llm.calls) == MAX_TOOL_ROUNDS

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

import os  # noqa: E402

import httpx  # noqa: E402

from hpca.config import LLMSettings  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402

LIVE_URL = os.environ.get("HPCA_TEST_LLM_URL", "http://localhost:51941/v1")
LIVE_MODEL = os.environ.get("HPCA_TEST_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")


def _backend_reachable() -> bool:
    try:
        return httpx.get(f"{LIVE_URL}/models", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


integration = pytest.mark.skipif(
    not _backend_reachable(), reason=f"LLM backend at {LIVE_URL} not reachable"
)


@integration
class TestGraphLive:
    async def test_full_tool_loop_with_live_model(self, tools):
        llm = LLMClient(
            LLMSettings(base_url=LIVE_URL, model=LIVE_MODEL, request_timeout_s=120)
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
            LLMSettings(base_url=LIVE_URL, model=LIVE_MODEL, request_timeout_s=120)
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
