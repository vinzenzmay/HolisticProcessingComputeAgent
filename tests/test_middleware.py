"""Tests for hpca.agent: tool registry and the validation/retry middleware (§4.3)."""

import json

import pytest
from pydantic import BaseModel, Field

from hpca.agent.middleware import (
    DecisionError,
    DirectResponse,
    ToolCall,
    decide,
    decision_schema,
)
from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse


class EchoParams(BaseModel):
    text: str = Field(description="Text to echo")


class CountParams(BaseModel):
    n: int = Field(ge=1, le=100)


async def echo_handler(args, ctx):
    return f"echo: {args.text}"


async def count_handler(args, ctx):
    return ", ".join(str(i) for i in range(1, args.n + 1))


@pytest.fixture
def tools():
    registry = ToolRegistry()
    registry.register(
        Tool(name="echo", description="Echo text", params=EchoParams, handler=echo_handler)
    )
    registry.register(
        Tool(
            name="count",
            description="Count to n",
            params=CountParams,
            handler=count_handler,
            destructive=True,
        )
    )
    return registry


class TestToolRegistry:
    def test_get_known(self, tools):
        assert tools.get("echo").name == "echo"

    def test_get_unknown_raises_with_available_names(self, tools):
        with pytest.raises(KeyError, match="echo"):
            tools.get("nope")

    def test_duplicate_name_rejected(self, tools):
        with pytest.raises(ValueError, match="echo"):
            tools.register(
                Tool(name="echo", description="x", params=EchoParams, handler=echo_handler)
            )

    def test_subset(self, tools):
        sub = tools.subset(["echo"])
        assert sub.names() == ["echo"]
        with pytest.raises(KeyError):
            sub.get("count")

    def test_subset_unknown_name_fails_fast(self, tools):
        with pytest.raises(KeyError):
            tools.subset(["echo", "missing"])


class TestDecisionSchema:
    def test_contains_respond_and_all_tools(self, tools):
        schema = decision_schema(tools)
        branches = schema["anyOf"]
        actions = []
        for b in branches:
            props = b["properties"]
            if props["action"]["const"] == "respond":
                actions.append("respond")
            else:
                actions.append(props["tool"]["const"])
        assert actions == ["respond", "echo", "count"]

    def test_tool_branch_embeds_param_schema(self, tools):
        schema = decision_schema(tools)
        echo_branch = schema["anyOf"][1]
        assert "text" in echo_branch["properties"]["arguments"]["properties"]


class FakeLLM:
    """Scripted LLM: returns canned outputs in order, records every call."""

    def __init__(self, outputs, supports_cd=True):
        self._outputs = list(outputs)
        self._supports_cd = supports_cd
        self.calls: list[dict] = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append({"messages": list(messages), "json_schema": json_schema})
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return self._supports_cd


def respond_json(text="hi"):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool="echo", **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


USER = [{"role": "user", "content": "do something"}]


class TestDecide:
    async def test_direct_response(self, tools):
        llm = FakeLLM([respond_json("hello")])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, DirectResponse)
        assert decision.text == "hello"

    async def test_valid_tool_call(self, tools):
        llm = FakeLLM([tool_json("echo", text="hi")])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, ToolCall)
        assert decision.tool.name == "echo"
        assert decision.arguments.text == "hi"

    async def test_constrained_decoding_passes_schema(self, tools):
        llm = FakeLLM([respond_json()])
        await decide(llm, USER, tools)
        assert llm.calls[0]["json_schema"] == decision_schema(tools)

    async def test_tool_instruction_always_present(self, tools):
        # The schema only constrains syntax; the model must also *read* which
        # tools exist, or it answers "I have no tools" (observed live).
        for supports_cd in (True, False):
            llm = FakeLLM([respond_json()], supports_cd=supports_cd)
            await decide(llm, USER, tools)
            first = llm.calls[0]["messages"][0]
            assert first["role"] == "system"
            assert "JSON" in first["content"]
            assert "echo" in first["content"] and "count" in first["content"]

    async def test_tool_instruction_merges_into_existing_system_message(self, tools):
        # vLLM/Qwen chat templates reject system messages after position 0.
        llm = FakeLLM([respond_json()])
        await decide(
            llm, [{"role": "system", "content": "Base prompt."}] + USER, tools
        )
        messages = llm.calls[0]["messages"]
        assert sum(1 for m in messages if m["role"] == "system") == 1
        assert messages[0]["role"] == "system"
        assert "Base prompt." in messages[0]["content"]
        assert "echo" in messages[0]["content"]

    async def test_no_constrained_decoding_omits_schema(self, tools):
        llm = FakeLLM([respond_json()], supports_cd=False)
        await decide(llm, USER, tools)
        assert llm.calls[0]["json_schema"] is None

    async def test_malformed_json_retries_with_error_feedback(self, tools):
        llm = FakeLLM(["not json {", respond_json("ok")])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, DirectResponse)
        retry_messages = llm.calls[1]["messages"]
        # previous bad output present as assistant turn, error explained after
        assert {"role": "assistant", "content": "not json {"} in retry_messages
        assert retry_messages[-1]["role"] == "user"
        assert "not valid JSON" in retry_messages[-1]["content"]

    async def test_unknown_tool_retry_names_available_tools(self, tools):
        llm = FakeLLM([tool_json("delete_everything"), tool_json("echo", text="x")])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, ToolCall)
        feedback = llm.calls[1]["messages"][-1]["content"]
        assert "delete_everything" in feedback
        assert "echo" in feedback and "count" in feedback

    async def test_invalid_arguments_retry_contains_pydantic_error(self, tools):
        llm = FakeLLM([tool_json("count", n=1000), tool_json("count", n=5)])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, ToolCall)
        assert decision.arguments.n == 5
        feedback = llm.calls[1]["messages"][-1]["content"]
        assert "less than or equal to 100" in feedback

    async def test_retries_bounded(self, tools):
        llm = FakeLLM(["bad"] * 10)
        with pytest.raises(DecisionError) as exc:
            await decide(llm, USER, tools, max_retries=2)
        assert len(llm.calls) == 3  # initial + 2 retries
        assert "3 attempts" in str(exc.value)

    async def test_execute_tool_call(self, tools):
        llm = FakeLLM([tool_json("echo", text="hi")])
        decision = await decide(llm, USER, tools)
        result = await decision.execute(ctx=None)
        assert result == "echo: hi"


# --------------------------------------------------------- integration tests

from hpca.config import LLMSettings  # noqa: E402
from hpca.agent.prompts import RESPOND_VS_TOOL_GUIDANCE  # noqa: E402
from hpca.llm import LLMClient  # noqa: E402

from tests.live_backend import LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@integration
class TestDecideLive:
    @pytest.fixture
    def llm(self):
        return LLMClient(
            LLMSettings(
                base_url=LIVE_URL,
                model=LIVE_MODEL,
                request_timeout_s=120,
                # these test routing, not reasoning; thinking is ~15x slower
                enable_thinking=False,
            )
        )

    async def test_picks_tool_call(self, tools, llm):
        messages = [
            {
                "role": "system",
                "content": RESPOND_VS_TOOL_GUIDANCE,
            },
            {"role": "user", "content": "Call the count tool with n=7."},
        ]
        decision = await decide(llm, messages, tools)
        assert isinstance(decision, ToolCall)
        assert decision.tool.name == "count"
        assert decision.arguments.n == 7

    async def test_picks_direct_response(self, tools, llm):
        messages = [
            {
                "role": "system",
                "content": RESPOND_VS_TOOL_GUIDANCE,
            },
            {"role": "user", "content": "Just say hi to me, no tools needed."},
        ]
        decision = await decide(llm, messages, tools)
        assert isinstance(decision, DirectResponse)
