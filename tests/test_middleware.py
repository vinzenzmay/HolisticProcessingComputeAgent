"""Tests for hpca.agent: tool registry and the validation/retry middleware (§4.3)."""

import json
from typing import get_origin

import pytest
from pydantic import BaseModel, Field

from hpca.agent.middleware import (
    DecisionError,
    DirectResponse,
    ToolCall,
    decide,
    decision_schema,
    format_instruction,
    inline_refs,
)
from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse, TruncatedOutput


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
        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return ChatResponse(content=output)

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

from tests.live_backend import LIVE_KEY, LIVE_MODEL, LIVE_URL, integration  # noqa: E402


@integration
class TestDecideLive:
    @pytest.fixture
    def llm(self):
        return LLMClient(
            LLMSettings(
                base_url=LIVE_URL,
                model=LIVE_MODEL,
                api_key=LIVE_KEY,
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


@integration
class TestLongArrayLive:
    """The artifact the argument-order rule exists for, against a real
    grammar: a model writing a long ``list[str]`` must reach the end of the
    array without a swallowed key (see the module docstring).

    Asserted on ``repairs`` rather than on the content, because that is the
    difference the ordering makes — ``_strip_key_echo`` would clean the call
    up either way, and a passing test with a non-empty ``repairs`` means the
    grammar is still swallowing keys and only the net is catching them.
    """

    @pytest.fixture
    def llm(self):
        return LLMClient(
            LLMSettings(
                base_url=LIVE_URL,
                model=LIVE_MODEL,
                api_key=LIVE_KEY,
                request_timeout_s=180,
                enable_thinking=False,
            )
        )

    @pytest.fixture
    def tools(self):
        from hpca.agent.builtin_tools import default_tool_registry
        from hpca.agent.file_tools import add_file_tools

        return add_file_tools(default_tool_registry()).subset(
            ["run_bash", "create_file"]
        )

    async def test_a_long_document_arrives_without_a_swallowed_key(
        self, tools, llm
    ):
        messages = [
            {"role": "system", "content": RESPOND_VS_TOOL_GUIDANCE},
            {
                "role": "user",
                "content": (
                    "The directory key 'project' is registered. Write a design "
                    "document to specs.md in it, at least 25 lines long, with "
                    "sections: what the tool does, why it is needed, the "
                    "command-line interface, and what was ruled out. Do it now "
                    "in one tool call."
                ),
            },
        ]
        decision = await decide(llm, messages, tools)
        assert isinstance(decision, ToolCall), decision
        lines = getattr(decision.arguments, "content_lines", [])
        assert len(lines) >= 10, lines  # a long array is the point
        assert decision.repairs == []


class TestNestedToolSchemas:
    """A tool whose params nest another model generates $ref/$defs pointing
    at the document root — which, once embedded as one branch of the decision
    envelope, is the envelope. vLLM rejects the request outright:
    "Grammar error: Pointer '/$defs/X' does not exist"."""

    def registry(self):
        from hpca.agent.memory_tools import add_memory_tools

        return add_memory_tools(ToolRegistry())

    def test_no_dangling_refs_in_the_decision_schema(self):
        blob = json.dumps(decision_schema(self.registry()))
        assert "$ref" not in blob
        assert "$defs" not in blob

    def test_every_registered_tool_is_self_contained(self):
        """Regression guard for the next nested tool, whichever it is."""
        from hpca.agent.builtin_tools import default_tool_registry
        from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
        from hpca.agent.file_tools import add_file_tools
        from hpca.agent.job_tools import add_job_tools
        from hpca.agent.memory_tools import add_memory_tools
        from hpca.agent.skill_tools import add_skill_tools

        registry = add_memory_tools(
            add_skill_tools(
                add_job_tools(
                    add_ask_docs(add_doc_tools(add_file_tools(default_tool_registry())))
                )
            )
        )
        blob = json.dumps(decision_schema(registry))
        assert "$ref" not in blob and "$defs" not in blob

    def test_the_nested_shape_survives_inlining(self):
        schema = decision_schema(self.registry())
        branch = next(
            b
            for b in schema["anyOf"]
            if b["properties"].get("tool", {}).get("const") == "memory"
        )
        item = branch["properties"]["arguments"]["properties"]["operations"]["items"]
        assert item["type"] == "object"
        assert set(item["properties"]) == {"op", "scope", "match", "text"}

    def test_example_shows_the_nested_shape_not_a_placeholder(self):
        """Shown `"operations": "<the changes>"` a small model writes exactly
        that string and the call fails validation."""
        text = format_instruction(self.registry())
        line = next(line for line in text.splitlines() if '"memory"' in line)
        arguments = json.loads(
            line[line.index('"arguments": ') + len('"arguments": ') : line.rindex("}} —") + 1]
        )
        assert isinstance(arguments["operations"], list)
        assert set(arguments["operations"][0]) == {"op", "scope", "match", "text"}


class TestInlineRefs:
    def test_plain_schema_unchanged(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert inline_refs(schema) == schema

    def test_definition_substituted(self):
        schema = {
            "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        }
        assert inline_refs(schema) == {
            "type": "object",
            "properties": {
                "inner": {"type": "object", "properties": {"x": {"type": "integer"}}}
            },
        }

    def test_siblings_of_a_ref_are_kept(self):
        schema = {
            "$defs": {"Inner": {"type": "object"}},
            "properties": {
                "inner": {"$ref": "#/$defs/Inner", "description": "the inner thing"}
            },
        }
        inner = inline_refs(schema)["properties"]["inner"]
        assert inner["type"] == "object"
        assert inner["description"] == "the inner thing"

    def test_nested_definitions_resolved(self):
        schema = {
            "$defs": {
                "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                "B": {"type": "string"},
            },
            "properties": {"a": {"$ref": "#/$defs/A"}},
        }
        result = inline_refs(schema)
        assert result["properties"]["a"]["properties"]["b"] == {"type": "string"}

    def test_recursive_definition_left_alone_rather_than_hanging(self):
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                }
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
        result = inline_refs(schema)  # must terminate
        assert result["properties"]["root"]["properties"]["child"] == {
            "$ref": "#/$defs/Node"
        }

    def test_unknown_ref_left_alone(self):
        schema = {"properties": {"a": {"$ref": "#/$defs/Missing"}}}
        assert inline_refs(schema)["properties"]["a"] == {"$ref": "#/$defs/Missing"}


class TestTruncatedDecision:
    """A decision cut off at max_tokens: nothing to parse, but not nothing to
    say about it. Writing a file is the case that reaches the cap honestly."""

    async def test_a_cut_off_call_is_retried_with_guidance(self, tools):
        llm = FakeLLM(
            [TruncatedOutput("truncated"), tool_json("echo", text="hi")]
        )
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, ToolCall)
        feedback = llm.calls[1]["messages"][-1]
        assert feedback["role"] == "user"  # never a system message (§4.3)
        assert "cut off" in feedback["content"]
        assert "parts" in feedback["content"]

    async def test_it_is_retried_once_not_until_the_budget_runs_out(self, tools):
        # A looping model would otherwise cost the full retry budget at the
        # token cap each time — minutes of hang for an output nobody can use.
        llm = FakeLLM([TruncatedOutput("truncated")] * 4)
        with pytest.raises(TruncatedOutput):
            await decide(llm, USER, tools)
        assert len(llm.calls) == 2


def full_registry() -> ToolRegistry:
    """Every tool the app assembles (see ``hpca.core.service``)."""
    from hpca.agent.builtin_tools import default_tool_registry
    from hpca.agent.doc_tools import add_ask_docs, add_doc_tools
    from hpca.agent.file_tools import add_file_tools
    from hpca.agent.job_tools import add_job_tools
    from hpca.agent.memory_tools import add_memory_tools
    from hpca.agent.modes import add_plan_tool
    from hpca.agent.skill_tools import add_skill_tools
    from hpca.agent.watch_tools import add_watch_tools

    registry = add_ask_docs(add_doc_tools(add_file_tools(default_tool_registry())))
    add_job_tools(registry)
    add_watch_tools(registry)
    add_skill_tools(registry)
    add_memory_tools(registry)
    add_plan_tool(registry)
    return registry


class TestArgumentOrder:
    """Array-valued arguments come last (see ``middleware`` module docstring)."""

    def test_no_scalar_argument_follows_an_array_one(self):
        offenders = []
        for tool in full_registry():
            after_array = ""
            for name, field in tool.params.model_fields.items():
                if get_origin(field.annotation) is list:
                    after_array = after_array or name
                elif after_array:
                    offenders.append(f"{tool.name}.{name} follows {after_array}")
        assert offenders == []


class LinesParams(BaseModel):
    timeout_s: int = Field(default=60)
    label: str = Field(default="")
    content_lines: list[str] = Field(min_length=1)


async def lines_handler(args, ctx):
    return "\n".join(args.content_lines)


@pytest.fixture
def lines_tools():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="run_bash",
            description="Run bash",
            params=LinesParams,
            handler=lines_handler,
        )
    )
    return registry


class TestKeyEcho:
    """A next-key the grammar swallowed into a string array (see the module
    docstring) is dropped before the arguments are validated."""

    async def call(self, lines_tools, lines, **rest):
        llm = FakeLLM([tool_json("run_bash", content_lines=lines, **rest)])
        return await decide(llm, USER, lines_tools)

    async def test_trailing_sibling_key_dropped(self, lines_tools):
        decision = await self.call(
            lines_tools, ["echo hi", "EOF", "timeout_s: 60"]
        )
        assert decision.arguments.content_lines == ["echo hi", "EOF"]

    async def test_the_drop_is_reported_not_silent(self, lines_tools):
        decision = await self.call(
            lines_tools, ["echo hi", "EOF", "timeout_s: 60"]
        )
        assert decision.repairs == [
            "dropped a trailing content_lines element that echoed the "
            "timeout_s argument: 'timeout_s: 60'"
        ]

    async def test_quoted_echo_dropped(self, lines_tools):
        decision = await self.call(lines_tools, ["echo hi", '"timeout_s": 60'])
        assert decision.arguments.content_lines == ["echo hi"]

    async def test_several_trailing_echoes_dropped(self, lines_tools):
        decision = await self.call(
            lines_tools, ["echo hi", '"label": "x"', "timeout_s: 60"]
        )
        assert decision.arguments.content_lines == ["echo hi"]

    async def test_a_line_naming_no_argument_is_kept(self, lines_tools):
        decision = await self.call(lines_tools, ["echo hi", "note: 60"])
        assert decision.arguments.content_lines == ["echo hi", "note: 60"]
        assert decision.repairs == []

    async def test_a_line_whose_value_is_not_json_is_kept(self, lines_tools):
        # YAML-ish prose in a heredoc; the echo always carries a JSON value.
        decision = await self.call(lines_tools, ["cat <<EOF", "timeout_s: soon"])
        assert decision.arguments.content_lines == ["cat <<EOF", "timeout_s: soon"]

    async def test_an_echo_that_is_not_last_is_kept(self, lines_tools):
        decision = await self.call(
            lines_tools, ["echo hi", "timeout_s: 60", "echo bye"]
        )
        assert len(decision.arguments.content_lines) == 3

    async def test_the_only_element_is_kept(self, lines_tools):
        # Emptying the array would turn a repair into a validation failure.
        decision = await self.call(lines_tools, ["timeout_s: 60"])
        assert decision.arguments.content_lines == ["timeout_s: 60"]

    async def test_a_real_argument_is_untouched(self, lines_tools):
        decision = await self.call(lines_tools, ["echo hi"], timeout_s=120)
        assert decision.arguments.timeout_s == 120
        assert decision.arguments.content_lines == ["echo hi"]
