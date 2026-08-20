"""Tests for hpca.agent: tool registry and the validation/retry middleware (§4.3)."""

import json
from typing import get_origin

import pytest
from pydantic import BaseModel, Field

from hpca.agent.middleware import (
    MAX_DECISION_TOKENS,
    TRUNCATION_FEEDBACK,
    DecisionError,
    DirectResponse,
    ToolCall,
    decide,
    decision_cap,
    decision_schema,
    format_instruction,
    inline_refs,
    uses_native_tools,
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
        self.calls.append(
            {
                "messages": list(messages),
                "json_schema": json_schema,
                "max_tokens": kwargs.get("max_tokens"),
            }
        )
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


class NativeLLM:
    """A backend speaking the native tool-calling protocol.

    Outputs are either a string (plain content, i.e. an answer) or a list of
    tool_calls dicts, the two shapes an OpenAI-style response comes in.
    """

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls: list[dict] = []
        # What the backend reports it spent. Only the silent-truncation tests
        # care: reaching max_tokens is the sole trace a parser leaves when it
        # drops the argument a generation died inside.
        self.usage: dict = {}

    async def chat(self, messages, *, json_schema=None, tools=None, **kwargs):
        self.calls.append(
            {
                "messages": list(messages),
                "json_schema": json_schema,
                "tools": tools,
                "max_tokens": kwargs.get("max_tokens"),
            }
        )
        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, str):
            return ChatResponse(content=output, usage=dict(self.usage))
        return ChatResponse(content="", tool_calls=output, usage=dict(self.usage))

    def uses_native_tools(self):
        return True

    async def supports_constrained_decoding(self):
        raise AssertionError("the native protocol must not probe for a grammar")


def native_call(tool="echo", call_id="call_1", **arguments):
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": json.dumps(arguments)},
        }
    ]


class TestDecideNative:
    async def test_a_tool_call_carries_its_id(self, tools):
        llm = NativeLLM([native_call("echo", text="hi")])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, ToolCall)
        assert decision.tool.name == "echo"
        assert decision.arguments.text == "hi"
        assert decision.call_id == "call_1"

    async def test_content_without_calls_is_an_answer(self, tools):
        decision = await decide(NativeLLM(["four BAMs, all indexed"]), USER, tools)
        assert isinstance(decision, DirectResponse)
        assert decision.text == "four BAMs, all indexed"

    async def test_tools_ride_the_array_not_the_grammar(self, tools):
        llm = NativeLLM(["ok"])
        await decide(llm, USER, tools)
        call = llm.calls[0]
        assert call["json_schema"] is None
        assert [t["function"]["name"] for t in call["tools"]] == ["echo", "count"]
        assert call["tools"][0]["function"]["description"].startswith("Echo text")
        assert "Example arguments" in call["tools"][0]["function"]["description"]

    async def test_the_instruction_stops_listing_tools(self, tools):
        # The tools array is the listing now. What survives in prose is only
        # what it cannot say — that a call is a real action.
        llm = NativeLLM(["ok"])
        await decide(llm, [{"role": "system", "content": "Base prompt."}] + USER, tools)
        system = llm.calls[0]["messages"][0]
        assert system["role"] == "system"
        assert "Base prompt." in system["content"]
        assert '{"action"' not in system["content"]

    async def test_invalid_arguments_retry_answers_on_the_tool_role(self, tools):
        # A chat template handed a call with no matching result renders a
        # broken conversation, so the complaint has to answer the call.
        llm = NativeLLM([native_call("count", n=1000), native_call("count", n=5)])
        decision = await decide(llm, USER, tools)
        assert isinstance(decision, ToolCall)
        assert decision.arguments.n == 5
        retry = llm.calls[1]["messages"]
        assert retry[-2]["role"] == "assistant" and retry[-2]["tool_calls"]
        assert retry[-1]["role"] == "tool"
        assert retry[-1]["tool_call_id"] == retry[-2]["tool_calls"][0]["id"]
        assert "less than or equal to 100" in retry[-1]["content"]

    async def test_unknown_tool_retries_with_the_available_names(self, tools):
        llm = NativeLLM([native_call("delete_everything"), native_call("echo", text="x")])
        assert isinstance(await decide(llm, USER, tools), ToolCall)
        feedback = llm.calls[1]["messages"][-1]["content"]
        assert "delete_everything" in feedback and "echo" in feedback

    async def test_unparseable_arguments_retry(self, tools):
        broken = [{"id": "c1", "type": "function",
                   "function": {"name": "echo", "arguments": "{not json"}}]
        llm = NativeLLM([broken, native_call("echo", text="ok")])
        assert isinstance(await decide(llm, USER, tools), ToolCall)
        assert "not valid JSON" in llm.calls[1]["messages"][-1]["content"]

    async def test_an_empty_response_is_refused_not_answered(self, tools):
        # "" is not an answer; without this it would end the turn silently.
        llm = NativeLLM(["   ", native_call("echo", text="x")])
        assert isinstance(await decide(llm, USER, tools), ToolCall)
        assert "neither a tool call nor an answer" in (
            llm.calls[1]["messages"][-1]["content"]
        )

    async def test_extra_calls_in_one_response_are_reported_not_dropped(self, tools):
        # The graph runs one call at a time (approval, round accounting), so
        # the rest go — but silently dropping them is how a half-done task
        # looks finished.
        both = native_call("echo", text="first") + native_call(
            "echo", call_id="call_2", text="second"
        )
        decision = await decide(NativeLLM([both]), USER, tools)
        assert decision.arguments.text == "first"
        assert any("dropped" in repair for repair in decision.repairs)


# --------------------------------------------------------- integration tests

from hpca.config import LLMSettings  # noqa: E402
from hpca.agent.prompts import (  # noqa: E402
    RESPOND_VS_TOOL_GUIDANCE,
    RESPOND_VS_TOOL_GUIDANCE_NATIVE,
    orchestrator_system_prompt,
)
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
        # the recovery it teaches is skeleton-then-fill, not resending
        assert "skeleton" in feedback["content"]
        assert "edit_file" in feedback["content"]

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


class TestTruncationSalvage:
    """A cut-off create_file/edit_file is salvaged, not thrown away: the
    complete prefix runs, the repair note says the write is partial."""

    @pytest.fixture
    def file_tools(self):
        from hpca.agent.file_tools import add_file_tools

        return add_file_tools(ToolRegistry())

    def truncated_create(self, n_lines, tail='"cut off mid-str'):
        lines = ",".join(f'"line {i}"' for i in range(n_lines))
        return (
            '{"action":"tool_call","tool":"create_file","arguments":'
            '{"path":"/work/specs.md","content_lines":['
            + lines + "," + tail
        )

    async def test_create_file_prefix_is_salvaged(self, file_tools):
        llm = FakeLLM([TruncatedOutput("t", partial=self.truncated_create(40))])
        decision = await decide(llm, USER, file_tools)
        assert isinstance(decision, ToolCall)
        assert decision.tool.name == "create_file"
        assert decision.arguments.content_lines[:40] == [f"line {i}" for i in range(40)]
        assert decision.arguments.content_lines[-1].startswith("TBD")
        assert decision.repairs and "INCOMPLETE" in decision.repairs[0]
        assert len(llm.calls) == 1  # no retry was spent

    async def test_edit_file_new_lines_prefix_is_salvaged(self, file_tools):
        new = ",".join(f'"new {i}"' for i in range(20))
        partial = (
            '{"action":"tool_call","tool":"edit_file","arguments":'
            '{"path":"specs.md","old_lines":["TBD: Build"],'
            '"new_lines":[' + new + ',"cut'
        )
        llm = FakeLLM([TruncatedOutput("t", partial=partial)])
        decision = await decide(llm, USER, file_tools)
        assert isinstance(decision, ToolCall)
        assert decision.tool.name == "edit_file"
        assert list(decision.arguments.old_lines) == ["TBD: Build"]
        assert len(decision.arguments.new_lines) == 21
        assert decision.arguments.new_lines[-1].startswith("TBD")
        assert decision.repairs and "TBD marker" in decision.repairs[0]

    async def test_cut_inside_old_lines_is_not_salvaged(self, file_tools):
        partial = (
            '{"action":"tool_call","tool":"edit_file","arguments":'
            '{"path":"specs.md","old_lines":["TBD: Bu'
        )
        llm = FakeLLM([TruncatedOutput("t", partial=partial), respond_json("ok")])
        # cut inside old_lines: no way to know the target lines, so no
        # salvage — the normal retry runs and the second decision answers
        decision = await decide(llm, USER, file_tools)
        assert isinstance(decision, DirectResponse)
        assert len(llm.calls) == 2  # retried instead of salvaging

    async def test_too_few_lines_take_the_retry_not_the_salvage(self, file_tools):
        llm = FakeLLM(
            [
                TruncatedOutput("t", partial=self.truncated_create(3)),
                TruncatedOutput("t", partial=self.truncated_create(3)),
            ]
        )
        with pytest.raises(TruncatedOutput):
            await decide(llm, USER, file_tools)
        assert len(llm.calls) == 2  # one retry, then surfaced

    async def test_garbage_partial_is_not_salvaged(self, file_tools):
        llm = FakeLLM(
            [
                TruncatedOutput("t", partial='{"acti'),
                TruncatedOutput("t", partial=""),
            ]
        )
        with pytest.raises(TruncatedOutput):
            await decide(llm, USER, file_tools)


class TestUsesNativeTools:
    """The single source both `decide` and the prompt builders read.

    They must never disagree: a prompt that spells out the envelope's
    {"action": "respond"} branch while the request goes out on the native
    channel names a format the model cannot emit, which is what took the core
    tier to 12/17 in specs-edit-eval.md §7.
    """

    def test_a_native_client_is_native(self):
        assert uses_native_tools(NativeLLM([])) is True

    def test_an_envelope_client_is_not(self):
        assert uses_native_tools(FakeLLM([])) is False

    def test_a_client_with_no_opinion_is_envelope(self):
        # A fake in a TUI test has no protocol; "envelope" is the answer that
        # keeps it working rather than an AttributeError at prompt-build time.
        assert uses_native_tools(object()) is False

    def test_the_prompt_follows_the_same_answer(self):
        native = orchestrator_system_prompt(native_tools=uses_native_tools(NativeLLM([])))
        envelope = orchestrator_system_prompt(native_tools=uses_native_tools(FakeLLM([])))
        assert RESPOND_VS_TOOL_GUIDANCE_NATIVE in native
        assert RESPOND_VS_TOOL_GUIDANCE not in native
        assert RESPOND_VS_TOOL_GUIDANCE in envelope
        assert RESPOND_VS_TOOL_GUIDANCE_NATIVE not in envelope


class TestNativeSilentTruncation:
    """A call that died at the cap, reported as a well-formed call.

    Measured against vLLM's qwen3_coder parser (2026-08-17): a create_file
    cut off part-way through content_lines comes back as
    {"dir_key": ..., "name": ...} with finish_reason "tool_calls" and no
    content. Only completion_tokens == max_tokens says what happened.
    """

    @pytest.fixture
    def file_tools(self):
        from hpca.agent.file_tools import add_file_tools

        return add_file_tools(ToolRegistry())

    def cut_off_call(self):
        return [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "create_file",
                    # content_lines: dropped by the parser, mid-generation
                    "arguments": json.dumps({"dir_key": "work", "name": "notes.md"}),
                },
            }
        ]

    async def test_the_cap_turns_a_validation_error_into_truncation_feedback(
        self, file_tools
    ):
        llm = NativeLLM([self.cut_off_call(), "done"])
        llm.usage = {"completion_tokens": 4096}
        await decide(llm, USER, file_tools, max_tokens=4096)
        # The retry must not say "content_lines: Field required" — that invites
        # the same too-long write again.
        retry = llm.calls[1]["messages"][-1]
        assert retry["role"] == "user"
        assert retry["content"] == TRUNCATION_FEEDBACK

    async def test_it_is_surfaced_once_the_truncation_budget_is_gone(
        self, file_tools
    ):
        llm = NativeLLM([self.cut_off_call(), self.cut_off_call()])
        llm.usage = {"completion_tokens": 4096}
        with pytest.raises(TruncatedOutput):
            await decide(llm, USER, file_tools, max_tokens=4096)

    async def test_a_short_generation_is_still_an_ordinary_shape_error(
        self, file_tools
    ):
        # Same missing field, but nowhere near the cap: the model really did
        # forget it, so the verbatim validation error is the right feedback.
        llm = NativeLLM([self.cut_off_call(), "done"])
        llm.usage = {"completion_tokens": 40}
        await decide(llm, USER, file_tools, max_tokens=4096)
        retry = llm.calls[1]["messages"][-1]
        assert retry["role"] == "tool"
        assert "content_lines" in retry["content"]

    async def test_a_valid_call_at_the_cap_is_still_kept(self, file_tools):
        # _hit_token_cap only reinterprets an already-failed validation; a
        # complete call that ends at the cap is real work.
        llm = NativeLLM(
            [
                native_call("create_file", path="n.md", content_lines=["x"])
            ]
        )
        llm.usage = {"completion_tokens": 4096}
        decision = await decide(llm, USER, file_tools, max_tokens=4096)
        assert isinstance(decision, ToolCall)


class TestDecisionCap:
    """The cap moves with the thinking level, because thinking is spent from it.

    The multipliers are a judgement call, not a fitted number (see
    DECISION_TOKEN_SCALE) — what these pin down is that the wiring works and
    that an explicit cap still wins.
    """

    def test_off_and_low_keep_the_base(self):
        assert decision_cap("off") == MAX_DECISION_TOKENS
        assert decision_cap("low") == MAX_DECISION_TOKENS

    def test_medium_and_xhigh_scale_up(self):
        assert decision_cap("medium") == 6144
        assert decision_cap("xhigh") == 8192

    def test_no_level_is_the_base(self):
        # A bare caller — every test predating the dial, and the aux callers.
        assert decision_cap(None) == MAX_DECISION_TOKENS
        assert decision_cap("") == MAX_DECISION_TOKENS

    def test_an_unknown_level_does_not_inflate_the_cap(self):
        # normalize_effort maps junk to "off"; the cap must not be the one
        # place a hand-edited settings file can buy an 8k generation.
        assert decision_cap("enormous") == MAX_DECISION_TOKENS

    async def test_decide_sends_the_scaled_cap(self, tools):
        llm = NativeLLM([respond_json("hi")])
        await decide(llm, USER, tools, effort="xhigh")
        assert llm.calls[0]["max_tokens"] == 8192

    async def test_an_explicit_cap_still_wins(self, tools):
        llm = NativeLLM([respond_json("hi")])
        await decide(llm, USER, tools, effort="xhigh", max_tokens=512)
        assert llm.calls[0]["max_tokens"] == 512
