"""Tests for hpca.agent.modes and the mode wiring in the graph (§3.5)."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from hpca.agent.graph import build_graph, run_turn
from hpca.agent.modes import (
    EXECUTION_TOOLS,
    MODES,
    add_plan_tool,
    destructive_approval_required,
    kickoff_message,
    mode_prompt_suffix,
    next_mode,
    parse_checklist,
    render_checklist,
    requires_execution_approval,
    skipped_message,
    tools_for_mode,
)
from hpca.agent.tools import Tool, ToolRegistry
from hpca.config import Settings
from hpca.db import connect, init_db
from hpca.llm import ChatResponse
from hpca.sessions import SessionStore


class BashParams(BaseModel):
    content_lines: list[str] = Field(description="Lines")


class KeyParams(BaseModel):
    registry_key: str = Field(description="Key")


async def bash_handler(args, ctx):
    return "ran"


async def key_handler(args, ctx):
    return "ran script"


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


@pytest.fixture
def tools():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="run_bash",
            description="Run a bash script",
            params=BashParams,
            handler=bash_handler,
        )
    )
    registry.register(
        Tool(
            name="run_script",
            description="Run a registered script",
            params=KeyParams,
            handler=key_handler,
        )
    )
    add_plan_tool(registry)
    return registry


def make_graph(llm, tools, mode):
    return build_graph(
        llm=llm, tools=tools, checkpointer=InMemorySaver(), mode_fn=lambda: mode
    )


def system_text(llm, call_index):
    return llm.calls[call_index]["messages"][0]["content"]


class TestHelpers:
    def test_next_mode_cycles_through_all(self):
        seen = []
        mode = MODES[0]
        for _ in MODES:
            seen.append(mode)
            mode = next_mode(mode)
        assert seen == list(MODES)
        assert mode == MODES[0]

    def test_next_mode_recovers_from_unknown(self):
        assert next_mode("nonsense") in MODES

    def test_execution_approval_by_mode(self):
        for name in EXECUTION_TOOLS:
            assert requires_execution_approval("manual", name)
            assert requires_execution_approval("plan", name)
            assert not requires_execution_approval("auto", name)
            assert not requires_execution_approval("full-auto", name)
            assert not requires_execution_approval(None, name)
        assert not requires_execution_approval("manual", "read_file")

    def test_destructive_gate_waived_only_in_full_auto(self):
        assert not destructive_approval_required("full-auto")
        for mode in ("manual", "auto", "plan", None):
            assert destructive_approval_required(mode)

    def test_plan_mode_withdraws_execution_tools(self, tools):
        offered = tools_for_mode(tools, "plan")
        assert "run_script" not in offered.names()
        assert "run_bash" in offered.names()  # look-around stays, but gated
        assert "update_plan" in offered.names()

    def test_other_modes_keep_the_registry(self, tools):
        assert tools_for_mode(tools, "manual") is tools
        assert tools_for_mode(tools, "auto") is tools
        assert tools_for_mode(tools, "full-auto") is tools
        assert tools_for_mode(tools, None) is tools

    def test_checklist_round_trip(self):
        steps = [
            {"text": "find the input BAM", "done": True},
            {"text": "write the script", "done": False},
        ]
        assert parse_checklist(render_checklist(steps)) == steps

    def test_parse_checklist_is_lenient(self):
        text = "- [x] first\n* [ ] second\n\nthird\n- fourth"
        steps = parse_checklist(text)
        assert [s["text"] for s in steps] == ["first", "second", "third", "fourth"]
        assert [s["done"] for s in steps] == [True, False, False, False]

    def test_suffix_carries_mode_guidance(self):
        assert "Manual mode" in mode_prompt_suffix("manual", None)
        assert "Auto mode" in mode_prompt_suffix("auto", None)
        assert "Full-auto mode" in mode_prompt_suffix("full-auto", None)
        assert "Plan mode" in mode_prompt_suffix("plan", None)
        assert mode_prompt_suffix(None, None) == ""

    def test_suffix_renders_the_plan_everywhere(self):
        plan = [{"text": "step one", "done": False}]
        for mode in MODES:
            assert "[ ] step one" in mode_prompt_suffix(mode, plan)

    def test_execution_instruction_only_outside_plan_mode(self):
        plan = [{"text": "step one", "done": False}]
        assert "unfinished steps in order" in mode_prompt_suffix("auto", plan)
        assert "unfinished steps in order" in mode_prompt_suffix("manual", plan)
        assert "unfinished steps in order" not in mode_prompt_suffix("plan", plan)

    def test_skip_message_forbids_retry(self):
        text = skipped_message("run_bash")
        assert "SKIPPED" in text
        assert "not retry" in text.lower() or "do not retry" in text.lower()

    def test_kickoff_names_the_mode(self):
        assert "auto mode" in kickoff_message("auto")


class TestManualMode:
    async def test_execution_tool_gates_with_script_preview(self, tools):
        llm = FakeLLM([tool_json("run_bash", content_lines=["echo hi"])])
        graph = make_graph(llm, tools, "manual")
        result = await run_turn(graph, session_id="s1", user_text="look around")
        assert result.interrupt is not None
        assert result.interrupt["kind"] == "execution"
        assert "echo hi" in result.interrupt["script"]

    async def test_approve_runs_the_tool(self, tools):
        llm = FakeLLM(
            [tool_json("run_bash", content_lines=["echo hi"]), respond_json("after")]
        )
        graph = make_graph(llm, tools, "manual")
        await run_turn(graph, session_id="s1", user_text="look around")
        result = await run_turn(
            graph, session_id="s1", resume=Command(resume={"approved": True})
        )
        assert result.reply == "after"
        assert any("run_bash: ran" in m["content"] for m in result.messages)

    async def test_skip_feeds_back_the_skip_message(self, tools):
        llm = FakeLLM(
            [tool_json("run_bash", content_lines=["echo hi"]), respond_json("ok")]
        )
        graph = make_graph(llm, tools, "manual")
        await run_turn(graph, session_id="s1", user_text="look around")
        result = await run_turn(
            graph, session_id="s1", resume=Command(resume={"approved": False})
        )
        skipped = [m for m in result.messages if "SKIPPED" in m["content"]]
        assert skipped, "the model must be told the user declined"
        assert not any("run_bash: ran" in m["content"] for m in result.messages)

    async def test_manual_guidance_in_system_prompt(self, tools):
        llm = FakeLLM([respond_json()])
        graph = make_graph(llm, tools, "manual")
        await run_turn(graph, session_id="s1", user_text="hi")
        assert "Manual mode is on" in system_text(llm, 0)


class TestAutoMode:
    async def test_execution_tool_runs_without_gate(self, tools):
        llm = FakeLLM(
            [tool_json("run_bash", content_lines=["echo hi"]), respond_json("did it")]
        )
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.interrupt is None
        assert result.reply == "did it"

    async def test_destructive_tool_still_gates(self, tools):
        tools.register(
            Tool(
                name="delete",
                description="Delete something",
                params=KeyParams,
                handler=key_handler,
                destructive=True,
            )
        )
        llm = FakeLLM([tool_json("delete", registry_key="x")])
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="delete x")
        assert result.interrupt is not None
        assert result.interrupt["kind"] == "destructive"


class TestFullAutoMode:
    def destructive_tools(self, tools):
        tools.register(
            Tool(
                name="delete",
                description="Delete something",
                params=KeyParams,
                handler=key_handler,
                destructive=True,
            )
        )
        return tools

    async def test_destructive_tool_runs_without_gate(self, tools):
        llm = FakeLLM(
            [tool_json("delete", registry_key="x"), respond_json("gone")]
        )
        graph = make_graph(llm, self.destructive_tools(tools), "full-auto")
        result = await run_turn(graph, session_id="s1", user_text="delete x")
        assert result.interrupt is None
        assert result.reply == "gone"
        assert any("delete: ran script" in m["content"] for m in result.messages)

    async def test_execution_tool_runs_without_gate(self, tools):
        llm = FakeLLM(
            [tool_json("run_bash", content_lines=["ls"]), respond_json("done")]
        )
        graph = make_graph(llm, tools, "full-auto")
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.interrupt is None
        assert result.reply == "done"

    async def test_full_auto_guidance_in_system_prompt(self, tools):
        llm = FakeLLM([respond_json()])
        graph = make_graph(llm, tools, "full-auto")
        await run_turn(graph, session_id="s1", user_text="hi")
        assert "Full-auto mode is on" in system_text(llm, 0)


class TestPlanMode:
    async def test_blocked_tools_are_not_offered(self, tools):
        llm = FakeLLM([respond_json()])
        graph = make_graph(llm, tools, "plan")
        await run_turn(graph, session_id="s1", user_text="plan something")
        # the tool listing (not the guidance prose) is what the model can call
        assert '"tool": "run_script"' not in system_text(llm, 0)
        assert '"tool": "update_plan"' in system_text(llm, 0)

    async def test_update_plan_lands_in_state_and_prompt(self, tools):
        steps = [
            {"text": "find inputs", "done": False},
            {"text": "write script", "done": False},
        ]
        llm = FakeLLM(
            [tool_json("update_plan", steps=steps), respond_json("here is the plan")]
        )
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan the run")
        assert result.plan == steps
        # the next decision after update_plan sees the checklist re-injected
        assert "Current plan checklist" in system_text(llm, 1)
        assert "[ ] find inputs" in system_text(llm, 1)

    async def test_plan_survives_into_the_next_turn(self, tools):
        steps = [{"text": "only step", "done": False}]
        llm = FakeLLM(
            [
                tool_json("update_plan", steps=steps),
                respond_json("planned"),
                respond_json("still here"),
            ]
        )
        graph = make_graph(llm, tools, "plan")
        await run_turn(graph, session_id="s1", user_text="plan it")
        result = await run_turn(graph, session_id="s1", user_text="thoughts?")
        assert result.plan == steps
        assert "[ ] only step" in system_text(llm, 2)

    async def test_look_around_gates_in_plan_mode(self, tools):
        llm = FakeLLM([tool_json("run_bash", content_lines=["ls"])])
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert result.interrupt is not None
        assert result.interrupt["kind"] == "execution"


class TestPersistence:
    def test_session_mode_round_trip(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        store = SessionStore(conn)
        session = store.create(profile="default", mode="auto")
        assert store.get(session.session_id).mode == "auto"
        store.set_mode(session.session_id, "plan")
        assert store.get(session.session_id).mode == "plan"
        conn.close()

    def test_default_mode_defaults_to_manual(self):
        assert Settings().agent.default_mode == "manual"

    def test_default_mode_configurable(self):
        settings = Settings.model_validate({"agent": {"default_mode": "auto"}})
        assert settings.agent.default_mode == "auto"
