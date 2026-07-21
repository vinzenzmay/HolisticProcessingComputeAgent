"""Tests for hpca.agent.modes and the mode wiring in the graph (§3.5)."""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from hpca.agent.builtin_tools import _bash_is_destructive
from hpca.agent.graph import MAX_CONTINUE_NUDGES, build_graph, run_turn
from hpca.agent.modes import (
    EXECUTION_TOOLS,
    MODES,
    PresentPlanParams,
    add_plan_tool,
    continue_nudge_for,
    destructive_approval_required,
    kickoff_message,
    looks_like_deferred_action,
    mode_prompt_suffix,
    next_mode,
    parse_checklist,
    present_plan_reply,
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
            # Mirror production wiring so mode tests exercise the real gate.
            is_destructive_call=_bash_is_destructive,
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
            # plan no longer blanket-gates execution tools: it leans on the
            # destructive gate instead, so benign look-around runs unattended.
            assert not requires_execution_approval("plan", name)
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
        assert "run_bash" in offered.names()  # look-around stays available
        # planning ends through present_plan; update_plan is execution-phase
        assert "present_plan" in offered.names()
        assert "update_plan" not in offered.names()

    def test_other_modes_offer_update_plan_not_present_plan(self, tools):
        for mode in ("manual", "auto", "full-auto", None):
            offered = tools_for_mode(tools, mode)
            assert "present_plan" not in offered.names()
            assert "update_plan" in offered.names()
            assert "run_script" in offered.names()

    def test_a_registry_without_plan_tools_is_returned_unchanged(self):
        bare = ToolRegistry()
        bare.register(
            Tool(
                name="run_bash",
                description="Run a bash script",
                params=BashParams,
                handler=bash_handler,
            )
        )
        assert tools_for_mode(bare, "auto") is bare
        assert tools_for_mode(bare, "plan") is bare

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

    def test_present_plan_needs_a_step_or_a_question(self):
        with pytest.raises(ValueError):
            PresentPlanParams()
        # either alone is enough
        assert PresentPlanParams(steps=[{"text": "s", "done": False}])
        assert PresentPlanParams(open_questions=["q"])

    def test_present_plan_reply_carries_summary_and_questions(self):
        args = PresentPlanParams(
            steps=[{"text": "s", "done": False}],
            summary="my plan",
            open_questions=["which build?"],
        )
        text = present_plan_reply(args)
        assert "my plan" in text
        assert "which build?" in text

    def test_present_plan_reply_has_a_default_when_no_summary(self):
        args = PresentPlanParams(steps=[{"text": "s", "done": False}])
        assert present_plan_reply(args).strip()

    def test_deferred_action_is_detected(self):
        for text in (
            "The README does not mention it. Let me dig deeper into the source.",
            "I'll grep the source for the subcommand.",
            "Next, I will read the CLI entrypoint.",
            "First I check the docs.\n\nNow let me look at the code.",
            "- Let me examine the alignment module.",
        ):
            assert looks_like_deferred_action(text), text

    def test_real_answers_and_questions_are_not_deferred_actions(self):
        for text in (
            "The command is `svirlpool cut`.",
            "Done — I cut the reads and wrote them to out.bam.",
            "Which reference build should I use?",
            "I found two candidates. Let me know which one you want.",
            "The tool will let me parse the interval, so it should work.",
            "",
        ):
            assert not looks_like_deferred_action(text), text

    def test_continue_nudge_is_mode_aware(self):
        # plan mode nudges any chat reply; other modes only a deferred action.
        assert continue_nudge_for("plan", "here is my answer") is not None
        assert continue_nudge_for("auto", "here is my answer") is None
        assert continue_nudge_for("auto", "Let me dig into the source.") is not None
        assert continue_nudge_for(None, "Let me dig into the source.") is not None


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
    async def test_planning_offers_present_plan_not_execution_or_update(self, tools):
        # present_plan is terminal, so one decision is enough to inspect the
        # offered tool listing without the nudge loop needing more outputs.
        llm = FakeLLM(
            [tool_json("present_plan", steps=[{"text": "s", "done": False}])]
        )
        graph = make_graph(llm, tools, "plan")
        await run_turn(graph, session_id="s1", user_text="plan something")
        # the tool listing (not the guidance prose) is what the model can call
        text = system_text(llm, 0)
        assert '"tool": "run_script"' not in text
        assert '"tool": "present_plan"' in text
        assert '"tool": "update_plan"' not in text  # execution-phase only

    async def test_present_plan_stores_the_checklist_and_ends_the_turn(self, tools):
        steps = [
            {"text": "find inputs", "done": False},
            {"text": "write script", "done": False},
        ]
        llm = FakeLLM(
            [tool_json("present_plan", steps=steps, summary="here is the plan")]
        )
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan the run")
        assert result.plan == steps
        assert result.reply == "here is the plan"
        assert result.interrupt is None
        assert len(llm.calls) == 1  # present_plan ends the turn in one decision

    async def test_present_plan_with_only_questions_asks_without_a_plan(self, tools):
        llm = FakeLLM(
            [tool_json("present_plan", open_questions=["Which reference build?"])]
        )
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert result.plan is None  # no checklist yet: nothing to approve
        assert "Which reference build?" in result.reply

    async def test_plan_survives_into_the_next_turn(self, tools):
        steps = [{"text": "only step", "done": False}]
        llm = FakeLLM(
            [
                tool_json("present_plan", steps=steps, summary="planned"),
                tool_json("present_plan", steps=steps, summary="still here"),
            ]
        )
        graph = make_graph(llm, tools, "plan")
        await run_turn(graph, session_id="s1", user_text="plan it")
        result = await run_turn(graph, session_id="s1", user_text="thoughts?")
        assert result.plan == steps
        # the next turn's first decision sees the stored checklist re-injected
        assert "[ ] only step" in system_text(llm, 1)

    async def test_bare_reply_is_nudged_then_the_action_is_taken(self, tools):
        # A narration-only reply must not end the plan turn: the model is
        # nudged and, on retry, takes the action it only described.
        seen = []

        async def peek_handler(args, ctx):
            seen.append(args.registry_key)
            return "peeked"

        tools.register(
            Tool(
                name="peek",
                description="Look at something (no side effects)",
                params=KeyParams,
                handler=peek_handler,
            )
        )
        llm = FakeLLM(
            [
                respond_json("Let me check the sniffles config first."),
                tool_json("peek", registry_key="sniffles"),
                tool_json("present_plan", steps=[{"text": "run it", "done": False}]),
            ]
        )
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert seen == ["sniffles"]  # the described action actually ran
        assert result.plan == [{"text": "run it", "done": False}]
        # the fumbled narration is fed back as a nudge, never persisted
        assert any(
            "[continue]" in m["content"] for m in llm.calls[1]["messages"]
        )
        assert not any(
            "Let me check the sniffles config" in m["content"]
            for m in result.messages
        )

    async def test_nudge_gives_up_after_the_budget(self, tools):
        # A model that only ever narrates cannot hang the turn: after the
        # nudge budget it ends with whatever it last said.
        llm = FakeLLM(
            [
                respond_json("Let me look at a."),
                respond_json("Let me look at b."),
                respond_json("Let me look at c."),
            ]
        )
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert len(llm.calls) == MAX_CONTINUE_NUDGES + 1
        assert result.reply == "Let me look at c."

    async def test_a_real_answer_ends_the_turn_outside_plan_mode(self, tools):
        llm = FakeLLM([respond_json("The command is `svirlpool cut`.")])
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.reply == "The command is `svirlpool cut`."
        assert len(llm.calls) == 1  # a genuine answer is not a deferred action

    async def test_deferred_action_is_nudged_then_taken_in_auto_mode(self, tools):
        # The exact failure this fixes: the model narrates "Let me dig deeper"
        # and stops. In every mode it is now fed back and, on retry, acts.
        seen = []

        async def peek_handler(args, ctx):
            seen.append(args.registry_key)
            return "peeked"

        tools.register(
            Tool(
                name="peek",
                description="Look at something (no side effects)",
                params=KeyParams,
                handler=peek_handler,
            )
        )
        llm = FakeLLM(
            [
                respond_json("Let me dig deeper into the source code to find it."),
                tool_json("peek", registry_key="svirlpool"),
                respond_json("Found it: `svirlpool cut`."),
            ]
        )
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="find the cut cmd")
        assert seen == ["svirlpool"]  # the narrated action actually ran
        assert result.reply == "Found it: `svirlpool cut`."
        # the fumbled narration is fed back as a nudge, never persisted
        assert any("[continue]" in m["content"] for m in llm.calls[1]["messages"])
        assert not any(
            "Let me dig deeper" in m["content"] for m in result.messages
        )

    async def test_benign_look_around_runs_unattended_in_plan_mode(self, tools):
        # A read-only look-around no longer gates in plan mode — it just runs,
        # and the turn ends normally when the model presents the plan.
        llm = FakeLLM(
            [
                tool_json("run_bash", content_lines=["ls"]),
                tool_json("present_plan", steps=[{"text": "s", "done": False}]),
            ]
        )
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert result.interrupt is None
        assert any("run_bash: ran" in m["content"] for m in result.messages)

    async def test_destructive_look_around_gates_in_plan_mode(self, tools):
        # A destructive one-shot still pauses — via the destructive gate now,
        # not the execution gate.
        llm = FakeLLM([tool_json("run_bash", content_lines=["rm -rf /data/x"])])
        graph = make_graph(llm, tools, "plan")
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert result.interrupt is not None
        assert result.interrupt["kind"] == "destructive"

    def _peek_tool(self, tools):
        async def peek(args, ctx):
            return "ok"

        tools.register(
            Tool(
                name="peek",
                description="Look at something (no side effects)",
                params=KeyParams,
                handler=peek,
            )
        )

    async def test_budget_exhaustion_forces_the_plan_handoff(self, tools):
        # Out of look-around budget mid-planning: the turn ends by presenting
        # the plan, not by trailing off in chat (the "30 steps" failure).
        self._peek_tool(tools)
        steps = [{"text": "run sawfish", "done": False}]
        llm = FakeLLM(
            [tool_json("peek", registry_key="x")] * 2
            + [tool_json("present_plan", steps=steps, summary="best plan for now")]
        )
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            mode_fn=lambda: "plan",
            max_tool_rounds=2,
        )
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert result.plan == steps
        assert result.reply == "best plan for now"
        # the budget call offered present_plan (not an empty registry)
        combined = " ".join(m["content"] for m in llm.calls[-1]["messages"])
        assert '"tool": "present_plan"' in combined
        assert "look-around steps" in combined

    async def test_budget_exhaustion_still_ends_if_the_model_will_not_present(
        self, tools
    ):
        self._peek_tool(tools)
        llm = FakeLLM(
            [tool_json("peek", registry_key="x")] * 2
            + [respond_json("here is what I found")]
        )
        graph = build_graph(
            llm=llm,
            tools=tools,
            checkpointer=InMemorySaver(),
            mode_fn=lambda: "plan",
            max_tool_rounds=2,
        )
        result = await run_turn(graph, session_id="s1", user_text="plan it")
        assert result.reply == "here is what I found"
        assert result.plan is None


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

    def test_session_backend_round_trip(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        store = SessionStore(conn)
        blob = '{"model": "qwen", "base_url": "http://x/v1"}'
        session = store.create(profile="default", backend=blob)
        assert store.get(session.session_id).backend == blob
        assert store.create(profile="default").backend == ""  # default
        store.set_backend(session.session_id, "")
        assert store.get(session.session_id).backend == ""
        conn.close()

    def test_default_mode_defaults_to_manual(self):
        assert Settings().agent.default_mode == "manual"

    def test_default_mode_configurable(self):
        settings = Settings.model_validate({"agent": {"default_mode": "auto"}})
        assert settings.agent.default_mode == "auto"
