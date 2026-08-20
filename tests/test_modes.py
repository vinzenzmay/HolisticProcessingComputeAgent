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
    add_plan_tool,
    continue_nudge_for,
    denied_message,
    destructive_approval_required,
    looks_like_deferred_action,
    mode_prompt_suffix,
    next_mode,
    render_checklist,
    requires_execution_approval,
    SCRIPT_PREVIEW_CHARS,
    script_preview,
    skipped_message,
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
            name="start_background_script",
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
            assert not requires_execution_approval("auto", name)
            assert not requires_execution_approval("full-auto", name)
            assert not requires_execution_approval(None, name)
        assert not requires_execution_approval("manual", "read_file")

    def test_destructive_gate_waived_only_in_full_auto(self):
        assert not destructive_approval_required("full-auto")
        for mode in ("manual", "auto", None):
            assert destructive_approval_required(mode)

    def test_checklist_renders_done_and_open_steps(self):
        steps = [
            {"text": "find the input BAM", "done": True},
            {"text": "write the script", "done": False},
        ]
        assert render_checklist(steps) == (
            "[x] find the input BAM\n[ ] write the script"
        )

    def test_suffix_carries_mode_guidance(self):
        assert "Manual mode" in mode_prompt_suffix("manual", None)
        assert "Auto mode" in mode_prompt_suffix("auto", None)
        assert "Full-auto mode" in mode_prompt_suffix("full-auto", None)
        assert mode_prompt_suffix(None, None) == ""

    def test_suffix_renders_the_plan_everywhere(self):
        plan = [{"text": "step one", "done": False}]
        for mode in MODES:
            assert "[ ] step one" in mode_prompt_suffix(mode, plan)

    def test_execution_instruction_rides_with_the_checklist(self):
        plan = [{"text": "step one", "done": False}]
        for mode in MODES:
            assert "unfinished steps in order" in mode_prompt_suffix(mode, plan)

    def test_skip_message_forbids_retry(self):
        text = skipped_message("run_bash")
        assert "SKIPPED" in text
        assert "not retry" in text.lower() or "do not retry" in text.lower()

    def test_a_reason_turns_the_refusal_into_a_correction(self):
        # A bare refusal is final; one the user explained is the opposite —
        # the reason is what the fixed version has to act on.
        text = skipped_message("run_bash", "wrong partition, use gpu")
        assert "SKIPPED" in text
        assert "wrong partition, use gpu" in text
        assert "do not retry" not in text.lower()

    def test_denial_message_carries_a_reason_too(self):
        plain = denied_message("delete")
        assert "DENIED" in plain
        assert "not executed" in plain
        explained = denied_message("delete", "that path holds the raw reads")
        assert "DENIED" in explained
        assert "that path holds the raw reads" in explained

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

    def test_continue_nudge_only_fires_on_a_deferred_action(self):
        # A real answer ends the turn in every mode; only an announced-but-
        # untaken next step is fed back.
        for mode in (*MODES, None):
            assert continue_nudge_for(mode, "here is my answer") is None
            assert continue_nudge_for(mode, "Let me dig into the source.") is not None


class TestScriptPreview:
    """What the approval prompt — and now every recorded call — shows as the
    thing that would run."""

    @pytest.fixture
    def ctx(self, tmp_path):
        import types

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "align.sh"
        script.write_text("#!/usr/bin/env bash\nbwa mem ref.fa in.fq\n")
        return types.SimpleNamespace(scripts_dir=scripts_dir, workdir=tmp_path)

    def test_written_lines_are_the_script(self, ctx):
        preview = script_preview(
            "run_bash", {"content_lines": ["ls /data", "wc -l"]}, ctx
        )
        assert preview == "ls /data\nwc -l"

    def test_a_kept_script_being_run_is_shown(self, ctx):
        preview = script_preview(
            "start_background_script", {"name": "align", "args": "-t 4"}, ctx
        )
        assert "bwa mem ref.fa in.fq" in preview
        assert "align.sh (args: -t 4)" in preview

    def test_an_edit_is_previewed_as_its_diff(self, ctx):
        # edit_file's "what would run" is the change it would make; the whole
        # file is neither the call nor something the user can judge it by.
        preview = script_preview(
            "edit_file",
            {"path": "align.sh", "old_lines": ["bwa mem"], "new_lines": ["bwa-mem2 mem"]},
            ctx,
        )
        assert preview.splitlines() == ["- bwa mem", "+ bwa-mem2 mem"]

    def test_a_long_script_is_shown_whole(self, ctx):
        # The prompt scrolls and the chat box collapses, so there is no layout
        # reason to cut a script the user is being asked to approve.
        lines = [f"echo {n}" for n in range(500)]
        preview = script_preview("run_bash", {"content_lines": lines}, ctx)
        assert preview.splitlines() == lines

    def test_a_pathological_script_keeps_its_head_and_its_tail(self, ctx):
        # Some cap has to exist — the call is checkpointed with every turn —
        # but a heredoc's last lines are what say whether it was closed
        # properly, which is exactly what cutting the tail throws away.
        lines = [f"echo {n}" for n in range(SCRIPT_PREVIEW_CHARS)]
        preview = script_preview("run_bash", {"content_lines": lines}, ctx)
        assert len(preview) < SCRIPT_PREVIEW_CHARS + 200
        assert preview.startswith("echo 0\n")
        assert preview.endswith(f"echo {SCRIPT_PREVIEW_CHARS - 1}")
        assert "omitted" in preview

    def test_a_registry_key_pointing_at_data_is_not_a_script(self, ctx):
        # read_file, delete_file and friends take a registry key too. Their
        # target is data, not something that runs: dumping its contents would
        # neither describe the call nor be what the user needs to judge it.
        for tool in ("read_file", "delete_file", "move_file", "copy_file"):
            assert script_preview(tool, {"registry_key": "align"}, ctx) is None


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

    async def test_a_skip_reason_rides_back_with_it(self, tools):
        llm = FakeLLM(
            [tool_json("run_bash", content_lines=["echo hi"]), respond_json("ok")]
        )
        graph = make_graph(llm, tools, "manual")
        await run_turn(graph, session_id="s1", user_text="look around")
        result = await run_turn(
            graph,
            session_id="s1",
            resume=Command(
                resume={"approved": False, "reason": "run it on the login node"}
            ),
        )
        skipped = [m for m in result.messages if "SKIPPED" in m["content"]]
        assert skipped and "run it on the login node" in skipped[0]["content"]

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

    async def test_destructive_look_around_gates(self, tools):
        # run_bash gates conditionally, on what its script would do — a
        # benign look-around runs unattended, `rm -rf` pauses.
        llm = FakeLLM([tool_json("run_bash", content_lines=["rm -rf /data/x"])])
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="clean up")
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


class TestChecklist:
    """update_plan outlived plan mode: it tracks progress through multi-step
    work in every mode, and the checklist rides the checkpoint, not the
    prompt — a prompt-only one is forgotten as soon as compaction folds it."""

    async def test_checklist_survives_into_the_next_turn(self, tools):
        steps = [{"text": "only step", "done": False}]
        llm = FakeLLM(
            [
                tool_json("update_plan", steps=steps),
                respond_json("noted"),
                respond_json("still here"),
            ]
        )
        graph = make_graph(llm, tools, "auto")
        await run_turn(graph, session_id="s1", user_text="track this")
        result = await run_turn(graph, session_id="s1", user_text="thoughts?")
        assert result.plan == steps
        # the next turn's first decision sees the stored checklist re-injected
        assert "[ ] only step" in system_text(llm, 2)


class TestContinueNudge:
    """A turn never ends on a reply that only announces the next step."""

    async def test_a_real_answer_ends_the_turn(self, tools):
        llm = FakeLLM([respond_json("The command is `svirlpool cut`.")])
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert result.reply == "The command is `svirlpool cut`."
        assert len(llm.calls) == 1  # a genuine answer is not a deferred action

    async def test_deferred_action_is_nudged_then_taken(self, tools):
        # The exact failure this fixes: the model narrates "Let me dig deeper"
        # and stops. It is fed back and, on retry, acts.
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
        graph = make_graph(llm, tools, "auto")
        result = await run_turn(graph, session_id="s1", user_text="go")
        assert len(llm.calls) == MAX_CONTINUE_NUDGES + 1
        assert result.reply == "Let me look at c."


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
