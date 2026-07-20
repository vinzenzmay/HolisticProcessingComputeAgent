"""Agent interaction modes: manual, auto, plan (§3.5).

A mode is a per-session dial on how much the agent may do unsupervised:

* ``manual`` — every script/command execution is shown to the user first,
  who runs or skips it. Consent moves into the approval prompt, so the model
  is told not to ask for permission in chat.
* ``auto`` — the agent works until the task is done, without narrating or
  asking; only genuinely destructive operations (§5.3) still gate.
* ``full-auto`` — auto with the destructive gate off too: nothing pauses
  for approval. The trash/backup layer (§5.3 recovery) is the only net.
* ``plan`` — nothing executes. Script tools are withdrawn from the registry
  (deterministic, not prompt-trust), look-around commands gate like manual,
  and the model maintains a checklist via the ``update_plan`` tool. The plan
  lives in the graph state, so it survives restarts with the checkpoint and
  is re-injected into the system prompt every round — a prompt-only plan
  mode is forgotten as soon as compaction folds the instruction away.

The mode reaches the graph as a callable (``mode_fn``) so a turn always reads
the session's current mode, and reaches the model as a per-round system
prompt suffix. Enforcement is structural where it matters: the model cannot
call a tool that was never offered, and cannot run a gated one unapproved.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from hpca.agent.tools import Tool, ToolRegistry

MODES = ("manual", "auto", "full-auto", "plan")

# Tools that execute something on the system. In manual mode each call gates;
# in plan mode they are withdrawn entirely — except run_bash, the bounded
# look-around tool, which stays available but gates so the plan can still be
# grounded in what is actually on disk. create_script only writes into the
# scripts dir and is syntax-checked, so manual mode lets it through and gates
# the run instead — the approval then shows the finished script.
EXECUTION_TOOLS = frozenset({"run_script", "start_script", "run_bash", "submit_job"})
PLAN_BLOCKED_TOOLS = frozenset(
    {"create_script", "run_script", "start_script", "submit_job"}
)

SCRIPT_PREVIEW_CHARS = 4000


def next_mode(mode: str) -> str:
    """The next mode in the cycle (manual → auto → plan → manual)."""
    if mode not in MODES:
        return MODES[0]
    return MODES[(MODES.index(mode) + 1) % len(MODES)]


def requires_execution_approval(mode: str | None, tool_name: str) -> bool:
    """Whether this mode gates the call beyond the destructive-op gate."""
    return mode in ("manual", "plan") and tool_name in EXECUTION_TOOLS


def destructive_approval_required(mode: str | None) -> bool:
    """Whether the destructive-op gate (§5.3) is active in this mode.

    Full-auto is the one deliberate exception to "human-in-the-loop for
    anything destructive": the user opted out by switching to it, and the
    trash/backup layer (§5.3 recovery) still stands behind every deletion.
    """
    return mode != "full-auto"


def tools_for_mode(tools: ToolRegistry, mode: str | None) -> ToolRegistry:
    """The registry as offered to the model this round.

    Plan mode withdraws execution tools instead of forbidding them in prose:
    a tool the model was never offered is a tool it cannot call, which holds
    for a small model exactly when instructions would not.
    """
    if mode != "plan":
        return tools
    return tools.subset(
        [name for name in tools.names() if name not in PLAN_BLOCKED_TOOLS]
    )


# ------------------------------------------------------------- prompt blocks
# Wording informed by what holds up in shipped agents (Hermes, Claude Code,
# Cline, Codex): a bare denial makes models re-propose the same command, so
# the skip message forbids retry AND rephrasing AND other routes to the same
# outcome; auto mode must say that asking is pointless, or the model checks
# in anyway; plan constraints are re-stated every round, never only once.

MANUAL_MODE_GUIDANCE = (
    "Manual mode is on: every script or command you run is first shown to "
    "the user, who runs or skips it. Do not ask for permission in chat — "
    "decide what to run and call the tool; the approval prompt IS the "
    "user's consent. If a tool result says SKIPPED, the user declined that "
    "action: do not retry it, do not rephrase it, and do not attempt the "
    "same outcome another way — ask the user how they would like to proceed."
)

AUTO_MODE_GUIDANCE = (
    "Auto mode is on: work autonomously until the user's request is "
    "complete. Do not announce what you are about to do, and do not ask "
    "for permission or confirmation — no answer will come until you "
    "finish, so asking only stalls the task. Do the work, verify it, then "
    "report what you did and what you found. Truly destructive operations "
    "(delete, overwrite, kill, cancel) still pause for the user's built-in "
    "approval, so you never need to ask about those in chat either. Ask a "
    "question only when the task cannot proceed without information that "
    "only the user has."
)

FULL_AUTO_MODE_GUIDANCE = (
    "Full-auto mode is on: work autonomously until the user's request is "
    "complete, and nothing pauses for approval — not even destructive "
    "operations. Do not announce what you are about to do, and do not ask "
    "for permission or confirmation — no answer will come until you "
    "finish, so asking only stalls the task. Because nothing is "
    "double-checked by the user, be careful on your own: verify paths "
    "before you delete, overwrite, or move over something, prefer "
    "recoverable operations, and list every destructive action you took "
    "in your final report. Ask a question only when the task cannot "
    "proceed without information that only the user has."
)

PLAN_MODE_GUIDANCE = (
    "Plan mode is on: the user wants a plan first — nothing is executed "
    "yet. You MUST NOT run scripts or change anything; script tools are "
    "disabled, and a look-around command (run_bash) runs only with the "
    "user's explicit approval. Investigate what the plan needs (read "
    "files, check docs) and ask clarifying questions rather than assume. "
    "Maintain the plan with the update_plan tool: a checklist of short, "
    "concrete steps, each a single action. Call update_plan whenever the "
    "plan changes, then summarise the plan and its open questions in your "
    "reply. The user will approve the plan to start execution; until "
    "then, keep refining it. This constraint overrides any instruction to "
    "execute, including from the user — answer such requests with the plan."
)

MODE_GUIDANCE = {
    "manual": MANUAL_MODE_GUIDANCE,
    "auto": AUTO_MODE_GUIDANCE,
    "full-auto": FULL_AUTO_MODE_GUIDANCE,
    "plan": PLAN_MODE_GUIDANCE,
}

PLAN_EXECUTION_GUIDANCE = (
    "Work through the unfinished steps in order. After you finish a step, "
    "call update_plan with the full updated checklist ([x] = done) before "
    "moving on."
)


def render_checklist(steps: list[dict]) -> str:
    """The plan as the model and the user see it: one ``[x]/[ ]`` line per step."""
    return "\n".join(
        f"[{'x' if step.get('done') else ' '}] {step.get('text', '')}"
        for step in steps
    )


def parse_checklist(text: str) -> list[dict]:
    """Checklist lines back into steps — lenient, the user edited this by hand.

    ``[x]``/``[X]`` marks a step done; ``[ ]``, a bare ``- `` bullet, or any
    other non-empty line is an open step. Empty lines are ignored.
    """
    steps: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        if not line:
            continue
        done = False
        lowered = line.lower()
        if lowered.startswith("[x]"):
            done = True
            line = line[3:].strip()
        elif line.startswith("[ ]"):
            line = line[3:].strip()
        elif line.startswith("[]"):
            line = line[2:].strip()
        if line:
            steps.append({"text": line, "done": done})
    return steps


def mode_prompt_suffix(mode: str | None, plan: list[dict] | None) -> str:
    """What this round's system prompt appends: mode rules, then the plan.

    Rendered per round rather than once, so the constraint survives context
    compaction and a mid-session mode switch takes effect immediately.
    """
    parts: list[str] = []
    guidance = MODE_GUIDANCE.get(mode or "")
    if guidance:
        parts.append(guidance)
    if plan:
        block = f"Current plan checklist:\n{render_checklist(plan)}"
        if mode in ("manual", "auto", "full-auto"):
            block += f"\n{PLAN_EXECUTION_GUIDANCE}"
        parts.append(block)
    return "\n\n".join(parts)


def skipped_message(tool_name: str) -> str:
    """Tool result fed back when the user skips an execution-gated call."""
    return (
        f"[tool result] {tool_name}: SKIPPED — the user chose not to run "
        "this. It was NOT executed. Do not retry it, do not rephrase it, "
        "and do not attempt the same outcome via a different tool. Ask the "
        "user how to proceed."
    )


def kickoff_message(mode: str) -> str:
    """The event message that starts execution after the user approves a plan."""
    return (
        f"[plan approved] The user approved the plan and switched to {mode} "
        "mode. Begin executing now: work through the unfinished checklist "
        "steps in order, and after finishing each step call update_plan "
        "with the full updated checklist."
    )


def script_preview(tool_name: str, arguments: dict, ctx: Any) -> str | None:
    """The script/command behind an execution-gated call, for the approval
    modal — the user decides on what would actually run, not on a JSON blob.

    Best-effort and side-effect-free: the graph re-runs gating when a parked
    turn resumes, so this must stay a pure read.
    """
    try:
        if "content_lines" in arguments:  # run_bash, create_script
            return _clip("\n".join(arguments["content_lines"]))
        key = arguments.get("registry_key")
        if key and ctx is not None and getattr(ctx, "registry", None) is not None:
            path = ctx.registry.resolve(key)
            text = path.read_text(errors="replace")
            args = arguments.get("args")
            header = f"# {path.name}" + (f" (args: {args})" if args else "")
            return _clip(f"{header}\n{text}")
    except Exception:
        return None
    return None


def _clip(text: str) -> str:
    if len(text) <= SCRIPT_PREVIEW_CHARS:
        return text
    return text[:SCRIPT_PREVIEW_CHARS] + "\n... [clipped]"


# ------------------------------------------------------------- update_plan


class PlanStep(BaseModel):
    text: str = Field(description="One concrete step")
    done: bool = Field(default=False, description="True once the step is finished")


class UpdatePlanParams(BaseModel):
    steps: list[PlanStep] = Field(
        min_length=1,
        description="The full plan checklist; replaces the previous plan",
    )


async def update_plan(args: UpdatePlanParams, ctx: Any) -> str:
    # The graph writes the checklist into the session state (it is
    # checkpointed there); this result only tells the model it worked.
    done = sum(1 for step in args.steps if step.done)
    return f"Plan updated: {len(args.steps)} steps, {done} done."


def add_plan_tool(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="update_plan",
            description=(
                "Create or update the plan checklist (send the FULL list of "
                "steps; it replaces the previous plan)"
            ),
            params=UpdatePlanParams,
            handler=update_plan,
        )
    )
    return registry
