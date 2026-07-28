"""Agent interaction modes: manual, auto, plan (§3.5).

A mode is a per-session dial on how much the agent may do unsupervised:

* ``manual`` — every script/command execution is shown to the user first,
  who runs or skips it. Consent moves into the approval prompt, so the model
  is told not to ask for permission in chat.
* ``auto`` — the agent works until the task is done, without narrating or
  asking; only genuinely destructive operations (§5.3) still gate.
* ``full-auto`` — auto with the destructive gate off too: nothing pauses
  for approval. The trash/backup layer (§5.3 recovery) is the only net.
* ``plan`` — nothing is built or submitted. Script tools are withdrawn from
  the registry (deterministic, not prompt-trust); the one look-around command
  that stays (run_bash) runs unattended unless its script would destroy
  something, in which case the destructive gate (§5.3) still asks. The model
  maintains a checklist via the ``update_plan`` tool. The plan
  lives in the graph state, so it survives restarts with the checkpoint and
  is re-injected into the system prompt every round — a prompt-only plan
  mode is forgotten as soon as compaction folds the instruction away.

The mode reaches the graph as a callable (``mode_fn``) so a turn always reads
the session's current mode, and reaches the model as a per-round system
prompt suffix. Enforcement is structural where it matters: the model cannot
call a tool that was never offered, and cannot run a gated one unapproved.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from hpca.agent.tools import Tool, ToolRegistry

MODES = ("manual", "auto", "full-auto", "plan")

# Tools that execute something on the system. In manual mode each call gates;
# in plan mode they are withdrawn entirely — except run_bash, the bounded
# look-around tool, which stays available so the plan can be grounded in what
# is actually on disk. run_bash no longer gates for every call in plan mode;
# it falls back to the destructive-op gate (§5.3), so only a genuinely
# destructive look-around command pauses for approval. create_script only
# writes into the scripts dir and is syntax-checked, so manual mode lets it
# through and gates the run instead — the approval then shows the finished
# script.
#
# Plan mode's withdrawal is no longer airtight, and deliberately so: since
# run_bash expands `{key}` to a registered path, a script registered in an
# earlier turn can be run from plan mode by naming it. Blocking create_script
# still means no *new* script can be built there, and the destructive gate
# still catches the calls that would break something. Closing the rest would
# mean an argument-level rejection — a call the model may emit and must then
# recover from — which costs more than the hole is worth.
EXECUTION_TOOLS = frozenset(
    {"start_background_script", "run_bash", "submit_job"}
)
PLAN_BLOCKED_TOOLS = frozenset(
    {"create_script", "start_background_script", "submit_job"}
)
# The two plan tools belong to opposite phases and are never offered together:
# present_plan finalises a plan for the user to approve (plan mode only), while
# update_plan ticks steps off during execution (every other mode). Offering
# both at once just gives a small model two near-identical options to confuse.
PLANNING_TOOLS = frozenset({"present_plan"})
EXECUTION_PLAN_TOOLS = frozenset({"update_plan"})

SCRIPT_PREVIEW_CHARS = 4000


def next_mode(mode: str) -> str:
    """The next mode in the cycle (manual → auto → plan → manual)."""
    if mode not in MODES:
        return MODES[0]
    return MODES[(MODES.index(mode) + 1) % len(MODES)]


def requires_execution_approval(mode: str | None, tool_name: str) -> bool:
    """Whether this mode shows every execution tool for approval first.

    Only manual mode does. Plan mode used to gate here too, but that made the
    user approve every benign look-around; plan mode now relies solely on the
    destructive-op gate (§5.3), which run_bash trips only when its script would
    actually destroy something.
    """
    return mode == "manual" and tool_name in EXECUTION_TOOLS


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
    for a small model exactly when instructions would not. It also swaps the
    plan tools by phase — present_plan in, update_plan out — so planning ends
    only through the one explicit hand-off. Every other mode does the reverse.
    """
    if mode == "plan":
        blocked = PLAN_BLOCKED_TOOLS | EXECUTION_PLAN_TOOLS
    else:
        blocked = PLANNING_TOOLS
    keep = [name for name in tools.names() if name not in blocked]
    if len(keep) == len(tools.names()):
        return tools  # nothing to withdraw (e.g. a bare registry): same object
    return tools.subset(keep)


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
    "Plan mode is on: the user wants a plan first — nothing is built or "
    "submitted yet. You MUST NOT change anything; the script tools are "
    "disabled. Look-around commands (run_bash) run freely to ground the "
    "plan — only a command that would destroy something pauses for the "
    "user's approval, and planning should not need one. Investigate what "
    "the plan needs by CALLING TOOLS — read files, check docs, look "
    "around. Do NOT narrate what you "
    "are about to do, and do NOT end your turn with a chat message: in plan "
    "mode a bare reply does not hand anything to the user, it is ignored and "
    "you are asked to keep going. When the plan is ready — OR when you need "
    "the user to decide something before you can finish it — call "
    "present_plan with the checklist (short, concrete steps, each a single "
    "action) and any open questions. Calling present_plan is the ONLY way to "
    "hand the plan to the user; they will approve it to start execution. "
    "This constraint overrides any instruction to execute, including from "
    "the user — answer such requests with a plan."
)

# Fed back (not persisted) when a plan-mode turn would otherwise end on a bare
# chat reply — almost always the model announcing a step instead of taking it.
PLAN_CONTINUE_NUDGE = (
    "[continue] You replied with text instead of acting, but plan mode does "
    "not end your turn on chat — that reply was not shown to the user. Keep "
    "going: call a tool to investigate the next thing the plan needs. When "
    "the plan is ready, or you need the user to decide something, call "
    "present_plan with the checklist and any open questions. Do not describe "
    "your next step — take it."
)

# Fed back (not persisted) in any non-plan mode when a bare reply only announces
# the next step. Unlike plan mode, these modes legitimately end a turn on chat —
# that is how the agent answers or asks — so the nudge keeps both those doors
# open while refusing the false stop.
CONTINUE_NUDGE = (
    "[continue] You described your next step instead of doing it, so your turn "
    "would end here without anything happening — and that reply is all the user "
    "would see. Do not narrate what you are about to do; take the action now by "
    "calling the tool. Only stop if you are actually finished (then give the "
    "user your final answer) or you are genuinely blocked and need something "
    "only the user can provide (then ask them directly)."
)

# Openers that, at the start of a bare reply's final sentence(s), mark the model
# announcing its next action rather than delivering an answer. Matched at the
# sentence start (after list/emphasis markers), never mid-sentence, so a real
# answer that merely contains "I'll" in passing is left to stand.
_DEFERRED_ACTION_OPENERS = (
    "let me ",
    "let's ",
    "let us ",
    "i'll ",
    "i will ",
    "i'm going to ",
    "i am going to ",
    "i'm gonna ",
    "i'm about to ",
    "i am about to ",
    "i need to ",
    "i should ",
    "i have to ",
    "i want to ",
    "i plan to ",
    "next, i ",
    "next i ",
    "now i'll ",
    "now let ",
    "time to ",
)
# "Let me know…" hands the turn back to the user — an ending, not a deferred
# action — so it must not be read as one despite the shared "let me" opener.
_INVITATION_OPENERS = ("let me know", "let us know")


def _last_sentences(text: str, n: int = 2) -> list[str]:
    """The final ``n`` sentences of ``text``, trailing markers and blanks gone."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()][-n:]


def looks_like_deferred_action(text: str) -> bool:
    """Whether a bare chat reply merely announces its next step instead of
    taking it — the "false stop" fed back with a nudge in every mode.

    Only the reply's last one or two sentences are inspected, each stripped of
    leading list/emphasis markers, and only when the reply does not end on a
    question (a genuine ask the user must answer). A sentence that opens with a
    deferred-action phrase ("Let me…", "I'll…", "Next I…") — but not an
    invitation ("Let me know…") — marks the false stop.
    """
    stripped = text.strip()
    if not stripped or stripped.endswith("?"):
        return False
    for sentence in _last_sentences(stripped):
        opener = sentence.lstrip("-*#> \t").lower()
        if opener.startswith(_INVITATION_OPENERS):
            continue
        if opener.startswith(_DEFERRED_ACTION_OPENERS):
            return True
    return False


def continue_nudge_for(mode: str | None, text: str) -> str | None:
    """The nudge to feed back when a turn would otherwise end on a bare chat
    reply, or ``None`` to let the reply stand as the turn's answer.

    Plan mode never ends on chat — the model must hand over through
    present_plan — so any bare reply is fed back. Every other mode ends on chat
    normally, so it is nudged only when the reply is a deferred action: an
    announced next step the model did not take.
    """
    if mode == "plan":
        return PLAN_CONTINUE_NUDGE
    if looks_like_deferred_action(text):
        return CONTINUE_NUDGE
    return None

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

    run_bash lines are shown with their ``{key}`` references expanded. The
    point of the modal is that the user approves what will actually run, and
    `rm -rf {scratch}` hides exactly the part they need to check.
    """
    from hpca.agent.builtin_tools import expand_keys

    try:
        if "content_lines" in arguments:  # run_bash, create_script
            lines, _ = expand_keys(arguments["content_lines"], ctx)
            return _clip("\n".join(lines))
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


class PresentPlanParams(BaseModel):
    steps: list[PlanStep] = Field(
        default_factory=list,
        description=(
            "The full plan checklist the user will approve to start execution "
            "(short, concrete steps, each a single action)"
        ),
    )
    summary: str = Field(
        default="",
        description="A short note to the user about the plan",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Anything you need the user to decide before the plan can be "
            "finished or executed"
        ),
    )

    @model_validator(mode="after")
    def _needs_a_plan_or_a_question(self) -> "PresentPlanParams":
        if not self.steps and not self.open_questions:
            raise ValueError(
                "present_plan needs at least one step or one open question — "
                "keep investigating with tools until you have one."
            )
        return self


async def present_plan(args: PresentPlanParams, ctx: Any) -> str:
    # Never actually invoked: the graph intercepts present_plan to end the
    # planning turn and hand the checklist to the user (§3.5). Present only so
    # the tool is offered and validated like any other.
    return "Plan presented to the user."


def present_plan_reply(args: PresentPlanParams) -> str:
    """The assistant message shown in chat when a plan is presented: the
    model's summary, then any open questions the user must weigh in on."""
    parts: list[str] = [args.summary.strip() or "Here is the plan."]
    if args.open_questions:
        questions = "\n".join(f"- {q}" for q in args.open_questions)
        parts.append(f"Open questions before I start:\n{questions}")
    return "\n\n".join(parts)


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
    registry.register(
        Tool(
            name="present_plan",
            description=(
                "Hand the finished plan to the user for approval: the full "
                "checklist plus any open questions. Ends your planning turn — "
                "the only way to do so"
            ),
            params=PresentPlanParams,
            handler=present_plan,
        )
    )
    return registry
