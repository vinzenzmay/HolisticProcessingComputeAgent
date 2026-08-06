"""Agent interaction modes: manual, auto, full-auto (§3.5).

A mode is a per-session dial on how much the agent may do unsupervised:

* ``manual`` — every script/command execution is shown to the user first,
  who runs or skips it. Consent moves into the approval prompt, so the model
  is told not to ask for permission in chat.
* ``auto`` — the agent works until the task is done, without narrating or
  asking; only genuinely destructive operations (§5.3) still gate.
* ``full-auto`` — auto with the destructive gate off too: nothing pauses
  for approval. The trash/backup layer (§5.3 recovery) is the only net.

There was a fourth, ``plan``: script tools withdrawn from the registry, a
checklist handed to the user through ``present_plan``, and approval switching
the session into manual or auto to execute it. It is gone — the shipped
``/plan`` skill does the job better, by grilling the user to a shared
understanding and writing a specs.md a fresh session can build from, without a
mode to enter and leave. Its structural guarantee had also never been as tight
as it read: ``run_bash`` stayed available and runs arbitrary bash, so a script
registered in an earlier turn could always be executed by naming its path.
Nothing migrates a session or settings file that still says ``plan`` — there
were none outside development when it was removed.

The mode reaches the graph as a callable (``mode_fn``) so a turn always reads
the session's current mode, and reaches the model as a per-round system
prompt suffix. Enforcement is structural where it matters: the model cannot
run a gated tool unapproved.

The ``update_plan`` checklist is NOT part of the retired mode: it tracks
progress through multi-step work in every mode, lives in the checkpointed
graph state, and is re-injected into the system prompt each round.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from hpca.agent.tools import Tool, ToolRegistry

MODES = ("manual", "auto", "full-auto")

# Tools that execute something on the system; in manual mode each call gates.
# create_script only writes into the scripts dir and is syntax-checked, so
# manual mode lets it through and gates the run instead — the approval then
# shows the finished script.
EXECUTION_TOOLS = frozenset(
    {"start_background_script", "run_bash", "submit_job"}
)

# How much of a script the preview keeps. It was 4000, which cut an ordinary
# hundred-line document mid-word: a user in manual mode was approving a script
# whose tail they could not see, and the record the chat keeps of what ran was
# missing the same part. Neither surface needs the cap for layout — the
# approval prompt scrolls, the chat box collapses — so what is left is the
# checkpoint, where every call is stored with the turn. Generous, not
# unbounded, and what remains is cut out of the MIDDLE (see ``_clip``).
SCRIPT_PREVIEW_CHARS = 40_000

# Tools whose call *is* a registered script: the file behind the key is the
# thing that would run, so it is what the user judges the call by. Every other
# tool taking a registry key points at data — a file to read, delete, move —
# and its contents are neither the call nor a script.
SCRIPT_FILE_TOOLS = frozenset({"start_background_script", "submit_job"})


def next_mode(mode: str) -> str:
    """The next mode in the cycle (manual → auto → full-auto → manual)."""
    if mode not in MODES:
        return MODES[0]
    return MODES[(MODES.index(mode) + 1) % len(MODES)]


def requires_execution_approval(mode: str | None, tool_name: str) -> bool:
    """Whether this mode shows every execution tool for approval first.

    Only manual mode does; auto and full-auto rely on the destructive-op gate
    (§5.3), which run_bash trips only when its script would actually destroy
    something.
    """
    return mode == "manual" and tool_name in EXECUTION_TOOLS


def destructive_approval_required(mode: str | None) -> bool:
    """Whether the destructive-op gate (§5.3) is active in this mode.

    Full-auto is the one deliberate exception to "human-in-the-loop for
    anything destructive": the user opted out by switching to it, and the
    trash/backup layer (§5.3 recovery) still stands behind every deletion.
    """
    return mode != "full-auto"


# ------------------------------------------------------------- prompt blocks
# Wording informed by what holds up in shipped agents (Hermes, Claude Code,
# Cline, Codex): a bare denial makes models re-propose the same command, so
# the skip message forbids retry AND rephrasing AND other routes to the same
# outcome; auto mode must say that asking is pointless, or the model checks
# in anyway; mode constraints are re-stated every round, never only once.

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

# Fed back (not persisted) when a bare reply only announces the next step.
# Every mode legitimately ends a turn on chat — that is how the agent answers
# or asks — so the nudge keeps both those doors open while refusing the false
# stop.
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

    Every mode ends on chat normally, so a reply is nudged only when it is a
    deferred action: an announced next step the model did not take.
    """
    if looks_like_deferred_action(text):
        return CONTINUE_NUDGE
    return None


MODE_GUIDANCE = {
    "manual": MANUAL_MODE_GUIDANCE,
    "auto": AUTO_MODE_GUIDANCE,
    "full-auto": FULL_AUTO_MODE_GUIDANCE,
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
        parts.append(
            f"Current plan checklist:\n{render_checklist(plan)}"
            f"\n{PLAN_EXECUTION_GUIDANCE}"
        )
    return "\n\n".join(parts)


def skipped_message(tool_name: str, reason: str = "") -> str:
    """Tool result fed back when the user skips an execution-gated call.

    Two opposite instructions, decided by whether the user said why. A bare
    refusal is final: nothing was given to work with, so another attempt is
    guessing, and guessing at what a user just refused is how a turn gets
    spent on three variants of the same rejected script. A refusal with a
    reason is the reverse — the reason is a correction, and acting on it is
    the whole point of having asked for it.
    """
    if not reason:
        return (
            f"[tool result] {tool_name}: SKIPPED — the user chose not to run "
            "this. It was NOT executed. Do not retry it, do not rephrase it, "
            "and do not attempt the same outcome via a different tool. Ask the "
            "user how to proceed."
        )
    return (
        f"[tool result] {tool_name}: SKIPPED — the user chose not to run this. "
        "It was NOT executed. They said why:\n"
        f"{reason}\n"
        "Treat that as the correction to make. Work it into a fixed version and "
        "put that up for approval — do not send back what was just refused. If "
        "the reason does not tell you enough to fix it, ask the user."
    )


def denied_message(tool_name: str, reason: str = "") -> str:
    """Tool result fed back when the user denies a destructive operation.

    The counterpart of ``skipped_message`` for the always-on gate (§5.3), and
    it reads the reason the same way: without one the operation is simply off
    the table, with one there is something to correct.
    """
    if not reason:
        return (
            f"[tool result] {tool_name}: DENIED by the user — "
            "the operation was not executed."
        )
    return (
        f"[tool result] {tool_name}: DENIED by the user — the operation was "
        "not executed. They said why:\n"
        f"{reason}\n"
        "Treat that as the correction to make. Propose an amended operation "
        "that answers it, or ask the user if the reason leaves you unsure — "
        "do not re-send what was just denied."
    )


def script_preview(tool_name: str, arguments: dict, ctx: Any) -> str | None:
    """The script or command behind a call: what would actually run, rather
    than a JSON blob.

    Shown in the approval prompt for a gated call, and recorded with every
    call so the chat can still show it once that prompt is answered (see
    ``hpca.transcript.call_text``).

    Best-effort and side-effect-free: it runs on every call, and again when a
    parked turn resumes and the graph re-runs gating, so it must stay a pure
    read and must never raise.

    run_bash lines are shown with their ``{key}`` references expanded. The
    point of the modal is that the user approves what will actually run, and
    `rm -rf {scratch}` hides exactly the part they need to check.

    For an edit the block is the diff, not the file: the change is the call.
    """
    from hpca.agent.builtin_tools import expand_keys

    try:
        if "content_lines" in arguments:  # run_bash, create_script
            lines, _ = expand_keys(arguments["content_lines"], ctx)
            return _clip("\n".join(lines))
        if tool_name == "edit_file":
            from hpca.agent.file_tools import edit_preview

            return _clip(edit_preview(arguments, ctx)) or None
        if tool_name not in SCRIPT_FILE_TOOLS:
            return None
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
    """Bound a preview by taking the middle out, never the end.

    A heredoc's last lines are what say whether it was closed properly, and a
    script's last command is usually the one that matters; a head-only clip
    throws away exactly the part worth checking. Whole lines on both sides, so
    the two halves still read as script.
    """
    if len(text) <= SCRIPT_PREVIEW_CHARS:
        return text
    half = SCRIPT_PREVIEW_CHARS // 2
    head = text[:half].rsplit("\n", 1)[0]
    tail = text[len(text) - half :].split("\n", 1)[-1]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [{omitted:,} characters omitted] ...\n{tail}"


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
