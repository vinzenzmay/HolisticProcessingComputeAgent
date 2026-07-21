"""Plan-mode handoff (§3.5).

Shown when a plan-mode turn ends with a checklist: the user reads the plan,
may edit it (it is a plain ``[ ]``/``[x]`` checklist in a text area), and
decides how to continue — execute on auto, execute step-by-step under manual
approval, or keep planning. The verdict is ``(mode, steps)`` or ``None`` for
"keep planning"; the transition itself is the app's job.

Like the approval gate, this renders inline in the chat column (see
``DecisionBar`` in ``app.py``) rather than as a modal that would cover the
other columns. These are the shared text/parse bits the bar embeds.
"""

from __future__ import annotations

from hpca.agent.modes import parse_checklist

PlanDecision = tuple[str, list[dict]] | None

PLAN_TITLE = "The agent proposes this plan"
# esc first (matches the esc/quit ordering); execute keys avoid ctrl+s
# (reserved: terminal XOFF) — step-by-step is ctrl+e.
PLAN_HINT = (
    "Edit freely ([x] marks a step done), then:\n"
    "esc  keep planning · ctrl+r  execute on auto · "
    "ctrl+e  execute step-by-step (each script asks first)"
)


def edited_plan_steps(text: str, fallback: list[dict]) -> list[dict]:
    """Parse the (possibly hand-edited) checklist the user is looking at.

    An emptied-out checklist is not a plan; keep what the agent wrote rather
    than executing nothing.
    """
    return parse_checklist(text) or fallback
