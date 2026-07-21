"""Text for a HITL confirmation on a gated operation (§5.3, §3.5).

Two kinds of gate share this rendering, told apart by ``payload["kind"]``:

* ``destructive`` — the always-on gate for destructive operations.
* ``execution`` — manual/plan mode showing a script or command before it
  runs; the user decides on the actual script text, not on a JSON blob.

The prompt is rendered inline in the chat column (see ``DecisionBar`` in
``app.py``), not as a full-screen modal — a decision waiting in one session
must not cover the other columns or block a session the user switched to.
The graph resumes with a bool either way; these helpers only build the text.
"""

from __future__ import annotations

import json


def approval_kind(payload: dict) -> str:
    return payload.get("kind", "destructive")


def approval_title(payload: dict) -> str:
    if approval_kind(payload) == "execution":
        return f"Run this — {payload.get('tool')}?"
    return "Destructive operation — approve?"


def approval_hint(payload: dict) -> str:
    if approval_kind(payload) == "execution":
        return "(y) run script · (n) skip script"
    return "(y) approve · (n) deny"


def approval_details(payload: dict) -> str:
    text = f"Tool: {payload.get('tool')}"
    # With a script shown below, the raw arguments would repeat it as a
    # JSON blob; without one they are all there is to judge the call by.
    if not payload.get("script"):
        text += f"\nArguments: {json.dumps(payload.get('arguments'), indent=2)}"
    description = payload.get("description", "")
    if description:
        text += f"\n{description}"
    details = payload.get("details")
    if details:  # resolved real paths (§5.3)
        text += f"\n\n{details}"
    return text


def approval_script(payload: dict) -> str | None:
    """The script/command preview, if this gate has one to show."""
    return payload.get("script")
