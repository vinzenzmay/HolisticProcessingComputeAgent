"""Text for a HITL confirmation on a gated operation (§5.3, §3.5).

Two kinds of gate share this rendering, told apart by ``payload["kind"]``:

* ``destructive`` — the always-on gate for destructive operations.
* ``execution`` — manual mode showing a script or command before it
  runs; the user decides on the actual script text, not on a JSON blob.

The prompt is rendered inline in the chat column (see ``DecisionBar`` in
``app.py``), not as a full-screen modal — a decision waiting in one session
must not cover the other columns or block a session the user switched to.
Saying no has a second step, the box asking why (``approval_reason_hint``);
the graph resumes with a verdict either way, and these helpers only build the
text.
"""

from __future__ import annotations

from hpca.transcript import call_arguments


def approval_kind(payload: dict) -> str:
    return payload.get("kind", "destructive")


def approval_title(payload: dict) -> str:
    """The question, naming the tool it is about.

    The destructive heading names it too, which it did not have to while the
    details below opened with a ``Tool: <name>`` line. That line is gone — it
    said nothing the heading could not — so the heading is now the only place
    the name appears, and a destructive tool that cannot describe its own call
    would otherwise put bare arguments on screen with nothing saying what they
    were arguments to.
    """
    if approval_kind(payload) == "execution":
        return f"Run this — {payload.get('tool')}?"
    return f"Destructive operation — {payload.get('tool')}?"


def approval_hint(payload: dict) -> str:
    if approval_kind(payload) == "execution":
        return "(y) run script · (n) skip script"
    return "(y) approve · (n) deny"


def approval_reason_title(payload: dict) -> str:
    """Replaces the question once it has been answered with "no".

    Asks about the *next* attempt rather than for a justification: what goes
    in the box is read by the model, and "what should be different" is the one
    thing it can act on.
    """
    if approval_kind(payload) == "execution":
        return "Skipped — what should be different about the script?"
    return "Denied — what should be different?"


def approval_reason_hint() -> str:
    """The keys the box takes. The same for either gate — what differs is the
    title above it."""
    return "(enter) send · (esc) no reason · (shift+enter) new line"


def approval_details(payload: dict) -> str:
    """What this particular call does — and nothing that only the agent cares
    about.

    Pointedly *not* shown is ``payload["description"]``: that is the tool's
    schema blurb, written for the model to pick the tool by ("Run a registered
    script as a tracked background process for work that outlives this
    turn..."). It describes the tool in general and never this call, so on a
    prompt asking about one concrete action it is prompt content leaking onto
    the screen. The tool's name is left out for the same reason it is not
    repeated twice — ``approval_title`` already says it in the heading.

    When the tool gave ``details`` that is the whole answer: it is the tool's
    own account of *this* call with the real paths already resolved ("edit
    /home/me/run.sh"), and the arguments it was built from would only say the
    same thing again, less clearly. Without ``details`` the arguments are all
    there is to judge by, so they are rendered instead.

    Returning "" is a real outcome, not a failure: for ``run_bash`` the command
    in the block below is the entire call, and there is nothing left to say
    above it.

    The arguments are rendered by :func:`hpca.transcript.call_arguments`, the
    same function the chat's own call row uses. The prompt and the record of
    what ran have to describe the same call in the same words — the prompt is
    gone the moment it is answered, and the row is where the user goes back to
    read it.
    """
    details = payload.get("details")
    if details:  # resolved real paths (§5.3)
        return str(details).strip()
    return call_arguments(
        payload.get("arguments") or {}, has_script=bool(payload.get("script"))
    )


def approval_script(payload: dict) -> str | None:
    """The script/command preview, if this gate has one to show."""
    return payload.get("script")
